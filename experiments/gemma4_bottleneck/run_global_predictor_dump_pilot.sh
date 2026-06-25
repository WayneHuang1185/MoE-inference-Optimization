#!/usr/bin/env bash
# Pilot data-collection run for the ExpertFlow-style global MoE-router predictor.
#
# Pipeline (Local -> Sync -> Remote):
#   1. (local)  rsync this repo to nthu-cs
#   2. (remote) ssh nthu-cs and run this script inside the gemma4-ram-bench
#               container (it dispatches podman runs by itself)
#   3. (remote) results land in experiments/gemma4_bottleneck/results/global_predictor_pilot_<ts>/
#
# All inference happens inside podman (CLAUDE.md §3). This script is a thin
# wrapper around run_router_prediction_batch.sh + prepare_global_predictor_dataset.py
# — it does NOT modify llama.cpp or any inference path (CLAUDE.md §1).
#
# Defaults are tuned for the pilot:
#   MAX_PROMPTS=100, N_PREDICT=1 (prefill-only labels), greedy implicit via
#   llama-server default benchmark settings.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-experiments/gemma4_bottleneck/results/global_predictor_pilot_${TS}}"
BATCH_DIR="${BATCH_DIR:-$OUT_ROOT/router_prediction_batch}"
DATASET_DIR="${DATASET_DIR:-$OUT_ROOT/dataset}"
MODEL="${MODEL:-models/gemma4-26B.gguf}"
PROMPT_DIR="${PROMPT_DIR:-experiments/gemma4_bottleneck/router_prediction_prompts}"
IMAGE="${IMAGE:-localhost/gemma4-ram-bench:24.04}"

MAX_PROMPTS="${MAX_PROMPTS:-100}"
CTX_SIZE="${CTX_SIZE:-8192}"
N_PREDICT="${N_PREDICT:-1}"          # prefill + 1 token; we only consume the prefill pass
RUNS="${RUNS:-1}"
WARMUP_RUNS="${WARMUP_RUNS:-0}"
THREADS="${THREADS:-$(nproc 2>/dev/null || echo 4)}"
# Relax the cgroup memory cap inherited from run_router_prediction_batch.sh
# (12g default was designed for RAM-bottleneck experiments). For data collection
# we want the 16 GB model fully resident, so allocate enough headroom for the
# model + KV cache + compute buffer + Python tooling.
MEMORY="${MEMORY:-48g}"
MEMORY_SWAP="${MEMORY_SWAP:-48g}"

N_LAYERS="${N_LAYERS:-30}"
N_EXPERTS="${N_EXPERTS:-128}"
TOP_K="${TOP_K:-8}"
HIDDEN="${HIDDEN:-2816}"

TOKENIZE_BIN="${TOKENIZE_BIN:-llama.cpp/build/bin/llama-tokenize}"

mkdir -p "$OUT_ROOT"

echo "[pilot] OUT_ROOT     = $OUT_ROOT"
echo "[pilot] BATCH_DIR    = $BATCH_DIR"
echo "[pilot] DATASET_DIR  = $DATASET_DIR"
echo "[pilot] MAX_PROMPTS  = $MAX_PROMPTS  N_PREDICT=$N_PREDICT  CTX_SIZE=$CTX_SIZE"

# --- Step 1: batch activation dump ----------------------------------------
echo "[pilot] step 1: batch activation dump via run_router_prediction_batch.sh"
PROMPT_DIR="$PROMPT_DIR" \
OUT_ROOT="$BATCH_DIR" \
MODEL="$MODEL" \
IMAGE="$IMAGE" \
THREADS="$THREADS" \
CTX_SIZE="$CTX_SIZE" \
N_PREDICT="$N_PREDICT" \
RUNS="$RUNS" \
WARMUP_RUNS="$WARMUP_RUNS" \
MAX_PROMPTS="$MAX_PROMPTS" \
N_LAYERS="$N_LAYERS" \
TRUE_K="$TOP_K" \
MEMORY="$MEMORY" \
MEMORY_SWAP="$MEMORY_SWAP" \
EXCLUDE_EDGE_LAYERS=0 \
bash experiments/gemma4_bottleneck/run_router_prediction_batch.sh

# --- Step 2: consolidate dumps into per-prompt npz ------------------------
echo "[pilot] step 2: consolidate dumps -> per-prompt .npz"
podman run --rm \
  -v "$ROOT_DIR:/workspace" -w /workspace \
  "$IMAGE" \
  python3 experiments/gemma4_bottleneck/prepare_global_predictor_dataset.py \
    --batch-dir   "$BATCH_DIR" \
    --out-dir     "$DATASET_DIR" \
    --model       "$MODEL" \
    --tokenize-bin "$TOKENIZE_BIN" \
    --layers      "$N_LAYERS" \
    --experts     "$N_EXPERTS" \
    --top-k       "$TOP_K" \
    --hidden      "$HIDDEN"

# --- Step 3: raw-dump footprint summary -----------------------------------
echo "[pilot] step 3: raw dump footprint"
RAW_BYTES="$(du -sb "$BATCH_DIR" 2>/dev/null | awk '{print $1}')"
DATASET_BYTES="$(du -sb "$DATASET_DIR" 2>/dev/null | awk '{print $1}')"
RAW_MB="$(python3 -c "print(f\"{$RAW_BYTES/1024/1024:.1f}\")")"
DATASET_MB="$(python3 -c "print(f\"{$DATASET_BYTES/1024/1024:.1f}\")")"

cat > "$OUT_ROOT/FOOTPRINT.md" <<EOF
# Pilot footprint

- raw activation dump (batch dir):  **${RAW_MB} MB** (\`$BATCH_DIR\`)
- consolidated dataset (npz dir):   **${DATASET_MB} MB** (\`$DATASET_DIR\`)

The raw dump can be removed once \`$DATASET_DIR\` is verified:
\`\`\`bash
rm -rf "$BATCH_DIR"
\`\`\`
EOF

echo "[pilot] done. summary:"
echo "  raw  = $RAW_MB MB at $BATCH_DIR"
echo "  data = $DATASET_MB MB at $DATASET_DIR"
echo "  report = $DATASET_DIR/REPORT.md"
