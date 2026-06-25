#!/usr/bin/env bash
# Fast variant of run_global_predictor_dump_pilot.sh:
#   * Runs llama-server ONCE inside a long-lived podman container.
#   * Feeds every prompt through /completion via the Python driver.
#   * Moves per-prompt dump files between requests so the existing
#     prepare_global_predictor_dataset.py consolidator works unchanged.
#
# Eliminates the per-prompt cold-start (podman + model mmap) overhead, dropping
# 100 prompts from a few hours down to a few minutes.
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
N_PREDICT="${N_PREDICT:-1}"
THREADS="${THREADS:-$(nproc 2>/dev/null || echo 32)}"
PORT="${PORT:-8080}"
CONTAINER_NAME="${CONTAINER_NAME:-gemma4-fast-server}"
MEMORY="${MEMORY:-48g}"
MEMORY_SWAP="${MEMORY_SWAP:-48g}"

N_LAYERS="${N_LAYERS:-30}"
N_EXPERTS="${N_EXPERTS:-128}"
TOP_K="${TOP_K:-8}"
HIDDEN="${HIDDEN:-2816}"
TOKENIZE_BIN="${TOKENIZE_BIN:-llama.cpp/build/bin/llama-tokenize}"

mkdir -p "$BATCH_DIR"

echo "[fast] OUT_ROOT    = $OUT_ROOT"
echo "[fast] BATCH_DIR   = $BATCH_DIR"
echo "[fast] DATASET_DIR = $DATASET_DIR"
echo "[fast] MAX_PROMPTS = $MAX_PROMPTS  CTX_SIZE=$CTX_SIZE  THREADS=$THREADS"

echo "[fast] step 1: drive prompts through one long-lived server (in-container)"
podman run --rm \
  --memory="$MEMORY" --memory-swap="$MEMORY_SWAP" \
  -v "$ROOT_DIR:/workspace" -w /workspace \
  "$IMAGE" \
  python3 experiments/gemma4_bottleneck/drive_global_predictor_fast.py \
    --prompt-dir   "$PROMPT_DIR" \
    --out-root     "$BATCH_DIR" \
    --model        "$MODEL" \
    --llama-server /workspace/llama.cpp/build/bin/llama-server \
    --port         "$PORT" \
    --threads      "$THREADS" \
    --ctx-size     "$CTX_SIZE" \
    --n-predict    "$N_PREDICT" \
    --max-prompts  "$MAX_PROMPTS"

echo "[fast] step 2: consolidate dumps -> per-prompt .npz"
podman run --rm \
  -v "$ROOT_DIR:/workspace" -w /workspace \
  "$IMAGE" \
  python3 experiments/gemma4_bottleneck/prepare_global_predictor_dataset.py \
    --batch-dir    "$BATCH_DIR" \
    --out-dir      "$DATASET_DIR" \
    --model        "$MODEL" \
    --tokenize-bin "$TOKENIZE_BIN" \
    --layers       "$N_LAYERS" \
    --experts      "$N_EXPERTS" \
    --top-k        "$TOP_K" \
    --hidden       "$HIDDEN"

echo "[fast] step 3: footprint"
RAW_BYTES="$(du -sb "$BATCH_DIR" 2>/dev/null | awk '{print $1}')"
DATASET_BYTES="$(du -sb "$DATASET_DIR" 2>/dev/null | awk '{print $1}')"
RAW_MB="$(python3 -c "print(f'{$RAW_BYTES/1024/1024:.1f}')")"
DATASET_MB="$(python3 -c "print(f'{$DATASET_BYTES/1024/1024:.1f}')")"
cat > "$OUT_ROOT/FOOTPRINT.md" <<EOF
# Pilot footprint (fast variant)

- raw activation dumps:           **${RAW_MB} MB** (\`$BATCH_DIR\`)
- consolidated dataset:           **${DATASET_MB} MB** (\`$DATASET_DIR\`)

Raw dumps can be removed once \`$DATASET_DIR\` is verified:
\`\`\`bash
rm -rf "$BATCH_DIR"
\`\`\`
EOF

echo "[fast] done."
echo "  raw  = $RAW_MB MB at $BATCH_DIR"
echo "  data = $DATASET_MB MB at $DATASET_DIR"
echo "  report = $DATASET_DIR/REPORT.md"
