#!/usr/bin/env bash
# Train per-layer-pair linear ridge predictor for delta = attn_out_(i+1) - attn_out_i.
# Hold-out by prompt: last N prompts (by prompt_id) go to eval. No new inference.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

INPUT_BATCH="${INPUT_BATCH:?set INPUT_BATCH to a previous batch results dir}"
OUT_ROOT="${OUT_ROOT:-experiments/gemma4_bottleneck/results/linear_predictor_$(date +%Y%m%d_%H%M%S)}"
PY="${PY:-experiments/gemma4_bottleneck/train_linear_predictor.py}"
IMAGE="${IMAGE:-localhost/gemma4-ram-bench:24.04}"
N_LAYERS="${N_LAYERS:-30}"
HIDDEN="${HIDDEN:-2816}"
EVAL_PROMPTS="${EVAL_PROMPTS:-4}"
LAMBDAS="${LAMBDAS:-1,10,100,1000,10000,100000}"

INPUT_MANIFEST="$INPUT_BATCH/manifest.csv"
[[ -f "$INPUT_MANIFEST" ]] || { echo "missing $INPUT_MANIFEST" >&2; exit 1; }

mkdir -p "$OUT_ROOT"
printf 'INPUT_BATCH=%s\nEVAL_PROMPTS=%s\nLAMBDAS=%s\nHIDDEN=%s\nN_LAYERS=%s\n' \
  "$INPUT_BATCH" "$EVAL_PROMPTS" "$LAMBDAS" "$HIDDEN" "$N_LAYERS" \
  > "$OUT_ROOT/meta"

podman run --rm \
  -v "$ROOT_DIR:/workspace" -w /workspace \
  "$IMAGE" \
  python3 "$PY" \
    --manifest "$INPUT_MANIFEST" \
    --output "$OUT_ROOT/linear_predictor.csv" \
    --hidden "$HIDDEN" \
    --n-layers "$N_LAYERS" \
    --eval-prompts "$EVAL_PROMPTS" \
    --lambdas "$LAMBDAS" \
  2>&1 | tee "$OUT_ROOT/run.log"

echo "wrote: $OUT_ROOT"
