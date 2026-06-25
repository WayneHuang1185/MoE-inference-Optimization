#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

INPUT_BATCH="${INPUT_BATCH:?set INPUT_BATCH}"
OUT_ROOT="${OUT_ROOT:-experiments/gemma4_bottleneck/results/chained_cosine_$(date +%Y%m%d_%H%M%S)}"
PY="${PY:-experiments/gemma4_bottleneck/eval_chained_cosine.py}"
IMAGE="${IMAGE:-localhost/gemma4-ram-bench:24.04}"
N_LAYERS="${N_LAYERS:-30}"
HIDDEN="${HIDDEN:-2816}"
N_TRAIN="${N_TRAIN:-18}"
N_VAL="${N_VAL:-9}"
N_TEST="${N_TEST:-9}"
LAMBDA="${LAMBDA:-1000}"
ANCHORS="${ANCHORS:-0,2,4,8,12,16,20}"

mkdir -p "$OUT_ROOT"
printf 'INPUT_BATCH=%s\nANCHORS=%s\n' "$INPUT_BATCH" "$ANCHORS" > "$OUT_ROOT/meta"

podman run --rm -v "$ROOT_DIR:/workspace" -w /workspace "$IMAGE" \
  python3 "$PY" \
    --manifest "$INPUT_BATCH/manifest.csv" \
    --output "$OUT_ROOT/cosine.csv" \
    --hidden "$HIDDEN" --n-layers "$N_LAYERS" \
    --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" \
    --lambda-ridge "$LAMBDA" --anchors "$ANCHORS" \
  2>&1 | tee "$OUT_ROOT/run.log"

echo "wrote: $OUT_ROOT"
