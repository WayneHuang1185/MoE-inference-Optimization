#!/usr/bin/env bash
# Causal running-mean predictor for delta inertia. Reuses existing dumps; no inference.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

INPUT_BATCH="${INPUT_BATCH:?set INPUT_BATCH to a previous batch results dir}"
OUT_ROOT="${OUT_ROOT:-experiments/gemma4_bottleneck/results/running_mean_inertia_$(date +%Y%m%d_%H%M%S)}"
PY="${PY:-experiments/gemma4_bottleneck/measure_running_mean_inertia.py}"
IMAGE="${IMAGE:-localhost/gemma4-ram-bench:24.04}"
N_LAYERS="${N_LAYERS:-30}"
KS="${KS:-1,2,4,8,16,32}"
HIDDEN="${HIDDEN:-2816}"

INPUT_MANIFEST="$INPUT_BATCH/manifest.csv"
[[ -f "$INPUT_MANIFEST" ]] || { echo "missing $INPUT_MANIFEST" >&2; exit 1; }

mkdir -p "$OUT_ROOT"
printf 'INPUT_BATCH=%s\nKS=%s\nHIDDEN=%s\nN_LAYERS=%s\n' \
  "$INPUT_BATCH" "$KS" "$HIDDEN" "$N_LAYERS" > "$OUT_ROOT/meta"

podman run --rm \
  -v "$ROOT_DIR:/workspace" -w /workspace \
  "$IMAGE" \
  python3 "$PY" \
    --manifest "$INPUT_MANIFEST" \
    --output "$OUT_ROOT/running_mean_inertia.csv" \
    --hidden "$HIDDEN" \
    --n-layers "$N_LAYERS" \
    --ks "$KS" \
  2>&1 | tee "$OUT_ROOT/run.log"

echo "wrote: $OUT_ROOT"
