#!/usr/bin/env bash
# Run delta-inertia analysis over an existing batch of activation dumps.
# Reads attn_out-{i} pairs from dumps; pools across prompts; reports per-layer
# variance ratios. No new inference.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

INPUT_BATCH="${INPUT_BATCH:?set INPUT_BATCH to a previous batch results dir}"
OUT_ROOT="${OUT_ROOT:-experiments/gemma4_bottleneck/results/delta_inertia_$(date +%Y%m%d_%H%M%S)}"
PY="${PY:-experiments/gemma4_bottleneck/measure_delta_inertia.py}"
IMAGE="${IMAGE:-localhost/gemma4-ram-bench:24.04}"
N_LAYERS="${N_LAYERS:-30}"
K_NN="${K_NN:-8}"
MAX_QUERIES="${MAX_QUERIES:-2000}"
REGIMES="${REGIMES:-prefill,decode}"
HIDDEN="${HIDDEN:-2816}"

INPUT_MANIFEST="$INPUT_BATCH/manifest.csv"
[[ -f "$INPUT_MANIFEST" ]] || { echo "missing $INPUT_MANIFEST" >&2; exit 1; }

mkdir -p "$OUT_ROOT"
printf 'INPUT_BATCH=%s\nK_NN=%s\nMAX_QUERIES=%s\nREGIMES=%s\nHIDDEN=%s\nN_LAYERS=%s\n' \
  "$INPUT_BATCH" "$K_NN" "$MAX_QUERIES" "$REGIMES" "$HIDDEN" "$N_LAYERS" \
  > "$OUT_ROOT/meta"

podman run --rm \
  -v "$ROOT_DIR:/workspace" -w /workspace \
  "$IMAGE" \
  python3 "$PY" \
    --manifest "$INPUT_MANIFEST" \
    --output "$OUT_ROOT/delta_inertia.csv" \
    --hidden "$HIDDEN" \
    --n-layers "$N_LAYERS" \
    --k-nn "$K_NN" \
    --max-queries "$MAX_QUERIES" \
    --regimes "$REGIMES" \
  2>&1 | tee "$OUT_ROOT/run.log"

echo "wrote: $OUT_ROOT"
