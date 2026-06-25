#!/usr/bin/env bash
# Evaluate decode-token top-k precision by MoE layer inside the RPP Docker image.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DOCKER_BIN="${DOCKER_BIN:-docker}"
IMAGE="${IMAGE:-localhost/gemma4-rpp-train:cpu}"
BUILD_IMAGE="${BUILD_IMAGE:-0}"
DATA_ROOT="${DATA_ROOT:-dataset/prompt1000/router_label_npz/npz}"
CHECKPOINT="${CHECKPOINT:-experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/checkpoint_best.pt}"
CONFIG="${CONFIG:-experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/config.json}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-experiments/gemma4_global_predictor}"
STATISTICS_DIR="${STATISTICS_DIR:-$OUT_ROOT/statistics/decode_layer_precision_${RUN_TIMESTAMP}}"
FIGURES_DIR="${FIGURES_DIR:-$OUT_ROOT/figures/decode_layer_precision_${RUN_TIMESTAMP}}"

if [[ "$BUILD_IMAGE" == "1" ]]; then
  "$DOCKER_BIN" build \
    -f experiments/gemma4_global_predictor/Dockerfile.rpp_torch \
    -t "$IMAGE" .
fi

mkdir -p "$STATISTICS_DIR" "$FIGURES_DIR"

"$DOCKER_BIN" run --rm \
  --shm-size "${SHM_SIZE:-64m}" \
  -v "$ROOT_DIR:/workspace" \
  -w /workspace \
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
  -e MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS:-4}}" \
  -e OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS:-4}}" \
  "$IMAGE" \
  python3 -m experiments.gemma4_global_predictor.eval_decode_layer_precision \
    --data-root "$DATA_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --out-dir "$STATISTICS_DIR" \
    --figures-dir "$FIGURES_DIR" \
    --max-samples "${MAX_SAMPLES:-1000}" \
    --decode-tokens "${DECODE_TOKENS:-5}" \
    --decode-window "${DECODE_WINDOW:-first}" \
    --topks "${TOPKS:-2,4,6,8,16}" \
    --batch-size "${BATCH_SIZE:-24}" \
    --num-workers "${NUM_WORKERS:-0}" \
    --device "${DEVICE:-auto}" \
    --log-every "${LOG_EVERY:-10}"

cp "$STATISTICS_DIR/REPORT.md" "$OUT_ROOT/REPORT.md"

echo "statistics: $STATISTICS_DIR"
echo "figures: $FIGURES_DIR"
