#!/usr/bin/env bash
# Offline 8G RPP expert-cache admission simulation.
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT_DIR"

DOCKER_BIN="${DOCKER_BIN:-docker}"
IMAGE="${IMAGE:-localhost/gemma4-rpp-train:cpu}"
BUILD_IMAGE="${BUILD_IMAGE:-0}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-experiments/gemma4_IO_behaviors/mem8G/statistics/rpp_admission_${RUN_TIMESTAMP}}"
DATA_ROOT="${DATA_ROOT:-dataset/prompt10000/router_label_npz/npz}"
CHECKPOINT="${CHECKPOINT:-experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/checkpoint_best.pt}"
CONFIG="${CONFIG:-experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/config.json}"
CAPACITY_CONFIG="${CAPACITY_CONFIG:-experiments/gemma4_IO_behaviors/mem8G/expert_capacity/statistics/capacity_estimate_budget_model_decode500_recal_20260519_1150/run_config.json}"

if [[ "$BUILD_IMAGE" == "1" ]]; then
  "$DOCKER_BIN" build \
    -f experiments/gemma4_global_predictor/Dockerfile.rpp_torch \
    -t "$IMAGE" .
fi

mkdir -p "$OUT_DIR"

"$DOCKER_BIN" run --rm \
  --memory=8g \
  --memory-swap=8g \
  --shm-size "${SHM_SIZE:-64m}" \
  -v "$ROOT_DIR:/workspace" \
  -w /workspace \
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
  -e MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS:-8}}" \
  -e OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS:-8}}" \
  "$IMAGE" \
  python3 experiments/gemma4_IO_behaviors/mem8G/utils/simulate_rpp_expert_admission.py \
    --data-root "$DATA_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --capacity-config "$CAPACITY_CONFIG" \
    --cache-capacity "${CACHE_CAPACITY:-0}" \
    --out-dir "$OUT_DIR" \
    --max-samples "${MAX_SAMPLES:-1000}" \
    --batch-size "${BATCH_SIZE:-24}" \
    --num-workers "${NUM_WORKERS:-0}" \
    --device "${DEVICE:-auto}" \
    --predict-topk "${PREDICT_TOPK:-8}" \
    --prefetch-budgets "${PREFETCH_BUDGETS:-60,120,180}" \
    --prefetch-thresholds "${PREFETCH_THRESHOLDS:-}" \
    --expert-bytes "${EXPERT_BYTES:-0}" \
    --log-every "${LOG_EVERY:-10}"

cp "$OUT_DIR/REPORT.md" experiments/gemma4_IO_behaviors/mem8G/REPORT.md
echo "rpp admission statistics: $OUT_DIR"
