#!/usr/bin/env bash
# Offline cache-aware request scheduler simulation inside the RPP Docker image.
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT_DIR"

DOCKER_BIN="${DOCKER_BIN:-docker}"
IMAGE="${IMAGE:-localhost/gemma4-rpp-train:cpu}"
BUILD_IMAGE="${BUILD_IMAGE:-0}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="experiments/gemma4_IO_behaviors/mem8G"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/statistics/cache_aware_scheduler_${RUN_TIMESTAMP}}"
FIGURES_DIR="${FIGURES_DIR:-$OUT_ROOT/figures/cache_aware_scheduler_${RUN_TIMESTAMP}}"
DATA_ROOT="${DATA_ROOT:-dataset/prompt10000/router_label_npz/npz}"
CHECKPOINT="${CHECKPOINT:-experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/checkpoint_best.pt}"
CONFIG="${CONFIG:-experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/config.json}"
RESIDENT_MATRIX="${RESIDENT_MATRIX:-auto}"

if [[ "$BUILD_IMAGE" == "1" ]]; then
  "$DOCKER_BIN" build \
    -f experiments/gemma4_global_predictor/Dockerfile.rpp_torch \
    -t "$IMAGE" .
fi

mkdir -p "$OUT_DIR" "$FIGURES_DIR"

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
  python3 experiments/gemma4_IO_behaviors/mem8G/utils/simulate_cache_aware_request_scheduler.py \
    --data-root "$DATA_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --resident-matrix "$RESIDENT_MATRIX" \
    --matrix-sample-index "${MATRIX_SAMPLE_INDEX:--1}" \
    --matrix-sample-label "${MATRIX_SAMPLE_LABEL:-}" \
    --out-root "$OUT_ROOT" \
    --out-dir "$OUT_DIR" \
    --figures-dir "$FIGURES_DIR" \
    --max-requests "${MAX_REQUESTS:-24}" \
    --pool-size "${POOL_SIZE:-24}" \
    --predict-topk "${PREDICT_TOPK:-8}" \
    --strategies "${STRATEGIES:-round_robin,rpp_similarity,cache_aware_greedy}" \
    --cache-capacity "${CACHE_CAPACITY:-auto}" \
    --base-decode-ms-per-token "${BASE_DECODE_MS_PER_TOKEN:-873.4268}" \
    --predictor-p95-ms "${PREDICTOR_P95_MS:-27.4934}" \
    --prefetch-p95-ms "${PREFETCH_P95_MS:-0.4506}" \
    --miss-cost-ms "${MISS_COST_MS:-0}" \
    --batch-size "${BATCH_SIZE:-24}" \
    --num-workers "${NUM_WORKERS:-0}" \
    --device "${DEVICE:-auto}" \
    --log-every "${LOG_EVERY:-10}" \
    --update-report

echo "cache-aware scheduler statistics: $OUT_DIR"
echo "cache-aware scheduler figures: $FIGURES_DIR"
