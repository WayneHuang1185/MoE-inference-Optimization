#!/usr/bin/env bash
# Train the Gemma4 global RoutingPathPredictor inside an isolated Docker image.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DOCKER_BIN="${DOCKER_BIN:-docker}"
IMAGE="${IMAGE:-localhost/gemma4-rpp-train:cpu}"
BUILD_IMAGE="${BUILD_IMAGE:-0}"
DATA_ROOT="${DATA_ROOT:-dataset/prompt1000/router_label_npz/npz}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-experiments/gemma4_bottleneck/results/rpp_train_${TS}}"

if [[ "$BUILD_IMAGE" == "1" ]]; then
  "$DOCKER_BIN" build \
    -f experiments/gemma4_global_predictor/Dockerfile.rpp_torch \
    -t "$IMAGE" .
fi

mkdir -p "$OUT_DIR"

"$DOCKER_BIN" run --rm \
  --shm-size "${SHM_SIZE:-64m}" \
  -v "$ROOT_DIR:/workspace" \
  -w /workspace \
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
  -e MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS:-4}}" \
  -e OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS:-4}}" \
  "$IMAGE" \
  python3 -m experiments.gemma4_global_predictor.train_rpp \
    --data-root "$DATA_ROOT" \
    --out-dir "$OUT_DIR" \
    --max-files "${MAX_FILES:-0}" \
    --max-seq-len "${MAX_SEQ_LEN:-512}" \
    --epochs "${EPOCHS:-5}" \
    --batch-size "${BATCH_SIZE:-4}" \
    --lr "${LR:-3e-4}" \
    --weight-decay "${WEIGHT_DECAY:-0.01}" \
    --grad-clip "${GRAD_CLIP:-1.0}" \
    --num-workers "${NUM_WORKERS:-0}" \
    --seed "${SEED:-0}" \
    --device "${DEVICE:-auto}" \
    --vocab-size "${VOCAB_SIZE:-262144}" \
    --embedding-mode "${EMBEDDING_MODE:-hash}" \
    --hash-vocab-size "${HASH_VOCAB_SIZE:-32768}" \
    --d-model "${D_MODEL:-32}" \
    --n-heads "${N_HEADS:-4}" \
    --encoder-layers "${ENCODER_LAYERS:-2}" \
    --decoder-layers "${DECODER_LAYERS:-2}" \
    --ffn-dim "${FFN_DIM:-2048}" \
    --dropout "${DROPOUT:-0.1}" \
    --pos-weight "${POS_WEIGHT:-auto}" \
    --bce-weight "${BCE_WEIGHT:-1.0}" \
    --kl-weight "${KL_WEIGHT:-0.1}" \
    --kl-schedule "${KL_SCHEDULE:-fixed}" \
    --kl-start "${KL_START:-0.05}" \
    --kl-end "${KL_END:-0.5}" \
    --kl-warmup-ratio "${KL_WARMUP_RATIO:-0.35}" \
    --temperature "${TEMPERATURE:-1.0}" \
    --log-every "${LOG_EVERY:-10}" \
    --resume "${RESUME:-}" \
    --split-strategy "${SPLIT_STRATEGY:-stratified}" \
    --train-frac "${TRAIN_FRAC:-0.8}" \
    --val-folds "${VAL_FOLDS:-5}" \
    --val-fold-index "${VAL_FOLD_INDEX:-0}" \
    --stratify-keys "${STRATIFY_KEYS:-task_type,source}"

echo "wrote: $OUT_DIR"
