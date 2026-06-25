#!/usr/bin/env bash
# Docker entrypoint for hosts that expose Docker and NVIDIA GPUs.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DOCKER_BIN="${DOCKER_BIN:-docker}"
IMAGE="${IMAGE:-gemma4-rpp-vast:cu124}"
BUILD_IMAGE="${BUILD_IMAGE:-0}"

if [[ "$BUILD_IMAGE" == "1" ]]; then
  "$DOCKER_BIN" build -t "$IMAGE" .
fi

"$DOCKER_BIN" run --rm \
  --gpus all \
  --ipc=host \
  --shm-size "${SHM_SIZE:-16g}" \
  -v "$ROOT_DIR:/workspace" \
  -w /workspace \
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}" \
  -e MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS:-32}}" \
  -e OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS:-32}}" \
  -e DATA_ROOT="${DATA_ROOT:-dataset/prompt10000/router_label_npz/npz}" \
  -e OUT_DIR="${OUT_DIR:-}" \
  -e RESUME="${RESUME:-checkpoints/current/checkpoint_best.pt}" \
  -e MAX_FILES="${MAX_FILES:-0}" \
  -e MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}" \
  -e EPOCHS="${EPOCHS:-30}" \
  -e BATCH_SIZE="${BATCH_SIZE:-8}" \
  -e LR="${LR:-3e-4}" \
  -e WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}" \
  -e GRAD_CLIP="${GRAD_CLIP:-1.0}" \
  -e NUM_WORKERS="${NUM_WORKERS:-4}" \
  -e SEED="${SEED:-0}" \
  -e SHUFFLE_SEED="${SHUFFLE_SEED:--1}" \
  -e DEVICE="${DEVICE:-auto}" \
  -e VOCAB_SIZE="${VOCAB_SIZE:-262144}" \
  -e EMBEDDING_MODE="${EMBEDDING_MODE:-hash}" \
  -e HASH_VOCAB_SIZE="${HASH_VOCAB_SIZE:-32768}" \
  -e D_MODEL="${D_MODEL:-32}" \
  -e N_HEADS="${N_HEADS:-4}" \
  -e ENCODER_LAYERS="${ENCODER_LAYERS:-2}" \
  -e DECODER_LAYERS="${DECODER_LAYERS:-2}" \
  -e FFN_DIM="${FFN_DIM:-2048}" \
  -e DROPOUT="${DROPOUT:-0.1}" \
  -e POS_WEIGHT="${POS_WEIGHT:-15.0}" \
  -e BCE_WEIGHT="${BCE_WEIGHT:-1.0}" \
  -e KL_WEIGHT="${KL_WEIGHT:-0.0}" \
  -e KL_SCHEDULE="${KL_SCHEDULE:-fixed}" \
  -e KL_START="${KL_START:-0.05}" \
  -e KL_END="${KL_END:-0.5}" \
  -e KL_WARMUP_RATIO="${KL_WARMUP_RATIO:-0.35}" \
  -e TEMPERATURE="${TEMPERATURE:-1.0}" \
  -e LOG_EVERY="${LOG_EVERY:-25}" \
  -e SPLIT_STRATEGY="${SPLIT_STRATEGY:-stratified}" \
  -e TRAIN_FRAC="${TRAIN_FRAC:-0.8}" \
  -e VAL_FOLDS="${VAL_FOLDS:-5}" \
  -e VAL_FOLD_INDEX="${VAL_FOLD_INDEX:-0}" \
  -e STRATIFY_KEYS="${STRATIFY_KEYS:-task_type,source}" \
  "$IMAGE" \
  bash /workspace/run_vast_train.sh
