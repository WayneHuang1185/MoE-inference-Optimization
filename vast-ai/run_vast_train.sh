#!/usr/bin/env bash
# Direct Vast.ai entrypoint. Use this inside a Vast.ai PyTorch/CUDA container.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x /venv/main/bin/python ]]; then
    PYTHON_BIN="/venv/main/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

echo "using python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import numpy
import torch
PY
then
  "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
  "$PYTHON_BIN" -m pip install numpy
  if command -v nvidia-smi >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip install torch --index-url https://download.pytorch.org/whl/cu124
  else
    "$PYTHON_BIN" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  fi
fi

TS="$(date +%Y%m%d_%H%M%S)"
DATA_ROOT="${DATA_ROOT:-dataset/prompt10000/router_label_npz/npz}"
OUT_DIR="${OUT_DIR:-outputs/rpp_train_vast_${TS}}"
RESUME="${RESUME-checkpoints/current/checkpoint_best.pt}"

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" -m experiments.gemma4_global_predictor.train_rpp \
  --data-root "$DATA_ROOT" \
  --out-dir "$OUT_DIR" \
  --max-files "${MAX_FILES:-0}" \
  --max-seq-len "${MAX_SEQ_LEN:-512}" \
  --epochs "${EPOCHS:-30}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --lr "${LR:-3e-4}" \
  --weight-decay "${WEIGHT_DECAY:-0.01}" \
  --grad-clip "${GRAD_CLIP:-1.0}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --seed "${SEED:-0}" \
  --shuffle-seed "${SHUFFLE_SEED:--1}" \
  --device "${DEVICE:-auto}" \
  --vocab-size "${VOCAB_SIZE:-262144}" \
  --embedding-mode "${EMBEDDING_MODE:-hash}" \
  --hash-vocab-size "${HASH_VOCAB_SIZE:-32768}" \
  --d-model "${D_MODEL:-32}" \
  --n-heads "${N_HEADS:-4}" \
  --encoder-layers "${ENCODER_LAYERS:-2}" \
  --decoder-layers "${DECODER_LAYERS:-2}" \
  --ffn-dim "${FFN_DIM:-2048}" \
  --head-hidden-dim "${HEAD_HIDDEN_DIM:-0}" \
  --dropout "${DROPOUT:-0.1}" \
  --pos-weight "${POS_WEIGHT:-15.0}" \
  --bce-weight "${BCE_WEIGHT:-1.0}" \
  --kl-weight "${KL_WEIGHT:-0.0}" \
  --kl-schedule "${KL_SCHEDULE:-fixed}" \
  --kl-start "${KL_START:-0.05}" \
  --kl-end "${KL_END:-0.5}" \
  --kl-warmup-ratio "${KL_WARMUP_RATIO:-0.35}" \
  --temperature "${TEMPERATURE:-1.0}" \
  --log-every "${LOG_EVERY:-25}" \
  --resume "$RESUME" \
  --split-strategy "${SPLIT_STRATEGY:-stratified}" \
  --train-frac "${TRAIN_FRAC:-0.8}" \
  --val-folds "${VAL_FOLDS:-5}" \
  --val-fold-index "${VAL_FOLD_INDEX:-0}" \
  --stratify-keys "${STRATIFY_KEYS:-task_type,source}"

echo "wrote: $OUT_DIR"
