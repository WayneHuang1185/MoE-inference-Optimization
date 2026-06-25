#!/usr/bin/env bash
# Direct Vast.ai grid-search entrypoint. Use this inside a Vast.ai PyTorch/CUDA container.
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
export PYTHON_BIN
export DATA_ROOT="${DATA_ROOT:-dataset/prompt10000/router_label_npz/npz}"
export GRID_OUT_ROOT="${GRID_OUT_ROOT:-outputs/rpp_grid_search_${TS}}"

"$PYTHON_BIN" -m experiments.gemma4_global_predictor.grid_search_rpp \
  --python-bin "$PYTHON_BIN" \
  --data-root "$DATA_ROOT" \
  --out-root "$GRID_OUT_ROOT"

echo "wrote grid-search outputs under: $GRID_OUT_ROOT"
