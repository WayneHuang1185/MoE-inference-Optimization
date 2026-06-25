#!/usr/bin/env bash
# Build/validate dataset/prompts_MTP_10000 on the remote Docker runtime.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

STAGE="${STAGE:-build-pairs}"

SOURCE_PROMPT_DIR="${SOURCE_PROMPT_DIR:-experiments/gemma4_bottleneck/prompts_MTP_10000_source_prompts}"
SOURCE_DATASET_DIR="${SOURCE_DATASET_DIR:-dataset/prompts_MTP_10000_source}"
OUT_DIR="${OUT_DIR:-dataset/prompts_MTP_10000}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments/gemma4_bottleneck/results}"
RUN_NAME="${RUN_NAME:-prompts_MTP_10000_$(date +%Y%m%d_%H%M%S)}"

PY_IMAGE="${PY_IMAGE:-python:3.12-slim}"
HF_IMAGE="${HF_IMAGE:-docker.io/pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime}"
HOST_HF_HOME="${HOST_HF_HOME:-$HOME/.cache/huggingface}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT_DIR/.cache/pip}"

TARGET_MODEL="${TARGET_MODEL:-google/gemma-4-26B-A4B-it}"
ASSISTANT_MODEL="${ASSISTANT_MODEL:-google/gemma-4-26B-A4B-it-assistant}"
TARGET_PAIRS="${TARGET_PAIRS:-10000}"
SOURCE_MULTIPLIER="${SOURCE_MULTIPLIER:-2}"
MAX_RECORDS="${MAX_RECORDS:-0}"
START_INDEX="${START_INDEX:-0}"
DRAFT_MAX="${DRAFT_MAX:-4}"
TOP_K_EXPERTS="${TOP_K_EXPERTS:-8}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-0}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
ROUTER_TRANSFORM="${ROUTER_TRANSFORM:-softmax}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
CLEAN="${CLEAN:-1}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"

mkdir -p "$HOST_HF_HOME" "$PIP_CACHE_DIR" "$RESULTS_ROOT"

echo "[prompts-mtp] stage        = $STAGE"
echo "[prompts-mtp] root         = $ROOT_DIR"
echo "[prompts-mtp] source db    = $SOURCE_DATASET_DIR/prompt_database.jsonl"
echo "[prompts-mtp] out dir      = $OUT_DIR"
echo "[prompts-mtp] results root = $RESULTS_ROOT"

case "$STAGE" in
  build-source)
    count_alpaca=$((3000 * SOURCE_MULTIPLIER))
    count_xsum=$((2000 * SOURCE_MULTIPLIER))
    count_wmt=$((2000 * SOURCE_MULTIPLIER))
    count_code=$((2000 * SOURCE_MULTIPLIER))
    count_math=$((1000 * SOURCE_MULTIPLIER))
    clean_args=()
    if [[ "$CLEAN" == "1" ]]; then
      clean_args+=(--clean)
    fi
    partial_args=()
    if [[ "$ALLOW_PARTIAL" == "1" ]]; then
      partial_args+=(--allow-partial)
    fi
    docker run --rm \
      -v "$ROOT_DIR:/workspace" \
      -v "$HOST_HF_HOME:/root/.cache/huggingface" \
      -v "$PIP_CACHE_DIR:/root/.cache/pip" \
      -w /workspace \
      -e HF_HOME=/root/.cache/huggingface \
      -e HF_DATASETS_CACHE=/root/.cache/huggingface/datasets \
      "$PY_IMAGE" \
      bash -lc 'python -m pip install --upgrade pip >/tmp/prompts_mtp_pip.log && python -m pip install "datasets>=2.20.0" "pyarrow>=15.0.0" >>/tmp/prompts_mtp_pip.log && python experiments/gemma4_bottleneck/build_global_rpp_prompts.py "$@"' \
      bash \
      --preset full \
      --out-dir "$SOURCE_PROMPT_DIR" \
      --results-root "$RESULTS_ROOT" \
      --run-name "${RUN_NAME}_source_prompts" \
      --streaming \
      --count-alpaca "$count_alpaca" \
      --count-xsum "$count_xsum" \
      --count-wmt16_de_en "$count_wmt" \
      --count-code_alpaca "$count_code" \
      --count-math "$count_math" \
      "${clean_args[@]}" \
      "${partial_args[@]}"
    ;;
  materialize-source)
    clean_args=()
    if [[ "$CLEAN" == "1" ]]; then
      clean_args+=(--clean)
    fi
    docker run --rm \
      -v "$ROOT_DIR:/workspace" \
      -w /workspace \
      "$PY_IMAGE" \
      python experiments/gemma4_bottleneck/materialize_prompt_database.py \
        --manifest "$SOURCE_PROMPT_DIR/prompts_manifest.jsonl" \
        --out-dir "$SOURCE_DATASET_DIR" \
        --results-root "$RESULTS_ROOT" \
        --run-name "${RUN_NAME}_source_database" \
        --dataset-name "prompts_MTP_10000_source" \
        "${clean_args[@]}"
    ;;
  build-pairs)
    gpu_args=()
    if [[ "${USE_GPU:-1}" == "1" ]]; then
      gpu_args+=(--gpus "${GPU_DEVICES:-all}")
    fi
    ipc_args=()
    if [[ -n "${IPC_MODE:-}" ]]; then
      ipc_args+=(--ipc="$IPC_MODE")
    fi
    clean_args=()
    if [[ "$CLEAN" == "1" ]]; then
      clean_args+=(--clean)
    fi
    script_args=(
      --target-model "$TARGET_MODEL"
      --assistant-model "$ASSISTANT_MODEL"
      --source-prompt-db "$SOURCE_DATASET_DIR/prompt_database.jsonl"
      --out-dir "$OUT_DIR"
      --results-root "$RESULTS_ROOT"
      --run-name "$RUN_NAME"
      --target-pairs "$TARGET_PAIRS"
      --max-records "$MAX_RECORDS"
      --start-index "$START_INDEX"
      --draft-max "$DRAFT_MAX"
      --top-k-experts "$TOP_K_EXPERTS"
      --max-prompt-tokens "$MAX_PROMPT_TOKENS"
      --device-map "$DEVICE_MAP"
      --torch-dtype "$TORCH_DTYPE"
      --router-transform "$ROUTER_TRANSFORM"
      --progress-every "$PROGRESS_EVERY"
    )
    if [[ -n "$ATTN_IMPLEMENTATION" ]]; then
      script_args+=(--attn-implementation "$ATTN_IMPLEMENTATION")
    fi
    if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
      script_args+=(--trust-remote-code)
    fi
    if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
      script_args+=(--local-files-only)
    fi
    inner='
set -euo pipefail
if [[ "'"$INSTALL_DEPS"'" == "1" ]]; then
  python -m pip install --upgrade pip
  python -m pip install -r experiments/MTP-draft/requirements.txt
fi
python dataset/utils/build_prompts_mtp_10000.py "$@"
'
    docker run --rm \
      "${gpu_args[@]}" \
      "${ipc_args[@]}" \
      --shm-size="${SHM_SIZE:-32g}" \
      -v "$ROOT_DIR:/workspace" \
      -v "$HOST_HF_HOME:/root/.cache/huggingface" \
      -w /workspace \
      -e HF_HOME=/root/.cache/huggingface \
      -e HF_TOKEN="${HF_TOKEN:-}" \
      -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}" \
      -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
      "$HF_IMAGE" \
      bash -lc "$inner" bash "${script_args[@]}" "${clean_args[@]}"
    ;;
  validate)
    expected_rows=$((TARGET_PAIRS * 2))
    docker run --rm \
      -v "$ROOT_DIR:/workspace" \
      -w /workspace \
      "$PY_IMAGE" \
      bash -lc 'python -m pip install numpy >/tmp/prompts_mtp_validate_pip.log && python dataset/utils/validate_prompts_mtp_10000.py "$@"' \
      bash \
      --root "$OUT_DIR" \
      --expected-pairs "$TARGET_PAIRS" \
      --expected-rows "$expected_rows" \
      --check-npz
    ;;
  *)
    echo "unknown STAGE=$STAGE" >&2
    exit 2
    ;;
esac

echo "[prompts-mtp] done stage=$STAGE"
