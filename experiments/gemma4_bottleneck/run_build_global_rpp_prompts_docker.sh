#!/usr/bin/env bash
# Build the Global RPP prompt corpus inside Docker.
#
# Run this from the repository root on the remote workstation after rsync.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

IMAGE="${IMAGE:-python:3.12-slim}"
PRESET="${PRESET:-pilot}"
RUN_NAME="${RUN_NAME:-global_rpp_prompts_${PRESET}_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-experiments/gemma4_bottleneck/global_rpp_prompts}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments/gemma4_bottleneck/results}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$ROOT_DIR/.cache/huggingface}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT_DIR/.cache/pip}"
SEED="${SEED:-20260513}"
STREAMING="${STREAMING:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"
CLEAN="${CLEAN:-1}"

mkdir -p "$HF_CACHE_DIR" "$PIP_CACHE_DIR" "$RESULTS_ROOT"

args=(
  --preset "$PRESET"
  --out-dir "$OUT_DIR"
  --results-root "$RESULTS_ROOT"
  --run-name "$RUN_NAME"
  --seed "$SEED"
)

if [[ "$STREAMING" == "1" ]]; then
  args+=(--streaming)
fi
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  args+=(--trust-remote-code)
fi
if [[ "$ALLOW_PARTIAL" == "1" ]]; then
  args+=(--allow-partial)
fi
if [[ "$CLEAN" == "1" ]]; then
  args+=(--clean)
fi

echo "[global-rpp-prompts] ROOT_DIR     = $ROOT_DIR"
echo "[global-rpp-prompts] IMAGE        = $IMAGE"
echo "[global-rpp-prompts] PRESET       = $PRESET"
echo "[global-rpp-prompts] RUN_NAME     = $RUN_NAME"
echo "[global-rpp-prompts] OUT_DIR      = $OUT_DIR"
echo "[global-rpp-prompts] RESULTS_ROOT = $RESULTS_ROOT"

docker run --rm \
  -v "$ROOT_DIR:/workspace" \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  -v "$PIP_CACHE_DIR:/root/.cache/pip" \
  -w /workspace \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_DATASETS_CACHE=/root/.cache/huggingface/datasets \
  "$IMAGE" \
  bash -lc 'python -m pip install --upgrade pip >/tmp/global_rpp_pip.log && python -m pip install "datasets>=2.20.0" "pyarrow>=15.0.0" >>/tmp/global_rpp_pip.log && python experiments/gemma4_bottleneck/build_global_rpp_prompts.py "$@"' \
  bash "${args[@]}"

DOCKER_VERSION="$(docker --version 2>&1 || true)"
if echo "$DOCKER_VERSION" | grep -qi podman; then
  {
    echo
    echo "## Runtime Note"
    echo
    echo "- The remote \`docker\` CLI is backed by Podman compatibility mode:"
    echo "  \`$DOCKER_VERSION\`"
  } >> "$RESULTS_ROOT/$RUN_NAME/REPORT.md"
fi

echo "[global-rpp-prompts] done"
echo "[global-rpp-prompts] manifest = $OUT_DIR/prompts_manifest.jsonl"
echo "[global-rpp-prompts] report   = $RESULTS_ROOT/$RUN_NAME/REPORT.md"
