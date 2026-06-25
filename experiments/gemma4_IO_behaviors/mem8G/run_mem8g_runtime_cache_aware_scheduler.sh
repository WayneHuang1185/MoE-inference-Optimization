#!/usr/bin/env bash
# Runtime cache-aware scheduler probe with optional selected-request fadvise.
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT_DIR"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="experiments/gemma4_IO_behaviors/mem8G"
UTIL_DIR="$OUT_ROOT/utils"
STATISTICS_DIR="${STATISTICS_DIR:-$OUT_ROOT/statistics/runtime_cache_aware_scheduler_${RUN_TIMESTAMP}}"
SERVER_IMAGE="${SERVER_IMAGE:-localhost/gemma4-ram-bench:24.04}"
SIDECAR_IMAGE="${SIDECAR_IMAGE:-localhost/gemma4-rpp-train:cpu}"
MODEL="${MODEL:-models/gemma4-26B.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-llama.cpp/build/bin/llama-server}"
PROMPT_DIR="${PROMPT_DIR:-experiments/gemma4_bottleneck/router_prediction_prompts}"
TENSOR_RANGES="${TENSOR_RANGES:-experiments/gemma4_bottleneck/results/gemma4_26b_tensor_ranges.csv}"
CHECKPOINT="${CHECKPOINT:-experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/checkpoint_best.pt}"
CONFIG="${CONFIG:-experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/config.json}"
RESIDENT_MATRIX="${RESIDENT_MATRIX:-auto}"
PORT="${PORT:-8080}"
THREADS="${THREADS:-$(nproc 2>/dev/null || echo 16)}"
CTX_SIZE="${CTX_SIZE:-8192}"
POOL_SIZE="${POOL_SIZE:-24}"
PROMPT_LIMIT="${PROMPT_LIMIT:-24}"
N_PREDICT="${N_PREDICT:-16}"
PREDICT_TOPK="${PREDICT_TOPK:-8}"
CACHE_CAPACITY="${CACHE_CAPACITY:-trace-schedule}"
CASES="${CASES:-round_robin:none,cache_aware_greedy:none,cache_aware_greedy:rpp_selected,rpp_similarity:none}"
PREFETCH_BUDGET="${PREFETCH_BUDGET:-30}"
PREFETCH_THRESHOLD="${PREFETCH_THRESHOLD:-0}"
ADVICE_CACHE_TOKENS="${ADVICE_CACHE_TOKENS:-4}"
TOUCH_BYTES="${TOUCH_BYTES:-0}"
OBSERVE_CACHE="${OBSERVE_CACHE:-0}"
OBSERVE_INTERVAL="${OBSERVE_INTERVAL:-1}"
PAGE_STRIDE="${PAGE_STRIDE:-32}"
RESIDENT_THRESHOLD="${RESIDENT_THRESHOLD:-0.95}"
DEVICE="${DEVICE:-auto}"
LLAMA_EXTRA_ARGS="${LLAMA_EXTRA_ARGS:-}"
DOCKER_GPU_ARGS="${DOCKER_GPU_ARGS:-}"
SERVER_CPUSET="${SERVER_CPUSET:-}"
SIDECAR_CPUSET="${SIDECAR_CPUSET:-}"

case " $LLAMA_EXTRA_ARGS " in
  *" --no-repack "*) ;;
  *) LLAMA_EXTRA_ARGS="${LLAMA_EXTRA_ARGS} --no-repack" ;;
esac

mkdir -p "$STATISTICS_DIR"
read -r -a DOCKER_GPU_ARGS_ARR <<< "$DOCKER_GPU_ARGS"
CONTAINER_NAME="gemma4-runtime-scheduler-${RUN_TIMESTAMP}-${PORT}"
SERVER_LOG="$STATISTICS_DIR/server.log"
PROBE_LOG="$STATISTICS_DIR/runtime_cache_aware_scheduler_probe.log"

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

cleanup

docker run -d \
  --name "$CONTAINER_NAME" \
  --network host \
  "${DOCKER_GPU_ARGS_ARR[@]}" \
  --memory=8g \
  --memory-swap=8g \
  --cap-add SYS_ADMIN \
  --security-opt seccomp=unconfined \
  -v "$ROOT_DIR:/workspace" \
  -w /workspace \
  -e MODEL="$MODEL" \
  -e LLAMA_SERVER="$LLAMA_SERVER" \
  -e PORT="$PORT" \
  -e THREADS="$THREADS" \
  -e CTX_SIZE="$CTX_SIZE" \
  -e POOL_SIZE="$POOL_SIZE" \
  -e LLAMA_EXTRA_ARGS="$LLAMA_EXTRA_ARGS" \
  -e SERVER_CPUSET="$SERVER_CPUSET" \
  "$SERVER_IMAGE" \
  bash -lc '
set -euo pipefail
mkdir -p /tmp/llama_slots_runtime_scheduler
python3 experiments/gemma4_IO_behaviors/mem48G/drop_file_cache.py "$MODEL" >/tmp/runtime_scheduler_drop_file_cache.log 2>&1 || true
export LD_LIBRARY_PATH="/workspace/llama.cpp/build/bin:${LD_LIBRARY_PATH:-}"
read -r -a LLAMA_EXTRA_ARGS_ARR <<< "${LLAMA_EXTRA_ARGS:-}"
LLAMA_CMD=(
  "$LLAMA_SERVER" \
  -m "$MODEL" \
  -c "${CTX_SIZE:-8192}" \
  -t "${THREADS:-16}" \
  -ngl 0 \
  -np "${POOL_SIZE:-24}" \
  --host 127.0.0.1 \
  --port "${PORT:-8080}" \
  --no-warmup \
  --slots \
  --slot-save-path /tmp/llama_slots_runtime_scheduler \
  "${LLAMA_EXTRA_ARGS_ARR[@]}"
)
if [[ -n "${SERVER_CPUSET:-}" ]] && command -v taskset >/dev/null 2>&1; then
  exec taskset -c "$SERVER_CPUSET" "${LLAMA_CMD[@]}"
fi
exec "${LLAMA_CMD[@]}"
' >/dev/null

python3 - "${PORT:-8080}" "$CONTAINER_NAME" <<'PY'
import subprocess
import sys
import time
import urllib.request

port = sys.argv[1]
container_name = sys.argv[2]
deadline = time.time() + 600
last = None
while time.time() < deadline:
    status = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", container_name], capture_output=True, text=True)
    if status.returncode == 0 and status.stdout.strip() == "false":
        print("server container exited before health became ready", file=sys.stderr)
        raise SystemExit(1)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as resp:
            if resp.status < 500:
                raise SystemExit(0)
    except Exception as exc:
        last = exc
    time.sleep(1)
print(f"server did not become ready: {last}", file=sys.stderr)
raise SystemExit(1)
PY

docker logs "$CONTAINER_NAME" > "$SERVER_LOG" 2>&1 || true

OBSERVE_FLAG=()
if [[ "$OBSERVE_CACHE" == "1" ]]; then
  OBSERVE_FLAG=(--observe-cache)
fi

docker run --rm \
  --network host \
  --memory=8g \
  --memory-swap=8g \
  -v "$ROOT_DIR:/workspace" \
  -w /workspace \
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
  -e MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS:-8}}" \
  -e OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS:-8}}" \
  -e SIDECAR_CPUSET="$SIDECAR_CPUSET" \
  "$SIDECAR_IMAGE" \
  bash -lc 'if [[ -n "${SIDECAR_CPUSET:-}" ]] && command -v taskset >/dev/null 2>&1; then exec taskset -c "$SIDECAR_CPUSET" "$@"; fi; exec "$@"' \
  bash \
  python3 "$UTIL_DIR/runtime_cache_aware_scheduler_probe.py" \
    --base-url "http://127.0.0.1:${PORT}" \
    --model "$MODEL" \
    --tensor-ranges "$TENSOR_RANGES" \
    --prompt-dir "$PROMPT_DIR" \
    --output-dir "$STATISTICS_DIR" \
    --out-root "$OUT_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --resident-matrix "$RESIDENT_MATRIX" \
    --matrix-sample-index "${MATRIX_SAMPLE_INDEX:--1}" \
    --matrix-sample-label "${MATRIX_SAMPLE_LABEL:-}" \
    --cache-capacity "$CACHE_CAPACITY" \
    --cases "$CASES" \
    --pool-size "$POOL_SIZE" \
    --limit "$PROMPT_LIMIT" \
    --n-predict "$N_PREDICT" \
    --predict-topk "$PREDICT_TOPK" \
    --prefetch-budget "$PREFETCH_BUDGET" \
    --prefetch-threshold "$PREFETCH_THRESHOLD" \
    --advice-cache-tokens "$ADVICE_CACHE_TOKENS" \
    --touch-bytes "$TOUCH_BYTES" \
    --page-stride "$PAGE_STRIDE" \
    --resident-threshold "$RESIDENT_THRESHOLD" \
    --observe-interval "$OBSERVE_INTERVAL" \
    --device "$DEVICE" \
    --log-every "${LOG_EVERY:-10}" \
    --sleep-after-erase "${SLEEP_AFTER_ERASE:-1}" \
    "${OBSERVE_FLAG[@]}" \
    --update-report \
  2>&1 | tee "$PROBE_LOG"

docker logs "$CONTAINER_NAME" > "$SERVER_LOG" 2>&1 || true
echo "runtime cache-aware scheduler statistics: $STATISTICS_DIR"
