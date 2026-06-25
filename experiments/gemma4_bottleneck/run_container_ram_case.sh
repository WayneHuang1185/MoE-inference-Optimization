#!/usr/bin/env bash
set -euo pipefail

CASE_NAME="${CASE_NAME:?CASE_NAME is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
THREADS="${THREADS:-$(nproc 2>/dev/null || echo 4)}"
CTX_SIZE="${CTX_SIZE:-8192}"
N_PREDICT="${N_PREDICT:-32}"
PROMPT_REPEAT="${PROMPT_REPEAT:-32}"
PROMPT="${PROMPT:-}"
PROMPT_FILE="${PROMPT_FILE:-}"
RUNS="${RUNS:-3}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
PORT="${PORT:-8080}"
MODEL="${MODEL:-models/gemma4-26B.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-llama.cpp/build/bin/llama-server}"
BENCH_SCRIPT="${BENCH_SCRIPT:-experiments/gemma4_bottleneck/benchmark_prefill_decode.py}"
ENABLE_TENSOR_SWAP_MONITOR="${ENABLE_TENSOR_SWAP_MONITOR:-0}"
ENABLE_ACTIVATION_DUMP="${ENABLE_ACTIVATION_DUMP:-0}"
TENSOR_MONITOR_SCRIPT="${TENSOR_MONITOR_SCRIPT:-experiments/gemma4_bottleneck/monitor_tensor_residency.py}"
TENSOR_RANGES="${TENSOR_RANGES:-experiments/gemma4_bottleneck/results/gemma4_26b_tensor_ranges.csv}"
TENSOR_MONITOR_INTERVAL="${TENSOR_MONITOR_INTERVAL:-1.0}"
TENSOR_MONITOR_PAGE_STRIDE="${TENSOR_MONITOR_PAGE_STRIDE:-1}"
TENSOR_MONITOR_INCLUDE_REGEX="${TENSOR_MONITOR_INCLUDE_REGEX:-}"

mkdir -p "$OUT_DIR"

SERVER_LOG="$OUT_DIR/server.log"
BENCH_LOG="$OUT_DIR/benchmark_stdout.log"
MEM_LOG="$OUT_DIR/cgroup_memory.csv"
META="$OUT_DIR/meta.env"

cat > "$META" <<META
case_name=$CASE_NAME
threads=$THREADS
ctx_size=$CTX_SIZE
n_predict=$N_PREDICT
prompt_repeat=$PROMPT_REPEAT
prompt_file=$PROMPT_FILE
runs=$RUNS
warmup_runs=$WARMUP_RUNS
port=$PORT
model=$MODEL
enable_tensor_swap_monitor=$ENABLE_TENSOR_SWAP_MONITOR
tensor_ranges=$TENSOR_RANGES
tensor_monitor_interval=$TENSOR_MONITOR_INTERVAL
tensor_monitor_page_stride=$TENSOR_MONITOR_PAGE_STRIDE
META

echo "ts_s,memory_current,memory_peak,memory_swap_current,memory_swap_peak" > "$MEM_LOG"

read_cgroup_file() {
  local path="$1"
  if [[ -f "$path" ]]; then
    cat "$path"
  else
    printf '0'
  fi
}

monitor_memory() {
  while true; do
    printf '%s,%s,%s,%s,%s\n' \
      "$(python3 - <<'PY'
import time
print(f"{time.time():.6f}")
PY
)" \
      "$(read_cgroup_file /sys/fs/cgroup/memory.current)" \
      "$(read_cgroup_file /sys/fs/cgroup/memory.peak)" \
      "$(read_cgroup_file /sys/fs/cgroup/memory.swap.current)" \
      "$(read_cgroup_file /sys/fs/cgroup/memory.swap.peak)" \
      >> "$MEM_LOG"
    sleep 0.5
  done
}

wait_until_ready() {
  python3 - "$PORT" <<'PY'
import sys
import time
import urllib.request

port = sys.argv[1]
url = f"http://127.0.0.1:{port}/health"
deadline = time.time() + 600
last_error = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=1) as resp:
            if resp.status < 500:
                raise SystemExit(0)
    except Exception as exc:
        last_error = exc
    time.sleep(1)
print(f"server did not become ready: {last_error}", file=sys.stderr)
raise SystemExit(1)
PY
}

export LD_LIBRARY_PATH="/workspace/llama.cpp/build/bin:${LD_LIBRARY_PATH:-}"

if [[ "$ENABLE_TENSOR_SWAP_MONITOR" == "1" ]]; then
  export GGML_TENSOR_RESIDENCY_LOG="$OUT_DIR/ggml_tensor_residency.csv"
  export GGML_TENSOR_RESIDENCY_MIN_BYTES="${GGML_TENSOR_RESIDENCY_MIN_BYTES:-4096}"
fi
if [[ "$ENABLE_ACTIVATION_DUMP" == "1" ]]; then
  mkdir -p "$OUT_DIR/activation_dump"
  export GGML_ACTIVATION_DUMP_DIR="$OUT_DIR/activation_dump"
fi

"$LLAMA_SERVER" \
  -m "$MODEL" \
  -c "$CTX_SIZE" \
  -t "$THREADS" \
  -ngl 0 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --no-warmup \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID="$!"
echo "server_pid=$SERVER_PID" >> "$META"

monitor_memory &
MONITOR_PID="$!"
TENSOR_MONITOR_PID=""

cleanup() {
  if [[ -n "$TENSOR_MONITOR_PID" ]]; then
    kill -TERM "$TENSOR_MONITOR_PID" 2>/dev/null || true
    wait "$TENSOR_MONITOR_PID" 2>/dev/null || true
  fi
  kill "$MONITOR_PID" 2>/dev/null || true
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -INT "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_until_ready

cp "/proc/$SERVER_PID/maps" "$OUT_DIR/proc_maps.txt" 2>/dev/null || true

if [[ "$ENABLE_TENSOR_SWAP_MONITOR" == "1" ]]; then
  if [[ ! -f "$TENSOR_RANGES" ]]; then
    echo "tensor ranges CSV not found: $TENSOR_RANGES" >&2
    exit 1
  fi
  TENSOR_MONITOR_ARGS=(
    --pid "$SERVER_PID"
    --model "$MODEL"
    --tensor-ranges "$TENSOR_RANGES"
    --model-name "$(basename "$MODEL")"
    --output-dir "$OUT_DIR/tensor_residency"
    --interval "$TENSOR_MONITOR_INTERVAL"
    --page-stride "$TENSOR_MONITOR_PAGE_STRIDE"
  )
  if [[ -n "$TENSOR_MONITOR_INCLUDE_REGEX" ]]; then
    TENSOR_MONITOR_ARGS+=(--include-regex "$TENSOR_MONITOR_INCLUDE_REGEX")
  fi
  python3 "$TENSOR_MONITOR_SCRIPT" "${TENSOR_MONITOR_ARGS[@]}" \
    > "$OUT_DIR/tensor_residency_monitor.log" 2>&1 &
  TENSOR_MONITOR_PID="$!"
  echo "tensor_monitor_pid=$TENSOR_MONITOR_PID" >> "$META"
fi

set +e
BENCH_ARGS=(
  --pid "$SERVER_PID"
  --url "http://127.0.0.1:${PORT}/completion"
  --runs "$RUNS"
  --warmup-runs "$WARMUP_RUNS"
  --n-predict "$N_PREDICT"
  --prompt-repeat "$PROMPT_REPEAT"
  --jsonl "$OUT_DIR/prefill_decode_benchmark.jsonl"
  --csv "$OUT_DIR/prefill_decode_benchmark.csv"
)
if [[ -n "$PROMPT_FILE" ]]; then
  BENCH_ARGS+=(--prompt-file "$PROMPT_FILE")
fi
if [[ -n "$PROMPT" ]]; then
  BENCH_ARGS+=(--prompt "$PROMPT")
fi

python3 "$BENCH_SCRIPT" \
  "${BENCH_ARGS[@]}" \
  2>&1 | tee "$BENCH_LOG"
BENCH_STATUS="${PIPESTATUS[0]}"
set -e

echo "benchmark_exit_code=$BENCH_STATUS" >> "$META"
exit "$BENCH_STATUS"
