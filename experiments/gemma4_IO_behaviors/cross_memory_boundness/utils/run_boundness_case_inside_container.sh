#!/usr/bin/env bash
set -euo pipefail

: "${EXP_DIR:?}"
: "${CASE_DIR:?}"
: "${MODEL:?}"
: "${LLAMA_SERVER:?}"
: "${PROMPT_DIR:?}"
: "${MEMORY_CAP:?}"

SERVER_LOG="$CASE_DIR/server.log"
RUNNER_LOG="$CASE_DIR/runner.log"
SLOT_DIR="/tmp/llama_cross_memory_slots"

mkdir -p "$CASE_DIR" "$SLOT_DIR"

if [[ "${DROP_MODEL_CACHE:-1}" == "1" ]]; then
  python3 experiments/gemma4_IO_behaviors/mem48G/drop_file_cache.py "$MODEL" \
    > "$CASE_DIR/drop_file_cache.log" 2>&1 || true
fi

export LD_LIBRARY_PATH="/workspace/llama.cpp/build/bin:${LD_LIBRARY_PATH:-}"
read -r -a LLAMA_EXTRA_ARGS_ARR <<< "${LLAMA_EXTRA_ARGS:-}"

"$LLAMA_SERVER" \
  -m "$MODEL" \
  -c "${CTX_SIZE:-8192}" \
  -t "${THREADS:-16}" \
  -ngl 0 \
  -np 1 \
  --host 127.0.0.1 \
  --port "${PORT:-8080}" \
  --no-warmup \
  --slots \
  --slot-save-path "$SLOT_DIR" \
  "${LLAMA_EXTRA_ARGS_ARR[@]}" \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID="$!"

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -INT "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

python3 - "${PORT:-8080}" <<'PY'
import sys
import time
import urllib.request

port = sys.argv[1]
deadline = time.time() + 600
last = None
while time.time() < deadline:
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

cp "/proc/$SERVER_PID/maps" "$CASE_DIR/proc_maps.txt" 2>/dev/null || true
{
  echo "memory_cap=$MEMORY_CAP"
  echo "server_pid=$SERVER_PID"
  echo "port=${PORT:-8080}"
  echo "threads=${THREADS:-16}"
  echo "ctx_size=${CTX_SIZE:-8192}"
  echo "n_predict=${N_PREDICT:-16}"
  echo "prompt_limit=${PROMPT_LIMIT:-30}"
  echo "llama_extra_args=${LLAMA_EXTRA_ARGS:-}"
  echo
  echo "## /sys/fs/cgroup/memory.max"
  cat /sys/fs/cgroup/memory.max 2>/dev/null || true
  echo
  echo "## /sys/fs/cgroup/memory.swap.max"
  cat /sys/fs/cgroup/memory.swap.max 2>/dev/null || true
  echo
  echo "## /sys/fs/cgroup/io.pressure"
  cat /sys/fs/cgroup/io.pressure 2>/dev/null || true
  echo
  echo "## /sys/fs/cgroup/memory.pressure"
  cat /sys/fs/cgroup/memory.pressure 2>/dev/null || true
} > "$CASE_DIR/run_environment.txt"

python3 "$EXP_DIR/utils/boundness_prompt_runner.py" \
  --pid "$SERVER_PID" \
  --base-url "http://127.0.0.1:${PORT:-8080}" \
  --prompt-dir "$PROMPT_DIR" \
  --statistics-dir "$CASE_DIR" \
  --memory-cap "$MEMORY_CAP" \
  --n-predict "${N_PREDICT:-16}" \
  --limit "${PROMPT_LIMIT:-30}" \
  2>&1 | tee "$RUNNER_LOG"

echo "statistics: $CASE_DIR"
