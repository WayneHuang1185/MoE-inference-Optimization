#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT_DIR"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="experiments/gemma4_IO_behaviors/mem8G"
UTIL_DIR="$OUT_DIR/utils"
STATISTICS_DIR="${STATISTICS_DIR:-$OUT_DIR/statistics/decode_expert_cache_fault_${RUN_TIMESTAMP}}"
FIGURES_DIR="${FIGURES_DIR:-$OUT_DIR/figures/matrix_${RUN_TIMESTAMP}}"
IMAGE="${IMAGE:-localhost/gemma4-ram-bench:24.04}"
MODEL="${MODEL:-models/gemma4-26B.gguf}"
MODEL_NAME="${MODEL_NAME:-gemma4-26B.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-llama.cpp/build/bin/llama-server}"
PROMPT_DIR="${PROMPT_DIR:-experiments/gemma4_bottleneck/router_prediction_prompts}"
TENSOR_RANGES="${TENSOR_RANGES:-experiments/gemma4_bottleneck/results/gemma4_26b_tensor_ranges.csv}"
PORT="${PORT:-8080}"
THREADS="${THREADS:-$(nproc 2>/dev/null || echo 16)}"
CTX_SIZE="${CTX_SIZE:-8192}"
N_PREDICT="${N_PREDICT:-5}"
PROMPT_LIMIT="${PROMPT_LIMIT:-1}"
PAGE_STRIDE="${PAGE_STRIDE:-16}"
RESIDENT_THRESHOLD="${RESIDENT_THRESHOLD:-0.95}"
PERF="${PERF:-0}"
LLAMA_EXTRA_ARGS="${LLAMA_EXTRA_ARGS:-}"
DOCKER_GPU_ARGS="${DOCKER_GPU_ARGS:-}"

case " $LLAMA_EXTRA_ARGS " in
  *" --no-repack "*) ;;
  *) LLAMA_EXTRA_ARGS="${LLAMA_EXTRA_ARGS} --no-repack" ;;
esac

mkdir -p "$STATISTICS_DIR" "$FIGURES_DIR"
read -r -a DOCKER_GPU_ARGS_ARR <<< "$DOCKER_GPU_ARGS"

docker run --rm \
  "${DOCKER_GPU_ARGS_ARR[@]}" \
  --memory=8g \
  --memory-swap=8g \
  --cap-add PERFMON \
  --cap-add SYS_ADMIN \
  --security-opt seccomp=unconfined \
  -v "$ROOT_DIR:/workspace" \
  -w /workspace \
  -e RUN_TIMESTAMP="$RUN_TIMESTAMP" \
  -e OUT_DIR="$OUT_DIR" \
  -e STATISTICS_DIR="$STATISTICS_DIR" \
  -e FIGURES_DIR="$FIGURES_DIR" \
  -e UTIL_DIR="$UTIL_DIR" \
  -e MODEL="$MODEL" \
  -e MODEL_NAME="$MODEL_NAME" \
  -e LLAMA_SERVER="$LLAMA_SERVER" \
  -e PROMPT_DIR="$PROMPT_DIR" \
  -e TENSOR_RANGES="$TENSOR_RANGES" \
  -e PORT="$PORT" \
  -e THREADS="$THREADS" \
  -e CTX_SIZE="$CTX_SIZE" \
  -e N_PREDICT="$N_PREDICT" \
  -e PROMPT_LIMIT="$PROMPT_LIMIT" \
  -e PAGE_STRIDE="$PAGE_STRIDE" \
  -e RESIDENT_THRESHOLD="$RESIDENT_THRESHOLD" \
  -e PERF="$PERF" \
  -e LLAMA_EXTRA_ARGS="$LLAMA_EXTRA_ARGS" \
  -e DOCKER_GPU_ARGS="$DOCKER_GPU_ARGS" \
  "$IMAGE" \
  bash -lc '
set -euo pipefail

mkdir -p "$STATISTICS_DIR" "$FIGURES_DIR" /tmp/llama_slots_decode_probe
SERVER_LOG="$STATISTICS_DIR/server.log"
PROBE_LOG="$STATISTICS_DIR/decode_expert_cache_probe.log"

python3 experiments/gemma4_IO_behaviors/mem48G/drop_file_cache.py "$MODEL" > "$STATISTICS_DIR/drop_file_cache.log" 2>&1 || true
python3 experiments/gemma4_IO_behaviors/mem48G/pfn_preflight.py --output "$STATISTICS_DIR/pfn_preflight.json" || true

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
  --slot-save-path /tmp/llama_slots_decode_probe \
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

python3 - "${PORT:-8080}" <<'"'"'PY'"'"'
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

cp "/proc/$SERVER_PID/maps" "$STATISTICS_DIR/proc_maps_before_probe.txt" 2>/dev/null || true

PERF_ARG=()
if [[ "${PERF:-0}" == "1" ]]; then
  PERF_ARG=(--perf)
fi

python3 "$UTIL_DIR/decode_expert_cache_probe.py" \
  --pid "$SERVER_PID" \
  --base-url "http://127.0.0.1:${PORT:-8080}" \
  --model "$MODEL" \
  --model-name "$MODEL_NAME" \
  --tensor-ranges "$TENSOR_RANGES" \
  --prompt-dir "$PROMPT_DIR" \
  --output-dir "$STATISTICS_DIR" \
  --n-predict "${N_PREDICT:-5}" \
  --limit "${PROMPT_LIMIT:-1}" \
  --page-stride "${PAGE_STRIDE:-16}" \
  --resident-threshold "${RESIDENT_THRESHOLD:-0.95}" \
  "${PERF_ARG[@]}" \
  2>&1 | tee "$PROBE_LOG"

mapfile -t MATRIX_FILES < <(find "$STATISTICS_DIR" -mindepth 2 -maxdepth 2 -name expert_cache_matrices.json | sort)
if [[ "${#MATRIX_FILES[@]}" -eq 1 ]]; then
  python3 "$UTIL_DIR/render_expert_cache_growth.py" \
    --matrices "${MATRIX_FILES[0]}" \
    --output "$FIGURES_DIR/expert_cache_matrices_growth.svg" \
    --title "8G decode expert page-cache growth"
elif [[ "${#MATRIX_FILES[@]}" -gt 1 ]]; then
  for matrix in "${MATRIX_FILES[@]}"; do
    prompt_dir="$(basename "$(dirname "$matrix")")"
    python3 "$UTIL_DIR/render_expert_cache_growth.py" \
      --matrices "$matrix" \
      --output "$FIGURES_DIR/${prompt_dir}_growth.svg" \
      --title "8G decode expert page-cache growth: ${prompt_dir}"
  done
fi

{
  echo "# mem8G Decode Expert Cache Fault"
  echo
  echo "- timestamp: ${RUN_TIMESTAMP:-unknown}"
  echo "- statistics: $STATISTICS_DIR"
  echo "- figures: $FIGURES_DIR"
  echo "- prompt_limit: ${PROMPT_LIMIT:-1}"
  echo "- n_predict: ${N_PREDICT:-5}"
  echo "- perf: ${PERF:-0}"
  echo
} > "$OUT_DIR/REPORT.md"

echo "decode expert cache/refault statistics: $STATISTICS_DIR"
echo "matrix figures: $FIGURES_DIR"
'
