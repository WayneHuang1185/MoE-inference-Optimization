#!/usr/bin/env bash
# Live in-server RPP page-cache-aware scheduler experiment.
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT_DIR"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="experiments/gemma4_IO_behaviors/mem8G"
UTIL_DIR="$OUT_ROOT/utils"
STATISTICS_DIR="${STATISTICS_DIR:-$OUT_ROOT/statistics/live_rpp_scheduler_compare_${RUN_TIMESTAMP}}"
REMOTE_HOST="${REMOTE_HOST:-nthu-cs}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/wayne/project}"
LIVE_IMAGE="${LIVE_IMAGE:-localhost/gemma4-rpp-train:cpu}"
MODEL="${MODEL:-models/gemma4-26B.gguf}"
TRUTH_ROOT="${TRUTH_ROOT:-experiments/gemma4_IO_behaviors/mem8G/statistics/truth_prefill_logits_5_short_n16_20260623_040619}"
TRUTH_MANIFEST="${TRUTH_MANIFEST:-$TRUTH_ROOT/router_label_npz/dump_pack_manifest.csv}"
TRUTH_PROMPTS="${TRUTH_PROMPTS:-$TRUTH_ROOT/selected_prompt_database.jsonl}"
PROMPT_DIR="${PROMPT_DIR:-$STATISTICS_DIR/prompts}"
ORACLE_TRACE="${ORACLE_TRACE:-$STATISTICS_DIR/oracle_live_trace.jsonl}"
TENSOR_RANGES="${TENSOR_RANGES:-experiments/gemma4_bottleneck/results/gemma4_26b_tensor_ranges.csv}"
CHECKPOINT="${CHECKPOINT:-experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/checkpoint_best.pt}"
CONFIG="${CONFIG:-experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/config.json}"
TORCHSCRIPT="${TORCHSCRIPT:-experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/rpp_last_token.ts.pt}"
BUILD_DIR="${BUILD_DIR:-llama.cpp/build-rpp-live-host}"
LLAMA_SERVER="${LLAMA_SERVER:-$BUILD_DIR/bin/llama-server}"
DEFAULT_TORCH_PYTHON="$ROOT_DIR/.venv-rpp-build/bin/python"
if [[ ! -x "$DEFAULT_TORCH_PYTHON" ]]; then
  DEFAULT_TORCH_PYTHON="python3"
fi
TORCH_PYTHON="${TORCH_PYTHON:-$DEFAULT_TORCH_PYTHON}"
PORT_BASE="${PORT_BASE:-8080}"
THREADS="${THREADS:-31}"
INFERENCE_CPU_MASK="${INFERENCE_CPU_MASK:-0-30}"
CTX_SIZE="${CTX_SIZE:-8192}"
POOL_SIZE="${POOL_SIZE:-24}"
BASELINE_POOL_SIZE="${BASELINE_POOL_SIZE:-$POOL_SIZE}"
LIVE_POOL_SIZE="${LIVE_POOL_SIZE:-$POOL_SIZE}"
BASELINE_DISPATCH_MODE="${BASELINE_DISPATCH_MODE:-waves}"
LIVE_DISPATCH_MODE="${LIVE_DISPATCH_MODE:-rolling}"
PROMPT_LIMIT="${PROMPT_LIMIT:-24}"
N_PREDICT="${N_PREDICT:-16}"
PREDICT_TOPK="${PREDICT_TOPK:-8}"
RPP_MAX_SEQ_LEN="${RPP_MAX_SEQ_LEN:-512}"
CACHE_THRESHOLD="${CACHE_THRESHOLD:-0.95}"
PAGE_STRIDE="${PAGE_STRIDE:-32}"
RPP_PREDICTOR_THREADS="${RPP_PREDICTOR_THREADS:-1}"
RPP_PREDICTOR_CPU_MASK="${RPP_PREDICTOR_CPU_MASK:-31}"
ACTIVE_UNION_MAX_SLOTS="${ACTIVE_UNION_MAX_SLOTS:-5}"
RPP_PREFETCH_IO_MODE="${RPP_PREFETCH_IO_MODE:-mmap}"
RPP_PREFETCH_PREFILL_ONLY="${RPP_PREFETCH_PREFILL_ONLY:-0}"
RPP_PREFETCH_COUNT_THRESHOLD="${RPP_PREFETCH_COUNT_THRESHOLD:-5}"
RPP_UBATCH_PREFETCH_MIN_COUNT="${RPP_UBATCH_PREFETCH_MIN_COUNT:-$RPP_PREFETCH_COUNT_THRESHOLD}"
RPP_UBATCH_PREFETCH_DENSITY_THRESHOLD="${RPP_UBATCH_PREFETCH_DENSITY_THRESHOLD:-0.5}"
RPP_PREFETCH_ASYNC="${RPP_PREFETCH_ASYNC:-0}"
RPP_PREFETCH_QUEUE_POLICY="${RPP_PREFETCH_QUEUE_POLICY:-priority}"
RPP_PREFETCH_FIFO_CANDIDATE_ORDER="${RPP_PREFETCH_FIFO_CANDIDATE_ORDER:-density}"
RPP_PREFETCH_QUEUE_CAP="${RPP_PREFETCH_QUEUE_CAP:-512}"
RPP_PREFETCH_MAX_STALENESS_UBATCHES="${RPP_PREFETCH_MAX_STALENESS_UBATCHES:-0}"
RPP_UBATCH_PREFETCH_MAX_EXPERTS="${RPP_UBATCH_PREFETCH_MAX_EXPERTS:-0}"
RPP_RECLAIM_MODE="${RPP_RECLAIM_MODE:-trace-priority}"
RPP_RECLAIM_QUEUE_CAP="${RPP_RECLAIM_QUEUE_CAP:-512}"
RPP_RECLAIM_LOOKAHEAD_UBATCHES="${RPP_RECLAIM_LOOKAHEAD_UBATCHES:-4}"
RPP_RECLAIM_PROTECT_UBATCHES="${RPP_RECLAIM_PROTECT_UBATCHES:-2}"
RPP_CROSS_UBATCH_LAYER_PREFETCH="${RPP_CROSS_UBATCH_LAYER_PREFETCH:-0}"
RPP_CROSS_UBATCH_LAYER_LOOKAHEAD="${RPP_CROSS_UBATCH_LAYER_LOOKAHEAD:-1}"
RPP_LAYER_FRONTIER_PREFETCH="${RPP_LAYER_FRONTIER_PREFETCH:-0}"
RPP_LAYER_FRONTIER_LOOKAHEAD_LAYERS="${RPP_LAYER_FRONTIER_LOOKAHEAD_LAYERS:-8}"
RPP_LAYER_FRONTIER_JOBS_PER_TICK="${RPP_LAYER_FRONTIER_JOBS_PER_TICK:-64}"
RPP_LAYER_FRONTIER_MAX_DISTANCE="${RPP_LAYER_FRONTIER_MAX_DISTANCE:-60}"
RPP_POST_CLIENT_SLEEP_S="${RPP_POST_CLIENT_SLEEP_S:-0}"
PERF_FAULTS="${PERF_FAULTS:-1}"
HOST_DROP_CACHES="${HOST_DROP_CACHES:-1}"
DECODE_FAULT_WINDOW="${DECODE_FAULT_WINDOW:-whole_request}"
DECODE_FAULT_WINDOW_TIMEOUT_S="${DECODE_FAULT_WINDOW_TIMEOUT_S:-600}"
RPP_REQUIRE_FULL_DECODE_BATCH="${RPP_REQUIRE_FULL_DECODE_BATCH:-0}"
BASELINE_MAX_DECODE_SLOTS="${BASELINE_MAX_DECODE_SLOTS:-$ACTIVE_UNION_MAX_SLOTS}"
DOCKER_GPU_ARGS="${DOCKER_GPU_ARGS:-}"
DOCKER_MEMORY_LIMIT="${DOCKER_MEMORY_LIMIT:-8g}"
DOCKER_MEMORY_SWAP_LIMIT="${DOCKER_MEMORY_SWAP_LIMIT:-$DOCKER_MEMORY_LIMIT}"
LLAMA_EXTRA_ARGS="${LLAMA_EXTRA_ARGS:-}"
BUILD_JOBS="${BUILD_JOBS:-2}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_LIVE="${RUN_LIVE:-1}"

SYNC_PATHS=(
  llama.cpp/common/common.h
  llama.cpp/common/arg.cpp
  llama.cpp/tools/server/CMakeLists.txt
  llama.cpp/tools/server/server-context.cpp
  llama.cpp/tools/server/server-rpp-live.cpp
  llama.cpp/tools/server/server-rpp-live.h
  experiments/gemma4_global_predictor/dataset.py
  experiments/gemma4_global_predictor/eval_rpp_checkpoint.py
  experiments/gemma4_global_predictor/export_rpp_torchscript.py
  experiments/gemma4_global_predictor/metrics.py
  experiments/gemma4_global_predictor/model.py
  experiments/gemma4_IO_behaviors/mem8G/run_mem8g_live_rpp_scheduler.sh
  experiments/gemma4_IO_behaviors/mem8G/utils/build_oracle_live_trace.py
  experiments/gemma4_IO_behaviors/mem8G/utils/live_rpp_fixed_pool_client.py
)

if [[ "${LIVE_RPP_REMOTE:-0}" != "1" ]]; then
  rsync -avR "${SYNC_PATHS[@]}" "$REMOTE_HOST:$REMOTE_ROOT/"
  ssh "$REMOTE_HOST" "cd '$REMOTE_ROOT' && LIVE_RPP_REMOTE=1 RUN_TIMESTAMP='$RUN_TIMESTAMP' STATISTICS_DIR='$STATISTICS_DIR' LIVE_IMAGE='$LIVE_IMAGE' MODEL='$MODEL' TRUTH_ROOT='$TRUTH_ROOT' TRUTH_MANIFEST='$TRUTH_MANIFEST' TRUTH_PROMPTS='$TRUTH_PROMPTS' PROMPT_DIR='$PROMPT_DIR' ORACLE_TRACE='$ORACLE_TRACE' TENSOR_RANGES='$TENSOR_RANGES' CHECKPOINT='$CHECKPOINT' CONFIG='$CONFIG' TORCHSCRIPT='$TORCHSCRIPT' BUILD_DIR='$BUILD_DIR' LLAMA_SERVER='$LLAMA_SERVER' TORCH_PYTHON='$TORCH_PYTHON' PORT_BASE='$PORT_BASE' THREADS='$THREADS' INFERENCE_CPU_MASK='$INFERENCE_CPU_MASK' CTX_SIZE='$CTX_SIZE' POOL_SIZE='$POOL_SIZE' BASELINE_POOL_SIZE='$BASELINE_POOL_SIZE' LIVE_POOL_SIZE='$LIVE_POOL_SIZE' BASELINE_DISPATCH_MODE='$BASELINE_DISPATCH_MODE' LIVE_DISPATCH_MODE='$LIVE_DISPATCH_MODE' PROMPT_LIMIT='$PROMPT_LIMIT' N_PREDICT='$N_PREDICT' PREDICT_TOPK='$PREDICT_TOPK' RPP_MAX_SEQ_LEN='$RPP_MAX_SEQ_LEN' CACHE_THRESHOLD='$CACHE_THRESHOLD' PAGE_STRIDE='$PAGE_STRIDE' RPP_PREDICTOR_THREADS='$RPP_PREDICTOR_THREADS' RPP_PREDICTOR_CPU_MASK='$RPP_PREDICTOR_CPU_MASK' ACTIVE_UNION_MAX_SLOTS='$ACTIVE_UNION_MAX_SLOTS' RPP_PREFETCH_IO_MODE='$RPP_PREFETCH_IO_MODE' RPP_PREFETCH_PREFILL_ONLY='$RPP_PREFETCH_PREFILL_ONLY' RPP_PREFETCH_COUNT_THRESHOLD='$RPP_PREFETCH_COUNT_THRESHOLD' RPP_UBATCH_PREFETCH_MIN_COUNT='$RPP_UBATCH_PREFETCH_MIN_COUNT' RPP_UBATCH_PREFETCH_DENSITY_THRESHOLD='$RPP_UBATCH_PREFETCH_DENSITY_THRESHOLD' RPP_PREFETCH_ASYNC='$RPP_PREFETCH_ASYNC' RPP_PREFETCH_QUEUE_POLICY='$RPP_PREFETCH_QUEUE_POLICY' RPP_PREFETCH_FIFO_CANDIDATE_ORDER='$RPP_PREFETCH_FIFO_CANDIDATE_ORDER' RPP_PREFETCH_QUEUE_CAP='$RPP_PREFETCH_QUEUE_CAP' RPP_PREFETCH_MAX_STALENESS_UBATCHES='$RPP_PREFETCH_MAX_STALENESS_UBATCHES' RPP_UBATCH_PREFETCH_MAX_EXPERTS='$RPP_UBATCH_PREFETCH_MAX_EXPERTS' RPP_RECLAIM_MODE='$RPP_RECLAIM_MODE' RPP_RECLAIM_QUEUE_CAP='$RPP_RECLAIM_QUEUE_CAP' RPP_RECLAIM_LOOKAHEAD_UBATCHES='$RPP_RECLAIM_LOOKAHEAD_UBATCHES' RPP_RECLAIM_PROTECT_UBATCHES='$RPP_RECLAIM_PROTECT_UBATCHES' RPP_CROSS_UBATCH_LAYER_PREFETCH='$RPP_CROSS_UBATCH_LAYER_PREFETCH' RPP_CROSS_UBATCH_LAYER_LOOKAHEAD='$RPP_CROSS_UBATCH_LAYER_LOOKAHEAD' RPP_LAYER_FRONTIER_PREFETCH='$RPP_LAYER_FRONTIER_PREFETCH' RPP_LAYER_FRONTIER_LOOKAHEAD_LAYERS='$RPP_LAYER_FRONTIER_LOOKAHEAD_LAYERS' RPP_LAYER_FRONTIER_JOBS_PER_TICK='$RPP_LAYER_FRONTIER_JOBS_PER_TICK' RPP_LAYER_FRONTIER_MAX_DISTANCE='$RPP_LAYER_FRONTIER_MAX_DISTANCE' RPP_POST_CLIENT_SLEEP_S='$RPP_POST_CLIENT_SLEEP_S' PERF_FAULTS='$PERF_FAULTS' HOST_DROP_CACHES='$HOST_DROP_CACHES' DECODE_FAULT_WINDOW='$DECODE_FAULT_WINDOW' DECODE_FAULT_WINDOW_TIMEOUT_S='$DECODE_FAULT_WINDOW_TIMEOUT_S' RPP_REQUIRE_FULL_DECODE_BATCH='$RPP_REQUIRE_FULL_DECODE_BATCH' BASELINE_MAX_DECODE_SLOTS='$BASELINE_MAX_DECODE_SLOTS' DOCKER_GPU_ARGS='$DOCKER_GPU_ARGS' DOCKER_MEMORY_LIMIT='$DOCKER_MEMORY_LIMIT' DOCKER_MEMORY_SWAP_LIMIT='$DOCKER_MEMORY_SWAP_LIMIT' LLAMA_EXTRA_ARGS='$LLAMA_EXTRA_ARGS' BUILD_JOBS='$BUILD_JOBS' RUN_BASELINE='$RUN_BASELINE' RUN_LIVE='$RUN_LIVE' bash '$OUT_ROOT/run_mem8g_live_rpp_scheduler.sh'"
  exit 0
fi

mkdir -p "$STATISTICS_DIR"
read -r -a DOCKER_GPU_ARGS_ARR <<< "$DOCKER_GPU_ARGS"

TORCH_CMAKE_PREFIX="$("$TORCH_PYTHON" -c 'import torch; print(torch.utils.cmake_prefix_path)' 2>/tmp/live_rpp_torch_import.err || true)"
if [[ -z "$TORCH_CMAKE_PREFIX" ]]; then
  echo "failed to import torch with TORCH_PYTHON=$TORCH_PYTHON; host build requires LibTorch/Torch CMake files" >&2
  cat /tmp/live_rpp_torch_import.err >&2 || true
  exit 1
fi
TORCH_LIB_DIR_REL="$("$TORCH_PYTHON" - <<'PY'
from pathlib import Path
import torch
root = Path.cwd().resolve()
lib = (Path(torch.__file__).resolve().parent / "lib")
print(lib.relative_to(root))
PY
)"
cmake -S llama.cpp -B "$BUILD_DIR" \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_WEBUI=OFF \
  -DLLAMA_RPP_LIVE=ON \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX"
cmake --build "$BUILD_DIR" --target llama-server -j"$BUILD_JOBS"
"$TORCH_PYTHON" experiments/gemma4_global_predictor/export_rpp_torchscript.py \
  --checkpoint "$CHECKPOINT" \
  --config "$CONFIG" \
  --output "$TORCHSCRIPT" \
  --device cpu \
  --trace

docker run --rm \
  --network host \
  --memory="$DOCKER_MEMORY_LIMIT" \
  --memory-swap="$DOCKER_MEMORY_SWAP_LIMIT" \
  -v "$ROOT_DIR:/workspace" \
  -w /workspace \
  "$LIVE_IMAGE" \
  python3 "$UTIL_DIR/build_oracle_live_trace.py" \
    --manifest "$TRUTH_MANIFEST" \
    --limit "$PROMPT_LIMIT" \
    --include-prefill \
    --no-decode \
    --output "$ORACLE_TRACE"

python3 - "$TRUTH_PROMPTS" "$TRUTH_MANIFEST" "$TRUTH_ROOT" "$PROMPT_DIR" "$STATISTICS_DIR/oracle_sample_ids.txt" "$STATISTICS_DIR/oracle_generation_settings.json" "$STATISTICS_DIR/oracle_expected_completions.json" "$PROMPT_LIMIT" <<'PY'
import csv
import json
import sys
from pathlib import Path
import numpy as np

prompt_jsonl = Path(sys.argv[1])
manifest_csv = Path(sys.argv[2])
truth_root = Path(sys.argv[3])
prompt_dir = Path(sys.argv[4])
sample_id_path = Path(sys.argv[5])
settings_path = Path(sys.argv[6])
expected_path = Path(sys.argv[7])
limit = int(sys.argv[8])
prompt_dir.mkdir(parents=True, exist_ok=True)

def decode_meta(meta_array):
    arr = np.asarray(meta_array)
    if arr.dtype == np.uint8:
        return json.loads(bytes(arr.tolist()).decode("utf-8"))
    value = arr.item() if arr.shape == () else arr
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value) if isinstance(value, str) else value

prompts = []
for line in prompt_jsonl.read_text(encoding="utf-8").splitlines():
    if line.strip():
        prompts.append(json.loads(line))
if limit > 0:
    prompts = prompts[:limit]
for i, row in enumerate(prompts):
    name = f"{i:04d}_{row.get('prompt_id', 'prompt')}.txt"
    (prompt_dir / name).write_text(row["prompt_text"], encoding="utf-8")

sample_ids = []
settings = {}
expected_completions = {}
with manifest_csv.open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        if row.get("status", "ok") == "ok" and row.get("sample_id"):
            sample_id = row["sample_id"]
            sample_ids.append(sample_id)
            npz_path = Path(row.get("npz_path", ""))
            if npz_path.exists():
                with np.load(npz_path, allow_pickle=False) as data:
                    meta = decode_meta(data["meta_json"]) if "meta_json" in data else {}
                    generation_settings = meta.get("generation_settings")
                    if isinstance(generation_settings, dict):
                        settings[sample_id] = generation_settings
            completion_path = truth_root / "generations" / "completions" / f"{sample_id}.json"
            if completion_path.exists():
                completion = json.loads(completion_path.read_text(encoding="utf-8"))
                expected_completions[sample_id] = str(completion.get("completion_text", ""))
if limit > 0:
    sample_ids = sample_ids[:limit]
    settings = {sample_id: settings[sample_id] for sample_id in sample_ids if sample_id in settings}
    expected_completions = {sample_id: expected_completions[sample_id] for sample_id in sample_ids if sample_id in expected_completions}
sample_id_path.write_text("\n".join(sample_ids) + "\n", encoding="utf-8")
settings_path.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
expected_path.write_text(json.dumps(expected_completions, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

wait_server() {
  local port="$1"
  local container="$2"
  python3 - "$port" "$container" <<'PY'
import subprocess
import sys
import time
import urllib.request

port = sys.argv[1]
container = sys.argv[2]
deadline = time.time() + 600
last = None
while time.time() < deadline:
    status = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", container], capture_output=True, text=True)
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
}

capture_faults() {
  local container="$1"
  local output="$2"
  python3 - "$container" "$output" <<'PY'
import json
import subprocess
import sys
import time
from pathlib import Path

container = sys.argv[1]
output = Path(sys.argv[2])

def run(args):
    return subprocess.run(args, capture_output=True, text=True, check=False)

data = {"container": container, "timestamp_s": time.time()}

stat = run(["docker", "exec", container, "cat", "/proc/1/stat"])
if stat.returncode == 0:
    text = stat.stdout.strip()
    end_comm = text.rfind(") ")
    if end_comm >= 0:
        fields = text[end_comm + 2:].split()
        # /proc/<pid>/stat: field 10=minflt, 12=majflt. fields starts at field 3.
        if len(fields) >= 10:
            data["proc_minflt"] = int(fields[7])
            data["proc_majflt"] = int(fields[9])
else:
    data["proc_stat_error"] = stat.stderr.strip()

mem = run(["docker", "exec", container, "cat", "/sys/fs/cgroup/memory.stat"])
if mem.returncode == 0:
    for line in mem.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in {"pgfault", "pgmajfault"}:
            data["cgroup_" + parts[0]] = int(parts[1])
else:
    data["cgroup_memory_stat_error"] = mem.stderr.strip()

output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

write_fault_delta() {
  local case_dir="$1"
  python3 - "$case_dir" <<'PY'
import json
import os
import sys
from pathlib import Path

case_dir = Path(sys.argv[1])
before = json.loads((case_dir / "faults_before.json").read_text(encoding="utf-8"))
after = json.loads((case_dir / "faults_after.json").read_text(encoding="utf-8"))
delta = {}
for key in ["proc_minflt", "proc_majflt", "cgroup_pgfault", "cgroup_pgmajfault"]:
    if key in before and key in after:
        delta[key + "_delta"] = after[key] - before[key]
mode = os.environ.get("DECODE_FAULT_WINDOW", "whole_request")
if mode == "after_prefill":
    delta["measurement"] = "server container faults after all active requests reached generating state"
else:
    delta["measurement"] = "server container faults during client request window"
(case_dir / "fault_delta.json").write_text(json.dumps(delta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

summary_path = case_dir / "summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(delta)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

drop_host_caches() {
  local case_dir="$1"
  if [[ "$HOST_DROP_CACHES" != "1" ]]; then
    echo "HOST_DROP_CACHES=$HOST_DROP_CACHES; skipped" > "$case_dir/drop_caches.log"
    return 0
  fi
  if command -v drop-caches >/dev/null 2>&1; then
    {
      date -Is
      drop-caches
      date -Is
    } > "$case_dir/drop_caches.log" 2>&1 || {
      echo "drop-caches failed with rc=$?" >> "$case_dir/drop_caches.log"
      return 1
    }
  else
    echo "drop-caches command not found on host" > "$case_dir/drop_caches.log"
  fi
}

start_perf_faults() {
  local container="$1"
  local case_dir="$2"
  if [[ "$PERF_FAULTS" != "1" ]]; then
    return 0
  fi
  if ! command -v perf >/dev/null 2>&1; then
    echo "perf not found on host" > "$case_dir/perf_faults.unavailable"
    return 0
  fi
  local pid
  pid="$(docker inspect -f '{{.State.Pid}}' "$container" 2>/dev/null || true)"
  if [[ -z "$pid" || "$pid" == "0" ]]; then
    echo "failed to resolve container host pid" > "$case_dir/perf_faults.unavailable"
    return 0
  fi
  echo "$pid" > "$case_dir/server_host_pid.txt"
  perf stat \
    -x, \
    -e page-faults,major-faults,minor-faults \
    -p "$pid" \
    -o "$case_dir/perf_faults.stat.csv" \
    -- sleep 3600 \
    > "$case_dir/perf_faults.stdout" \
    2> "$case_dir/perf_faults.stderr" &
  echo "$!" > "$case_dir/perf_faults.pid"
  sleep 0.25
  if ! kill -0 "$(cat "$case_dir/perf_faults.pid")" >/dev/null 2>&1; then
    echo "perf stat exited before client run" >> "$case_dir/perf_faults.unavailable"
  fi
}

stop_perf_faults() {
  local case_dir="$1"
  if [[ ! -f "$case_dir/perf_faults.pid" ]]; then
    return 0
  fi
  local pid
  pid="$(cat "$case_dir/perf_faults.pid")"
  if kill -0 "$pid" >/dev/null 2>&1; then
    kill -INT "$pid" >/dev/null 2>&1 || true
    wait "$pid" || true
  fi
  python3 - "$case_dir" <<'PY'
import json
import sys
from pathlib import Path

case_dir = Path(sys.argv[1])
path = case_dir / "perf_faults.stat.csv"
metrics = {}
if path.exists():
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 3:
            continue
        value, _unit, event = cols[:3]
        try:
            metrics["perf_" + event.replace("-", "_")] = int(float(value))
        except ValueError:
            pass
if metrics:
    (case_dir / "perf_faults.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path = case_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(metrics)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_case() {
  local case_name="$1"
  local port="$2"
  local server_pool_size="$3"
  local client_pool_size="$4"
  local dispatch_mode="$5"
  shift 5
  local case_dir="$STATISTICS_DIR/$case_name"
  local container="gemma4-live-rpp-${RUN_TIMESTAMP}-${case_name}-${port}"
  local decode_window_marker="$case_dir/decode_window_ready"
  local decode_window_continue="$case_dir/decode_window_continue"
  local decode_window_active_min="$client_pool_size"
  local case_status=0
  if [[ "$PROMPT_LIMIT" -gt 0 && "$PROMPT_LIMIT" -lt "$decode_window_active_min" ]]; then
    decode_window_active_min="$PROMPT_LIMIT"
  fi
  local docker_decode_env=()
  if [[ "$DECODE_FAULT_WINDOW" == "after_prefill" ]]; then
    docker_decode_env=(
      -e "RPP_DECODE_WINDOW_MARKER=/workspace/$decode_window_marker"
      -e "RPP_DECODE_WINDOW_CONTINUE=/workspace/$decode_window_continue"
      -e "RPP_DECODE_WINDOW_ACTIVE_MIN=$decode_window_active_min"
      -e "RPP_DECODE_WINDOW_TIMEOUT_MS=$((DECODE_FAULT_WINDOW_TIMEOUT_S * 1000))"
    )
  fi
  if [[ "$RPP_REQUIRE_FULL_DECODE_BATCH" == "1" ]]; then
    docker_decode_env+=(-e "RPP_REQUIRE_FULL_DECODE_BATCH=1")
  fi
  mkdir -p "$case_dir"
  rm -f "$decode_window_marker" "$decode_window_continue"
  docker rm -f "$container" >/dev/null 2>&1 || true
  drop_host_caches "$case_dir"

  docker run -d \
    --name "$container" \
    --network host \
    --memory="$DOCKER_MEMORY_LIMIT" \
    --memory-swap="$DOCKER_MEMORY_SWAP_LIMIT" \
    "${DOCKER_GPU_ARGS_ARR[@]}" \
    "${docker_decode_env[@]}" \
    -v "$ROOT_DIR:/workspace" \
    -w /workspace \
    "$LIVE_IMAGE" \
    bash -lc '
set -euo pipefail
export LD_LIBRARY_PATH="/workspace/'"$BUILD_DIR"'/bin:/workspace/'"$TORCH_LIB_DIR_REL"':${LD_LIBRARY_PATH:-}"
export RPP_LIVE_OUT_DIR="/workspace/'"$case_dir"'/rpp_live"
export RPP_PREFETCH_IO_MODE="'"$RPP_PREFETCH_IO_MODE"'"
export RPP_PREFETCH_PREFILL_ONLY="'"$RPP_PREFETCH_PREFILL_ONLY"'"
export RPP_UBATCH_PREFETCH_MIN_COUNT="'"$RPP_UBATCH_PREFETCH_MIN_COUNT"'"
export RPP_UBATCH_PREFETCH_DENSITY_THRESHOLD="'"$RPP_UBATCH_PREFETCH_DENSITY_THRESHOLD"'"
export RPP_PREFETCH_ASYNC="'"$RPP_PREFETCH_ASYNC"'"
export RPP_PREFETCH_QUEUE_POLICY="'"$RPP_PREFETCH_QUEUE_POLICY"'"
export RPP_PREFETCH_FIFO_CANDIDATE_ORDER="'"$RPP_PREFETCH_FIFO_CANDIDATE_ORDER"'"
export RPP_PREFETCH_QUEUE_CAP="'"$RPP_PREFETCH_QUEUE_CAP"'"
export RPP_PREFETCH_MAX_STALENESS_UBATCHES="'"$RPP_PREFETCH_MAX_STALENESS_UBATCHES"'"
export RPP_UBATCH_PREFETCH_MAX_EXPERTS="'"$RPP_UBATCH_PREFETCH_MAX_EXPERTS"'"
export RPP_RECLAIM_MODE="'"$RPP_RECLAIM_MODE"'"
export RPP_RECLAIM_QUEUE_CAP="'"$RPP_RECLAIM_QUEUE_CAP"'"
export RPP_RECLAIM_LOOKAHEAD_UBATCHES="'"$RPP_RECLAIM_LOOKAHEAD_UBATCHES"'"
export RPP_RECLAIM_PROTECT_UBATCHES="'"$RPP_RECLAIM_PROTECT_UBATCHES"'"
export RPP_CROSS_UBATCH_LAYER_PREFETCH="'"$RPP_CROSS_UBATCH_LAYER_PREFETCH"'"
export RPP_CROSS_UBATCH_LAYER_LOOKAHEAD="'"$RPP_CROSS_UBATCH_LAYER_LOOKAHEAD"'"
export RPP_LAYER_FRONTIER_PREFETCH="'"$RPP_LAYER_FRONTIER_PREFETCH"'"
export RPP_LAYER_FRONTIER_LOOKAHEAD_LAYERS="'"$RPP_LAYER_FRONTIER_LOOKAHEAD_LAYERS"'"
export RPP_LAYER_FRONTIER_JOBS_PER_TICK="'"$RPP_LAYER_FRONTIER_JOBS_PER_TICK"'"
export RPP_LAYER_FRONTIER_MAX_DISTANCE="'"$RPP_LAYER_FRONTIER_MAX_DISTANCE"'"
export RPP_MAX_SEQ_LEN="'"$RPP_MAX_SEQ_LEN"'"
python3 experiments/gemma4_IO_behaviors/mem48G/drop_file_cache.py "'"$MODEL"'" >/tmp/live_rpp_drop_file_cache.log 2>&1 || true
if command -v taskset >/dev/null 2>&1; then
  exec taskset -c "'"$INFERENCE_CPU_MASK"'" "'"$LLAMA_SERVER"'" \
    -m "'"$MODEL"'" \
    -c "'"$CTX_SIZE"'" \
    -t "'"$THREADS"'" \
    -ngl 0 \
    -np "'"$server_pool_size"'" \
    --host 127.0.0.1 \
    --port "'"$port"'" \
    --no-warmup \
    --slots \
    --no-repack \
    --rpp-predictor-threads "'"$RPP_PREDICTOR_THREADS"'" \
    --rpp-predictor-cpu-mask "'"$RPP_PREDICTOR_CPU_MASK"'" \
    '"$LLAMA_EXTRA_ARGS"' \
    '"$*"'
fi
echo "warning: taskset not found; llama-server will inherit container CPU affinity" >&2
exec "'"$LLAMA_SERVER"'" \
  -m "'"$MODEL"'" \
  -c "'"$CTX_SIZE"'" \
  -t "'"$THREADS"'" \
  -ngl 0 \
  -np "'"$server_pool_size"'" \
  --host 127.0.0.1 \
  --port "'"$port"'" \
  --no-warmup \
  --slots \
  --no-repack \
  --rpp-predictor-threads "'"$RPP_PREDICTOR_THREADS"'" \
  --rpp-predictor-cpu-mask "'"$RPP_PREDICTOR_CPU_MASK"'" \
  '"$LLAMA_EXTRA_ARGS"' \
  '"$*"'
'

  wait_server "$port" "$container"
  docker logs "$container" > "$case_dir/server_start.log" 2>&1 || true

  run_client() {
    docker run --rm \
      --network host \
      --memory="$DOCKER_MEMORY_LIMIT" \
      --memory-swap="$DOCKER_MEMORY_SWAP_LIMIT" \
      -v "$ROOT_DIR:/workspace" \
      -w /workspace \
      "$LIVE_IMAGE" \
      python3 "$UTIL_DIR/live_rpp_fixed_pool_client.py" \
        --base-url "http://127.0.0.1:${port}" \
        --model "$MODEL" \
        --prompt-dir "$PROMPT_DIR" \
        --output-dir "$case_dir" \
        --case-name "$case_name" \
        --limit "$PROMPT_LIMIT" \
        --pool-size "$client_pool_size" \
        --dispatch-mode "$dispatch_mode" \
        --n-predict "$N_PREDICT" \
        --oracle-sample-id-file "$STATISTICS_DIR/oracle_sample_ids.txt" \
        --oracle-settings-file "$STATISTICS_DIR/oracle_generation_settings.json" \
        --expected-completions-file "$STATISTICS_DIR/oracle_expected_completions.json"
  }

  if [[ "$DECODE_FAULT_WINDOW" == "after_prefill" ]]; then
    run_client 2>&1 | tee "$case_dir/client.log" &
    local client_pid="$!"
    local deadline=$((SECONDS + DECODE_FAULT_WINDOW_TIMEOUT_S))
    while [[ ! -f "$decode_window_marker" ]]; do
      if ! kill -0 "$client_pid" >/dev/null 2>&1; then
        wait "$client_pid" || true
        docker logs "$container" > "$case_dir/server.log" 2>&1 || true
        docker rm -f "$container" >/dev/null 2>&1 || true
        echo "client exited before decode window marker" >&2
        return 1
      fi
      if [[ "$SECONDS" -ge "$deadline" ]]; then
        docker logs "$container" > "$case_dir/server.log" 2>&1 || true
        docker rm -f "$container" >/dev/null 2>&1 || true
        echo "timed out waiting for decode window marker: $decode_window_marker" >&2
        return 1
      fi
      sleep 1
    done
    capture_faults "$container" "$case_dir/faults_before.json"
    start_perf_faults "$container" "$case_dir"
    date -Is > "$decode_window_continue"
    set +e
    wait "$client_pid"
    case_status=$?
    set -e
  else
    capture_faults "$container" "$case_dir/faults_before.json"
    start_perf_faults "$container" "$case_dir"
    set +e
    run_client 2>&1 | tee "$case_dir/client.log"
    case_status=${PIPESTATUS[0]}
    set -e
  fi

  if [[ "$RPP_POST_CLIENT_SLEEP_S" -gt 0 ]]; then
    sleep "$RPP_POST_CLIENT_SLEEP_S"
  fi

  stop_perf_faults "$case_dir"
  capture_faults "$container" "$case_dir/faults_after.json"
  write_fault_delta "$case_dir"
  docker logs "$container" > "$case_dir/server.log" 2>&1 || true
  docker rm -f "$container" >/dev/null 2>&1 || true
  if [[ "$case_status" -ne 0 ]]; then
    return "$case_status"
  fi
}

case_index=0
if [[ "$RUN_BASELINE" == "1" ]]; then
  run_case "baseline_fifo_k${BASELINE_MAX_DECODE_SLOTS}" "$((PORT_BASE + case_index))" \
    "$BASELINE_POOL_SIZE" "$BASELINE_POOL_SIZE" "$BASELINE_DISPATCH_MODE" \
    --rpp-scheduler fifo \
    --rpp-max-decode-slots "$BASELINE_MAX_DECODE_SLOTS"
  case_index=$((case_index + 1))
fi

live_case_name="decode_rpp_top${PREDICT_TOPK}_${RPP_PREFETCH_IO_MODE}_ubatch_k${ACTIVE_UNION_MAX_SLOTS}"
if [[ "$RUN_LIVE" == "1" ]]; then
  if [[ "$RPP_PREFETCH_ASYNC" == "1" ]]; then
    live_case_name="decode_rpp_top${PREDICT_TOPK}_${RPP_PREFETCH_IO_MODE}_async_ubatch_k${ACTIVE_UNION_MAX_SLOTS}"
  fi

  run_case "$live_case_name" "$((PORT_BASE + case_index))" \
    "$LIVE_POOL_SIZE" "$LIVE_POOL_SIZE" "$LIVE_DISPATCH_MODE" \
    --rpp-live-model "$TORCHSCRIPT" \
    --rpp-oracle-trace "$ORACLE_TRACE" \
    --rpp-tensor-ranges "$TENSOR_RANGES" \
    --rpp-scheduler decode_rpp_top8_mmap_ubatch \
    --rpp-max-decode-slots "$ACTIVE_UNION_MAX_SLOTS" \
    --rpp-predict-topk "$PREDICT_TOPK" \
    --rpp-prefetch-count-threshold "$RPP_PREFETCH_COUNT_THRESHOLD" \
    --rpp-reclaim-mode "$RPP_RECLAIM_MODE" \
    --rpp-cache-threshold "$CACHE_THRESHOLD" \
    --rpp-page-stride "$PAGE_STRIDE"
  case_index=$((case_index + 1))
fi

python3 - "$STATISTICS_DIR" "$OUT_ROOT/REPORT.md" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
report_path = Path(sys.argv[2])
rows = []
for path in sorted(root.glob("*/summary.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows.append(data)
summary = {"cases": rows}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = [
    "# Live RPP Scheduler Comparison",
    "",
    f"- statistics_dir: `{root}`",
    f"- inference threads: `{os.environ.get('THREADS', '')}`",
    f"- inference CPU mask: `{os.environ.get('INFERENCE_CPU_MASK', '')}`",
    f"- predictor threads: `{os.environ.get('RPP_PREDICTOR_THREADS', '')}`",
    f"- predictor CPU mask: `{os.environ.get('RPP_PREDICTOR_CPU_MASK', '')}`",
    f"- baseline pool size: `{os.environ.get('BASELINE_POOL_SIZE', '')}`",
    f"- baseline dispatch mode: `{os.environ.get('BASELINE_DISPATCH_MODE', '')}`",
    f"- live pool size: `{os.environ.get('LIVE_POOL_SIZE', '')}`",
    f"- live dispatch mode: `{os.environ.get('LIVE_DISPATCH_MODE', '')}`",
    f"- docker memory limit: `{os.environ.get('DOCKER_MEMORY_LIMIT', '')}`",
    f"- docker memory swap limit: `{os.environ.get('DOCKER_MEMORY_SWAP_LIMIT', '')}`",
    f"- decode fault window: `{os.environ.get('DECODE_FAULT_WINDOW', '')}`",
    f"- require full decode batch: `{os.environ.get('RPP_REQUIRE_FULL_DECODE_BATCH', '')}`",
    f"- run live case: `{os.environ.get('RUN_LIVE', '')}`",
    f"- prefetch io mode: `{os.environ.get('RPP_PREFETCH_IO_MODE', '')}`",
    f"- prefetch async: `{os.environ.get('RPP_PREFETCH_ASYNC', '')}`",
    f"- prefetch queue policy: `{os.environ.get('RPP_PREFETCH_QUEUE_POLICY', '')}`",
    f"- prefetch FIFO candidate order: `{os.environ.get('RPP_PREFETCH_FIFO_CANDIDATE_ORDER', '')}`",
    f"- prefetch queue cap: `{os.environ.get('RPP_PREFETCH_QUEUE_CAP', '')}`",
    f"- prefetch max staleness ubatches: `{os.environ.get('RPP_PREFETCH_MAX_STALENESS_UBATCHES', '')}`",
    f"- ubatch prefetch max experts: `{os.environ.get('RPP_UBATCH_PREFETCH_MAX_EXPERTS', '')}`",
    f"- reclaim mode: `{os.environ.get('RPP_RECLAIM_MODE', '')}`",
    f"- reclaim lookahead ubatches: `{os.environ.get('RPP_RECLAIM_LOOKAHEAD_UBATCHES', '')}`",
    f"- cross ubatch layer prefetch: `{os.environ.get('RPP_CROSS_UBATCH_LAYER_PREFETCH', '')}`",
    f"- cross ubatch layer lookahead: `{os.environ.get('RPP_CROSS_UBATCH_LAYER_LOOKAHEAD', '')}`",
    f"- layer frontier prefetch: `{os.environ.get('RPP_LAYER_FRONTIER_PREFETCH', '')}`",
    f"- layer frontier lookahead layers: `{os.environ.get('RPP_LAYER_FRONTIER_LOOKAHEAD_LAYERS', '')}`",
    f"- layer frontier jobs per tick: `{os.environ.get('RPP_LAYER_FRONTIER_JOBS_PER_TICK', '')}`",
    f"- layer frontier max distance: `{os.environ.get('RPP_LAYER_FRONTIER_MAX_DISTANCE', '')}`",
    f"- prefetch prefill only: `{os.environ.get('RPP_PREFETCH_PREFILL_ONLY', '')}`",
    f"- predict topk: `{os.environ.get('PREDICT_TOPK', '')}`",
    "- cases: " + ", ".join(f"`{row.get('case_name', '')}`" for row in rows),
    "",
    "| case | wall s | tok/s | p50 s | p95 s | predicted tokens |",
    "|---|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['case_name']} | {row['total_wall_s']:.6f} | {row['predicted_tok_s']:.6f} | "
        f"{row['completion_p50_s']:.6f} | {row['completion_p95_s']:.6f} | {row['total_predicted_tokens']} |"
    )
lines.append("")
text = "\n".join(lines)
(root / "REPORT.md").write_text(text, encoding="utf-8")
report_path.write_text(text, encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
PY

echo "live RPP scheduler statistics: $STATISTICS_DIR"
