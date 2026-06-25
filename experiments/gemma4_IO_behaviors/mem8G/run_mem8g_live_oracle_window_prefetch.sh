#!/usr/bin/env bash
# Live oracle-window RPP prefetch experiment for the 5-sample truth dataset.
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT_DIR"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="experiments/gemma4_IO_behaviors/mem8G"
UTIL_DIR="$OUT_ROOT/utils"
STATISTICS_DIR="${STATISTICS_DIR:-$OUT_ROOT/statistics/live_oracle_window_prefetch_${RUN_TIMESTAMP}}"
REMOTE_HOST="${REMOTE_HOST:-nthu-cs}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/wayne/project}"
LIVE_IMAGE="${LIVE_IMAGE:-localhost/gemma4-rpp-train:cpu}"
MEMORY_CAP="${MEMORY_CAP:-8g}"
MODEL="${MODEL:-models/gemma4-26B.gguf}"
TENSOR_RANGES="${TENSOR_RANGES:-experiments/gemma4_bottleneck/results/gemma4_26b_tensor_ranges.csv}"
TRUTH_ROOT="${TRUTH_ROOT:-experiments/gemma4_IO_behaviors/mem8G/statistics/truth_prefill_logits_5_short_n16_20260623_040619}"
TRUTH_MANIFEST="${TRUTH_MANIFEST:-$TRUTH_ROOT/router_label_npz/dump_pack_manifest.csv}"
TRUTH_PROMPTS="${TRUTH_PROMPTS:-$TRUTH_ROOT/selected_prompt_database.jsonl}"
BUILD_IN_DOCKER="${BUILD_IN_DOCKER:-0}"
if [[ -z "${BUILD_DIR:-}" ]]; then
  if [[ "$BUILD_IN_DOCKER" == "1" ]]; then
    BUILD_DIR="llama.cpp/build-rpp-live"
  else
    BUILD_DIR="llama.cpp/build-rpp-live-host"
  fi
fi
LLAMA_SERVER="${LLAMA_SERVER:-$BUILD_DIR/bin/llama-server}"
PORT="${PORT:-8096}"
THREADS="${THREADS:-31}"
INFERENCE_CPU_MASK="${INFERENCE_CPU_MASK:-0-30}"
CTX_SIZE="${CTX_SIZE:-8192}"
POOL_SIZE="${POOL_SIZE:-5}"
PROMPT_LIMIT="${PROMPT_LIMIT:-5}"
N_PREDICT="${N_PREDICT:-16}"
RPP_WINDOW_SIZE="${RPP_WINDOW_SIZE:-10}"
RPP_PREFETCH_COUNT_THRESHOLD="${RPP_PREFETCH_COUNT_THRESHOLD:-5}"
RPP_RECLAIM_MODE="${RPP_RECLAIM_MODE:-log-only}"
RPP_PREFETCH_PREFILL_ONLY="${RPP_PREFETCH_PREFILL_ONLY:-0}"
RPP_PREFETCH_IO_MODE="${RPP_PREFETCH_IO_MODE:-readahead}"
RPP_UBATCH_PREFETCH_MIN_COUNT="${RPP_UBATCH_PREFETCH_MIN_COUNT:-$RPP_PREFETCH_COUNT_THRESHOLD}"
RPP_UBATCH_PREFETCH_DENSITY_THRESHOLD="${RPP_UBATCH_PREFETCH_DENSITY_THRESHOLD:-0.5}"
DOCKER_GPU_ARGS="${DOCKER_GPU_ARGS:-}"
LLAMA_EXTRA_ARGS="${LLAMA_EXTRA_ARGS:-}"
BUILD_JOBS="${BUILD_JOBS:-2}"
RUN_BASELINE="${RUN_BASELINE:-0}"
BASELINE_GATE_MODE="${BASELINE_GATE_MODE:-strict}"

SYNC_PATHS=(
  llama.cpp/common/common.h
  llama.cpp/common/arg.cpp
  llama.cpp/tools/server/CMakeLists.txt
  llama.cpp/tools/server/server-context.cpp
  llama.cpp/tools/server/server-rpp-live.cpp
  llama.cpp/tools/server/server-rpp-live.h
  llama.cpp/tools/server/server-task.cpp
  llama.cpp/tools/server/server-task.h
  experiments/gemma4_IO_behaviors/mem8G/run_mem8g_live_oracle_window_prefetch.sh
  experiments/gemma4_IO_behaviors/mem8G/utils/build_oracle_live_trace.py
  experiments/gemma4_IO_behaviors/mem8G/utils/live_rpp_fixed_pool_client.py
)

if [[ "${LIVE_RPP_REMOTE:-0}" != "1" ]]; then
  rsync -avR "${SYNC_PATHS[@]}" "$REMOTE_HOST:$REMOTE_ROOT/"
  ssh "$REMOTE_HOST" "cd '$REMOTE_ROOT' && LIVE_RPP_REMOTE=1 RUN_TIMESTAMP='$RUN_TIMESTAMP' STATISTICS_DIR='$STATISTICS_DIR' LIVE_IMAGE='$LIVE_IMAGE' MEMORY_CAP='$MEMORY_CAP' MODEL='$MODEL' TENSOR_RANGES='$TENSOR_RANGES' TRUTH_ROOT='$TRUTH_ROOT' TRUTH_MANIFEST='$TRUTH_MANIFEST' TRUTH_PROMPTS='$TRUTH_PROMPTS' BUILD_DIR='$BUILD_DIR' LLAMA_SERVER='$LLAMA_SERVER' PORT='$PORT' THREADS='$THREADS' INFERENCE_CPU_MASK='$INFERENCE_CPU_MASK' CTX_SIZE='$CTX_SIZE' POOL_SIZE='$POOL_SIZE' PROMPT_LIMIT='$PROMPT_LIMIT' N_PREDICT='$N_PREDICT' RPP_WINDOW_SIZE='$RPP_WINDOW_SIZE' RPP_PREFETCH_COUNT_THRESHOLD='$RPP_PREFETCH_COUNT_THRESHOLD' RPP_RECLAIM_MODE='$RPP_RECLAIM_MODE' RPP_PREFETCH_PREFILL_ONLY='$RPP_PREFETCH_PREFILL_ONLY' RPP_PREFETCH_IO_MODE='$RPP_PREFETCH_IO_MODE' RPP_UBATCH_PREFETCH_MIN_COUNT='$RPP_UBATCH_PREFETCH_MIN_COUNT' RPP_UBATCH_PREFETCH_DENSITY_THRESHOLD='$RPP_UBATCH_PREFETCH_DENSITY_THRESHOLD' DOCKER_GPU_ARGS='$DOCKER_GPU_ARGS' LLAMA_EXTRA_ARGS='$LLAMA_EXTRA_ARGS' BUILD_JOBS='$BUILD_JOBS' BUILD_IN_DOCKER='$BUILD_IN_DOCKER' RUN_BASELINE='$RUN_BASELINE' BASELINE_GATE_MODE='$BASELINE_GATE_MODE' bash '$OUT_ROOT/run_mem8g_live_oracle_window_prefetch.sh'"
  exit 0
fi

mkdir -p "$STATISTICS_DIR"
read -r -a DOCKER_GPU_ARGS_ARR <<< "$DOCKER_GPU_ARGS"

if [[ "$BUILD_IN_DOCKER" == "1" ]]; then
  docker run --rm \
    --network host \
    --memory="$MEMORY_CAP" \
    --memory-swap="$MEMORY_CAP" \
    "${DOCKER_GPU_ARGS_ARR[@]}" \
    -v "$ROOT_DIR:/workspace" \
    -w /workspace \
    "$LIVE_IMAGE" \
    bash -lc '
set -euo pipefail
if ! command -v cmake >/dev/null 2>&1 || ! command -v c++ >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends cmake build-essential ninja-build pkg-config
fi
cmake -S llama.cpp -B "'"$BUILD_DIR"'" \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_WEBUI=OFF \
  -DGGML_OPENMP=OFF \
  -DLLAMA_RPP_LIVE=ON
cmake --build "'"$BUILD_DIR"'" --target llama-server -j"'"$BUILD_JOBS"'"
'
else
  cmake -S llama.cpp -B "$BUILD_DIR" \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_WEBUI=OFF \
    -DGGML_OPENMP=OFF \
    -DLLAMA_RPP_LIVE=ON
  cmake --build "$BUILD_DIR" --target llama-server -j"$BUILD_JOBS"
fi

docker run --rm \
  --network host \
  --memory="$MEMORY_CAP" \
  --memory-swap="$MEMORY_CAP" \
  -v "$ROOT_DIR:/workspace" \
  -w /workspace \
  "$LIVE_IMAGE" \
  python3 "$UTIL_DIR/build_oracle_live_trace.py" \
    --manifest "$TRUTH_MANIFEST" \
    --limit "$PROMPT_LIMIT" \
    --include-prefill \
    --output "$STATISTICS_DIR/oracle_live_trace.jsonl"

python3 - "$TRUTH_PROMPTS" "$TRUTH_MANIFEST" "$TRUTH_ROOT" "$STATISTICS_DIR/prompts" "$STATISTICS_DIR/oracle_sample_ids.txt" "$STATISTICS_DIR/oracle_generation_settings.json" "$STATISTICS_DIR/oracle_expected_completions.json" "$PROMPT_LIMIT" <<'PY'
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

CONTAINER="gemma4-live-oracle-window-${RUN_TIMESTAMP}-${PORT}"
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
        raise SystemExit("server container exited before health became ready")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as resp:
            if resp.status < 500:
                raise SystemExit(0)
    except Exception as exc:
        last = exc
    time.sleep(1)
raise SystemExit(f"server did not become ready: {last}")
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
import sys
from pathlib import Path

case_dir = Path(sys.argv[1])
before = json.loads((case_dir / "faults_before.json").read_text(encoding="utf-8"))
after = json.loads((case_dir / "faults_after.json").read_text(encoding="utf-8"))
delta = {"measurement": "server container faults during fixed-pool client request window"}
for key in ["proc_minflt", "proc_majflt", "cgroup_pgfault", "cgroup_pgmajfault"]:
    if key in before and key in after:
        delta[key + "_delta"] = after[key] - before[key]
(case_dir / "fault_delta.json").write_text(json.dumps(delta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_case() {
  local case_name="$1"
  local case_port="$2"
  local case_dir="$3"
  local mode="$4"
  local send_oracle_ids="$5"
  local container="gemma4-live-oracle-window-${RUN_TIMESTAMP}-${case_name}-${case_port}"
  local rpp_cli="--rpp-scheduler fifo"
  if [[ "$mode" == "oracle" ]]; then
    rpp_cli="--rpp-scheduler oracle-window --rpp-oracle-trace /workspace/${STATISTICS_DIR}/oracle_live_trace.jsonl --rpp-tensor-ranges ${TENSOR_RANGES} --rpp-window-size ${RPP_WINDOW_SIZE} --rpp-prefetch-count-threshold ${RPP_PREFETCH_COUNT_THRESHOLD} --rpp-reclaim-mode ${RPP_RECLAIM_MODE}"
  fi
  mkdir -p "$case_dir"
  docker rm -f "$container" >/dev/null 2>&1 || true
  docker run -d \
    --name "$container" \
    --network host \
    --memory="$MEMORY_CAP" \
    --memory-swap="$MEMORY_CAP" \
    "${DOCKER_GPU_ARGS_ARR[@]}" \
    -v "$ROOT_DIR:/workspace" \
    -w /workspace \
    "$LIVE_IMAGE" \
    bash -lc '
set -euo pipefail
export LD_LIBRARY_PATH="/workspace/'"$BUILD_DIR"'/bin:${LD_LIBRARY_PATH:-}"
export RPP_LIVE_OUT_DIR="/workspace/'"$case_dir"'"
export RPP_PREFETCH_PREFILL_ONLY="'"$RPP_PREFETCH_PREFILL_ONLY"'"
export RPP_PREFETCH_IO_MODE="'"$RPP_PREFETCH_IO_MODE"'"
export RPP_UBATCH_PREFETCH_MIN_COUNT="'"$RPP_UBATCH_PREFETCH_MIN_COUNT"'"
export RPP_UBATCH_PREFETCH_DENSITY_THRESHOLD="'"$RPP_UBATCH_PREFETCH_DENSITY_THRESHOLD"'"
python3 experiments/gemma4_IO_behaviors/mem48G/drop_file_cache.py "'"$MODEL"'" >/tmp/live_oracle_window_drop_file_cache.log 2>&1 || true
if command -v taskset >/dev/null 2>&1; then
  exec taskset -c "'"$INFERENCE_CPU_MASK"'" "'"$LLAMA_SERVER"'" \
    -m "'"$MODEL"'" \
    -c "'"$CTX_SIZE"'" \
    -t "'"$THREADS"'" \
    -ngl 0 \
    -np "'"$POOL_SIZE"'" \
    --host 127.0.0.1 \
    --port "'"$case_port"'" \
    --no-warmup \
    --slots \
    --no-repack \
    '"$rpp_cli"' \
    '"$LLAMA_EXTRA_ARGS"'
fi
exec "'"$LLAMA_SERVER"'" \
  -m "'"$MODEL"'" \
  -c "'"$CTX_SIZE"'" \
  -t "'"$THREADS"'" \
  -ngl 0 \
  -np "'"$POOL_SIZE"'" \
  --host 127.0.0.1 \
  --port "'"$case_port"'" \
  --no-warmup \
  --slots \
  --no-repack \
  '"$rpp_cli"' \
  '"$LLAMA_EXTRA_ARGS"'
'

  wait_server "$case_port" "$container"
  capture_faults "$container" "$case_dir/faults_before.json"

  local oracle_client_args=()
  if [[ "$send_oracle_ids" == "1" ]]; then
    oracle_client_args=(
      --oracle-sample-id-file "$STATISTICS_DIR/oracle_sample_ids.txt"
      --oracle-settings-file "$STATISTICS_DIR/oracle_generation_settings.json"
      --expected-completions-file "$STATISTICS_DIR/oracle_expected_completions.json"
    )
  fi
  docker run --rm \
    --network host \
    --memory="$MEMORY_CAP" \
    --memory-swap="$MEMORY_CAP" \
    -v "$ROOT_DIR:/workspace" \
    -w /workspace \
    "$LIVE_IMAGE" \
    python3 "$UTIL_DIR/live_rpp_fixed_pool_client.py" \
      --base-url "http://127.0.0.1:${case_port}" \
      --model "$MODEL" \
      --prompt-dir "$STATISTICS_DIR/prompts" \
      --output-dir "$case_dir/client" \
      --case-name "$case_name" \
      --limit "$PROMPT_LIMIT" \
      --pool-size "$POOL_SIZE" \
      --dispatch-mode waves \
      --n-predict "$N_PREDICT" \
      "${oracle_client_args[@]}" \
    2>&1 | tee "$case_dir/client.log"

  capture_faults "$container" "$case_dir/faults_after.json"
  write_fault_delta "$case_dir"
  docker logs "$container" > "$case_dir/server.log" 2>&1 || true
  docker rm -f "$container" >/dev/null 2>&1 || true
}

run_case "oracle_window" "$PORT" "$STATISTICS_DIR/oracle_window" "oracle" "1"

if python3 - "$STATISTICS_DIR" "$BASELINE_GATE_MODE" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
baseline_gate_mode = sys.argv[2]
oracle_summary = json.loads((run_dir / "oracle_window" / "summary.json").read_text(encoding="utf-8"))
client_summary = json.loads((run_dir / "oracle_window" / "client" / "summary.json").read_text(encoding="utf-8"))
lead_path = run_dir / "oracle_window" / "ubatch_prefetch_lead_trace.csv"
event_path = run_dir / "oracle_window" / "ubatch_prefetch_events.csv"
ubatch_prefetch_bad_leads = 0
ubatch_prefetch_wrong_targets = 0
ubatch_prefetch_bad_thresholds = 0
ubatch_prefetch_min_lead_us = None
import csv
import math
if lead_path.exists():
    with lead_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            lead_us = int(row.get("lead_us") or 0)
            ubatch_prefetch_min_lead_us = lead_us if ubatch_prefetch_min_lead_us is None else min(ubatch_prefetch_min_lead_us, lead_us)
            if lead_us <= 0:
                ubatch_prefetch_bad_leads += 1
if event_path.exists():
    with event_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("target_decode_call_id") != row.get("source_decode_call_id"):
                ubatch_prefetch_wrong_targets += 1
            if row.get("target_ubatch_id") != row.get("current_ubatch_id"):
                ubatch_prefetch_wrong_targets += 1
            counter = int(row.get("counter") or 0)
            token_count = int(row.get("target_token_count") or 0)
            required_count = int(row.get("required_count") or 0)
            min_count = int(row.get("min_count") or 0)
            density_threshold = float(row.get("density_threshold") or 0.0)
            expected_required = max(min_count, math.ceil(max(1, token_count) * density_threshold))
            if required_count != expected_required or counter < required_count:
                ubatch_prefetch_bad_thresholds += 1
validation = {
    "baseline_gate_mode": baseline_gate_mode,
    "global_order_prediction_mismatches": int(oracle_summary.get("global_order_prediction_mismatches", 0)),
    "ubatch_prediction_mismatches": int(oracle_summary.get("ubatch_prediction_mismatches", 0)),
    "prefill_token_slot_mismatches": int(oracle_summary.get("prefill_token_slot_mismatches", 0)),
    "decode_token_slot_mismatches": int(oracle_summary.get("decode_token_slot_mismatches", 0)),
    "token_slot_mismatches": int(oracle_summary.get("token_slot_mismatches", 0)),
    "oracle_mismatches": int(oracle_summary.get("oracle_mismatches", 0)),
    "expected_content_mismatches": int(client_summary.get("expected_content_mismatches", 0)),
    "ubatch_prefetch_events": int(oracle_summary.get("ubatch_prefetch_events", 0)),
    "ubatch_prefetch_lead_rows": int(oracle_summary.get("ubatch_prefetch_lead_rows", 0)),
    "ubatch_prefetch_bad_leads": ubatch_prefetch_bad_leads,
    "ubatch_prefetch_wrong_targets": ubatch_prefetch_wrong_targets,
    "ubatch_prefetch_bad_thresholds": ubatch_prefetch_bad_thresholds,
    "ubatch_prefetch_min_lead_us": ubatch_prefetch_min_lead_us,
    "allowed_for_baseline_comparison": False,
}
if baseline_gate_mode == "order-only":
    validation["allowed_for_baseline_comparison"] = (
        validation["global_order_prediction_mismatches"] == 0 and
        validation["ubatch_prediction_mismatches"] == 0 and
        validation["prefill_token_slot_mismatches"] == 0 and
        validation["ubatch_prefetch_bad_leads"] == 0 and
        validation["ubatch_prefetch_wrong_targets"] == 0 and
        validation["ubatch_prefetch_bad_thresholds"] == 0
    )
else:
    validation["allowed_for_baseline_comparison"] = (
        validation["global_order_prediction_mismatches"] == 0 and
        validation["ubatch_prediction_mismatches"] == 0 and
        validation["token_slot_mismatches"] == 0 and
        validation["oracle_mismatches"] == 0 and
        validation["expected_content_mismatches"] == 0 and
        validation["ubatch_prefetch_bad_leads"] == 0 and
        validation["ubatch_prefetch_wrong_targets"] == 0 and
        validation["ubatch_prefetch_bad_thresholds"] == 0
    )
(run_dir / "validation_summary.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
raise SystemExit(0 if validation["allowed_for_baseline_comparison"] else 1)
PY
then
  VALIDATION_OK=1
else
  VALIDATION_OK=0
fi

if [[ "$RUN_BASELINE" == "1" && "$VALIDATION_OK" == "1" ]]; then
  run_case "baseline_fifo" "$((PORT + 1))" "$STATISTICS_DIR/baseline_fifo" "baseline" "1"
else
  echo "baseline comparison skipped: RUN_BASELINE=$RUN_BASELINE VALIDATION_OK=$VALIDATION_OK"
fi

python3 - "$STATISTICS_DIR" "$OUT_ROOT/REPORT.md" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
top_report = Path(sys.argv[2])
summary_path = run_dir / "oracle_window" / "summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
client_summary_path = run_dir / "oracle_window" / "client" / "summary.json"
client_summary = json.loads(client_summary_path.read_text(encoding="utf-8")) if client_summary_path.exists() else {}
validation_path = run_dir / "validation_summary.json"
validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
baseline_fault_path = run_dir / "baseline_fifo" / "fault_delta.json"
baseline_faults = json.loads(baseline_fault_path.read_text(encoding="utf-8")) if baseline_fault_path.exists() else {}
oracle_faults = json.loads((run_dir / "oracle_window" / "fault_delta.json").read_text(encoding="utf-8"))

comparison = {}
for key in ["proc_minflt_delta", "proc_majflt_delta", "cgroup_pgfault_delta", "cgroup_pgmajfault_delta"]:
    b = baseline_faults.get(key)
    o = oracle_faults.get(key)
    if isinstance(b, int) and isinstance(o, int):
        comparison[key] = {
            "baseline": b,
            "oracle_window": o,
            "delta": o - b,
            "reduction": b - o,
            "reduction_pct": ((b - o) / b * 100.0) if b else 0.0,
        }

combined = {
    "baseline_faults": baseline_faults,
    "oracle_faults": oracle_faults,
    "page_fault_comparison": comparison,
    "oracle_summary": summary,
    "oracle_client_summary": client_summary,
    "validation": validation,
}
(run_dir / "comparison_summary.json").write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8")

main_fault = comparison.get("cgroup_pgfault_delta") or comparison.get("proc_minflt_delta") or {}
lines = [
    "",
    "## Live Oracle Window Prefetch",
    "",
    f"- statistics_dir: `{run_dir}`",
    f"- memory_cap: `{__import__('os').environ.get('MEMORY_CAP', '')}`",
    f"- window_size: `{summary.get('window_size', '')}`",
    f"- prefetch_count_threshold: `{summary.get('prefetch_count_threshold', '')}`",
    f"- prefetch_io_mode: `{summary.get('prefetch_io_mode', '')}`",
    f"- ubatch_prefetch_min_count: `{summary.get('ubatch_prefetch_min_count', '')}`",
    f"- ubatch_prefetch_density_threshold: `{summary.get('ubatch_prefetch_density_threshold', '')}`",
    f"- prefetch_prefill_only: `{summary.get('prefetch_prefill_only', '')}`",
    f"- prefetch_stop_decode_call: `{summary.get('prefetch_stop_decode_call', '')}`",
    f"- prefetch_events: `{summary.get('prefetch_events', '')}`",
    f"- ubatch_prefetch_events: `{summary.get('ubatch_prefetch_events', '')}`",
    f"- ubatch_prefetch_lead_rows: `{summary.get('ubatch_prefetch_lead_rows', '')}`",
    f"- ubatch_prefetch_bad_leads: `{validation.get('ubatch_prefetch_bad_leads', '')}`",
    f"- ubatch_prefetch_wrong_targets: `{validation.get('ubatch_prefetch_wrong_targets', '')}`",
    f"- ubatch_prefetch_bad_thresholds: `{validation.get('ubatch_prefetch_bad_thresholds', '')}`",
    f"- ubatch_prefetch_min_lead_us: `{validation.get('ubatch_prefetch_min_lead_us', '')}`",
    f"- fadvise_calls: `{summary.get('fadvise_calls', '')}`",
    f"- fadvise_errors: `{summary.get('fadvise_errors', '')}`",
    f"- advised_bytes: `{summary.get('advised_bytes', '')}`",
    f"- reclaim_candidates: `{summary.get('reclaim_candidates', '')}`",
    f"- global_order_prediction_mismatches: `{summary.get('global_order_prediction_mismatches', '')}`",
    f"- ubatch_prediction_mismatches: `{summary.get('ubatch_prediction_mismatches', '')}`",
    f"- prefill_token_slot_mismatches: `{summary.get('prefill_token_slot_mismatches', '')}`",
    f"- decode_token_slot_mismatches: `{summary.get('decode_token_slot_mismatches', '')}`",
    f"- token_slot_mismatches: `{summary.get('token_slot_mismatches', '')}`",
    f"- oracle_mismatches: `{summary.get('oracle_mismatches', '')}`",
    f"- expected_content_mismatches: `{client_summary.get('expected_content_mismatches', '')}`",
    f"- baseline_gate_mode: `{validation.get('baseline_gate_mode', '')}`",
    f"- allowed_for_baseline_comparison: `{validation.get('allowed_for_baseline_comparison', '')}`",
    f"- baseline_page_faults: `{main_fault.get('baseline', '')}`",
    f"- oracle_page_faults: `{main_fault.get('oracle_window', '')}`",
    f"- page_fault_reduction: `{main_fault.get('reduction', '')}`",
    f"- page_fault_reduction_pct: `{main_fault.get('reduction_pct', '')}`",
    "",
]
with top_report.open("a", encoding="utf-8") as f:
    f.write("\n".join(lines))
PY

echo "live oracle-window prefetch statistics: $STATISTICS_DIR"
