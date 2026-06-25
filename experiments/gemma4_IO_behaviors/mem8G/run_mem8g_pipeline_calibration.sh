#!/usr/bin/env bash
# Calibrate mem8G CPU-only inference parallelism and RPP sidecar overhead.
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT_DIR"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="experiments/gemma4_IO_behaviors/mem8G"
STATISTICS_DIR="${STATISTICS_DIR:-$OUT_DIR/statistics/pipeline_calibration_${RUN_TIMESTAMP}}"
FIGURES_DIR="${FIGURES_DIR:-$OUT_DIR/figures/pipeline_calibration_${RUN_TIMESTAMP}}"
SERVER_IMAGE="${SERVER_IMAGE:-localhost/gemma4-ram-bench:24.04}"
SIDECAR_IMAGE="${SIDECAR_IMAGE:-localhost/gemma4-rpp-train:cpu}"
MODEL="${MODEL:-models/gemma4-26B.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-llama.cpp/build/bin/llama-server}"
BENCH_SCRIPT="${BENCH_SCRIPT:-experiments/gemma4_bottleneck/benchmark_prefill_decode.py}"
PROMPT_DIR="${PROMPT_DIR:-experiments/gemma4_bottleneck/router_prediction_prompts}"
PROMPT_FILE="${PROMPT_FILE:-$PROMPT_DIR/00_moe_intro.txt}"
TENSOR_RANGES="${TENSOR_RANGES:-experiments/gemma4_bottleneck/results/gemma4_26b_tensor_ranges.csv}"
CHECKPOINT="${CHECKPOINT:-experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/checkpoint_best.pt}"
CONFIG="${CONFIG:-experiments/gemma4_bottleneck/results/rpp_best_h512_d128_continue_60_20260516_1346/config.json}"
BASE_PORT="${BASE_PORT:-8090}"
THREAD_SWEEP="${THREAD_SWEEP:-8,16,24,32}"
THREAD_RUNS="${THREAD_RUNS:-3}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
RUNTIME_REPEATS="${RUNTIME_REPEATS:-3}"
N_PREDICT="${N_PREDICT:-64}"
PROMPT_LIMIT="${PROMPT_LIMIT:-1}"
CTX_SIZE="${CTX_SIZE:-8192}"
PREFETCH_BUDGET="${PREFETCH_BUDGET:-30}"
PREFETCH_THRESHOLD="${PREFETCH_THRESHOLD:-0}"
PREFETCH_INTERVAL="${PREFETCH_INTERVAL:-1}"
ADVICE_CACHE_TOKENS="${ADVICE_CACHE_TOKENS:-4}"
PREDICT_TOPK="${PREDICT_TOPK:-8}"
TOUCH_BYTES="${TOUCH_BYTES:-0}"
DEVICE="${DEVICE:-auto}"
LLAMA_EXTRA_ARGS="${LLAMA_EXTRA_ARGS:-}"
DOCKER_GPU_ARGS="${DOCKER_GPU_ARGS:-}"

case " $LLAMA_EXTRA_ARGS " in
  *" --no-repack "*) ;;
  *) LLAMA_EXTRA_ARGS="${LLAMA_EXTRA_ARGS} --no-repack" ;;
esac

mkdir -p "$STATISTICS_DIR" "$FIGURES_DIR"
read -r -a DOCKER_GPU_ARGS_ARR <<< "$DOCKER_GPU_ARGS"

to_container_path() {
  local path="$1"
  case "$path" in
    "$ROOT_DIR"/*) printf '/workspace/%s' "${path#$ROOT_DIR/}" ;;
    /*) printf '%s' "$path" ;;
    *) printf '/workspace/%s' "$path" ;;
  esac
}

write_cpu_environment() {
  local out="$STATISTICS_DIR/cpu_environment.txt"
  {
    echo "# Host"
    date -Is
    uname -a
    echo
    echo "## nproc"
    nproc || true
    echo
    echo "## lscpu"
    lscpu || true
    echo
    echo "## top CPU users before experiment"
    ps -eo pid,ppid,pcpu,pmem,comm,args --sort=-pcpu | head -30 || true
    echo
    echo "# mem8G Docker cgroup view"
  } > "$out"

  docker run --rm \
    --memory=8g \
    --memory-swap=8g \
    -v "$ROOT_DIR:/workspace" \
    -w /workspace \
    "$SERVER_IMAGE" \
    bash -lc '
set -euo pipefail
echo "## nproc"
nproc || true
echo
echo "## lscpu"
lscpu || true
echo
echo "## cgroup cpu"
for f in /sys/fs/cgroup/cpu.max /sys/fs/cgroup/cpuset.cpus /sys/fs/cgroup/cpuset.cpus.effective; do
  [[ -f "$f" ]] && printf "%s: %s\n" "$f" "$(cat "$f")"
done
echo
echo "## cgroup memory"
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.swap.max /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.swap.current; do
  [[ -f "$f" ]] && printf "%s: %s\n" "$f" "$(cat "$f")"
done
echo
echo "## top CPU users inside calibration container"
ps -eo pid,ppid,pcpu,pmem,comm,args --sort=-pcpu | head -30 || true
' >> "$out" 2>&1 || true
}

wait_until_ready() {
  local port="$1"
  local container_name="$2"
  python3 - "$port" "$container_name" <<'PY'
import subprocess
import sys
import time
import urllib.request

port = sys.argv[1]
container_name = sys.argv[2]
deadline = time.time() + 600
last = None
while time.time() < deadline:
    status = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
        capture_output=True,
        text=True,
    )
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

run_thread_case() {
  local threads="$1"
  local port="$2"
  local case_dir="$STATISTICS_DIR/thread_${threads}"
  local container_name="gemma4-pipeline-calib-thread-${RUN_TIMESTAMP}-${threads}-${port}"
  local server_log="$case_dir/server.log"
  local bench_log="$case_dir/benchmark_stdout.log"
  mkdir -p "$case_dir"

  docker rm -f "$container_name" >/dev/null 2>&1 || true
  docker run -d \
    --name "$container_name" \
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
    -e PORT="$port" \
    -e THREADS="$threads" \
    -e CTX_SIZE="$CTX_SIZE" \
    -e LLAMA_EXTRA_ARGS="$LLAMA_EXTRA_ARGS" \
    "$SERVER_IMAGE" \
    bash -lc '
set -euo pipefail
python3 experiments/gemma4_IO_behaviors/mem48G/drop_file_cache.py "$MODEL" >/tmp/pipeline_calibration_drop_file_cache.log 2>&1 || true
export LD_LIBRARY_PATH="/workspace/llama.cpp/build/bin:${LD_LIBRARY_PATH:-}"
read -r -a LLAMA_EXTRA_ARGS_ARR <<< "${LLAMA_EXTRA_ARGS:-}"
exec "$LLAMA_SERVER" \
  -m "$MODEL" \
  -c "${CTX_SIZE:-8192}" \
  -t "${THREADS:-16}" \
  -ngl 0 \
  -np 1 \
  --host 127.0.0.1 \
  --port "${PORT:-8080}" \
  --no-warmup \
  "${LLAMA_EXTRA_ARGS_ARR[@]}"
' >/dev/null

  set +e
  wait_until_ready "$port" "$container_name"
  local ready_status="$?"
  set -e
  docker logs "$container_name" > "$server_log" 2>&1 || true
  if [[ "$ready_status" != "0" ]]; then
    docker rm -f "$container_name" >/dev/null 2>&1 || true
    return "$ready_status"
  fi

  local server_pid
  server_pid="$(docker exec "$container_name" bash -lc "pgrep -n -f '[l]lama-server'")"
  {
    echo "threads=$threads"
    echo "port=$port"
    echo "server_pid=$server_pid"
    echo "n_predict=$N_PREDICT"
    echo "runs=$THREAD_RUNS"
    echo "warmup_runs=$WARMUP_RUNS"
    echo "prompt_file=$PROMPT_FILE"
  } > "$case_dir/meta.env"

  set +e
  docker exec "$container_name" \
    python3 "$(to_container_path "$BENCH_SCRIPT")" \
      --pid "$server_pid" \
      --url "http://127.0.0.1:${port}/completion" \
      --runs "$THREAD_RUNS" \
      --warmup-runs "$WARMUP_RUNS" \
      --n-predict "$N_PREDICT" \
      --prompt-file "$(to_container_path "$PROMPT_FILE")" \
      --jsonl "$(to_container_path "$case_dir/prefill_decode_benchmark.jsonl")" \
      --csv "$(to_container_path "$case_dir/prefill_decode_benchmark.csv")" \
    2>&1 | tee "$bench_log"
  local bench_status="${PIPESTATUS[0]}"
  set -e

  docker logs "$container_name" > "$server_log" 2>&1 || true
  docker rm -f "$container_name" >/dev/null 2>&1 || true
  return "$bench_status"
}

run_runtime_case() {
  local mode="$1"
  local repeat="$2"
  local port="$3"
  local case_dir="$STATISTICS_DIR/runtime_${mode}_r${repeat}"
  mkdir -p "$case_dir"
  echo "running runtime mode=${mode} repeat=${repeat} port=${port}"
  STATISTICS_DIR="$case_dir" \
  FIGURES_DIR="$FIGURES_DIR" \
  RUN_TIMESTAMP="${RUN_TIMESTAMP}_${mode}_r${repeat}" \
  SERVER_IMAGE="$SERVER_IMAGE" \
  SIDECAR_IMAGE="$SIDECAR_IMAGE" \
  MODEL="$MODEL" \
  LLAMA_SERVER="$LLAMA_SERVER" \
  PROMPT_DIR="$PROMPT_DIR" \
  TENSOR_RANGES="$TENSOR_RANGES" \
  CHECKPOINT="$CHECKPOINT" \
  CONFIG="$CONFIG" \
  PORT="$port" \
  THREADS="${RUNTIME_THREADS:-$(nproc 2>/dev/null || echo 16)}" \
  CTX_SIZE="$CTX_SIZE" \
  N_PREDICT="$N_PREDICT" \
  PROMPT_LIMIT="$PROMPT_LIMIT" \
  PREFETCH_MODE="$mode" \
  PREFETCH_BUDGET="$PREFETCH_BUDGET" \
  PREFETCH_THRESHOLD="$PREFETCH_THRESHOLD" \
  PREFETCH_INTERVAL="$PREFETCH_INTERVAL" \
  ADVICE_CACHE_TOKENS="$ADVICE_CACHE_TOKENS" \
  PREDICT_TOPK="$PREDICT_TOPK" \
  TOUCH_BYTES="$TOUCH_BYTES" \
  DEVICE="$DEVICE" \
  LLAMA_EXTRA_ARGS="$LLAMA_EXTRA_ARGS" \
  DOCKER_GPU_ARGS="$DOCKER_GPU_ARGS" \
  "$OUT_DIR/run_mem8g_runtime_rpp_prefetch.sh"
}

aggregate_results() {
  python3 - "$STATISTICS_DIR" "$FIGURES_DIR" "$OUT_DIR/REPORT.md" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

statistics_dir = Path(sys.argv[1])
figures_dir = Path(sys.argv[2])
top_report = Path(sys.argv[3])
figures_dir.mkdir(parents=True, exist_ok=True)

def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0

thread_rows: list[dict[str, object]] = []
for case_dir in sorted(statistics_dir.glob("thread_*")):
    try:
        threads = int(case_dir.name.split("_", 1)[1])
    except (IndexError, ValueError):
        continue
    for row in read_csv(case_dir / "prefill_decode_benchmark.csv"):
        out = dict(row)
        out["threads_setting"] = threads
        out["case"] = case_dir.name
        if row.get("phase") == "decode":
            pred_n = float(row.get("predicted_n") or 0.0)
            pred_ms = float(row.get("predicted_ms") or 0.0)
            out["decode_ms_per_token"] = pred_ms / pred_n if pred_n > 0 else ""
        else:
            out["decode_ms_per_token"] = ""
        thread_rows.append(out)

thread_fields = [
    "case",
    "threads_setting",
    "run",
    "phase",
    "bound_guess",
    "wall_s",
    "user_cpu_s",
    "system_cpu_s",
    "total_cpu_s",
    "cpu_parallelism",
    "block_io_delay_s",
    "minor_faults",
    "major_faults",
    "read_mb",
    "read_bytes",
    "read_syscalls",
    "prompt_n",
    "prompt_ms",
    "predicted_n",
    "predicted_ms",
    "decode_ms_per_token",
    "boundary_source",
]
write_csv(statistics_dir / "thread_sweep.csv", thread_rows, thread_fields)

mode_rows: list[dict[str, object]] = []
event_rows: list[dict[str, object]] = []
for case_dir in sorted(statistics_dir.glob("runtime_*_r*")):
    stem, repeat = case_dir.name.rsplit("_r", 1)
    mode = stem.removeprefix("runtime_")
    for row in read_csv(case_dir / "mode_summary.csv"):
        out = dict(row)
        out["case"] = case_dir.name
        out["mode"] = row.get("mode") or mode
        out["repeat"] = repeat
        mode_rows.append(out)
    for event_csv in sorted(case_dir.glob("prompt_*/runtime_prefetch_events.csv")):
        for row in read_csv(event_csv):
            out = dict(row)
            out["case"] = case_dir.name
            out["mode"] = mode
            out["repeat"] = repeat
            event_rows.append(out)

mode_fields = [
    "case",
    "repeat",
    "prompt_index",
    "prompt_name",
    "mode",
    "tokens",
    "wall_s",
    "wall_ms_per_token",
    "tokens_per_second_wall",
    "predicted_per_token_ms",
    "predict_p50_ms",
    "predict_p95_ms",
    "predict_max_ms",
    "prefetch_p50_ms",
    "prefetch_p95_ms",
    "prefetch_max_ms",
    "token_delta_p50_ms",
    "token_delta_p95_ms",
    "token_delta_max_ms",
    "candidate_count_mean",
    "fadvise_calls",
    "fadvise_errors",
    "advised_bytes",
    "touched_bytes",
    "skipped_cached",
    "skipped_cached_bytes",
    "predict_s_total",
    "prefetch_s_total",
    "prefetch_budget",
    "prefetch_threshold",
    "predict_topk",
]
write_csv(statistics_dir / "mode_summary.csv", mode_rows, mode_fields)

event_fields = [
    "case",
    "repeat",
    "mode",
    "prompt_index",
    "prompt_name",
    "token_index",
    "token_count",
    "generated_chars",
    "candidate_count",
    "fadvise_calls",
    "fadvise_errors",
    "advised_bytes",
    "touched_bytes",
    "skipped_cached",
    "skipped_cached_bytes",
    "predict_s",
    "prefetch_s",
    "token_elapsed_s",
    "token_delta_s",
    "elapsed_s",
]
write_csv(statistics_dir / "runtime_prefetch_events.csv", event_rows, event_fields)

decode_by_thread: dict[int, list[dict[str, object]]] = {}
for row in thread_rows:
    if row.get("phase") != "decode":
        continue
    threads = int(row["threads_setting"])
    decode_by_thread.setdefault(threads, []).append(row)

thread_summary = []
for threads, rows in sorted(decode_by_thread.items()):
    cpu_parallelism = mean([float(r.get("cpu_parallelism") or 0.0) for r in rows])
    wall_s = mean([float(r.get("wall_s") or 0.0) for r in rows])
    decode_ms = mean([float(r.get("decode_ms_per_token") or 0.0) for r in rows if r.get("decode_ms_per_token") != ""])
    io_delay_s = mean([float(r.get("block_io_delay_s") or 0.0) for r in rows])
    read_mb = mean([float(r.get("read_mb") or 0.0) for r in rows])
    thread_summary.append({
        "threads_setting": threads,
        "decode_cpu_parallelism_avg": cpu_parallelism,
        "decode_wall_s_avg": wall_s,
        "decode_ms_per_token_avg": decode_ms,
        "decode_block_io_delay_s_avg": io_delay_s,
        "decode_read_mb_avg": read_mb,
    })

best_parallel = max(thread_summary, key=lambda r: r["decode_cpu_parallelism_avg"], default=None)
best_latency = min(
    [r for r in thread_summary if r["decode_ms_per_token_avg"] > 0],
    key=lambda r: r["decode_ms_per_token_avg"],
    default=None,
)

mode_summary: dict[str, dict[str, float]] = {}
for mode in sorted({str(r.get("mode", "")) for r in mode_rows if r.get("mode")}):
    rows = [r for r in mode_rows if r.get("mode") == mode]
    mode_summary[mode] = {
        "wall_ms_per_token_avg": mean([float(r.get("wall_ms_per_token") or 0.0) for r in rows]),
        "tokens_per_second_wall_avg": mean([float(r.get("tokens_per_second_wall") or 0.0) for r in rows]),
        "predict_p95_ms_avg": mean([float(r.get("predict_p95_ms") or 0.0) for r in rows]),
        "prefetch_p95_ms_avg": mean([float(r.get("prefetch_p95_ms") or 0.0) for r in rows]),
        "candidate_count_mean_avg": mean([float(r.get("candidate_count_mean") or 0.0) for r in rows]),
        "runs": float(len(rows)),
    }

none = mode_summary.get("none", {})
predict_only = mode_summary.get("predict_only", {})
rpp = mode_summary.get("rpp", {})
decode_ms_reference = (
    float(best_latency["decode_ms_per_token_avg"])
    if best_latency
    else float(none.get("wall_ms_per_token_avg") or 0.0)
)
predict_plus_prefetch_p95 = float(rpp.get("predict_p95_ms_avg") or 0.0) + float(rpp.get("prefetch_p95_ms_avg") or 0.0)

analysis = {
    "decode_ms_per_token_reference": decode_ms_reference,
    "predict_plus_prefetch_p95_ms": predict_plus_prefetch_p95,
    "can_overlap_by_p95": bool(decode_ms_reference > 0 and predict_plus_prefetch_p95 < decode_ms_reference),
    "predict_only_wall_delta_pct_vs_none": (
        ((float(predict_only.get("wall_ms_per_token_avg") or 0.0) / float(none.get("wall_ms_per_token_avg") or 1.0)) - 1.0) * 100.0
        if none else None
    ),
    "rpp_wall_delta_pct_vs_predict_only": (
        ((float(rpp.get("wall_ms_per_token_avg") or 0.0) / float(predict_only.get("wall_ms_per_token_avg") or 1.0)) - 1.0) * 100.0
        if predict_only else None
    ),
    "rpp_wall_delta_pct_vs_none": (
        ((float(rpp.get("wall_ms_per_token_avg") or 0.0) / float(none.get("wall_ms_per_token_avg") or 1.0)) - 1.0) * 100.0
        if none else None
    ),
}

summary = {
    "statistics_dir": str(statistics_dir),
    "figures_dir": str(figures_dir),
    "thread_summary": thread_summary,
    "best_parallelism_thread": best_parallel,
    "best_latency_thread": best_latency,
    "mode_summary": mode_summary,
    "analysis": analysis,
}
(statistics_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = [
    "# mem8G Pipeline Calibration",
    "",
    f"- statistics: `{statistics_dir}`",
    f"- figures: `{figures_dir}`",
    "",
    "## CPU Parallelism Sweep",
    "",
    "| threads | decode_cpu_parallelism | decode_ms/token | decode_wall_s | io_delay_s | read_mb |",
    "|---:|---:|---:|---:|---:|---:|",
]
for row in thread_summary:
    lines.append(
        f"| {row['threads_setting']} | {row['decode_cpu_parallelism_avg']:.2f} | "
        f"{row['decode_ms_per_token_avg']:.3f} | {row['decode_wall_s_avg']:.3f} | "
        f"{row['decode_block_io_delay_s_avg']:.3f} | {row['decode_read_mb_avg']:.1f} |"
    )
if best_parallel:
    lines.extend([
        "",
        (
            f"Peak observed decode CPU parallelism was `{best_parallel['decode_cpu_parallelism_avg']:.2f}x` "
            f"at `THREADS={best_parallel['threads_setting']}`."
        ),
    ])

lines.extend([
    "",
    "## Runtime Sidecar Modes",
    "",
    "| mode | runs | wall_ms/token | tok/s | predict_p95_ms | prefetch_p95_ms | candidates/token |",
    "|---|---:|---:|---:|---:|---:|---:|",
])
for mode, row in mode_summary.items():
    lines.append(
        f"| {mode} | {int(row['runs'])} | {row['wall_ms_per_token_avg']:.3f} | "
        f"{row['tokens_per_second_wall_avg']:.3f} | {row['predict_p95_ms_avg']:.3f} | "
        f"{row['prefetch_p95_ms_avg']:.3f} | {row['candidate_count_mean_avg']:.1f} |"
    )

lines.extend([
    "",
    "## Calibration Criteria",
    "",
    f"- decode reference: `{analysis['decode_ms_per_token_reference']:.3f} ms/token`",
    f"- predictor + prefetch p95: `{analysis['predict_plus_prefetch_p95_ms']:.3f} ms/token`",
    f"- p95 overlap criterion: `{'PASS' if analysis['can_overlap_by_p95'] else 'FAIL'}`",
])
if analysis["predict_only_wall_delta_pct_vs_none"] is not None:
    lines.append(f"- predict_only wall delta vs none: `{analysis['predict_only_wall_delta_pct_vs_none']:.2f}%`")
if analysis["rpp_wall_delta_pct_vs_predict_only"] is not None:
    lines.append(f"- rpp wall delta vs predict_only: `{analysis['rpp_wall_delta_pct_vs_predict_only']:.2f}%`")
if analysis["rpp_wall_delta_pct_vs_none"] is not None:
    lines.append(f"- rpp wall delta vs none: `{analysis['rpp_wall_delta_pct_vs_none']:.2f}%`")

lines.extend([
    "",
    "## Output Files",
    "",
    "- `cpu_environment.txt`",
    "- `thread_sweep.csv`",
    "- `runtime_prefetch_events.csv`",
    "- `mode_summary.csv`",
    "- `summary.json`",
])

report = "\n".join(lines) + "\n"
(statistics_dir / "REPORT.md").write_text(report, encoding="utf-8")
top_report.write_text(report, encoding="utf-8")
PY
}

write_cpu_environment

IFS=',' read -r -a THREAD_VALUES <<< "$THREAD_SWEEP"
thread_case_index=0
for threads in "${THREAD_VALUES[@]}"; do
  threads="${threads//[[:space:]]/}"
  [[ -z "$threads" ]] && continue
  port=$((BASE_PORT + thread_case_index))
  echo "running thread sweep THREADS=${threads} port=${port}"
  run_thread_case "$threads" "$port"
  thread_case_index=$((thread_case_index + 1))
done

runtime_case_index=0
for repeat in $(seq 1 "$RUNTIME_REPEATS"); do
  for mode in none predict_only rpp; do
    port=$((BASE_PORT + 100 + repeat * 10 + runtime_case_index))
    run_runtime_case "$mode" "$repeat" "$port"
    runtime_case_index=$((runtime_case_index + 1))
  done
done

aggregate_results
echo "pipeline calibration statistics: $STATISTICS_DIR"
echo "pipeline calibration figures: $FIGURES_DIR"
