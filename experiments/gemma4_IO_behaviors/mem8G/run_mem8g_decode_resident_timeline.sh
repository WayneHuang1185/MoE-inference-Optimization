#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT_DIR"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="experiments/gemma4_IO_behaviors/mem8G"
UTIL_DIR="$OUT_DIR/utils"
STATISTICS_DIR="${STATISTICS_DIR:-$OUT_DIR/statistics/layer_transition_${RUN_TIMESTAMP}}"
FIGURES_DIR="${FIGURES_DIR:-$OUT_DIR/figures/layer_transition_${RUN_TIMESTAMP}}"
IMAGE="${IMAGE:-localhost/gemma4-ram-bench:24.04}"
MODEL="${MODEL:-models/gemma4-26B.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-llama.cpp/build/bin/llama-server}"
PROMPT_DIR="${PROMPT_DIR:-experiments/gemma4_bottleneck/router_prediction_prompts}"
TENSOR_RANGES="${TENSOR_RANGES:-experiments/gemma4_bottleneck/results/gemma4_26b_tensor_ranges.csv}"
PORT="${PORT:-8080}"
THREADS="${THREADS:-$(nproc 2>/dev/null || echo 16)}"
CTX_SIZE="${CTX_SIZE:-12288}"
N_PREDICT="${N_PREDICT:-10000}"
PROMPT_LIMIT="${PROMPT_LIMIT:-1}"
PAGE_STRIDE="${PAGE_STRIDE:-16}"
RESIDENT_THRESHOLD="${RESIDENT_THRESHOLD:-0.95}"
PREDICTED_TOP_K="${PREDICTED_TOP_K:-8}"
CAPACITY_CONFIG="${CAPACITY_CONFIG:-$OUT_DIR/expert_capacity/statistics/capacity_estimate_budget_model_decode500_recal_20260519_1150/run_config.json}"
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
  -e PREDICTED_TOP_K="$PREDICTED_TOP_K" \
  -e CAPACITY_CONFIG="$CAPACITY_CONFIG" \
  -e PERF="$PERF" \
  -e LLAMA_EXTRA_ARGS="$LLAMA_EXTRA_ARGS" \
  "$IMAGE" \
  bash -lc '
set -euo pipefail

mkdir -p "$STATISTICS_DIR" "$FIGURES_DIR" /tmp/llama_slots_resident_timeline
SERVER_LOG="$STATISTICS_DIR/server.log"
PROBE_LOG="$STATISTICS_DIR/decode_resident_timeline_probe.log"

python3 experiments/gemma4_IO_behaviors/mem48G/drop_file_cache.py "$MODEL" > "$STATISTICS_DIR/drop_file_cache.log" 2>&1 || true
python3 experiments/gemma4_IO_behaviors/mem48G/pfn_preflight.py --output "$STATISTICS_DIR/pfn_preflight.json" || true

export LD_LIBRARY_PATH="/workspace/llama.cpp/build/bin:${LD_LIBRARY_PATH:-}"
read -r -a LLAMA_EXTRA_ARGS_ARR <<< "${LLAMA_EXTRA_ARGS:-}"

"$LLAMA_SERVER" \
  -m "$MODEL" \
  -c "${CTX_SIZE:-12288}" \
  -t "${THREADS:-16}" \
  -ngl 0 \
  -np 1 \
  --host 127.0.0.1 \
  --port "${PORT:-8080}" \
  --no-warmup \
  --slots \
  --slot-save-path /tmp/llama_slots_resident_timeline \
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

python3 "$UTIL_DIR/decode_resident_timeline_probe.py" \
  --base-url "http://127.0.0.1:${PORT:-8080}" \
  --model "$MODEL" \
  --tensor-ranges "$TENSOR_RANGES" \
  --prompt-dir "$PROMPT_DIR" \
  --output-dir "$STATISTICS_DIR" \
  --n-predict "${N_PREDICT:-10000}" \
  --limit "${PROMPT_LIMIT:-1}" \
  --page-stride "${PAGE_STRIDE:-16}" \
  --resident-threshold "${RESIDENT_THRESHOLD:-0.95}" \
  --predicted-top-k "${PREDICTED_TOP_K:-8}" \
  --capacity-config "${CAPACITY_CONFIG:-}" \
  2>&1 | tee "$PROBE_LOG"

mapfile -t TIMELINES < <(find "$STATISTICS_DIR" -mindepth 2 -maxdepth 2 -name resident_timeline.csv | sort)
mapfile -t LAYER_TRANSITIONS < <(find "$STATISTICS_DIR" -mindepth 2 -maxdepth 2 -name resident_layer_transition_summary.csv | sort)
if [[ "${#TIMELINES[@]}" -eq 1 ]]; then
  python3 "$UTIL_DIR/plot_resident_timeline.py" \
    --timeline "${TIMELINES[0]}" \
    --output-dir "$FIGURES_DIR" \
    --title-prefix "8G decode resident timeline"
elif [[ "${#TIMELINES[@]}" -gt 1 ]]; then
  for timeline in "${TIMELINES[@]}"; do
    prompt_dir="$(basename "$(dirname "$timeline")")"
    python3 "$UTIL_DIR/plot_resident_timeline.py" \
      --timeline "$timeline" \
      --output-dir "$FIGURES_DIR/$prompt_dir" \
      --title-prefix "8G decode resident timeline: ${prompt_dir}"
  done
fi
if [[ "${#LAYER_TRANSITIONS[@]}" -eq 1 ]]; then
  python3 "$UTIL_DIR/plot_layer_transition_concentration.py" \
    --layer-transitions "${LAYER_TRANSITIONS[0]}" \
    --output-dir "$FIGURES_DIR" \
    --statistics-output-dir "$STATISTICS_DIR" \
    --title-prefix "8G layer transition"
elif [[ "${#LAYER_TRANSITIONS[@]}" -gt 1 ]]; then
  for layer_transition in "${LAYER_TRANSITIONS[@]}"; do
    prompt_dir="$(basename "$(dirname "$layer_transition")")"
    python3 "$UTIL_DIR/plot_layer_transition_concentration.py" \
      --layer-transitions "$layer_transition" \
      --output-dir "$FIGURES_DIR/$prompt_dir" \
      --statistics-output-dir "$(dirname "$layer_transition")" \
      --title-prefix "8G layer transition: ${prompt_dir}"
  done
fi

python3 - "$OUT_DIR/REPORT.md" "$STATISTICS_DIR" "$FIGURES_DIR" "$RUN_TIMESTAMP" "$N_PREDICT" "$PROMPT_LIMIT" "$RESIDENT_THRESHOLD" "$PAGE_STRIDE" <<'"'"'PY'"'"'
import json
import sys
from pathlib import Path

report, stats_dir, figures_dir, ts, n_predict, prompt_limit, threshold, stride = sys.argv[1:]
summary_paths = sorted(Path(stats_dir).glob("prompt_*/summary.json"))
if not summary_paths:
    raise SystemExit("no summary.json files found")
summary = json.loads(summary_paths[0].read_text())
meta = json.loads((summary_paths[0].parent / "run_meta.json").read_text())
actual = summary["actual_resident"]
pred = summary["predicted_resident"]
prompt_name_value = meta.get("prompt_name")
actual_mean = actual["mean"]
actual_min = actual["min"]
actual_max = actual["max"]
actual_final = actual["final"]
pred_mean = pred["mean"]
pred_min = pred["min"]
pred_max = pred["max"]
pred_final = pred["final"]
total_swap_in = summary["total_swap_in"]
total_swap_out = summary["total_swap_out"]
top_concentration_path = Path(stats_dir) / "layer_transition_concentration_summary.json"
prompt_concentration_path = summary_paths[0].parent / "layer_transition_concentration_summary.json"
concentration_path = top_concentration_path if top_concentration_path.exists() else prompt_concentration_path
concentration = json.loads(concentration_path.read_text()) if concentration_path.exists() else {}
def cstat(name, field):
    return float(concentration.get(name, {}).get(field, 0))
gained_active_mean = cstat("gained_active_layers", "mean")
gained_active_final = cstat("gained_active_layers", "final")
lost_active_mean = cstat("lost_active_layers", "mean")
lost_active_final = cstat("lost_active_layers", "final")
gained_top1_mean = cstat("gained_top1_layer_share", "mean")
gained_top1_final = cstat("gained_top1_layer_share", "final")
lost_top1_mean = cstat("lost_top1_layer_share", "mean")
lost_top1_final = cstat("lost_top1_layer_share", "final")
gained_entropy_mean = cstat("gained_entropy_normalized", "mean")
gained_entropy_final = cstat("gained_entropy_normalized", "final")
lost_entropy_mean = cstat("lost_entropy_normalized", "mean")
lost_entropy_final = cstat("lost_entropy_normalized", "final")
lines = [
    "# mem8G Decode Resident Timeline",
    "",
    f"- timestamp: {ts}",
    "- docker_memory_limit: 8g",
    f"- n_predict: {n_predict}",
    f"- prompt_limit: {prompt_limit}",
    f"- prompt_name: {prompt_name_value}",
    f"- resident_threshold: {threshold}",
    f"- page_stride: {stride}",
    f"- statistics: {stats_dir}",
    f"- figures: {figures_dir}",
    "",
    "## Summary",
    "",
    f"- actual resident mean/min/max/final: {actual_mean:.3f} / {actual_min} / {actual_max} / {actual_final}",
    f"- predicted resident mean/min/max/final: {pred_mean:.3f} / {pred_min} / {pred_max} / {pred_final}",
    f"- total swap-in count: {total_swap_in}",
    f"- total swap-out count: {total_swap_out}",
    "",
    "## Layer Transition Concentration",
    "",
    f"- gained active layers mean/final: {gained_active_mean:.3f} / {gained_active_final:.3f}",
    f"- lost active layers mean/final: {lost_active_mean:.3f} / {lost_active_final:.3f}",
    f"- gained top1 layer share mean/final: {gained_top1_mean:.3f} / {gained_top1_final:.3f}",
    f"- lost top1 layer share mean/final: {lost_top1_mean:.3f} / {lost_top1_final:.3f}",
    f"- gained entropy normalized mean/final: {gained_entropy_mean:.3f} / {gained_entropy_final:.3f}",
    f"- lost entropy normalized mean/final: {lost_entropy_mean:.3f} / {lost_entropy_final:.3f}",
    "",
]
Path(report).write_text("\n".join(lines), encoding="utf-8")
PY

echo "resident timeline statistics: $STATISTICS_DIR"
echo "resident timeline figures: $FIGURES_DIR"
'
