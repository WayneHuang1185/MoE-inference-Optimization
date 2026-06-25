#!/usr/bin/env bash
# Run a small runtime matrix for baseline and conservative RPP prefetch configs.
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT_DIR"

OUT_DIR="experiments/gemma4_IO_behaviors/mem8G"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
MATRIX_DIR="${MATRIX_DIR:-$OUT_DIR/statistics/runtime_rpp_prefetch_matrix_${RUN_TIMESTAMP}}"
REPEATS="${REPEATS:-1}"
BASE_PORT="${BASE_PORT:-8080}"
N_PREDICT="${N_PREDICT:-16}"
PROMPT_LIMIT="${PROMPT_LIMIT:-1}"

mkdir -p "$MATRIX_DIR"

run_case() {
  local case_name="$1"
  local repeat="$2"
  shift 2

  local port=$((BASE_PORT + repeat * 10 + CASE_INDEX))
  local case_dir="$MATRIX_DIR/${case_name}_r${repeat}"
  echo "running case=${case_name} repeat=${repeat} port=${port}"
  STATISTICS_DIR="$case_dir" \
  RUN_TIMESTAMP="${RUN_TIMESTAMP}_${case_name}_r${repeat}" \
  PORT="$port" \
  N_PREDICT="$N_PREDICT" \
  PROMPT_LIMIT="$PROMPT_LIMIT" \
  "$@" \
  "$OUT_DIR/run_mem8g_runtime_rpp_prefetch.sh"
}

for repeat in $(seq 1 "$REPEATS"); do
  CASE_INDEX=0
  run_case baseline "$repeat" env PREFETCH_MODE=none

  CASE_INDEX=1
  run_case rpp_b30 "$repeat" env PREFETCH_MODE=rpp PREFETCH_BUDGET=30 PREFETCH_THRESHOLD=0 ADVICE_CACHE_TOKENS=4

  CASE_INDEX=2
  run_case rpp_b60 "$repeat" env PREFETCH_MODE=rpp PREFETCH_BUDGET=60 PREFETCH_THRESHOLD=0 ADVICE_CACHE_TOKENS=4

  CASE_INDEX=3
  run_case rpp_thr095 "$repeat" env PREFETCH_MODE=rpp PREFETCH_BUDGET=0 PREFETCH_THRESHOLD=0.95 ADVICE_CACHE_TOKENS=4
done

python3 - "$MATRIX_DIR" "$OUT_DIR/REPORT.md" <<'PY'
import json
import sys
from pathlib import Path

matrix_dir = Path(sys.argv[1])
report_path = Path(sys.argv[2])

rows = []
for path in sorted(matrix_dir.glob("*/summary.json")):
    case_name = path.parent.name
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        continue
    for item in data:
        row = {
            "case": case_name,
            "prompt": item.get("prompt_name", ""),
            "mode": item.get("mode", ""),
            "tokens": int(item.get("tokens", 0)),
            "wall_s": float(item.get("wall_s", 0.0)),
            "tok_s": float(item.get("tokens_per_second_wall", 0.0)),
            "candidate_mean": float(item.get("candidate_count_mean", 0.0)),
            "fadvise_calls": int(item.get("fadvise_calls", 0)),
            "advised_mb": float(item.get("advised_bytes", 0)) / (1024 * 1024),
            "cached_skip": int(item.get("skipped_cached", 0)),
            "predict_s": float(item.get("predict_s_total", 0.0)),
            "prefetch_s": float(item.get("prefetch_s_total", 0.0)),
        }
        rows.append(row)

baseline = next((r for r in rows if r["case"].startswith("baseline_")), None)
for row in rows:
    if baseline and row["wall_s"] > 0:
        row["wall_delta_pct"] = ((row["wall_s"] / baseline["wall_s"]) - 1.0) * 100.0
    else:
        row["wall_delta_pct"] = 0.0

lines = [
    "# mem8G Runtime RPP Prefetch Matrix",
    "",
    f"- statistics: `{matrix_dir}`",
    f"- baseline reference: `{baseline['case'] if baseline else 'missing'}`",
    "",
    "| case | prompt | tokens | wall_s | wall_delta_pct | tok/s | candidate_mean | fadvise_calls | advised_mb | cached_skip | predict_s | prefetch_s |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['case']} | {row['prompt']} | {row['tokens']} | "
        f"{row['wall_s']:.3f} | {row['wall_delta_pct']:.2f} | {row['tok_s']:.3f} | "
        f"{row['candidate_mean']:.1f} | {row['fadvise_calls']} | {row['advised_mb']:.1f} | "
        f"{row['cached_skip']} | {row['predict_s']:.3f} | {row['prefetch_s']:.3f} |"
    )

matrix_report = matrix_dir / "REPORT.md"
matrix_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
report_path.write_text(matrix_report.read_text(encoding="utf-8"), encoding="utf-8")
print(f"runtime RPP prefetch matrix report: {matrix_report}")
PY
