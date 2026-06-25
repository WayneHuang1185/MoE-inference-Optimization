#!/usr/bin/env bash
# Estimate expert page-cache capacity from existing decode residency matrices.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MEM_NAME="$(basename "$MEM_DIR")"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$ROOT_DIR"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
MEM_ROOT="${MEM_ROOT:-experiments/gemma4_IO_behaviors/$MEM_NAME}"
OUT_ROOT="${OUT_ROOT:-$MEM_ROOT/expert_capacity}"
STATISTICS_DIR="${STATISTICS_DIR:-$OUT_ROOT/statistics/capacity_estimate_${RUN_TIMESTAMP}}"
FIGURES_DIR="${FIGURES_DIR:-$OUT_ROOT/figures/capacity_estimate_${RUN_TIMESTAMP}}"
MATRIX_GLOB="${MATRIX_GLOB:-$MEM_ROOT/statistics/decode_expert_cache_fault_*/**/expert_cache_matrices.json}"
TENSOR_RANGES="${TENSOR_RANGES:-experiments/gemma4_bottleneck/results/gemma4_26b_tensor_ranges.csv}"
DECODE_TARGETS="${DECODE_TARGETS:-5,50,100}"
PROJECTION_MAX_TOKEN="${PROJECTION_MAX_TOKEN:-10000}"
BASELINE_DECODE_TOKENS="${BASELINE_DECODE_TOKENS:-5}"
CALIBRATION_START_TOKEN="${CALIBRATION_START_TOKEN:-50}"

if [[ -z "${KV_BYTES_PER_TOKEN:-}" ]]; then
  KV_BYTES_PER_TOKEN="$(
    python3 - "$MEM_ROOT" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
logs = sorted(root.glob("statistics/decode_expert_cache_fault_*/server.log"))
if not logs:
    print(225280)
    raise SystemExit
text = logs[-1].read_text(encoding="utf-8", errors="replace").splitlines()
pending_cells = None
total = 0.0
for line in text:
    m = re.search(r"llama_kv_cache: size =\s*([0-9.]+)\s*MiB\s*\(\s*(\d+)\s*cells", line)
    if m:
        total += float(m.group(1)) * 1024 * 1024 / max(int(m.group(2)), 1)
print(int(round(total)) if total > 0 else 225280)
PY
  )"
  KV_SOURCE="server_log"
else
  KV_SOURCE="env"
fi
SEQUENCES="${SEQUENCES:-1}"
GENERATE_FIGURES="${GENERATE_FIGURES:-1}"

mkdir -p "$STATISTICS_DIR" "$FIGURES_DIR"

FIGURE_ARGS=()
if [[ "$GENERATE_FIGURES" == "0" ]]; then
  FIGURE_ARGS+=(--skip-figures)
fi

python3 "$OUT_ROOT/utils/estimate_expert_capacity.py" \
  --input "$MEM_NAME=$MATRIX_GLOB" \
  --tensor-ranges "$TENSOR_RANGES" \
  --out-dir "$STATISTICS_DIR" \
  --figures-dir "$FIGURES_DIR" \
  --decode-targets "$DECODE_TARGETS" \
  --projection-max-token "$PROJECTION_MAX_TOKEN" \
  --baseline-decode-tokens "$BASELINE_DECODE_TOKENS" \
  --calibration-start-token "$CALIBRATION_START_TOKEN" \
  --kv-bytes-per-token "$KV_BYTES_PER_TOKEN" \
  --kv-source "$KV_SOURCE" \
  --sequences "$SEQUENCES" \
  "${FIGURE_ARGS[@]}"

cp "$STATISTICS_DIR/REPORT.md" "$OUT_ROOT/REPORT.md"

echo "statistics: $STATISTICS_DIR"
echo "figures: $FIGURES_DIR"
