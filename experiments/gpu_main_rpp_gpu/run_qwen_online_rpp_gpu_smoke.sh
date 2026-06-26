#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/hazcashi/lab}"
RUNTIME_DIR="${RUNTIME_DIR:-${ROOT}/rpp_runtime_implementation}"
LLAMA_DIR="${LLAMA_DIR:-${RUNTIME_DIR}/llama.cpp}"
BUILD_DIR="${BUILD_DIR:-${LLAMA_DIR}/build-rpp-cuda124}"
OUT_DIR="${OUT_DIR:-${RUNTIME_DIR}/outputs/qwen36_rpp_gpu}"

PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
MODEL="${MODEL:-${ROOT}/model/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/experience/RPP/qwen36_rpp/results/rpp_train_d64/checkpoint_best.pt}"
RPP_CONFIG="${RPP_CONFIG:-${ROOT}/experience/RPP/qwen36_rpp/results/rpp_train_d64/config.json}"
RPP_MODEL_PY="${RPP_MODEL_PY:-${ROOT}/experience/RPP/qwen36_rpp/model.py}"
PAGE_MAP="${PAGE_MAP:-${OUT_DIR}/expert_page_map.csv}"

SIDE_PORT="${SIDE_PORT:-18181}"
SERVER_PORT="${SERVER_PORT:-18135}"
TOP_K="${TOP_K:-2}"
CACHE_MIB="${CACHE_MIB:-1024}"
STAGING_MIB="${STAGING_MIB:-64}"
N_PREDICT="${N_PREDICT:-6}"
CTX_SIZE="${CTX_SIZE:-256}"
THREADS="${THREADS:-8}"
PROMPT="${PROMPT:-Explain RPP in one short sentence.}"

STAMP="$(date +%m%d_%H%M%S)"
TRACE="${TRACE:-${OUT_DIR}/rpp_gpu_online_trace_${STAMP}.jsonl}"
METRICS="${METRICS:-${OUT_DIR}/sidecar_metrics_${STAMP}.jsonl}"
SIDECAR_LOG="${SIDECAR_LOG:-${OUT_DIR}/sidecar_${STAMP}.log}"
SERVER_LOG="${SERVER_LOG:-${OUT_DIR}/server_${STAMP}.log}"

mkdir -p "${OUT_DIR}"

if [[ ! -x "${BUILD_DIR}/bin/llama-server" ]]; then
  echo "missing llama-server: ${BUILD_DIR}/bin/llama-server" >&2
  echo "build first: cmake --build ${BUILD_DIR} --target llama-server -j \$(nproc)" >&2
  exit 1
fi

if [[ ! -f "${PAGE_MAP}" ]]; then
  "${PYTHON}" "${RUNTIME_DIR}/scripts/build_expert_page_map.py" \
    --model "${MODEL}" \
    --out "${PAGE_MAP}"
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SIDECAR_PID:-}" ]] && kill -0 "${SIDECAR_PID}" 2>/dev/null; then
    kill "${SIDECAR_PID}" 2>/dev/null || true
    wait "${SIDECAR_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

"${PYTHON}" "${RUNTIME_DIR}/scripts/rpp_sidecar.py" \
  --checkpoint "${CHECKPOINT}" \
  --config "${RPP_CONFIG}" \
  --model-py "${RPP_MODEL_PY}" \
  --device cpu \
  --top-k "${TOP_K}" \
  --host 127.0.0.1 \
  --port "${SIDE_PORT}" \
  --metrics-jsonl "${METRICS}" \
  >"${SIDECAR_LOG}" 2>&1 &
SIDECAR_PID="$!"

"${PYTHON}" - <<PY
import socket, time
port = int("${SIDE_PORT}")
deadline = time.time() + 120
while time.time() < deadline:
    with socket.socket() as s:
        s.settimeout(0.5)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(0)
    time.sleep(0.5)
raise SystemExit("sidecar did not open port")
PY

cd "${LLAMA_DIR}"
LD_LIBRARY_PATH="${BUILD_DIR}/bin:${LD_LIBRARY_PATH:-}" \
"${BUILD_DIR}/bin/llama-server" \
  -m "${MODEL}" \
  --ctx-size "${CTX_SIZE}" \
  --threads "${THREADS}" \
  --n-gpu-layers 999 \
  --cpu-moe \
  --rpp-mode online \
  --rpp-sidecar-url "http://127.0.0.1:${SIDE_PORT}" \
  --rpp-sidecar-timeout-ms 60000 \
  --rpp-page-map "${PAGE_MAP}" \
  --rpp-prefetch-depth 1 \
  --rpp-prefetch-top-k "${TOP_K}" \
  --rpp-host-prefetch pretouch \
  --rpp-prefetch-threads 1 \
  --rpp-gpu-correction on \
  --rpp-gpu-compute on \
  --rpp-gpu-cache-mib "${CACHE_MIB}" \
  --rpp-gpu-staging-mib "${STAGING_MIB}" \
  --rpp-gpu-copy-workers 1 \
  --rpp-gpu-queue-policy deadline \
  --rpp-trace "${TRACE}" \
  --host 127.0.0.1 \
  --port "${SERVER_PORT}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID="$!"

"${PYTHON}" - <<PY
import socket, time
port = int("${SERVER_PORT}")
deadline = time.time() + 240
while time.time() < deadline:
    with socket.socket() as s:
        s.settimeout(0.5)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(0)
    time.sleep(0.5)
raise SystemExit("llama-server did not open port")
PY

"${PYTHON}" - <<PY
import json, urllib.request
payload = {
    "prompt": "${PROMPT}",
    "n_predict": int("${N_PREDICT}"),
    "temperature": 0,
    "cache_prompt": False,
}
req = urllib.request.Request(
    "http://127.0.0.1:${SERVER_PORT}/completion",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=420) as resp:
    data = json.loads(resp.read().decode())
print(json.dumps({
    "content": data.get("content"),
    "tokens_predicted": data.get("tokens_predicted"),
    "tokens_evaluated": data.get("tokens_evaluated"),
    "timings": data.get("timings"),
}, ensure_ascii=False, indent=2))
PY

"${PYTHON}" - <<PY
import json
from pathlib import Path

trace = Path("${TRACE}")
metrics = Path("${METRICS}")
rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
metric_rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()] if metrics.exists() else []
print(json.dumps({
    "trace": str(trace),
    "metrics": str(metrics),
    "events": len(rows),
    "correction_success": sum(1 for row in rows if row.get("gpu_correction_success")),
    "prediction_found": sum(1 for row in rows if row.get("prediction_found")),
    "prefetched_total": sum(len(row.get("prefetched_experts", [])) for row in rows),
    "hit_experts_total": sum(len(row.get("hit_experts", [])) for row in rows),
    "missing_total": sum(len(row.get("missing_experts", [])) for row in rows),
    "evictions": sum(row.get("gpu_correction_evictions", 0) for row in rows),
    "loaded_on_demand": sum(row.get("gpu_correction_loaded_on_demand", 0) for row in rows),
    "ready_hits": sum(row.get("gpu_correction_ready_hits", 0) for row in rows),
    "resident_max": max((row.get("gpu_correction_resident_entries", 0) for row in rows), default=0),
    "slots": max((row.get("gpu_correction_cache_slots", 0) for row in rows), default=0),
    "sidecar_requests": len(metric_rows),
    "last_sidecar_inference_ms": metric_rows[-1].get("inference_ms") if metric_rows else None,
}, ensure_ascii=False, indent=2))
PY
