#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT_DIR"

EXP_DIR="experiments/gemma4_IO_behaviors/cross_memory_boundness"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
STATISTICS_DIR="${STATISTICS_DIR:-$EXP_DIR/statistics/cross_memory_boundness_${RUN_TIMESTAMP}}"
FIGURES_DIR="${FIGURES_DIR:-$EXP_DIR/figures/cross_memory_boundness_${RUN_TIMESTAMP}}"
IMAGE="${IMAGE:-localhost/gemma4-ram-bench:24.04}"
MODEL="${MODEL:-models/gemma4-26B.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-llama.cpp/build/bin/llama-server}"
PROMPT_MANIFEST="${PROMPT_MANIFEST:-experiments/gemma4_bottleneck/global_rpp_prompts/prompts_manifest.jsonl}"
MEMORY_CAPS="${MEMORY_CAPS:-24g,10g,8g,6g}"
BASE_PORT="${BASE_PORT:-8240}"
THREADS="${THREADS:-$(nproc 2>/dev/null || echo 16)}"
CTX_SIZE="${CTX_SIZE:-8192}"
N_PREDICT="${N_PREDICT:-16}"
PROMPT_LIMIT="${PROMPT_LIMIT:-30}"
MAX_PROMPT_CHARS="${MAX_PROMPT_CHARS:-256}"
DROP_MODEL_CACHE="${DROP_MODEL_CACHE:-1}"
LLAMA_EXTRA_ARGS="${LLAMA_EXTRA_ARGS:-}"
DOCKER_BIN="${DOCKER_BIN:-docker}"

case " $LLAMA_EXTRA_ARGS " in
  *" --no-repack "*) ;;
  *) LLAMA_EXTRA_ARGS="${LLAMA_EXTRA_ARGS} --no-repack" ;;
esac

IFS=',' read -r -a MEMORY_CAP_ARRAY <<< "$MEMORY_CAPS"
mkdir -p "$STATISTICS_DIR" "$FIGURES_DIR" "$STATISTICS_DIR/selected_prompts"

python3 "$EXP_DIR/utils/select_short_prompts.py" \
  --manifest "$PROMPT_MANIFEST" \
  --output-dir "$STATISTICS_DIR/selected_prompts" \
  --selection-json "$STATISTICS_DIR/prompt_selection.json" \
  --limit "$PROMPT_LIMIT" \
  --max-chars "$MAX_PROMPT_CHARS" \
  --root "$ROOT_DIR"

python3 - "$STATISTICS_DIR/run_config.json" "$MEMORY_CAPS" "$N_PREDICT" "$PROMPT_LIMIT" "$MAX_PROMPT_CHARS" "$MODEL" "$LLAMA_EXTRA_ARGS" "$THREADS" "$CTX_SIZE" <<'PY'
import json
import sys

path, caps, n_predict, prompt_limit, max_chars, model, extra, threads, ctx = sys.argv[1:]
config = {
    "memory_caps": caps.split(","),
    "n_predict": int(n_predict),
    "prompt_limit": int(prompt_limit),
    "max_prompt_chars": int(max_chars),
    "model": model,
    "llama_extra_args": extra,
    "threads": int(threads),
    "ctx_size": int(ctx),
}
open(path, "w", encoding="utf-8").write(json.dumps(config, indent=2) + "\n")
PY

write_cpu_environment() {
  {
    date -Is
    uname -a
    echo
    nproc || true
    echo
    lscpu || true
    echo
    cat /proc/pressure/io 2>/dev/null || true
    echo
    cat /proc/pressure/memory 2>/dev/null || true
  } > "$STATISTICS_DIR/host_environment.txt"
}

write_cpu_environment

for i in "${!MEMORY_CAP_ARRAY[@]}"; do
  memory_cap="${MEMORY_CAP_ARRAY[$i]}"
  port="$((BASE_PORT + i))"
  case_dir="$STATISTICS_DIR/case_${memory_cap}"
  mkdir -p "$case_dir"
  echo "[cross-memory] running memory=${memory_cap} port=${port} case_dir=${case_dir}"
  "$DOCKER_BIN" run --rm \
    --network host \
    --memory="$memory_cap" \
    --memory-swap="$memory_cap" \
    --cap-add SYS_ADMIN \
    --security-opt seccomp=unconfined \
    -v "$ROOT_DIR:/workspace" \
    -w /workspace \
    -e EXP_DIR="$EXP_DIR" \
    -e CASE_DIR="$case_dir" \
    -e MODEL="$MODEL" \
    -e LLAMA_SERVER="$LLAMA_SERVER" \
    -e PROMPT_DIR="$STATISTICS_DIR/selected_prompts" \
    -e MEMORY_CAP="$memory_cap" \
    -e PORT="$port" \
    -e THREADS="$THREADS" \
    -e CTX_SIZE="$CTX_SIZE" \
    -e N_PREDICT="$N_PREDICT" \
    -e PROMPT_LIMIT="$PROMPT_LIMIT" \
    -e DROP_MODEL_CACHE="$DROP_MODEL_CACHE" \
    -e LLAMA_EXTRA_ARGS="$LLAMA_EXTRA_ARGS" \
    "$IMAGE" \
    bash "$EXP_DIR/utils/run_boundness_case_inside_container.sh"
done

python3 "$EXP_DIR/utils/summarize_cross_memory_boundness.py" \
  --statistics-dir "$STATISTICS_DIR" \
  --config-json "$STATISTICS_DIR/run_config.json"

{
  echo
  echo "## Cross-Memory Boundness ${RUN_TIMESTAMP}"
  echo
  echo "- statistics: \`$STATISTICS_DIR\`"
  echo "- memory caps: \`$MEMORY_CAPS\`"
  echo "- prompt_limit: \`$PROMPT_LIMIT\`, n_predict: \`$N_PREDICT\`, max_prompt_chars: \`$MAX_PROMPT_CHARS\`"
  echo "- report: \`$STATISTICS_DIR/REPORT.md\`"
} >> "$EXP_DIR/REPORT.md"

echo "statistics: $STATISTICS_DIR"
echo "report:     $STATISTICS_DIR/REPORT.md"
