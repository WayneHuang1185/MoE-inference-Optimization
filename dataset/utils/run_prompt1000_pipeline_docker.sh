#!/usr/bin/env bash
# Run prompt1000 Global RPP stages inside the remote Docker-compatible runtime.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

IMAGE="${IMAGE:-localhost/gemma4-ram-bench:24.04}"
STAGE="${STAGE:-generate}"
MODEL="${MODEL:-models/gemma4-26B.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-/workspace/llama.cpp/build/bin/llama-server}"
TOKENIZE_BIN="${TOKENIZE_BIN:-llama.cpp/build/bin/llama-tokenize}"
PORT="${PORT:-8080}"
THREADS="${THREADS:-$(nproc 2>/dev/null || echo 32)}"
CTX_SIZE="${CTX_SIZE:-8192}"
LLAMA_EXTRA_ARGS="${LLAMA_EXTRA_ARGS:-}"
MEMORY="${MEMORY:-48g}"
MEMORY_SWAP="${MEMORY_SWAP:-48g}"

DB="${DB:-dataset/prompt1000/prompt_database.jsonl}"
GEN_DIR="${GEN_DIR:-dataset/prompt1000/generations}"
DUMP_DIR="${DUMP_DIR:-dataset/prompt1000/label_dumps}"
NPZ_DIR="${NPZ_DIR:-dataset/prompt1000/router_label_npz}"

N_GENERATIONS="${N_GENERATIONS:-1}"
N_PREDICT="${N_PREDICT:-10}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
SEED_BASE="${SEED_BASE:-20260513}"
MAX_RECORDS="${MAX_RECORDS:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CLEAN_MANIFEST="${CLEAN_MANIFEST:-0}"
DELETE_RAW_AFTER_PACK="${DELETE_RAW_AFTER_PACK:-1}"

common_run=(
  docker run --rm
  --memory="$MEMORY" --memory-swap="$MEMORY_SWAP"
  -v "$ROOT_DIR:/workspace"
  -w /workspace
  -e LLAMA_EXTRA_ARGS="$LLAMA_EXTRA_ARGS"
  "$IMAGE"
  python3 dataset/utils/rpp_prompt1000_pipeline.py
)

skip_args=()
if [[ "$SKIP_EXISTING" == "1" ]]; then
  skip_args+=(--skip-existing)
fi

manifest_args=()
if [[ "$CLEAN_MANIFEST" == "1" ]]; then
  manifest_args+=(--clean-manifest)
fi

echo "[prompt1000-pipeline] stage       = $STAGE"
echo "[prompt1000-pipeline] root        = $ROOT_DIR"
echo "[prompt1000-pipeline] image       = $IMAGE"
echo "[prompt1000-pipeline] db          = $DB"

case "$STAGE" in
  generate)
    "${common_run[@]}" generate \
      --db "$DB" \
      --out-dir "$GEN_DIR" \
      --model "$MODEL" \
      --llama-server "$LLAMA_SERVER" \
      --port "$PORT" \
      --threads "$THREADS" \
      --ctx-size "$CTX_SIZE" \
      --n-generations "$N_GENERATIONS" \
      --n-predict "$N_PREDICT" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --seed-base "$SEED_BASE" \
      --max-records "$MAX_RECORDS" \
      "${skip_args[@]}" \
      "${manifest_args[@]}"
    ;;
  dump-labels)
    "${common_run[@]}" dump-labels \
      --generations-dir "$GEN_DIR" \
      --out-dir "$DUMP_DIR" \
      --model "$MODEL" \
      --llama-server "$LLAMA_SERVER" \
      --port "$PORT" \
      --threads "$THREADS" \
      --ctx-size "$CTX_SIZE" \
      --max-samples "$MAX_SAMPLES" \
      "${skip_args[@]}" \
      "${manifest_args[@]}"
    ;;
  dump-pack-labels)
    raw_args=()
    if [[ "$DELETE_RAW_AFTER_PACK" == "1" ]]; then
      raw_args+=(--delete-raw-after-pack)
    fi
    "${common_run[@]}" dump-pack-labels \
      --generations-dir "$GEN_DIR" \
      --out-dir "$NPZ_DIR" \
      --model "$MODEL" \
      --llama-server "$LLAMA_SERVER" \
      --tokenize-bin "$TOKENIZE_BIN" \
      --port "$PORT" \
      --threads "$THREADS" \
      --ctx-size "$CTX_SIZE" \
      --max-samples "$MAX_SAMPLES" \
      "${skip_args[@]}" \
      "${manifest_args[@]}" \
      "${raw_args[@]}"
    ;;
  pack-labels)
    "${common_run[@]}" pack-labels \
      --dumps-dir "$DUMP_DIR" \
      --out-dir "$NPZ_DIR" \
      --model "$MODEL" \
      --tokenize-bin "$TOKENIZE_BIN" \
      --max-samples "$MAX_SAMPLES" \
      "${skip_args[@]}"
    ;;
  *)
    echo "unknown STAGE=$STAGE" >&2
    exit 2
    ;;
esac

echo "[prompt1000-pipeline] done stage=$STAGE"
