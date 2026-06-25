#!/usr/bin/env bash
# Run prefetch-cache simulation over an existing batch of activation dumps.
# Reuses dumps under $INPUT_BATCH/<prompt_id>/activation_dump/ — no new inference.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

INPUT_BATCH="${INPUT_BATCH:?set INPUT_BATCH to a previous batch results dir}"
OUT_ROOT="${OUT_ROOT:-experiments/gemma4_bottleneck/results/prefetch_cache_$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-models/gemma4-26B.gguf}"
TENSOR_RANGES="${TENSOR_RANGES:-experiments/gemma4_bottleneck/results/gemma4_26b_tensor_ranges.csv}"
SIM_PY="${SIM_PY:-experiments/gemma4_bottleneck/simulate_prefetch_cache.py}"
SUMM_PY="${SUMM_PY:-experiments/gemma4_bottleneck/summarize_prefetch_cache.py}"
IMAGE="${IMAGE:-localhost/gemma4-ram-bench:24.04}"
N_LAYERS="${N_LAYERS:-30}"
CACHE_SIZES="${CACHE_SIZES:-16,24,32,48,64,96}"
PREFETCH_BUDGETS="${PREFETCH_BUDGETS:-8,16,24}"
EXCLUDE_EDGE_LAYERS="${EXCLUDE_EDGE_LAYERS:-1}"

INPUT_MANIFEST="$INPUT_BATCH/manifest.csv"
[[ -f "$INPUT_MANIFEST" ]] || { echo "missing $INPUT_MANIFEST" >&2; exit 1; }

mkdir -p "$OUT_ROOT"
MANIFEST="$OUT_ROOT/manifest.csv"
echo "prompt_id,prompt_file,out_dir,status" > "$MANIFEST"

printf 'CACHE_SIZES=%s\nPREFETCH_BUDGETS=%s\nINPUT_BATCH=%s\n' \
  "$CACHE_SIZES" "$PREFETCH_BUDGETS" "$INPUT_BATCH" > "$OUT_ROOT/sim.meta"

tail -n +2 "$INPUT_MANIFEST" | tr -d '\r' | while IFS=, read -r prompt_id prompt_file orig_out_dir orig_status; do
  if [[ "$orig_status" != "ok" ]]; then
    echo "skip $prompt_id (orig_status=$orig_status)"
    continue
  fi
  dump_dir="$orig_out_dir/activation_dump"
  [[ -d "$dump_dir" ]] || { echo "skip $prompt_id (no dump dir)"; continue; }

  out_dir="$OUT_ROOT/$prompt_id"
  mkdir -p "$out_dir"
  printf 'prompt_id=%s\nprompt_file=%s\nsource_dump=%s\n' \
    "$prompt_id" "$prompt_file" "$dump_dir" > "$out_dir/prompt.meta"

  if podman run --rm \
      -v "$ROOT_DIR:/workspace" -w /workspace \
      "$IMAGE" \
      python3 "$SIM_PY" \
        --dump-dir "$dump_dir" \
        --model "$MODEL" \
        --tensor-ranges "$TENSOR_RANGES" \
        --output "$out_dir/prefetch_cache_metrics.csv" \
        --layers "$N_LAYERS" \
        --cache-sizes "$CACHE_SIZES" \
        --prefetch-budgets "$PREFETCH_BUDGETS"; then
    new_status="ok"
  else
    new_status="sim_failed"
  fi
  echo "$prompt_id,$prompt_file,$out_dir,$new_status" >> "$MANIFEST"
done

SUMMARIZE_ARGS=(--manifest "$MANIFEST" --output-dir "$OUT_ROOT" --n-layers "$N_LAYERS")
[[ "$EXCLUDE_EDGE_LAYERS" == "1" ]] && SUMMARIZE_ARGS+=(--exclude-edge-layers)

podman run --rm -v "$ROOT_DIR:/workspace" -w /workspace "$IMAGE" \
  python3 "$SUMM_PY" "${SUMMARIZE_ARGS[@]}"

echo "wrote: $OUT_ROOT"
