#!/usr/bin/env bash
# Re-analyze existing activation dumps from a previous run_router_prediction_batch.sh
# invocation with DoLa-style contrast candidates enabled (Phase 1).
#
# Reuses dumps under $INPUT_BATCH/<prompt_id>/activation_dump/ — no new inference.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

INPUT_BATCH="${INPUT_BATCH:?set INPUT_BATCH to a previous batch results dir}"
OUT_ROOT="${OUT_ROOT:-experiments/gemma4_bottleneck/results/dola_phase1_$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-models/gemma4-26B.gguf}"
TENSOR_RANGES="${TENSOR_RANGES:-experiments/gemma4_bottleneck/results/gemma4_26b_tensor_ranges.csv}"
ANALYZER_PY="${ANALYZER_PY:-experiments/gemma4_bottleneck/analyze_router_prediction.py}"
SUMMARIZER_PY="${SUMMARIZER_PY:-experiments/gemma4_bottleneck/summarize_router_prediction_batch.py}"
IMAGE="${IMAGE:-localhost/gemma4-ram-bench:24.04}"
TRUE_K="${TRUE_K:-8}"
EXCLUDE_EDGE_LAYERS="${EXCLUDE_EDGE_LAYERS:-1}"
N_LAYERS="${N_LAYERS:-30}"
CONTRAST_PAIRS="${CONTRAST_PAIRS:-all}"
CONTRAST_ALPHAS="${CONTRAST_ALPHAS:-0.1,0.3,0.5,0.7,1.0}"
CROSSLAYER_SOURCE="${CROSSLAYER_SOURCE:-}"
CROSSLAYER_OFFSETS="${CROSSLAYER_OFFSETS:-1,2,3,5,8,12}"
CROSSLAYER_ALPHAS="${CROSSLAYER_ALPHAS:-0.1,0.3,0.5,0.7,1.0}"

INPUT_MANIFEST="$INPUT_BATCH/manifest.csv"
[[ -f "$INPUT_MANIFEST" ]] || { echo "missing $INPUT_MANIFEST" >&2; exit 1; }

mkdir -p "$OUT_ROOT"
MANIFEST="$OUT_ROOT/manifest.csv"
echo "prompt_id,prompt_file,out_dir,status" > "$MANIFEST"

printf 'CONTRAST_PAIRS=%s\nCONTRAST_ALPHAS=%s\nCROSSLAYER_SOURCE=%s\nCROSSLAYER_OFFSETS=%s\nCROSSLAYER_ALPHAS=%s\nINPUT_BATCH=%s\n' \
  "$CONTRAST_PAIRS" "$CONTRAST_ALPHAS" "$CROSSLAYER_SOURCE" "$CROSSLAYER_OFFSETS" "$CROSSLAYER_ALPHAS" "$INPUT_BATCH" > "$OUT_ROOT/contrast.meta"

tail -n +2 "$INPUT_MANIFEST" | tr -d '\r' | while IFS=, read -r prompt_id prompt_file orig_out_dir orig_status; do
  if [[ "$orig_status" != "ok" ]]; then
    echo "skip $prompt_id (orig_status=$orig_status)"
    continue
  fi
  dump_dir="$orig_out_dir/activation_dump"
  if [[ ! -d "$dump_dir" ]]; then
    echo "skip $prompt_id (no dump dir at $dump_dir)"
    continue
  fi

  out_dir="$OUT_ROOT/$prompt_id"
  mkdir -p "$out_dir"
  printf 'prompt_id=%s\nprompt_file=%s\nsource_dump=%s\n' \
    "$prompt_id" "$prompt_file" "$dump_dir" > "$out_dir/prompt.meta"

  if podman run --rm \
      -v "$ROOT_DIR:/workspace" -w /workspace \
      "$IMAGE" \
      python3 "$ANALYZER_PY" \
        --dump-dir "$dump_dir" \
        --model "$MODEL" \
        --tensor-ranges "$TENSOR_RANGES" \
        --output "$out_dir/router_prediction_metrics.csv" \
        --true-k "$TRUE_K" \
        --layers "$N_LAYERS" \
        --contrast-pairs "$CONTRAST_PAIRS" \
        --contrast-alphas "$CONTRAST_ALPHAS" \
        --crosslayer-source "$CROSSLAYER_SOURCE" \
        --crosslayer-offsets "$CROSSLAYER_OFFSETS" \
        --crosslayer-alphas "$CROSSLAYER_ALPHAS"; then
    new_status="ok"
  else
    new_status="analyze_failed"
  fi
  echo "$prompt_id,$prompt_file,$out_dir,$new_status" >> "$MANIFEST"
done

SUMMARIZE_ARGS=(--manifest "$MANIFEST" --output-dir "$OUT_ROOT" --n-layers "$N_LAYERS")
if [[ "$EXCLUDE_EDGE_LAYERS" == "1" ]]; then
  SUMMARIZE_ARGS+=(--exclude-edge-layers)
fi
podman run --rm -v "$ROOT_DIR:/workspace" -w /workspace "$IMAGE" \
  python3 "$SUMMARIZER_PY" "${SUMMARIZE_ARGS[@]}"

if [[ "$EXCLUDE_EDGE_LAYERS" == "1" ]]; then
  EDGE_DIR="$OUT_ROOT/with_edge_layers"
  mkdir -p "$EDGE_DIR"
  podman run --rm -v "$ROOT_DIR:/workspace" -w /workspace "$IMAGE" \
    python3 "$SUMMARIZER_PY" --manifest "$MANIFEST" --output-dir "$EDGE_DIR" --n-layers "$N_LAYERS"
fi

echo "wrote: $OUT_ROOT"
