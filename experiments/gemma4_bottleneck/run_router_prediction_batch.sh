#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PROMPT_DIR="${PROMPT_DIR:-experiments/gemma4_bottleneck/router_prediction_prompts}"
OUT_ROOT="${OUT_ROOT:-experiments/gemma4_bottleneck/results/router_prediction_batch_$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-models/gemma4-26B.gguf}"
TENSOR_RANGES="${TENSOR_RANGES:-experiments/gemma4_bottleneck/results/gemma4_26b_tensor_ranges.csv}"
ANALYZER_PY="${ANALYZER_PY:-experiments/gemma4_bottleneck/analyze_router_prediction.py}"
SUMMARIZER_PY="${SUMMARIZER_PY:-experiments/gemma4_bottleneck/summarize_router_prediction_batch.py}"
IMAGE="${IMAGE:-localhost/gemma4-ram-bench:24.04}"
MEMORY="${MEMORY:-12g}"
MEMORY_SWAP="${MEMORY_SWAP:-20g}"
THREADS="${THREADS:-1}"
CTX_SIZE="${CTX_SIZE:-8192}"
# N_PREDICT > 1 so we capture decode passes too — prefill alone is not the
# regime where expert prefetch matters most.
N_PREDICT="${N_PREDICT:-8}"
RUNS="${RUNS:-1}"
WARMUP_RUNS="${WARMUP_RUNS:-0}"
TRUE_K="${TRUE_K:-8}"
PORT_BASE="${PORT_BASE:-8080}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
EXCLUDE_EDGE_LAYERS="${EXCLUDE_EDGE_LAYERS:-1}"
N_LAYERS="${N_LAYERS:-30}"
CONTRAST_PAIRS="${CONTRAST_PAIRS:-}"
CONTRAST_ALPHAS="${CONTRAST_ALPHAS:-0.1,0.3,0.5,0.7,1.0}"

mkdir -p "$OUT_ROOT"

mapfile -t PROMPTS < <(find "$PROMPT_DIR" -maxdepth 1 -type f -name '*.txt' | sort)
if [[ "${#PROMPTS[@]}" -eq 0 ]]; then
  echo "no prompt files found in $PROMPT_DIR" >&2
  exit 1
fi

if [[ "$MAX_PROMPTS" -gt 0 && "$MAX_PROMPTS" -lt "${#PROMPTS[@]}" ]]; then
  PROMPTS=("${PROMPTS[@]:0:$MAX_PROMPTS}")
fi

MANIFEST="$OUT_ROOT/manifest.csv"
echo "prompt_id,prompt_file,out_dir,status" > "$MANIFEST"

idx=0
for prompt_file in "${PROMPTS[@]}"; do
  prompt_id="$(basename "$prompt_file" .txt)"
  out_dir="$OUT_ROOT/$prompt_id"
  port="$((PORT_BASE + idx))"
  mkdir -p "$out_dir"
  printf 'prompt_id=%s\nprompt_file=%s\n' "$prompt_id" "$prompt_file" > "$out_dir/prompt.meta"

  status="ok"
  if ! podman run --rm \
      --memory="$MEMORY" \
      --memory-swap="$MEMORY_SWAP" \
      -v "$ROOT_DIR:/workspace" \
      -w /workspace \
      "$IMAGE" \
      bash -lc "CASE_NAME=router_pred_${prompt_id} OUT_DIR=$out_dir ENABLE_ACTIVATION_DUMP=1 MODEL=$MODEL CTX_SIZE=$CTX_SIZE RUNS=$RUNS WARMUP_RUNS=$WARMUP_RUNS N_PREDICT=$N_PREDICT PROMPT_FILE=$prompt_file THREADS=$THREADS PORT=$port experiments/gemma4_bottleneck/run_container_ram_case.sh"; then
    status="run_failed"
  fi

  if [[ "$status" == "ok" ]]; then
    if ! podman run --rm \
        -v "$ROOT_DIR:/workspace" -w /workspace \
        "$IMAGE" \
        python3 "$ANALYZER_PY" \
          --dump-dir "$out_dir/activation_dump" \
          --model "$MODEL" \
          --tensor-ranges "$TENSOR_RANGES" \
          --output "$out_dir/router_prediction_metrics.csv" \
          --true-k "$TRUE_K" \
          --layers "$N_LAYERS" \
          --contrast-pairs "$CONTRAST_PAIRS" \
          --contrast-alphas "$CONTRAST_ALPHAS"; then
      status="analyze_failed"
    fi
  fi

  echo "$prompt_id,$prompt_file,$out_dir,$status" >> "$MANIFEST"
  idx="$((idx + 1))"
done

SUMMARIZE_ARGS=(--manifest "$MANIFEST" --output-dir "$OUT_ROOT" --n-layers "$N_LAYERS")
if [[ "$EXCLUDE_EDGE_LAYERS" == "1" ]]; then
  SUMMARIZE_ARGS+=(--exclude-edge-layers)
fi
podman run --rm -v "$ROOT_DIR:/workspace" -w /workspace "$IMAGE" \
  python3 "$SUMMARIZER_PY" "${SUMMARIZE_ARGS[@]}"

# Also write a side-by-side summary that keeps edge layers, for sanity-checking.
if [[ "$EXCLUDE_EDGE_LAYERS" == "1" ]]; then
  EDGE_DIR="$OUT_ROOT/with_edge_layers"
  mkdir -p "$EDGE_DIR"
  podman run --rm -v "$ROOT_DIR:/workspace" -w /workspace "$IMAGE" \
    python3 "$SUMMARIZER_PY" --manifest "$MANIFEST" --output-dir "$EDGE_DIR" --n-layers "$N_LAYERS"
fi

echo "wrote batch results: $OUT_ROOT"
