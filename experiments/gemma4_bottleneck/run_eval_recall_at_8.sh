#!/usr/bin/env bash
# Recall@8 eval for linear-corrected attn_out_i vs naive / DoLa-static-k / oracle.
# Hold-out by prompt; same split as train_linear_predictor.sh.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

INPUT_BATCH="${INPUT_BATCH:?set INPUT_BATCH to a previous batch results dir}"
OUT_ROOT="${OUT_ROOT:-experiments/gemma4_bottleneck/results/recall_at_8_$(date +%Y%m%d_%H%M%S)}"
PY="${PY:-experiments/gemma4_bottleneck/eval_recall_at_8.py}"
IMAGE="${IMAGE:-localhost/gemma4-ram-bench:24.04}"
MODEL="${MODEL:-models/gemma4-26B.gguf}"
TENSOR_RANGES="${TENSOR_RANGES:-experiments/gemma4_bottleneck/results/gemma4_26b_tensor_ranges.csv}"
N_LAYERS="${N_LAYERS:-30}"
HIDDEN="${HIDDEN:-2816}"
EVAL_PROMPTS="${EVAL_PROMPTS:-4}"
LAMBDA="${LAMBDA:-1000}"
DOLA_K="${DOLA_K:-5}"
DOLA_ALPHA="${DOLA_ALPHA:-0.1}"

INPUT_MANIFEST="$INPUT_BATCH/manifest.csv"
[[ -f "$INPUT_MANIFEST" ]] || { echo "missing $INPUT_MANIFEST" >&2; exit 1; }

mkdir -p "$OUT_ROOT"
printf 'INPUT_BATCH=%s\nEVAL_PROMPTS=%s\nLAMBDA=%s\nDOLA_K=%s\nDOLA_ALPHA=%s\nHIDDEN=%s\nN_LAYERS=%s\nMODEL=%s\n' \
  "$INPUT_BATCH" "$EVAL_PROMPTS" "$LAMBDA" "$DOLA_K" "$DOLA_ALPHA" "$HIDDEN" "$N_LAYERS" "$MODEL" \
  > "$OUT_ROOT/meta"

podman run --rm \
  -v "$ROOT_DIR:/workspace" -w /workspace \
  "$IMAGE" \
  python3 "$PY" \
    --manifest "$INPUT_MANIFEST" \
    --model "$MODEL" \
    --tensor-ranges "$TENSOR_RANGES" \
    --output "$OUT_ROOT/recall_at_8.csv" \
    --hidden "$HIDDEN" \
    --n-layers "$N_LAYERS" \
    --eval-prompts "$EVAL_PROMPTS" \
    --lambda-ridge "$LAMBDA" \
    --dola-k "$DOLA_K" \
    --dola-alpha "$DOLA_ALPHA" \
  2>&1 | tee "$OUT_ROOT/run.log"

echo "wrote: $OUT_ROOT"
