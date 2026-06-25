#!/usr/bin/env bash
# Pack the existing Global RPP prompt corpus into dataset/prompt1000.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

IMAGE="${IMAGE:-python:3.12-slim}"
MANIFEST="${MANIFEST:-experiments/gemma4_bottleneck/global_rpp_prompts/prompts_manifest.jsonl}"
OUT_DIR="${OUT_DIR:-dataset/prompt1000}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments/gemma4_bottleneck/results}"
RUN_NAME="${RUN_NAME:-prompt1000_materialize_$(date +%Y%m%d_%H%M%S)}"
DATASET_NAME="${DATASET_NAME:-}"
CLEAN="${CLEAN:-1}"

args=(
  --manifest "$MANIFEST"
  --out-dir "$OUT_DIR"
  --results-root "$RESULTS_ROOT"
  --run-name "$RUN_NAME"
)
if [[ -n "$DATASET_NAME" ]]; then
  args+=(--dataset-name "$DATASET_NAME")
fi
if [[ "$CLEAN" == "1" ]]; then
  args+=(--clean)
fi

mkdir -p "$OUT_DIR" "$RESULTS_ROOT"

echo "[prompt1000] ROOT_DIR     = $ROOT_DIR"
echo "[prompt1000] MANIFEST     = $MANIFEST"
echo "[prompt1000] OUT_DIR      = $OUT_DIR"
echo "[prompt1000] RESULTS_ROOT = $RESULTS_ROOT"
echo "[prompt1000] RUN_NAME     = $RUN_NAME"

docker run --rm \
  -v "$ROOT_DIR:/workspace" \
  -w /workspace \
  "$IMAGE" \
  python experiments/gemma4_bottleneck/materialize_prompt_database.py "${args[@]}"

DOCKER_VERSION="$(docker --version 2>&1 || true)"
if echo "$DOCKER_VERSION" | grep -qi podman; then
  {
    echo
    echo "## Runtime Note"
    echo
    echo "- The remote \`docker\` CLI is backed by Podman compatibility mode:"
    echo "  \`$DOCKER_VERSION\`"
  } >> "$OUT_DIR/REPORT.md"
  cp "$OUT_DIR/REPORT.md" "$RESULTS_ROOT/$RUN_NAME/REPORT.md"
fi

echo "[prompt1000] done"
echo "[prompt1000] database = $OUT_DIR/prompt_database.jsonl"
echo "[prompt1000] metadata = $OUT_DIR/metadata.json"
echo "[prompt1000] report   = $OUT_DIR/REPORT.md"
