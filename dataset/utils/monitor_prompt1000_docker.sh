#!/usr/bin/env bash
# Read-only monitor for dataset/prompt1000 pipeline progress.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

IMAGE="${IMAGE:-python:3.12-slim}"
ROOT="${ROOT:-dataset/prompt1000}"
TAIL="${TAIL:-20}"
WATCH="${WATCH:-0}"

docker run --rm \
  -v "$ROOT_DIR:/workspace" \
  -w /workspace \
  --pid=host \
  "$IMAGE" \
  python dataset/utils/monitor_prompt1000.py \
    --root "$ROOT" \
    --tail "$TAIL" \
    --watch "$WATCH"
