#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat >&2 <<'EOF'
usage:
  run_under_memory_limit.sh 14G -- python experiments/.../run_prefetch_benchmark.py ...

This wrapper runs a command inside a transient systemd user scope with
MemoryMax/MemoryHigh set. It is intended for local memory-pressure experiments.
EOF
  exit 2
fi

MEMORY_LIMIT="$1"
shift
if [[ "${1:-}" == "--" ]]; then
  shift
fi
if [[ $# -lt 1 ]]; then
  echo "missing command after memory limit" >&2
  exit 2
fi

if ! command -v systemd-run >/dev/null 2>&1; then
  echo "systemd-run is not available; cannot create a memory-limited scope" >&2
  exit 1
fi

if ! systemctl --user show-environment >/dev/null 2>&1; then
  cat >&2 <<'EOF'
systemd --user is not available in this shell.

Run the benchmark without this wrapper, or use Docker/cgroup on a Linux host.
This script intentionally fails instead of silently running without a memory limit.
EOF
  exit 1
fi

UNIT="rpp-mixed-prefetch-$(date +%Y%m%d-%H%M%S)-$$"
echo "running in systemd user scope: ${UNIT}"
echo "memory limit: ${MEMORY_LIMIT}"

exec systemd-run --user --scope --collect \
  -p "MemoryMax=${MEMORY_LIMIT}" \
  -p "MemoryHigh=${MEMORY_LIMIT}" \
  -p "MemoryAccounting=yes" \
  "$@"
