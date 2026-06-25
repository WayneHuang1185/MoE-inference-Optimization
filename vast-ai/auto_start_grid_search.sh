#!/usr/bin/env bash
# Poll a Vast.ai instance until it is SSH-ready, then sync code and start grid search.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$ROOT_DIR/.." && pwd)"

INSTANCE_ID="${INSTANCE_ID:-36821321}"
POLL_SECONDS="${POLL_SECONDS:-1800}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/vast-ai}"
GRID_HALVING_EPOCHS="${GRID_HALVING_EPOCHS:-2,8,20}"
GRID_HALVING_KEEP="${GRID_HALVING_KEEP:-6,2}"
GRID_LOG_NAME="${GRID_LOG_NAME:-grid_search_autostart_$(date +%Y%m%d_%H%M%S).log}"

VASTAI_BIN="${VASTAI_BIN:-$PROJECT_DIR/.venv-vastai/bin/vastai}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$PROJECT_DIR/.vastai-config}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_DIR/.vastai-cache}"

LOCAL_LOG="${LOCAL_LOG:-$ROOT_DIR/auto_start_grid_search.log}"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOCAL_LOG"
}

instance_json() {
  "$VASTAI_BIN" show instances --raw --no-color
}

instance_field() {
  python3 - "$INSTANCE_ID" "$1" "$2" <<'PY'
import json
import sys

instance_id = int(sys.argv[1])
field = sys.argv[2]
data = json.loads(sys.argv[3])
for row in data:
    if int(row.get("id", -1)) == instance_id:
        value = row.get(field, "")
        print("" if value is None else value)
        break
PY
}

ssh_parts() {
  local url
  url="$("$VASTAI_BIN" ssh-url "$INSTANCE_ID" --no-color)"
  python3 - "$url" <<'PY'
import re
import sys

text = sys.argv[1].strip()
match = re.match(r"ssh://([^@]+)@([^:]+):(\d+)", text)
if not match:
    raise SystemExit(f"cannot parse ssh url: {text!r}")
print(match.group(1), match.group(2), match.group(3))
PY
}

remote_ssh() {
  local user="$1"
  local host="$2"
  local port="$3"
  shift 3
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p "$port" "$user@$host" "$@"
}

sync_package() {
  local user="$1"
  local host="$2"
  local port="$3"
  rsync -az --delete \
    --exclude 'dataset/' \
    --exclude 'outputs/' \
    --exclude 'source_results/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $port" \
    "$ROOT_DIR/" "$user@$host:$REMOTE_DIR/"
}

start_grid() {
  local user="$1"
  local host="$2"
  local port="$3"
  remote_ssh "$user" "$host" "$port" "
    set -e
    cd '$REMOTE_DIR'
    mkdir -p outputs
    chmod +x run_vast_grid_search.sh
    if pgrep -af 'experiments.gemma4_global_predictor.grid_search_rpp|run_vast_grid_search.sh' >/dev/null; then
      echo already_running
      pgrep -af 'experiments.gemma4_global_predictor.grid_search_rpp|run_vast_grid_search.sh'
      exit 0
    fi
    if nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
      echo gpu_busy
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
      exit 3
    fi
    nohup env GRID_HALVING_EPOCHS='$GRID_HALVING_EPOCHS' GRID_HALVING_KEEP='$GRID_HALVING_KEEP' ./run_vast_grid_search.sh > 'outputs/$GRID_LOG_NAME' 2>&1 &
    echo \$! > outputs/grid_search.pid
    echo started pid=\$(cat outputs/grid_search.pid) log=outputs/$GRID_LOG_NAME
  "
}

log "monitor start instance=$INSTANCE_ID poll=${POLL_SECONDS}s epochs=$GRID_HALVING_EPOCHS keep=$GRID_HALVING_KEEP"

while true; do
  json="$(instance_json || true)"
  if [[ -n "$json" ]]; then
    cur_state="$(instance_field cur_state "$json" || true)"
    actual_status="$(instance_field actual_status "$json" || true)"
    intended_status="$(instance_field intended_status "$json" || true)"
    gpu_util="$(instance_field gpu_util "$json" || true)"
    log "instance state cur=${cur_state:-unknown} actual=${actual_status:-unknown} intended=${intended_status:-unknown} gpu_util=${gpu_util:-unknown}"
  else
    log "could not fetch instance state"
  fi

  "$VASTAI_BIN" start instance "$INSTANCE_ID" --no-color >>"$LOCAL_LOG" 2>&1 || true

  if read -r ssh_user ssh_host ssh_port < <(ssh_parts); then
    if remote_ssh "$ssh_user" "$ssh_host" "$ssh_port" "echo ssh_ready && nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits" >>"$LOCAL_LOG" 2>&1; then
      log "ssh ready at $ssh_user@$ssh_host:$ssh_port; syncing package"
      sync_package "$ssh_user" "$ssh_host" "$ssh_port" >>"$LOCAL_LOG" 2>&1
      log "package synced; starting grid search"
      if start_grid "$ssh_user" "$ssh_host" "$ssh_port" >>"$LOCAL_LOG" 2>&1; then
        log "grid search started or already running; monitor exiting"
        exit 0
      else
        log "remote GPU busy or start failed; will retry after ${POLL_SECONDS}s"
      fi
    else
      log "ssh not ready yet"
    fi
  else
    log "ssh url unavailable"
  fi

  sleep "$POLL_SECONDS"
done
