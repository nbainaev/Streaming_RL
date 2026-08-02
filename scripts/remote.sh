#!/usr/bin/env bash
# Bridge helper for working with the stream_rl SSH server.
# Usage:
#   scripts/remote.sh push                     # sync local stream_rl/ -> server
#   scripts/remote.sh pull                      # sync server logs/ -> local
#   scripts/remote.sh run <session> "<cmd>"      # start/replace a tmux session running <cmd>
#   scripts/remote.sh attach <session>           # attach to a tmux session
#   scripts/remote.sh tail <session>             # tail tmux pane output without attaching
#   scripts/remote.sh ls                         # list running tmux sessions
#   scripts/remote.sh kill <session>             # kill a tmux session
set -euo pipefail

HOST=stream_rl
REMOTE_DIR=projects/stream_rl
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/stream_rl"

cmd="${1:-}"
shift || true

case "$cmd" in
  push)
    # No rsync on Windows/Git Bash; sync via tar over ssh instead, skipping
    # venv/cache/git/logs so we never ship the environment or old results.
    tar -C "$LOCAL_DIR" \
      --exclude .venv --exclude .git --exclude __pycache__ \
      --exclude '*.egg-info' --exclude logs --exclude logs_remote \
      -cf - . | ssh "$HOST" "mkdir -p $REMOTE_DIR && tar -C $REMOTE_DIR -xf -"
    ;;
  pull)
    mkdir -p "$LOCAL_DIR/logs_remote"
    ssh "$HOST" "cd $REMOTE_DIR && tar -cf - logs" | tar -C "$LOCAL_DIR/logs_remote" --strip-components=1 -xf -
    ;;
  run)
    session="${1:?session name required}"
    run_cmd="${2:?command required}"
    ssh "$HOST" "tmux kill-session -t '$session' 2>/dev/null; cd $REMOTE_DIR && mkdir -p logs && tmux new-session -d -s '$session' \"source .venv/bin/activate && ($run_cmd) 2>&1 | tee logs/${session}.log\""
    echo "Started tmux session '$session' on $HOST."
    ;;
  attach)
    session="${1:?session name required}"
    ssh -t "$HOST" "tmux attach -t '$session'"
    ;;
  tail)
    session="${1:?session name required}"
    ssh "$HOST" "tail -n 100 -f $REMOTE_DIR/logs/${session}.log"
    ;;
  ls)
    ssh "$HOST" "tmux ls 2>/dev/null || echo 'no sessions'"
    ;;
  kill)
    session="${1:?session name required}"
    ssh "$HOST" "tmux kill-session -t '$session'"
    ;;
  *)
    echo "Usage: $0 {push|pull|run <session> <cmd>|attach <session>|tail <session>|ls|kill <session>}" >&2
    exit 1
    ;;
esac
