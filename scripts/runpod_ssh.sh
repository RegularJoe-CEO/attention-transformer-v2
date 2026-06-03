#!/usr/bin/env bash
# Shared RunPod SSH/rsync helpers. Source from other scripts:
#   source "$(dirname "$0")/runpod_ssh.sh"

_RUNPOD_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]:-$0}")" && pwd)"
if [[ -f "$_RUNPOD_SCRIPT_DIR/runpod.env" ]]; then
  # shellcheck source=runpod.env
  source "$_RUNPOD_SCRIPT_DIR/runpod.env"
fi

RUNPOD_SSH_HOST="${RUNPOD_SSH_HOST:-}"
RUNPOD_SSH_IDENTITY="${RUNPOD_SSH_IDENTITY:-${HOME}/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/attention-transformer-v2}"

if [[ -z "$RUNPOD_SSH_HOST" ]]; then
  echo "Missing RUNPOD_SSH_HOST in scripts/runpod.env"
  exit 1
fi

if [[ ! -f "$RUNPOD_SSH_IDENTITY" ]]; then
  echo "SSH key not found: $RUNPOD_SSH_IDENTITY"
  exit 1
fi

# Expand ~ in identity path
RUNPOD_SSH_IDENTITY="${RUNPOD_SSH_IDENTITY/#\~/$HOME}"

export RUNPOD_SSH_OPTS=(-i "$RUNPOD_SSH_IDENTITY" -o StrictHostKeyChecking=accept-new)

runpod_ssh() {
  ssh "${RUNPOD_SSH_OPTS[@]}" "$RUNPOD_SSH_HOST" "$@"
}

runpod_rsync() {
  rsync -az -e "ssh -i ${RUNPOD_SSH_IDENTITY} -o StrictHostKeyChecking=no" "$@"
}

runpod_echo() {
  echo "RunPod: $RUNPOD_SSH_HOST"
  echo "Key:    $RUNPOD_SSH_IDENTITY"
  echo "Remote: $REMOTE_DIR"
}