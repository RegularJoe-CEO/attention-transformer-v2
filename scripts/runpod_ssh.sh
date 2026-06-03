#!/usr/bin/env bash
# RunPod SSH/rsync — reads scripts/runpod.env (from pod-setup paste)

_RUNPOD_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]:-$0}")" && pwd)"
if [[ -f "$_RUNPOD_SCRIPT_DIR/runpod.env" ]]; then
  # shellcheck source=runpod.env
  source "$_RUNPOD_SCRIPT_DIR/runpod.env"
fi

RUNPOD_SSH_USER="${RUNPOD_SSH_USER:-root}"
RUNPOD_SSH_HOST="${RUNPOD_SSH_HOST:-}"
RUNPOD_SSH_PORT="${RUNPOD_SSH_PORT:-}"
RUNPOD_SSH_IDENTITY="${RUNPOD_SSH_IDENTITY:-${HOME}/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/attention-transformer-v2}"
RUNPOD_SSH_IDENTITY="${RUNPOD_SSH_IDENTITY/#\~/$HOME}"

if [[ -z "$RUNPOD_SSH_HOST" ]]; then
  echo "No RunPod SSH configured. Run once:"
  echo "  scripts/pod-setup"
  echo "Then paste the line from RunPod → Connect → SSH over exposed TCP"
  exit 1
fi

if [[ ! -f "$RUNPOD_SSH_IDENTITY" ]]; then
  echo "SSH key not found: $RUNPOD_SSH_IDENTITY"
  echo "RunPod may have given a different -i path in the copied line — re-run pod-setup"
  exit 1
fi

_build_ssh_opts() {
  RUNPOD_SSH_OPTS=(-i "$RUNPOD_SSH_IDENTITY" -o StrictHostKeyChecking=accept-new)
  if [[ -n "$RUNPOD_SSH_PORT" ]]; then
    RUNPOD_SSH_OPTS+=(-p "$RUNPOD_SSH_PORT")
  fi
}

runpod_ssh() {
  _build_ssh_opts
  ssh "${RUNPOD_SSH_OPTS[@]}" "${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}" "$@"
}

runpod_rsync() {
  local port_arg=()
  if [[ -n "$RUNPOD_SSH_PORT" ]]; then
    port_arg=(-p "$RUNPOD_SSH_PORT")
  fi
  rsync -az -e "ssh -i ${RUNPOD_SSH_IDENTITY} ${port_arg[*]} -o StrictHostKeyChecking=no" "$@"
}

runpod_echo() {
  echo "SSH: ${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}${RUNPOD_SSH_PORT:+ port $RUNPOD_SSH_PORT}"
  echo "Key: $RUNPOD_SSH_IDENTITY"
  echo "Remote: $REMOTE_DIR"
}

runpod_ssh_display() {
  _build_ssh_opts
  echo -n "ssh"
  for o in "${RUNPOD_SSH_OPTS[@]}"; do printf ' %q' "$o"; done
  printf ' %q\n' "${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}"
}