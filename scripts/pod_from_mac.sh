#!/usr/bin/env bash
# One-command Mac → RunPod: clone private v2 repo (via gh token) + run GPU tests.
# No GitHub password — uses your existing `gh auth login` on this Mac.
#
# Setup once:
#   export RUNPOD_SSH='root@POD_IP -p SSH_PORT'   # RunPod → Connect → SSH
#
# Then every iteration:
#   bash scripts/pod_from_mac.sh              # full test
#   bash scripts/pod_from_mac.sh --sync-only  # rsync only, no test
#   bash scripts/pod_from_mac.sh --quick      # rsync + cuda_verify only

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="${REMOTE_DIR:-/workspace/attention-transformer-v2}"
MODE="${1:-full}"

if [[ -z "${RUNPOD_SSH:-}" ]]; then
  echo "Set RUNPOD_SSH from RunPod → Connect → SSH over exposed TCP"
  echo "  export RUNPOD_SSH='root@203.0.113.1 -p 22154'"
  exit 1
fi

if ! command -v gh >/dev/null 2>&1 || ! gh auth status >/dev/null 2>&1; then
  echo "Run on your Mac first: gh auth login"
  exit 1
fi

# Token from macOS keyring (never commit this script output)
GITHUB_TOKEN="$(gh auth token)"

# shellcheck disable=SC2086
ssh $RUNPOD_SSH "mkdir -p /workspace && command -v git >/dev/null || (apt-get update -y && apt-get install -y git)"

if [[ "$MODE" != "--sync-only" ]] && ! ssh $RUNPOD_SSH "test -d $REMOTE_DIR/.git" 2>/dev/null; then
  echo "Cloning private repo on pod (token auth, not password)..."
  # shellcheck disable=SC2086
  ssh $RUNPOD_SSH "rm -rf $REMOTE_DIR && git clone --depth 1 https://x-access-token:${GITHUB_TOKEN}@github.com/RegularJoe-CEO/attention-transformer-v2.git $REMOTE_DIR"
fi

echo "Rsyncing latest Desktop tree → pod..."
# shellcheck disable=SC2086
rsync -az -e "ssh $RUNPOD_SSH" \
  --exclude target --exclude .git \
  "$ROOT/" "$RUNPOD_SSH:$REMOTE_DIR/"

if [[ "$MODE" == "--sync-only" ]]; then
  echo "Sync done."
  exit 0
fi

if [[ "$MODE" == "--quick" ]]; then
  # shellcheck disable=SC2086
  ssh $RUNPOD_SSH "export PATH=/usr/local/cuda/bin:\$PATH CUDA_ARCH=90 && cd $REMOTE_DIR && \
    cargo build --release --features cuda 2>/dev/null || (source \$HOME/.cargo/env 2>/dev/null; cargo build --release --features cuda) && \
    cargo run --release --features cuda --example cuda_verify"
  exit 0
fi

echo "Running full runpod_v2_test.sh..."
# shellcheck disable=SC2086
ssh $RUNPOD_SSH "chmod +x $REMOTE_DIR/scripts/runpod_v2_test.sh && INSTALL_DIR=$REMOTE_DIR bash $REMOTE_DIR/scripts/runpod_v2_test.sh"