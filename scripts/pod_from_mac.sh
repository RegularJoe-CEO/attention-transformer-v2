#!/usr/bin/env bash
# Mac → RunPod: clone (gh token) + rsync + test. SSH from scripts/runpod.env
#
#   scripts/pod              # full
#   scripts/pod quick
#   scripts/pod sync

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
# shellcheck source=runpod_ssh.sh
source "$DIR/runpod_ssh.sh"

MODE="${1:-full}"

if ! command -v gh >/dev/null 2>&1 || ! gh auth status >/dev/null 2>&1; then
  echo "On Mac run: gh auth login   (account RegularJoe-CEO)"
  exit 1
fi

GITHUB_TOKEN="$(gh auth token)"
runpod_echo

runpod_ssh "mkdir -p /workspace && command -v git >/dev/null || (apt-get update -y && apt-get install -y git)"

if [[ "$MODE" != "--sync-only" ]] && ! runpod_ssh "test -d $REMOTE_DIR/.git" 2>/dev/null; then
  echo "Cloning attention-transformer-v2 on pod..."
  runpod_ssh "rm -rf $REMOTE_DIR && git clone --depth 1 https://x-access-token:${GITHUB_TOKEN}@github.com/RegularJoe-CEO/attention-transformer-v2.git $REMOTE_DIR"
fi

echo "Rsync Desktop → pod..."
runpod_rsync --exclude target --exclude .git "$ROOT/" "$RUNPOD_SSH_HOST:$REMOTE_DIR/"

if [[ "$MODE" == "--sync-only" ]]; then
  echo "Sync done."
  exit 0
fi

if [[ "$MODE" == "--quick" ]]; then
  runpod_ssh "export PATH=/usr/local/cuda/bin:\$PATH CUDA_ARCH=${CUDA_ARCH:-90} && cd $REMOTE_DIR && \
    (command -v cargo >/dev/null || (curl -sSf https://sh.rustup.rs | sh -s -- -y && source \$HOME/.cargo/env)) && \
    cargo build --release --features cuda && \
    cargo run --release --features cuda --example cuda_verify"
  exit 0
fi

echo "Full GPU test..."
runpod_ssh "chmod +x $REMOTE_DIR/scripts/runpod_v2_test.sh && INSTALL_DIR=$REMOTE_DIR bash $REMOTE_DIR/scripts/runpod_v2_test.sh"