#!/usr/bin/env bash
# Mac: commit + push, then pod pull + cuda_verify

set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
# shellcheck source=runpod_ssh.sh
source "$DIR/runpod_ssh.sh"

MSG="${1:-wip}"
cd "$ROOT"
git add -A
if ! git diff --staged --quiet; then
  git commit -m "$MSG"
fi
git push origin main

runpod_ssh "cd $REMOTE_DIR && git pull && export PATH=/usr/local/cuda/bin:\$PATH CUDA_ARCH=${CUDA_ARCH:-90} && \
  cargo run --release --features cuda --example cuda_verify"