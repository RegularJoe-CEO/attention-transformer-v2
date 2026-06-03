#!/usr/bin/env bash
# Mac: commit+push to GitHub, then pod git pull + quick CUDA verify.
# Requires: gh auth on Mac, RUNPOD_SSH, repo already cloned on pod once.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MSG="${1:-wip}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/attention-transformer-v2}"

cd "$ROOT"
git add -A
if git diff --staged --quiet; then
  echo "Nothing to commit."
else
  git commit -m "$MSG"
fi
git push origin main

if [[ -z "${RUNPOD_SSH:-}" ]]; then
  echo "Pushed. Set RUNPOD_SSH and re-run, or on pod: cd $REMOTE_DIR && git pull && bash scripts/runpod_v2_test.sh"
  exit 0
fi

# shellcheck disable=SC2086
ssh $RUNPOD_SSH "cd $REMOTE_DIR && git pull && export PATH=/usr/local/cuda/bin:\$PATH CUDA_ARCH=90 && \
  cargo run --release --features cuda --example cuda_verify"