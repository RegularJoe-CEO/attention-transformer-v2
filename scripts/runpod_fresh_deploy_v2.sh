#!/usr/bin/env bash
# Fresh deploy attention-transformer-v2 on RunPod (clone + build + cuda_verify + bench)

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/RegularJoe-CEO/attention-transformer-v2.git}"
INSTALL_DIR="${INSTALL_DIR:-/workspace/attention-transformer-v2}"
BRANCH="${BRANCH:-main}"

export PATH="/usr/local/cuda/bin:${PATH:-}"
export CUDA_ARCH="${CUDA_ARCH:-90}"
export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"

LOGFILE="${INSTALL_DIR}/runpod_fresh_deploy_v2_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$INSTALL_DIR")"
exec > >(tee -a "$LOGFILE") 2>&1

echo "attention-transformer-v2 fresh deploy — $(date -u)"
nvidia-smi

if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
fi
# shellcheck source=/dev/null
source "$CARGO_HOME/env" 2>/dev/null || true
export PATH="$CARGO_HOME/bin:$PATH"

if [[ -d "$INSTALL_DIR" ]]; then
  mv "$INSTALL_DIR" "${INSTALL_DIR}.bak.$(date +%Y%m%d_%H%M%S)"
fi

git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
cd "$INSTALL_DIR"
INSTALL_DIR="$INSTALL_DIR" bash scripts/runpod_v2_test.sh