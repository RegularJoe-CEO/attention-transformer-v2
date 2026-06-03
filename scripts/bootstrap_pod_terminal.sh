#!/usr/bin/env bash
# Run INSIDE RunPod web terminal (you are root@bff7ccc1fa32, not on your Mac).
# Does NOT use ~/Desktop — uses /workspace on the pod.

set -euo pipefail

export PATH="/usr/local/cuda/bin:${PATH:-}"
export CUDA_ARCH="${CUDA_ARCH:-90}"
export INSTALL_DIR="${INSTALL_DIR:-/workspace/attention-transformer-v2}"

echo "=== RunPod pod bootstrap (attention-transformer-v2) ==="
hostname
nvidia-smi || { echo "No GPU visible"; exit 1; }

apt-get update -y || true
apt-get install -y git curl build-essential pkg-config libssl-dev || true

if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
fi
# shellcheck source=/dev/null
source "${HOME}/.cargo/env"
cargo --version

if [[ ! -d "$INSTALL_DIR/.git" ]]; then
  if [[ -z "${GITHUB_TOKEN:-}" ]]; then
    echo ""
    echo "Private repo: set a GitHub PAT (repo scope) from https://github.com/settings/tokens"
    echo "  export GITHUB_TOKEN=ghp_xxxxxxxx"
    echo "Then re-run:  bash bootstrap_pod_terminal.sh"
    echo ""
    echo "Or clone manually:"
    echo "  cd /workspace"
    echo "  git clone https://x-access-token:\$GITHUB_TOKEN@github.com/RegularJoe-CEO/attention-transformer-v2.git"
    exit 1
  fi
  rm -rf "$INSTALL_DIR"
  git clone --depth 1 "https://x-access-token:${GITHUB_TOKEN}@github.com/RegularJoe-CEO/attention-transformer-v2.git" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"
git pull origin main || true
bash scripts/runpod_v2_test.sh