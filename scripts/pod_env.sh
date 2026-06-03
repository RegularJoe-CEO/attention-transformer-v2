#!/usr/bin/env bash
# Source on RunPod every new terminal:  source /workspace/attention-transformer-v2/scripts/pod_env.sh

export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
# shellcheck source=/dev/null
[[ -f "$CARGO_HOME/env" ]] && source "$CARGO_HOME/env"
export PATH="$CARGO_HOME/bin:/usr/local/cuda/bin:${PATH:-}"
export CUDA_ARCH="${CUDA_ARCH:-90}"
export INSTALL_DIR="${INSTALL_DIR:-/workspace/attention-transformer-v2}"

if ! command -v cargo >/dev/null 2>&1; then
  echo "pod_env: installing Rust..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
  # shellcheck source=/dev/null
  source "$CARGO_HOME/env"
  export PATH="$CARGO_HOME/bin:/usr/local/cuda/bin:${PATH:-}"
fi
command -v cargo >/dev/null && echo "pod_env: cargo $(cargo -V)"
command -v nvcc >/dev/null && echo "pod_env: nvcc $(nvcc --version | grep release | head -1)" || echo "pod_env: WARN nvcc not in PATH"