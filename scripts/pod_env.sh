#!/usr/bin/env bash
# Source on RunPod every new terminal:  source /workspace/attention-transformer-v2/scripts/pod_env.sh

export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
# shellcheck source=/dev/null
[[ -f "$CARGO_HOME/env" ]] && source "$CARGO_HOME/env"
export PATH="$CARGO_HOME/bin:/usr/local/cuda/bin:${PATH:-}"
export CUDA_ARCH="${CUDA_ARCH:-90}"
export INSTALL_DIR="${INSTALL_DIR:-/workspace/attention-transformer-v2}"

command -v cargo >/dev/null && echo "pod_env: cargo $(cargo --version | cut -d' ' -f2)" || echo "pod_env: cargo NOT FOUND — run: curl -sSf https://sh.rustup.rs | sh -s -- -y && source \$HOME/.cargo/env"
command -v nvcc >/dev/null && echo "pod_env: nvcc ok" || echo "pod_env: nvcc not in PATH"