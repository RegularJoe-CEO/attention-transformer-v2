#!/usr/bin/env bash
# attention-transformer-v2 — full RunPod test (AUDIT + TRADE)
# Run ON the pod after the repo is at INSTALL_DIR (default /workspace/attention-transformer-v2).

set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-/workspace/attention-transformer-v2}"
export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
# shellcheck source=/dev/null
[[ -f "$CARGO_HOME/env" ]] && source "$CARGO_HOME/env"
export PATH="$CARGO_HOME/bin:/usr/local/cuda/bin:${PATH:-}"
export CUDA_ARCH="${CUDA_ARCH:-90}"

LOGFILE="${INSTALL_DIR}/runpod_v2_test_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "════════════════════════════════════════════════════════════════════════"
echo " attention-transformer-v2 — RunPod full test"
echo " $(date -u)"
echo " dir=$INSTALL_DIR CUDA_ARCH=$CUDA_ARCH"
echo "════════════════════════════════════════════════════════════════════════"

cd "$INSTALL_DIR"

echo "=== GPU ==="
nvidia-smi

echo "=== CUDA toolkit ==="
command -v nvcc && nvcc --version || { echo "FATAL: nvcc required for TRADE lane"; exit 1; }

echo "=== Rust ==="
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
fi
# shellcheck source=/dev/null
source "$CARGO_HOME/env" 2>/dev/null || true
export PATH="$CARGO_HOME/bin:$PATH"
rustc --version
cargo --version

echo "=== CPU AUDIT gates ==="
cargo test --release 2>&1 | tail -20
cargo run --release --example production_demo
cargo run --release --example hyperwall_bench

echo "=== Build TRADE (cuda) ==="
cargo build --release --features cuda

echo "=== CUDA verify (AUDIT receipt vs CPU) ==="
export LUXI_RECEIPT_AUDIT=1
cargo run --release --features cuda --example cuda_verify

echo "=== CUDA bench ==="
cargo run --release --features cuda --example cuda_bench -- 200 1024 1024 16 || true
cargo run --release --features cuda --example cuda_layer_bench -- 5 128 1024 16 256 || true

echo "=== Quant gate (if present) ==="
if [[ -x scripts/runpod_quant_gate.sh ]]; then
  bash scripts/runpod_quant_gate.sh || true
fi

echo "=== Done ==="
echo "Log: $LOGFILE"
echo "Copy this log back for TRADE ms/layer and receipt lines."