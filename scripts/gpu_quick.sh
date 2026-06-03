#!/usr/bin/env bash
# TRADE lane quick test (energy/speed path) — RunPod only.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=pod_env.sh
source "$DIR/pod_env.sh"
cd "$INSTALL_DIR"
git pull 2>/dev/null || true
echo "=== Build TRADE ==="
export CUDA_ARCH="${CUDA_ARCH:-90}"
cargo build --release --features cuda,flash-bridge
echo "=== cuda_verify (AUDIT vs GPU tolerance) ==="
cargo run --release --features cuda,flash-bridge --example cuda_verify
echo "=== cuda_bench @ seq=1024 (headline attention ms) ==="
cargo run --release --features cuda,flash-bridge --example cuda_bench -- 200 1024 1024 16
echo "=== cuda_layer_bench @ seq=1024 (full geodesic layer) ==="
cargo run --release --features cuda,flash-bridge --example cuda_layer_bench -- 20 1024 1024 16 256
echo "=== cuda_bench @ seq=8192 (v7 tiled auto) ==="
cargo run --release --features cuda,flash-bridge --example cuda_bench -- 30 8192 1024 16
echo "=== production_demo (sacred receipt) ==="
cargo run --release --example production_demo