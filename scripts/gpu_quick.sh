#!/usr/bin/env bash
# TRADE lane quick test (energy/speed path) — RunPod only.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=pod_env.sh
source "$DIR/pod_env.sh"
cd "$INSTALL_DIR"
git pull 2>/dev/null || true
echo "=== Build TRADE ==="
cargo build --release --features cuda
echo "=== cuda_verify (AUDIT vs GPU tolerance) ==="
cargo run --release --features cuda --example cuda_verify
echo "=== cuda_bench (ms/layer — headline metric) ==="
cargo run --release --features cuda --example cuda_bench -- 200 1024 1024 16