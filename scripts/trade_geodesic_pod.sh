#!/usr/bin/env bash
# TRADE v3 proof: Flash attention + geodesic-shaped layer (competitive path).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/pod_env.sh"
cd "$INSTALL_DIR"

ITERS="${1:-50}"
SEQ="${2:-1024}"
HIDDEN="${3:-1024}"
HEADS="${4:-16}"
MLP="${5:-256}"

echo "=== TRADE v3 geodesic + Flash ==="
python3 -c "import flash_attn, torch" 2>/dev/null || pip install -q flash-attn einops torch

echo "=== Build (cuda + flash-bridge) ==="
cargo build --release --features cuda,flash-bridge

echo "--- Rust TRADE attention (flash-bridge @ LUXI_TRADE_ATTN=$LUXI_TRADE_ATTN) ---"
cargo run --release --features cuda,flash-bridge --example cuda_bench -- 50 "$SEQ" "$HIDDEN" "$HEADS" 2>&1 | tee /tmp/cuda_bench_trade.log
grep -E "TRADE attention|flash-bridge|KERNEL-ONLY|median|Error|panic|bridge" /tmp/cuda_bench_trade.log || true

echo "--- Flash attention only (TRADE v3 core) ---"
python3 benchmarks/trade_attn_flash_only.py 50 "$SEQ" "$HIDDEN" "$HEADS"

echo "--- Full layer: geodesic + Flash (Python TRADE v3) ---"
python3 integrations/trade_geodesic_flash.py "$ITERS" "$SEQ" "$HIDDEN" "$HEADS" "$MLP"

echo "--- Geodesic full layer (Rust + flash-bridge) ---"
cargo run --release --features cuda,flash-bridge --example cuda_layer_bench -- 50 "$SEQ" "$HIDDEN" "$HEADS" "$MLP" 2>&1 | grep -E "TOTAL|Waller|attn|median"

echo "--- Head-to-head script ---"
bash "$DIR/compare_flash_pod.sh" 50 "$SEQ" "$HIDDEN" "$HEADS"