#!/usr/bin/env bash
# Head-to-head: Waller cuda_bench vs PyTorch SDPA / flash-attn on the same pod.
#
# Usage (same arg order as cuda_bench):
#   bash scripts/compare_flash_pod.sh [ITERS] [SEQ] [HIDDEN] [HEADS]
# Example:
#   bash scripts/compare_flash_pod.sh 200 1024 1024 16
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=pod_env.sh
source "$DIR/pod_env.sh"
cd "$INSTALL_DIR"

ITERS="${1:-200}"
SEQ="${2:-1024}"
HIDDEN="${3:-1024}"
HEADS="${4:-16}"

echo "=== compare_flash_pod ==="
echo " cuda_bench order: iters=$ITERS seq=$SEQ hidden=$HIDDEN heads=$HEADS"

if ! python3 -c "import torch" 2>/dev/null; then
  echo "Installing torch for Flash baseline..."
  pip install -q torch
fi

BENCH_LOG=$(mktemp)
cargo run --release --features cuda --example cuda_bench -- "$ITERS" "$SEQ" "$HIDDEN" "$HEADS" 2>&1 | tee "$BENCH_LOG"

WALLER_MS=$(grep -A12 "KERNEL-ONLY" "$BENCH_LOG" | grep "median" | head -1 | sed -E 's/.*:[[:space:]]*([0-9.]+).*/\1/')
if [[ -z "$WALLER_MS" || "$WALLER_MS" == "0" ]]; then
  echo "WARN: could not parse KERNEL-ONLY median from cuda_bench"
  python3 benchmarks/compare_flash.py "$SEQ" "$HIDDEN" "$HEADS" "$ITERS"
else
  echo " Parsed Waller KERNEL-ONLY median: ${WALLER_MS} ms"
  python3 benchmarks/compare_flash.py "$SEQ" "$HIDDEN" "$HEADS" "$ITERS" "$WALLER_MS"
fi

rm -f "$BENCH_LOG"

echo ""
echo "Re-run anytime: bash scripts/compare_flash_pod.sh $ITERS $SEQ $HIDDEN $HEADS"