#!/usr/bin/env bash
# Head-to-head: Waller cuda_bench vs PyTorch SDPA / flash-attn on the same pod.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=pod_env.sh
source "$DIR/pod_env.sh"
cd "$INSTALL_DIR"

SEQ="${1:-1024}"
HIDDEN="${2:-1024}"
HEADS="${3:-16}"
ITERS="${4:-200}"

echo "=== compare_flash_pod ==="
echo " shape: seq=$SEQ hidden=$HIDDEN heads=$HEADS iters=$ITERS"

if ! python3 -c "import torch" 2>/dev/null; then
  echo "Installing torch for Flash baseline..."
  pip install -q torch
fi

BENCH_LOG=$(mktemp)
cargo run --release --features cuda --example cuda_bench -- "$ITERS" "$SEQ" "$HIDDEN" "$HEADS" 2>&1 | tee "$BENCH_LOG"

WALLER_MS=$(grep -A10 "KERNEL-ONLY" "$BENCH_LOG" | awk '/median/ {print $4; exit}')
WALLER_MS="${WALLER_MS%ms}"
if [[ -z "$WALLER_MS" ]]; then
  echo "WARN: could not parse KERNEL-ONLY median from cuda_bench; pass manually to compare_flash.py"
  python3 benchmarks/compare_flash.py "$SEQ" "$HIDDEN" "$HEADS" "$ITERS"
else
  echo " Parsed Waller KERNEL-ONLY median: ${WALLER_MS} ms"
  python3 benchmarks/compare_flash.py "$SEQ" "$HIDDEN" "$HEADS" "$ITERS" "$WALLER_MS"
fi

rm -f "$BENCH_LOG"

echo ""
echo "Optional true Flash-2: pip install flash-attn  (compile ~minutes on first install)"
echo "Then re-run: bash scripts/compare_flash_pod.sh $SEQ $HIDDEN $HEADS $ITERS"