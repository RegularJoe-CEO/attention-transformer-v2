#!/usr/bin/env bash
# Energy + power bench on RunPod (must run from repo; sources pod_env).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=pod_env.sh
source "$DIR/pod_env.sh"
cd "$INSTALL_DIR"

OUT="${1:-energy.csv}"
DMON_LOG="${2:-/tmp/nvidia_dmon_pwr.log}"

echo "=== energy_bench_pod ==="
echo " repo=$INSTALL_DIR"
echo " dmon log=$DMON_LOG"
echo " energy sweep -> $OUT"

rm -f "$DMON_LOG"
nvidia-smi dmon -s p -d 1 >"$DMON_LOG" 2>&1 &
DMON_PID=$!
cleanup() {
  kill "$DMON_PID" 2>/dev/null || true
  wait "$DMON_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== cuda_bench (200 iters @ seq=1024) ==="
cargo run --release --features cuda --example cuda_bench -- 200 1024 1024 16

echo "=== energy_sweep -> $OUT ==="
cargo run --release --example energy_sweep >"$OUT"

echo "=== median GPU power (W) from dmon ==="
if [[ -f "$DMON_LOG" ]]; then
  awk 'NF && $2 ~ /^[0-9]+$/ {sum+=$2; n++} END {if(n) printf " median power: %.1f W (%d samples)\n", sum/n, n; else print " (no dmon samples)"}' "$DMON_LOG" || true
  echo " full log: $DMON_LOG"
fi
echo " energy table: $OUT"