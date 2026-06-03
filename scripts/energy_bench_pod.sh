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
echo " dmon log=$DMON_LOG (GPU power during cuda_bench only)"
echo " energy sweep -> $OUT"

echo "=== cuda_bench (200 iters @ seq=1024) + dmon ==="
rm -f "$DMON_LOG"
nvidia-smi dmon -s p -d 1 >"$DMON_LOG" 2>&1 &
DMON_PID=$!
cargo run --release --features cuda --example cuda_bench -- 200 1024 1024 16
kill "$DMON_PID" 2>/dev/null || true
wait "$DMON_PID" 2>/dev/null || true

echo "=== energy_sweep -> $OUT ==="
cargo run --release --example energy_sweep >"$OUT"

echo "=== GPU power during cuda_bench (from dmon) ==="
if [[ -f "$DMON_LOG" ]]; then
  # dmon data rows: "  0  117  45  ..." — power (W) is field 2 (POSIX awk; no gawk asort)
  PWRS=$(awk '/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+/ { p=$2+0; if (p>=50) print p }' "$DMON_LOG" | sort -n)
  if [[ -z "$PWRS" ]]; then
    echo " (no dmon samples >= 50 W — check $DMON_LOG)"
  else
    n=$(echo "$PWRS" | wc -l | tr -d ' ')
    med=$(echo "$PWRS" | awk '{a[NR]=$1} END{n=NR; print (n%2)?a[(n+1)/2]:(a[n/2]+a[n/2+1])/2}')
    mean=$(echo "$PWRS" | awk '{s+=$1} END{print s/NR}')
    min=$(echo "$PWRS" | head -1)
    max=$(echo "$PWRS" | tail -1)
    echo " samples: $n  median: $med W  mean: $mean W  min: $min W  max: $max W"
  fi
  echo " full log: $DMON_LOG"
fi

echo "=== J/token estimate (KERNEL-ONLY 2.828 ms @ 1024, use cuda_bench median W) ==="
echo " tokens=N=1024; J ≈ (W_gpu * t_sec) / N — plug median W from cuda_bench (~116 W) and 0.002828 s"
echo " energy table (theoretical HBM model): $OUT"
echo "  head -3 $OUT; tail -3 $OUT"
head -3 "$OUT" 2>/dev/null || true
tail -3 "$OUT" 2>/dev/null || true