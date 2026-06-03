#!/usr/bin/env bash
# Commercial proof kit — run all v2 gates locally (CPU) + optional CUDA on NVIDIA hosts.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${ROOT}/target/commercial_proof_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
log() { echo "[commercial_proof] $*" | tee -a "$OUT/summary.log"; }

log "=== attention-transformer-v2 commercial proof ==="
log "cwd=$ROOT"

log "cargo test --release"
cargo test --release 2>&1 | tee "$OUT/cargo_test.log"

log "production_demo (sacred receipt)"
cargo run --release --example production_demo 2>&1 | tee "$OUT/production_demo.log"

log "hyperwall_bench"
cargo run --release --example hyperwall_bench 2>&1 | tee "$OUT/hyperwall_bench.log"

log "long_context_bench"
cargo run --release --example long_context_bench 2>&1 | tee "$OUT/long_context_bench.log"

log "llama_demo"
cargo run --release --example llama_demo 2>&1 | tee "$OUT/llama_demo.log"

if command -v nvcc >/dev/null 2>&1; then
  log "CUDA detected — cuda_verify + cuda_bench"
  export CUDA_ARCH="${CUDA_ARCH:-90}"
  cargo run --release --features cuda --example cuda_verify 2>&1 | tee "$OUT/cuda_verify.log" || true
  cargo run --release --features cuda --example cuda_bench 2>&1 | tee "$OUT/cuda_bench.log" || true
  if [[ -x "$ROOT/scripts/runpod_quant_gate.sh" ]]; then
    bash "$ROOT/scripts/runpod_quant_gate.sh" 2>&1 | tee "$OUT/runpod_quant_gate.log" || true
  fi
else
  log "No nvcc — skip GPU gates (schedule H100 RunPod for TRADE proof)"
fi

if command -v python3 >/dev/null 2>&1; then
  log "PyTorch receipt smoke"
  PYTHONPATH="${ROOT}/integrations/torch:${PYTHONPATH:-}" python3 -c "
import torch
from waller_attention import waller_attention_torch
from luxiedge_ops import sha256_receipt_hex
q = torch.randn(4, 8)
k = torch.randn(4, 8)
v = torch.randn(4, 8)
out, r = waller_attention_torch(q, k, v)
print('receipt', r[:16], 'out', tuple(out.shape))
" 2>&1 | tee "$OUT/pytorch_smoke.log" || true
fi

log "Artifacts: $OUT"
log "DONE"