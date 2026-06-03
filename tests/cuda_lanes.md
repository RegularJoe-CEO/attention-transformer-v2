# CUDA lane tests (manual — requires NVIDIA + `--features cuda`)

Locked reference: [`docs/QUANT_TRADE_LOCKED.md`](../docs/QUANT_TRADE_LOCKED.md)

## Quick regression (RunPod)

```bash
export CUDA_ARCH=90 PATH="/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH"
bash scripts/runpod_quant_gate.sh
```

## AUDIT receipt gate

```bash
LUXI_RECEIPT_AUDIT=1 cargo run --release --features cuda --example cuda_verify
```

Expect: `FULL DECODER CUDA PATH VERIFIED`, receipt `0ae659948eabc3fa1212b84d9a2006c707c28ba4209ce28410df676d38d37ada`, `max_diff 0.00e0`.

## TRADE throughput (locked H100 @ seq=1024)

| Bench | What it measures | Locked median |
|-------|------------------|---------------|
| `cuda_bench` | Waller only (Q,K,V on GPU) | **~4 ms** |
| `cuda_layer_bench` | One full layer | **~6.8 ms** |
| `cuda_quant_bench` | 12-layer GPU stack | **~69 ms** (~5.8 ms/layer) |

```bash
cargo run --release --features cuda --example cuda_layer_bench -- 20 1024 1024 16 256
cargo run --release --features cuda --example cuda_quant_bench -- 20 1024 12
cargo run --release --features cuda --example cuda_bench -- 50 1024 1024 16
```

Phase breakdown: `LUXI_CUDA_PHASE_TIMING=1` with `cuda_layer_bench`.

**Do not use:** `LUXI_CUDA_ROW_FUSED=1` (~135 ms/layer), `LUXI_CUDA_FUSED_ROW_MLP=1` (~27 ms MLP).

## NPOW scaling witness (CPU — any machine)

```bash
LUXI_NPOW_FAST=1 cargo run --release --example npow_scaling_proof
```

Expect: `NPOW memory scaling proof PASS`, `NPOW scaling proof complete`, WNSM `max_diff 0.00e0`. Included in `runpod_quant_gate.sh`.

## Full regression

```bash
bash scripts/runpod_substantial_test.sh
bash scripts/runpod_geodesic_bench.sh
```