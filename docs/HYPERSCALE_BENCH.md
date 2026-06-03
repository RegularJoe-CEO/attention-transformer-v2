# HyperScale benchmark (v2)

## One-command proof kit

```bash
bash scripts/commercial_proof.sh
```

## GPU TRADE headline (RunPod H100)

```bash
source scripts/pod_env.sh
bash scripts/gpu_quick.sh
```

| Metric | Command | What it measures |
|--------|---------|------------------|
| Attention kernel | `cuda_bench -- 200 1024 1024 16` | **KERNEL-ONLY** row = sustained ceiling; **DEVICE-RESIDENT** = kernel+D2H |
| Full layer | `cuda_layer_bench -- 20 1024 1024 16 256` | Geodesic TRADE (LN1 + packed QKV + waller+wo + GPU MLP) |
| Long context | `cuda_bench -- 30 8192 1024 16` | Auto **v7 tiled** cuBLAS path (`seq ≥ 2048`) |
| Receipt gate | `cuda_verify` | Standard B + full decoder parity |

### Kernel selection (TRADE)

- **Register waller** (default @ `seq < 2048`): fused online-softmax, `head_dim ∈ {16,32,64,128}`.
- **v7 tiled** (auto @ `seq ≥ 2048`): persistent workspace, cuBLAS QK tiles + streaming softmax.
- Override: `LUXI_CUDA_V7=1` force, `LUXI_CUDA_V7=0` disable, `LUXI_CUDA_V7_AUTO=0` keep register at long seq.

### Locked H100 reference (attention-only, seq=1024, iters=200)

| Lane | Median ms | Notes |
|------|-----------|-------|
| DEVICE-RESIDENT | ~3.1 | QKV on GPU, kernel+D2H |
| KERNEL-ONLY | ~2.5–2.8 | No D2H (new in v2 sprint) |
| PERSISTENT | ~4.1 | Full H2D each iter |
| Full geodesic layer | ~6.8 | See `docs/QUANT_TRADE_LOCKED.md` |

Record `nvidia-smi` power during `cuda_bench` (printed when available).

## CPU AUDIT (any machine)

```bash
cargo run --release --example hyperwall_bench
cargo run --release --features rayon --example hyperwall_bench
cargo run --release --example long_context_bench
cargo run --release --example llama_demo
cargo run --release --example scaling_sweep > scaling.csv
cargo run --release --example energy_sweep > energy.csv
```

## FlashAttention baseline (external)

Compare on the same host using `../flash-attention/benchmarks/` when present.