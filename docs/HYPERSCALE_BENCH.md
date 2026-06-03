# HyperScale benchmark (v2)

## CPU AUDIT (any machine)

```bash
cargo run --release --example hyperwall_bench
cargo run --release --example scaling_sweep > scaling.csv
cargo run --release --example energy_sweep > energy.csv
```

## GPU TRADE (H100 RunPod)

```bash
export CUDA_ARCH=90
cargo run --release --features cuda --example cuda_verify
cargo run --release --features cuda --example cuda_bench
bash scripts/runpod_quant_gate.sh
```

Record `nvidia-smi` power draw during runs; attach logs to this doc when H100 numbers are captured.

## FlashAttention baseline (external, read-only repo)

Compare on the same host using `../flash-attention/benchmarks/` after checking out `triangle-engine-benchmark` if present.