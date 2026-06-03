# Substantial Test Report — RunPod (certified)

| Field | Value |
|-------|-------|
| **UTC** | 2026-06-02T02:08:37Z |
| **Commit** | `8946803` |
| **GPU** | NVIDIA H100 NVL, driver 580.126.09, 95830 MiB |
| **CUDA** | 12.4 |
| **Rust** | 1.96.0 |
| **Overall** | **PASS** (`exit_code=0`) |
| **Log** | `test_logs/substantial_2026-06-02T02-08-37Z.log` |

## How to re-run

```bash
cd /workspace/attention-transformer
git fetch origin main && git reset --hard origin/main
export PATH="/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH"
export CUDA_ARCH=90
source "$HOME/.cargo/env" 2>/dev/null || true
bash scripts/runpod_substantial_test.sh
echo exit_code=$?   # expect 0
```

## Gate summary

| Test | Result |
|------|--------|
| cargo build `--features cuda` | **PASS** |
| cargo build `--features cuda,cuda-quant` | **PASS** |
| cargo test `--release` (21 tests) | **PASS** |
| cuda_verify (default) | **PASS** |
| cuda_verify (`LUXI_CUDA_GPU_LAYER=1`) | **PASS** |
| cuda_bench 500×1024×1024×16 | **PASS** |
| cuda_bench + `LUXI_WALLER_SMEM=1` (100 iters) | **PASS** (informational; slower) |
| cuda_quant_stack | **PASS** |
| **OVERALL_GATE** | **PASS** |

## Decoder receipt (verified both verify runs)

```text
0ae659948eabc3fa1212b84d9a2006c707c28ba4209ce28410df676d38d37ada
```

- **Receipts match:** true  
- **Max diff:** 0.00e0  
- Isolated Waller Standard B: max_abs 5.96e-8 (per-head harness; expected)

## Throughput — cuda_bench (500×1024×1024×16)

| Metric | Value |
|--------|-------|
| Phase mean H2D | 0.684 ms |
| Phase mean kernel | 3.146 ms |
| Phase mean D2H | 0.197 ms |
| Power (nvidia-smi median) | 118.9 W |
| Checksum guard | 0.044448 |

*(See log for PERSISTENT / DEVICE-RESIDENT medians — typically ~4.1 ms / ~3.1 ms, ~1040–1380 GFLOP/s on this pod.)*

## SMEM kernel (`LUXI_WALLER_SMEM=1`, 100 iters)

| Mode | Median |
|------|--------|
| PERSISTENT | 7.300 ms |
| DEVICE-RESIDENT | 6.609 ms |

**Do not enable for production** on this workload — register kernel is ~2× faster.

## cuda_quant_stack

```text
Lane B INT8 GEMM max_abs(cpu,gpu): 7.451e-9
KV cache step out[0]: 0.030000
forward_cuda len=512 y[0]=-1.751183
```

## Production env (receipt-safe)

| Variable | Use |
|----------|-----|
| *(unset)* | Default legacy path |
| `LUXI_CUDA_GPU_LAYER=1` | Optimized: CPU QKV + GPU Waller |
| `LUXI_CUDA_GPU_QKV=1` | **Avoid** (receipt drift) |
| `LUXI_WALLER_SMEM=1` | **Avoid** (slower on H100) |
| `LUXI_CUDA_MEGA=1` | Verify before audit use |