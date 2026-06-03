#!/usr/bin/env python3
"""Same-pod attention baseline: PyTorch SDPA (+ optional flash-attn) vs Waller cuda_bench shape.

Usage (on RunPod, after source scripts/pod_env.sh):
  python3 benchmarks/compare_flash.py [SEQ] [HIDDEN] [HEADS] [ITERS]

Fairness notes:
  - Waller TRADE cuda_bench is f32, single batch, causal.
  - Flash-Attn ships f16/bf16; SDPA may pick cuDNN Flash kernel on H100.
  - Compare f32 SDPA row for dtype parity; f16 Flash row for "production Flash speed".
"""

from __future__ import annotations

import math
import sys
import time

import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func

    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False


def median_ms(samples: list[float]) -> float:
    s = sorted(samples)
    n = len(s)
    return s[n // 2]


def bench_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    iters: int,
    warmup: int,
) -> float:
    scale = 1.0 / math.sqrt(q.shape[-1])
    for _ in range(warmup):
        F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=scale
        )
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=scale
        )
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return median_ms(times)


def bench_flash(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    iters: int,
    warmup: int,
) -> float:
    # flash_attn expects (batch, seqlen, nheads, headdim)
    for _ in range(warmup):
        flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return median_ms(times)


def main() -> None:
    seq = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    hidden = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    heads = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    warmup = min(20, max(5, iters // 10))

    if hidden % heads != 0:
        raise SystemExit("hidden must be divisible by heads")
    head_dim = hidden // heads
    batch = 1

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    print("=" * 60)
    print(" FLASH / SDPA baseline (the only opponent that matters)")
    print("=" * 60)
    print(f" batch=1 seq={seq} hidden={hidden} heads={heads} head_dim={head_dim}")
    print(f" iters={iters} warmup={warmup} device={torch.cuda.get_device_name(0)}")
    print("-" * 60)

    waller_ms = None
    if len(sys.argv) > 5:
        waller_ms = float(sys.argv[5])
        print(f" Waller KERNEL-ONLY (from cuda_bench): {waller_ms:.3f} ms")
        print("-" * 60)

    for dtype_name, dtype in [("fp32", torch.float32), ("fp16", torch.float16)]:
        q = torch.randn(batch, seq, heads, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(batch, seq, heads, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(batch, seq, heads, head_dim, device="cuda", dtype=dtype)

        sdpa_med = bench_sdpa(q, k, v, iters, warmup)
        print(f" PyTorch SDPA ({dtype_name}, causal)  median: {sdpa_med:7.3f} ms")

        if HAS_FLASH and dtype in (torch.float16, torch.bfloat16):
            flash_med = bench_flash(q, k, v, iters, warmup)
            print(f" flash_attn ({dtype_name}, causal)   median: {flash_med:7.3f} ms")
        elif dtype == torch.float16 and not HAS_FLASH:
            print(
                " flash_attn: not installed (pip install flash-attn on pod for true Flash-2 row)"
            )

        if waller_ms is not None and dtype == torch.float32:
            ratio = waller_ms / sdpa_med
            print(f" Waller f32 / SDPA f32 ratio: {ratio:.2f}x  (>1.0 = Waller slower)")

    print("=" * 60)
    print(" Run Waller: cargo run --release --features cuda --example cuda_bench --")
    print(f"   {iters} {seq} {hidden} {heads}")
    print(" Then re-run with KERNEL-ONLY median ms as 5th arg to this script.")
    print("=" * 60)


if __name__ == "__main__":
    main()