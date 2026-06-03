#!/usr/bin/env python3
"""Same-pod attention baseline: PyTorch SDPA (+ flash-attn) vs Waller cuda_bench shape.

Usage:
  python3 benchmarks/compare_flash.py SEQ HIDDEN HEADS ITERS [WALLER_MS]

On H100, default SDPA often dispatches to the Flash/cuDNN backend (very fast).
We also print SDPA-math (Flash disabled) for a less misleading f32 row.
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

# Match cuda_bench FLOP model: 4 * seq^2 * head_dim * heads
def attn_gflops(seq: int, head_dim: int, heads: int) -> float:
    return 4.0 * (seq**2) * head_dim * heads / 1e9


def median_ms(samples: list[float]) -> float:
    s = sorted(samples)
    return s[len(s) // 2]


def bench_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    iters: int,
    warmup: int,
    *,
    enable_flash: bool,
    enable_mem_efficient: bool,
    enable_math: bool,
) -> float:
    scale = 1.0 / math.sqrt(q.shape[-1])
    with torch.backends.cuda.sdp_kernel(
        enable_flash=enable_flash,
        enable_mem_efficient=enable_mem_efficient,
        enable_math=enable_math,
    ):
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


def print_row(label: str, ms: float, gflops: float, waller_ms: float | None) -> None:
    gflops_per_s = gflops / (ms / 1000.0) if ms > 0 else 0.0
    print(f" {label:<32} median: {ms:7.3f} ms  (~{gflops_per_s:6.0f} GFLOP/s)")
    if waller_ms is not None:
        print(f"   Waller f32 / this row: {waller_ms / ms:.2f}x  (>1 = Waller slower)")


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
    gflops = attn_gflops(seq, head_dim, heads)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    print("=" * 60)
    print(" FLASH / SDPA baseline (the only opponent that matters)")
    print("=" * 60)
    print(f" batch=1 seq={seq} hidden={hidden} heads={heads} head_dim={head_dim}")
    print(f" iters={iters} warmup={warmup} device={torch.cuda.get_device_name(0)}")
    print(f" nominal attn GFLOPs/iter: {gflops:.3f} (cuda_bench formula)")
    print("-" * 60)

    waller_ms = float(sys.argv[5]) if len(sys.argv) > 5 else None
    if waller_ms is not None:
        print(f" Waller KERNEL-ONLY (cuda_bench): {waller_ms:.3f} ms")
        print_row("Waller (effective)", waller_ms, gflops, None)
        print("-" * 60)

    for dtype_name, dtype in [("fp32", torch.float32), ("fp16", torch.float16)]:
        print(f" [{dtype_name}]")
        q = torch.randn(batch, seq, heads, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(batch, seq, heads, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(batch, seq, heads, head_dim, device="cuda", dtype=dtype)

        # What PyTorch picks by default on H100 (usually Flash-sdpa)
        sdpa_auto = bench_sdpa(
            q, k, v, iters, warmup, enable_flash=True, enable_mem_efficient=True, enable_math=True
        )
        print_row("SDPA auto (Flash/mem/math allowed)", sdpa_auto, gflops, waller_ms)

        # Flash disabled — closer to "classic" f32 matmul attention (still cuBLAS-tuned)
        sdpa_math = bench_sdpa(
            q, k, v, iters, warmup, enable_flash=False, enable_mem_efficient=False, enable_math=True
        )
        print_row("SDPA math-only (Flash OFF)", sdpa_math, gflops, waller_ms)

        if HAS_FLASH and dtype == torch.float16:
            flash_med = bench_flash(q, k, v, iters, warmup)
            print_row("flash_attn 2.x causal", flash_med, gflops, waller_ms)
        elif dtype == torch.float16:
            print(" flash_attn: not installed")

    print("=" * 60)
    print(" Verdict: if SDPA auto ~0.05 ms and Waller ~2.8 ms, Flash-class kernels win")
    print(" on this H100 today. Waller f32 register path ~50x slower on wall-clock.")
    print(" Next: FP16/BF16 TRADE, FP8, or compete on WNSM + J/token — not naive energy.csv.")
    print("=" * 60)


if __name__ == "__main__":
    main()