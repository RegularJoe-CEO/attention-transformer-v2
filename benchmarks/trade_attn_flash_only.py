#!/usr/bin/env python3
"""TRADE attention: flash_attn only (same shape as cuda_bench)."""
from __future__ import annotations

import math
import sys
import time

import torch

try:
    from flash_attn import flash_attn_func
except ImportError as e:
    raise SystemExit("pip install flash-attn") from e


def main() -> None:
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seq = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    hidden = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    heads = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    head_dim = hidden // heads
    warmup = min(20, max(5, iters // 10))

    for dtype in (torch.float16, torch.bfloat16):
        q = torch.randn(1, seq, heads, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(1, seq, heads, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(1, seq, heads, head_dim, device="cuda", dtype=dtype)
        for _ in range(warmup):
            flash_attn_func(q, k, v, causal=True)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            flash_attn_func(q, k, v, causal=True)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        times.sort()
        med = times[len(times) // 2]
        gflops = 4 * seq * seq * head_dim * heads / 1e9 / (med / 1000.0)
        print(f"flash_attn {dtype} median_ms={med:.4f} gflops={gflops:.0f}")


if __name__ == "__main__":
    main()