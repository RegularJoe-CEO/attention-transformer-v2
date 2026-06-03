#!/usr/bin/env python3
"""TRADE v3: geodesic-shaped layer with Flash attention (competitive path).

Approximates v2 geodesic TRADE: LN1 → QKV → Flash causal → wo → res+LN2 → MLP
Uses torch for LN/GEMM (cuBLAS); flash_attn for attention.

Usage:
  python3 integrations/trade_geodesic_flash.py ITERS SEQ HIDDEN HEADS MLP
"""
from __future__ import annotations

import math
import sys
import time

import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
except ImportError as e:
    raise SystemExit("pip install flash-attn") from e


def ln(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps) * gamma + beta


def main() -> None:
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    seq = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    hidden = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    heads = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    mlp_dim = int(sys.argv[5]) if len(sys.argv) > 5 else 256
    head_dim = hidden // heads
    warmup = min(10, max(3, iters // 5))
    dtype = torch.float16
    device = "cuda"

    # Random weights (fixed across iters)
    wq = torch.randn(hidden, hidden, device=device, dtype=dtype) * 0.02
    wk = torch.randn(hidden, hidden, device=device, dtype=dtype) * 0.02
    wv = torch.randn(hidden, hidden, device=device, dtype=dtype) * 0.02
    wo = torch.randn(hidden, hidden, device=device, dtype=dtype) * 0.02
    w_fc = torch.randn(mlp_dim, hidden, device=device, dtype=dtype) * 0.02
    w_proj = torch.randn(hidden, mlp_dim, device=device, dtype=dtype) * 0.02
    ln1_g = torch.ones(hidden, device=device, dtype=dtype)
    ln1_b = torch.zeros(hidden, device=device, dtype=dtype)
    ln2_g = torch.ones(hidden, device=device, dtype=dtype)
    ln2_b = torch.zeros(hidden, device=device, dtype=dtype)

    x = torch.randn(1, seq, hidden, device=device, dtype=dtype)

    def layer_forward(inp: torch.Tensor) -> torch.Tensor:
        h = ln(inp, ln1_g, ln1_b)
        q = h @ wq
        k = h @ wk
        v = h @ wv
        qh = q.view(1, seq, heads, head_dim)
        kh = k.view(1, seq, heads, head_dim)
        vh = v.view(1, seq, heads, head_dim)
        attn = flash_attn_func(qh, kh, vh, causal=True)
        attn_flat = attn.reshape(1, seq, hidden)
        proj = attn_flat @ wo
        combined = inp + proj
        h2 = ln(combined, ln2_g, ln2_b)
        mlp_h = F.gelu(h2 @ w_fc.T)
        out = mlp_h @ w_proj.T
        return out

    for _ in range(warmup):
        layer_forward(x)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        layer_forward(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    med = times[len(times) // 2]

    print("=" * 60)
    print(" TRADE v3 geodesic + Flash (PyTorch GEMM + flash_attn)")
    print("=" * 60)
    print(f" iters={iters} seq={seq} hidden={hidden} heads={heads} mlp={mlp_dim}")
    print(f" FULL LAYER median: {med:.3f} ms")
    print(" Compare: cuda_layer_bench geodesic ~6.9 ms (Waller f32 attn)")
    print("=" * 60)


if __name__ == "__main__":
    main()