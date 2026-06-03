"""
Waller attention — PyTorch-facing wrapper (CPU AUDIT via pure Python reference).

When `libattention_transformer` is built (`cargo build --release`), set
LUXIEDGE_LIB to load the FFI fast path.
"""

from __future__ import annotations

import ctypes
import os
import struct
from pathlib import Path
from typing import Optional, Tuple

import torch

from luxiedge_ops import sha256_receipt_hex


def _load_lib() -> Optional[ctypes.CDLL]:
    path = os.environ.get("LUXIEDGE_LIB")
    if path and Path(path).exists():
        return ctypes.CDLL(path)
    for candidate in (
        Path("target/release/libattention_transformer.dylib"),
        Path("target/release/libattention_transformer.so"),
        Path("target/release/attention_transformer.dll"),
    ):
        if candidate.exists():
            return ctypes.CDLL(str(candidate))
    return None


def _waller_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """Small-seq reference (matches Rust AUDIT). For production use FFI or Rust bench."""
    seq, hd = q.shape
    out = torch.zeros_like(q)
    for i in range(seq):
        scores = torch.full((seq,), float("-inf"), device=q.device, dtype=q.dtype)
        for j in range(i + 1):
            scores[j] = (q[i] * k[j]).sum() * scale
        w = torch.softmax(scores[: i + 1], dim=0)
        out[i] = (w.unsqueeze(1) * v[: i + 1]).sum(dim=0)
    return out


def waller_attention_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
) -> Tuple[torch.Tensor, str]:
    if not causal:
        raise NotImplementedError("only causal Waller supported")
    q = q.detach().float().contiguous()
    k = k.detach().float().contiguous()
    v = v.detach().float().contiguous()
    assert q.shape == k.shape == v.shape
    seq_len, head_dim = q.shape
    scale = head_dim ** -0.5
    lib = _load_lib()
    if lib is not None and seq_len * head_dim <= 65536:
        out = torch.empty_like(q)
        fn = lib.luxiedge_waller_attention_f32
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
        ]
        fn.restype = ctypes.c_int
        rc = fn(
            q.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            k.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            v.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            seq_len,
            head_dim,
            scale,
        )
        if rc == 0:
            return out, sha256_receipt_hex(out)
    out = _waller_reference(q, k, v, scale)
    return out, sha256_receipt_hex(out)