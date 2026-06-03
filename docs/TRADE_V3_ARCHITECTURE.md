# TRADE v3 — competitive + energy-efficient (corrected)

## What went wrong in v2.0 TRADE

We shipped **register f32 Waller** as the default GPU attention kernel. That path is for **AUDIT / receipt proof**, not for fighting Flash. On H100 @ seq=1024 it is **~50× slower** than `flash_attn` / SDPA.

**Determinism is not the product.** Ops/joule at commercial speed is.

## Novel combination (from your audit repos)

| Source | Idea | Role in TRADE v3 |
|--------|------|------------------|
| **flash-attention** | IO-aware tiled attention | **TRADE attention core (speed)** |
| **v2 geodesic** | Packed LN1+QKV GEMM, one stream, quant stack | **HBM around attention** |
| **WNSM / NPOW** | Null-space payload in MLP | **Energy others don't have** |
| **FP8 shadow** (v1/v2) | E4M3 matmul, f32 receipts | **Lower joules on GEMMs** |
| **Waller register** (v2) | Bit-exact causal reference | **AUDIT only** — not TRADE default |
| **GAE v7 / APA** | Tiled / pyramid experiments | Research — verify before production |

## TRADE v3 stack (production)

```
H2D (once)
  → GPU LN1
  → packed QKV GEMM (geodesic)
  → Flash-Attn 2 causal (fp16/bf16)   ← speed
  → wo GEMM
  → residual + LN2
  → fused MLP (+ WNSM inject)
  → D2H (once)
```

AUDIT lane (CI): CPU or CUDA **Waller f32** vs textbook, receipts.

## Env

| Variable | Values |
|----------|--------|
| `LUXI_TRADE_ATTN` | `flash` (default TRADE), `waller` (legacy register), `v7` (tiled cuBLAS) |
| `LUXI_RECEIPT_AUDIT` | `1` → forces CPU/waller, disables flash default |

## Proof on RunPod

```bash
source scripts/pod_env.sh
bash scripts/trade_geodesic_pod.sh 200 1024 1024 16
bash scripts/compare_flash_pod.sh 200 1024 1024 16
```

Target: **attention ≤ Flash fp16** (same order ms), **full layer < 2 ms** aspirational on H100 @ 1024 with fusion.