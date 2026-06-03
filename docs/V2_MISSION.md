# v2 mission: maximum ops/joule (not “determinism first”)

## Primary goal

**The most energy-efficient attention transformer at commercial speed** — measured as:

- **HBM bytes moved per token** (Waller O(N) vs standard O(N²) scores)
- **Wall-clock ms/layer and TFLOPS** on GPU (TRADE lane)
- **Joules per token** (NVML on RunPod / datacenter)

Determinism is a **byproduct of the AUDIT lane**, not the product goal. v1 optimized for trading-desk receipts; **v2 optimizes for TRADE**.

## Two lanes (pick the lane for the job)

| Lane | Hardware | Optimizes for | Use when |
|------|----------|---------------|----------|
| **TRADE** | CUDA fused kernels | **ops/joule, ms/layer, throughput** | Production inference, hyperscale |
| **AUDIT** | CPU f32 reference | Bit-exact regression vs textbook attention | Prove correctness once; CI gate |

TRADE does **not** need to match AUDIT receipts in production — it must match **within tolerance** on verify (`cuda_verify`), then win on **speed and energy**.

## Where the energy comes from (not from SHA-256)

1. **Waller operator** — never materialize N×N attention matrix → linear HBM traffic
2. **Fused TRADE layer** — LN+QKV+attention+MLP in minimal passes
3. **WNSM null-space** — audit/NPOW payloads in MLP slack without extra round-trips
4. **Recurrent KV** — flat decode cost per token
5. **FP8 / INT8 TRADE** (optional) — lower bytes moved on H100

Receipts and deterministic CPU math are **verification tools** — they do not consume energy on the hot TRADE path unless you enable AUDIT in production.

## Headline benchmarks (RunPod)

```bash
source scripts/pod_env.sh
cargo run --release --features cuda --example cuda_bench
cargo run --release --example energy_sweep > energy.csv
```

Publish: ms/layer, TFLOPS, GB moved, W/token from `nvidia-smi`.
## TRADE v3 (corrected — Jun 2026)

- **AUDIT:** Waller f32 + receipts (`LUXI_RECEIPT_AUDIT=1`).
- **TRADE default:** `LUXI_TRADE_ATTN=fp16` → fp16 tiled cuBLAS attention (competitive vs register Waller).
- **Flash-class:** `LUXI_TRADE_ATTN=flash` or Python `integrations/trade_geodesic_flash.py` on pod.
- **Do not** compare `energy.csv` multipliers to Flash — compare measured J/token @ same dtype.

See `docs/TRADE_V3_ARCHITECTURE.md`, `docs/NOVEL_COMBINATIONS.md`.
