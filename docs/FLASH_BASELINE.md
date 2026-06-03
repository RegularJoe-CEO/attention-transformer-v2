# Flash is the only opponent that matters

`energy.csv` compares Waller vs **materialized N×N scores** (textbook attention). **FlashAttention does not do that** — it also streams tiles with online softmax. **Do not cite energy_reduction_x vs Flash.**

## Same-pod compare (RunPod)

```bash
cd /workspace/attention-transformer-v2
git pull
source scripts/pod_env.sh
# Args = ITERS SEQ HIDDEN HEADS (same as cuda_bench)
bash scripts/compare_flash_pod.sh 200 1024 1024 16
```

**Common mistake:** `200 1024 1024 16` means iters=200, seq=1024, heads=**16** — not seq=200, heads=1024.

Outputs:

| Row | What it is |
|-----|------------|
| **Waller KERNEL-ONLY** | Your TRADE f32 register kernel (`cuda_bench`) |
| **PyTorch SDPA fp32** | Apples-to-apples dtype (cuDNN may still use fast path) |
| **PyTorch SDPA fp16** | Production-ish PyTorch |
| **flash_attn fp16** | Real Flash-2 (after `pip install flash-attn`) |

**Ratio > 1.0** on `Waller f32 / SDPA f32` = Waller slower than that baseline.

## What winning means

- **Speed:** KERNEL-ONLY ms ≤ Flash/SDPA at same seq, heads, **and dtype** (or document f32 vs f16 gap).
- **Energy:** J/token from `(W × t) / N` on both sides during the same timed loop — not `energy.csv` alone.

## If Flash is faster (expected today at 8k+)

The product claim shifts to:

- **WNSM** payload bytes (Flash does not do null-space transport)
- **AUDIT** bit-exact lane
- **Closing the gap:** FP8 TRADE, better v7 tiles, fusion — measured again with this script