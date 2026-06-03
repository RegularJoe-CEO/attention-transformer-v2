# Novel combinations (audit harvest → TRADE v3)

| Repo | Mechanism | TRADE v3 use |
|------|-----------|--------------|
| flash-attention / vLLM | IO-aware causal attention | `LUXI_TRADE_ATTN=flash`, `compare_flash_pod.sh` |
| attention-transformer-v2 geodesic | LN1 + packed QKV + one stream + GPU MLP | `forward_cuda` geodesic path |
| WNSM / NPOW | Null-space MLP payload | Energy vs standard KV/MLP traffic |
| FP8 shadow (v1/v2) | E4M3 GEMM + f32 receipts | Next: fp16 QKV + fp8 MLP on H100 |
| Waller register | Bit-exact causal reference | AUDIT only |
| GAE v7 | Tiled f32 online softmax | `LUXI_TRADE_ATTN=v7` long-ctx |
| Luxi geodesic (ship) | Fused layer discipline | Same as v2 geodesic TRADE |
| Hyperstack | O(N log N) research | Not production until benchmarked vs Flash |
| APA / pyramid (GAE) | Approximate attention | Research — verify accuracy before TRADE |

**Shipped combo:** geodesic fusion around **fp16 tiled** or **Flash** attention + WNSM energy story.
