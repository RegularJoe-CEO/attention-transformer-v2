# vLLM integration (Phase 3)

v2 exposes Waller attention for hyperscale serving via a custom attention backend.

## Status

- **Stub:** `luxi_waller_attention.py` registers a backend name `LUXI_WALLER` for future vLLM worker hooks.
- **Requirement:** Build v2 cdylib + PyTorch op (`integrations/torch/waller_attention.py`) first.

## Target metrics

- `vllm bench throughput` within 90–110% of FlashAttention @ 8K context
- Memory win documented @ 128K+ via `long_context_bench` / H100 `cuda_bench`

## Wiring (when ready)

1. Patch vLLM attention selector to call `waller_attention_torch` for prefill/decode.
2. Run `scripts/commercial_proof.sh` on the same GPU as baseline vLLM.