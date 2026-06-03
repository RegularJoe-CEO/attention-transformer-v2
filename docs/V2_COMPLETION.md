# v2 program completion status

All implementation work lives in **attention-transformer-v2** only. Audit repos under `_audit_repos/` are unchanged.

## Phase 0 — Gates

| Gate | Status |
|------|--------|
| `cargo test --release` | 33 tests pass |
| `production_demo` sacred receipt | Pass |
| `waller_standard_equivalence` | Pass @ seq 16, 512 |
| `cargo clippy -D warnings` | Pass |
| FlashAttention baseline CSV | Run on H100 host with `../flash-attention` |

## Phase 1 — GPU TRADE

| Item | Status |
|------|--------|
| `cuda_src/waller_v7_trade.cu` + cublas | Implemented |
| `waller_v7_trade_cuda` Rust FFI | Implemented |
| `mega_fused_layer` in `waller_operator.cu` | Already wired (doc stub updated) |
| `waller_operator_parallel` | Real rayon path, bit-identical test |
| `hyperwall_bench` + energy mem ratio | Implemented |

**H100 required:** `CUDA_ARCH=90 cargo run --release --features cuda --example cuda_verify`

## Phase 2 — Model coverage

| Item | Status |
|------|--------|
| LLaMA RoPE + RMSNorm | `src/rope_rmsnorm.rs` |
| LLaMA layer prefill | `src/llama/mod.rs` + `llama_demo` |
| Long-context bench | `long_context_bench` |
| GPT-2 CUDA TRADE | Inherited from v1 snapshot |
| RULER / InfiniteBench eval | Run manually from audit clones |

## Phase 3 — Commercial surfaces

| Item | Status |
|------|--------|
| FFI `luxiedge_waller_attention_f32` | Implemented |
| PyTorch `waller_attention.py` | Implemented |
| vLLM stub | `integrations/vllm/luxi_waller_attention.py` |
| `scripts/commercial_proof.sh` | Implemented |

## Next on H100 (speed claims)

1. `bash scripts/commercial_proof.sh` with nvcc
2. Compare `cuda_bench` vs FlashAttention on same box
3. Wire vLLM worker to `LUXI_WALLER` backend
4. Publish `docs/HYPERSCALE_BENCH.md` with NVML joules