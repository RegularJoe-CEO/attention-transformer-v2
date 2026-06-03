# attention-transformer-v2 lineage

This repository is the **only** write target for the HyperScale energy-efficient transformer program. Audit clones under `~/Desktop/eRock/_audit_repos/` stay **read-only**.

## Harvest map

| Audit repo | Role in v2 | Integration status |
|------------|------------|-------------------|
| `attention-transformer` | v1 base copied at fork | Core crate (Waller, WNSM, CUDA, GPT-2) |
| `Geodesic-Attention-Engine-GAE-` | CUDA TRADE kernels, equivalence tests | `standard_attention` gate; `waller_v7` port planned |
| `Adiabatic-Transform-Engine-ATE-` | Geometric null-space MLP | Planned TRADE fusion |
| `Luxi` | RoPE, RMSNorm, quant matmul | Planned LLaMA profile |
| `luxi-quant-engine` | Receipts, WGPU reference | Partial via existing FFI |
| `flash-attention` | Baseline benchmarks | External compare only |
| `vllm` | Serving shell | Phase 3 plugin |
| `waller-eval` | Archived evidence | Reference only |

## Version policy

- **v1** (`attention-transformer`): frozen reference; no edits during v2 work.
- **v2** (this repo): all new code, docs, benches, PyTorch/vLLM bridges.

## Legal

GAE CUDA is AGPL-3.0 in the audit tree. Production TRADE kernels in v2 must be **algorithm ports** with independent legal review before commercial ship—not verbatim AGPL file copies in release binaries.