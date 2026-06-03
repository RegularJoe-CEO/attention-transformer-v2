# CUDA TRADE port plan (v2 only)

Reference audit snapshot: `../Geodesic-Attention-Engine-GAE-/src/waller_v7.cu` (AGPL — do not ship verbatim).

v2 will add a clean-room `waller_v7_trade.cu` implementing the same tiled online-softmax + cuBLAS contract, wired via `build.rs` and `src/gpu/cuda.rs`.

Gate: `cuda_verify` AUDIT receipt unchanged; TRADE ms/layer logged in `hyperwall_bench` on H100.