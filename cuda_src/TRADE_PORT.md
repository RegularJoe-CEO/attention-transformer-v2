# CUDA TRADE port plan (v2 only)

Reference audit snapshot: `../Geodesic-Attention-Engine-GAE-/src/waller_v7.cu` (AGPL — do not ship verbatim).

v2 ships clean-room `waller_v7_trade.cu` (tiled online-softmax + cuBLAS), wired via `build.rs` and `src/gpu/cuda.rs`.

## v7 long-context path (implemented)

- `launch_waller_v7_trade` — multi-head loop, **persistent** `d_scores` / `d_m` / `d_l` (no per-call `cudaMalloc`).
- Auto-dispatch from `launch_waller_operator` when `waller_v7_should_use` (default `seq_len >= 2048`).
- Env: `LUXI_CUDA_V7=1` force, `LUXI_CUDA_V7=0` disable, `LUXI_CUDA_V7_AUTO=0` keep register kernel at long seq.
- Short seq (`< 2048`): register `waller_multihead_hd_t_kernel` stays default (fastest @ 1024 on H100).

Gate: `cuda_verify` AUDIT receipt unchanged; TRADE in `cuda_bench` / `cuda_layer_bench` on H100 (`bash scripts/gpu_quick.sh`).