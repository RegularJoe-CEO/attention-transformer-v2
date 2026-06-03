# Geodesic Sweep — Deterministic 10× Layer Path (Design)

**Status (2026-06):** Quant TRADE **locked** at ~**6.8 ms/layer** @ seq=1024 (vs ~466 ms pre-geodesic). Batched MLP + GPU QKV + parallel wo shipped; mega-kernel sweep remains optional. Numbers: [`QUANT_TRADE_LOCKED.md`](QUANT_TRADE_LOCKED.md).

**Constraints (fixed):**
- Same tensors in/out: `input[seq × hidden]` → `output[seq × hidden]`, same weight layout, same `sha256_of_f32_slice` receipt contract for **Lane AUDIT**.
- Fully deterministic: fixed traversal order, no atomics on reductions, Welford LN matches CPU, Waller matches `waller_operator.rs`.

**Goal:** ~**46 ms/layer** at seq=1024, hidden=1024, 16 heads on H100 (vs ~466 ms TRADE today) — ~10× — without changing what the trader sees at the API boundary.

---

## 1. Diagnosis: why today is slow (not attention)

| Phase | ms @ 1024 | Root cause |
|-------|-----------|------------|
| CPU QKV | ~433 | Host GEMM + PCIe + 3 launches worth of math in wrong place |
| GPU attn | ~5 | Already won |
| GPU MLP | ~28 | Acceptable |
| **TOTAL** | **~466** | **Data visits the wrong memory hierarchy** |

Attention is not the enemy. **Phase boundaries and host QKV** are. FlashAttention wins by fusion; we win by **one geodesic sweep per layer on device** with the Waller operator as the attention spine (deterministic, O(N) HBM per row).

---

## 2. Design principle: one semantic pass, no phase loops

“No loops” means **no orchestration loops** across CPU/GPU/kernels:

```
Today (6+ steps, 2 PCIe round-trips per layer):
  CPU QKV → H2D QKV → kernel Waller → kernel wo → kernel LN2 → kernel MLP → D2H

Geodesic Sweep (1 launch per layer, 0 host math):
  H2D x (once) → geodesic_layer_sweep → D2H y (once)
```

Inside the kernel, **causal attention is not a nested loop over N²** — it is a **fixed-order column sweep** (already how Waller works): for query row `r`, visit columns `0..=r` once, update `(m, s, acc)` registers. That is a **state machine**, not an algorithmic loop in the orchestration sense.

**Bifurcation (allowed):**
- `if (audit_lane) return cpu_reference_layer(x);` — receipt gate, cold path.
- `else geodesic_layer_sweep(x);` — hot path, 100% device.

---

## 3. The Geodesic Sweep (single kernel per layer)

### 3.1 Grid geometry (match the machine)

| CUDA block | Owns | Warps do |
|------------|------|----------|
| `(blockIdx.x = row, blockIdx.y = head)` | One query row, one head | Register-resident Q; stream K/V along causal prefix |

One block = one **geodesic** through space-time (token index × head). No block synchronizes across rows (causal independence). Same geometry as `waller_multihead_hd_t_kernel`, extended **forward** through the rest of the layer without returning to HBM.

### 3.2 Register pipeline (smooth single pass)

For each `(row, head)` block, in **fixed order**:

```
1. LOAD row x[row,:] → smem (or registers if h≤128)
2. WELFORD LN1 → q̂ (local normed slice for this row)
3. HADAMARD_BUTTERFLY(q̂) → q̃   [optional, add-only, deterministic; preconditions GEMM]
4. QKV READOUT:
     - Not 3× GEMM. One of:
       (A) Prepacked W_qkv [h × 3h] single matmul via WMMA/tensor core with INT32 accum → deterministic dequant, or
       (B) Fused proj_dot unroll for h=64 head slices (already proven bit-exact at hd=64)
5. WALLER SWEEP col = 0..row:
     - Online softmax update (m, s, acc_h) — no N×N matrix
     - Fused wo: acc_h weighted into head slice of attn_proj[row,:] in registers (eliminate separate wo GEMM)
6. BARRIER (block row only): all heads wrote slices → full attn_proj[row,:] in smem
7. RESIDUAL + WELFORD LN2 → mlp_in[row,:]
8. MLP EXPAND (mlp_dim in smem) → GELU → PROJECT + residual → WELFORD LN2 → write output[row,:]
```

**Warp/bend:** Step 5 merges **attention + output projection** — the geometric “bend” that removes a full `[seq,h]×[h,h]` matmul. Output projection is linear; it commutes with the per-head accumulation if you apply `wo` head-slice-wise as values are produced (weights pre-partitioned per head).

### 3.3 Decoder = no layer loop on host

```
d_x = upload(tokens_embedded)   // once
for layer in 0..L:
    geodesic_layer_sweep<<<grid>>>(d_x, d_weights[layer], d_x)  // in-place ping-pong or d_x/d_y
d_out = d_x
D2H once
```

Host **never** loops over `seq` for math. Only `L` launches (or one multi-layer kernel later).

---

## 4. Geometric pieces (why “geometric” is literal)

| Idea | Role | Determinism |
|------|------|-------------|
| **Waller geodesic** | Shortest causal path through keys: one forward sweep per query | Serial `col` order fixed |
| **Walsh–Hadamard** | Orthogonal energy redistribution before QKV; add/sub only | Fixed butterfly stages |
| **Null-space WNSM** | Payload in MLP null space (unchanged) | Already 0.00e0 proven |
| **Online softmax state** | `(m,s,acc)` = sufficient statistics | Same as Flash, but receipt-locked order |

Hadamard is not decoration — it **reduces per-channel dynamic range** so a deterministic INT32 GEMM (Lane B) or WMMA path matches f32 CPU within audit tolerance, or bit-exact f32 accum in fixed order.

---

## 5. How we get 10× (accounting)

| Savings | Mechanism | Approx gain |
|---------|-----------|-------------|
| Remove host QKV | One device GEMM / fused proj | **~9×** on current 466 ms |
| Fuse wo into sweep | −1 matmul, −1 HBM pass | ~1.1× |
| 1 launch vs 4–6 | Launch latency + sync | ~1.05× |
| Persistent d_x + weights | No per-layer alloc | ~1.1× on multi-layer |
| **Combined** | | **~10× → ~40–50 ms/layer** |

Waller alone cannot shrink 433 ms — **moving QKV and orchestration onto the geodesic** does.

---

## 6. Receipt strategy (two lanes, one API)

| Lane | Implementation | Receipt |
|------|----------------|---------|
| **AUDIT** | Existing CPU `forward()` / current AUDIT CUDA split | `0ae65994…` (frozen) |
| **GEODESIC** | `geodesic_layer_sweep` | New namespace `GEODESIC_V1` until proven bit-exact; then merge |

Prove GEODESIC vs CPU on H100 with `cuda_verify` extended — same harness, env `LUXI_GEODESIC=1`.

---

## 7. Implementation order (minimal risk)

1. **P0 — Kill host QKV:** ✅ `layer_forward_cuda_geodesic_gpu_post` — `pack_w_qkv_host`, device `launch_ln1_rows` + one `launch_matmul_f32` → `[seq,3h]`.
2. **P1 — Waller+wo:** TRADE default = parallel waller + wo GEMM (~4 ms @ seq=1024). Row-fused opt-in `LUXI_CUDA_ROW_FUSED=1` (~105 ms).
3. **P2 — Tiled deterministic GEMM:** ✅ `launch_matmul_f32_geodesic` — 32×32 SMEM tiles, fixed `kt`/`kk` order (opt-out `LUXI_CUDA_NAIVE_GEMM=1`).
4. **P3 — `geodesic_layer_sweep` v1:** single launch LN1+QKV+Waller+MLP.
5. **P4 — Multi-layer persistent decoder** (one H2D / one D2H).

---

## 8. What we explicitly do not do

- No approximate attention, no sampling, no Flash “tiling” that reorders sums across blocks.
- No `par_iter().sum()` on host for receipt paths.
- No second model, no speculative loop over draft tokens on the hot path.

---

## 9. Success metrics

```bash
# Must still pass
LUXI_RECEIPT_AUDIT=1 cargo run --release --features cuda --example cuda_verify

# Target
cargo run --release --features cuda --example cuda_layer_bench -- 20 1024 1024 16 256
# TOTAL median < 50 ms on H100 NVL
```

---

*This is the path: bend attention and projection into one causal geodesic, bend the layer into one machine sweep, bifurcate only for audit.*