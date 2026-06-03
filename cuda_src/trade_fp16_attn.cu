// trade_fp16_attn.cu — TRADE v3 fp16 tiled causal attention (cuBLAS HGEMM + online softmax).
// Competitive speed path; AUDIT uses f32 Waller in waller_operator.cu.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <algorithm>

#ifndef LUXI_CUDA_MIN
#define LUXI_CUDA_MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

static const int FP16_TILE = 64;

static cublasHandle_t fp16_cublas_handle() {
    static cublasHandle_t h = nullptr;
    if (h == nullptr) {
        cublasCreate(&h);
    }
    return h;
}

__global__ void fp16_init_ml(__half* m, __half* l, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        m[i] = __float2half(-1e4f);
        l[i] = __float2half(0.0f);
    }
}

__global__ void f32_to_f16_kernel(const float* src, __half* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2half(src[i]);
    }
}

__global__ void f16_to_f32_kernel(const __half* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __half2float(src[i]);
    }
}

// Online softmax update (fp16 scores, fp32 accum for stability).
__global__ void fp16_fused_softmax_v_update(
    const __half* scores,
    const __half* V,
    float* Output,
    __half* m,
    __half* l,
    int row_start,
    int col_start,
    int tile_rows,
    int tile_cols,
    int tile_stride,
    int seq_len,
    int head_dim,
    int hidden,
    int head_off,
    float scale
) {
    int local_row = blockIdx.x;
    if (local_row >= tile_rows) return;
    int row = row_start + local_row;

    float m_i = __half2float(m[row]);
    float l_i = __half2float(l[row]);

    for (int c = 0; c < tile_cols; c++) {
        int col = col_start + c;
        if (col > row) continue;
        float s = __half2float(scores[local_row * tile_stride + c]) * scale;
        float m_new = fmaxf(m_i, s);
        float alpha = expf(m_i - m_new);
        float p = expf(s - m_new);
        l_i = l_i * alpha + p;
        m_i = m_new;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            int out_idx = row * hidden + head_off + d;
            float v_val = __half2float(V[col * hidden + head_off + d]);
            Output[out_idx] = Output[out_idx] * alpha + p * v_val;
        }
    }
    if (threadIdx.x == 0) {
        m[row] = __float2half(m_i);
        l[row] = __float2half(l_i);
    }
}

__global__ void fp16_normalize_output(
    float* Output,
    const __half* l,
    int seq_len,
    int head_dim,
    int hidden,
    int head_off
) {
    int row = blockIdx.x;
    if (row >= seq_len) return;
    float inv_l = 1.0f / fmaxf(__half2float(l[row]), 1e-8f);
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        int idx = row * hidden + head_off + d;
        Output[idx] *= inv_l;
    }
}

struct Fp16Ws {
    __half* d_q = nullptr;
    __half* d_k = nullptr;
    __half* d_v = nullptr;
    __half* d_scores = nullptr;
    __half* d_m = nullptr;
    __half* d_l = nullptr;
    float* d_out_f32 = nullptr;
    int cap = 0;
};

static Fp16Ws g_fp16_ws;

static int fp16_ensure(int seq_len, int total, int /*head_dim*/) {
    int need = std::max(seq_len, total);
    if (need <= g_fp16_ws.cap && g_fp16_ws.d_q) return 0;
    auto freep = [](auto*& p) { if (p) cudaFree(p); p = nullptr; };
    freep(g_fp16_ws.d_q);
    freep(g_fp16_ws.d_k);
    freep(g_fp16_ws.d_v);
    freep(g_fp16_ws.d_scores);
    freep(g_fp16_ws.d_m);
    freep(g_fp16_ws.d_l);
    freep(g_fp16_ws.d_out_f32);
    int grow = g_fp16_ws.cap > 0 ? g_fp16_ws.cap : 512;
    while (grow < need) grow *= 2;
    g_fp16_ws.cap = grow;
    size_t elem = (size_t)g_fp16_ws.cap;
    if (cudaMalloc(&g_fp16_ws.d_q, elem * sizeof(__half)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_fp16_ws.d_k, elem * sizeof(__half)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_fp16_ws.d_v, elem * sizeof(__half)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_fp16_ws.d_out_f32, elem * sizeof(float)) != cudaSuccess) return -1;
    size_t score_bytes = (size_t)FP16_TILE * FP16_TILE * sizeof(__half);
    if (cudaMalloc(&g_fp16_ws.d_scores, score_bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&g_fp16_ws.d_m, (size_t)seq_len * sizeof(__half)) != cudaSuccess) return -1;
    if (cudaMalloc(&g_fp16_ws.d_l, (size_t)seq_len * sizeof(__half)) != cudaSuccess) return -1;
    return 0;
}

static void fp16_one_head(
    const __half* Qh,
    const __half* Kh,
    const __half* Vh,
    float* Output,
    int seq_len,
    int head_dim,
    int hidden,
    int head_off,
    float scale,
    cudaStream_t stream
) {
    if (fp16_ensure(seq_len, seq_len * hidden, head_dim) != 0) return;

    fp16_init_ml<<<(seq_len + 255) / 256, 256, 0, stream>>>(g_fp16_ws.d_m, g_fp16_ws.d_l, seq_len);

    cublasHandle_t handle = fp16_cublas_handle();
    cublasSetStream(handle, stream);
    const __half alpha_h = __float2half(1.0f);
    const __half beta_h = __float2half(0.0f);

    for (int row_start = 0; row_start < seq_len; row_start += FP16_TILE) {
        int tile_rows = LUXI_CUDA_MIN(FP16_TILE, seq_len - row_start);
        int max_col = row_start + tile_rows;
        for (int col_start = 0; col_start < max_col; col_start += FP16_TILE) {
            int tile_cols = LUXI_CUDA_MIN(FP16_TILE, seq_len - col_start);
            cublasHgemm(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                tile_cols,
                tile_rows,
                head_dim,
                &alpha_h,
                Kh + col_start * hidden + head_off,
                hidden,
                Qh + row_start * hidden + head_off,
                hidden,
                &beta_h,
                g_fp16_ws.d_scores,
                FP16_TILE);
            fp16_fused_softmax_v_update<<<tile_rows, 128, 0, stream>>>(
                g_fp16_ws.d_scores,
                Vh,
                Output,
                g_fp16_ws.d_m,
                g_fp16_ws.d_l,
                row_start,
                col_start,
                tile_rows,
                tile_cols,
                FP16_TILE,
                seq_len,
                head_dim,
                hidden,
                head_off,
                scale);
        }
    }
    fp16_normalize_output<<<seq_len, 128, 0, stream>>>(
        Output, g_fp16_ws.d_l, seq_len, head_dim, hidden, head_off);
}

extern "C" void launch_trade_fp16_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* Output,
    int seq_len,
    int head_dim,
    int num_heads,
    float scale,
    cudaStream_t stream
) {
    if (seq_len <= 0 || head_dim <= 0 || num_heads <= 0) return;
    const int hidden = head_dim * num_heads;
    const int total = seq_len * hidden;
    if (fp16_ensure(seq_len, total, head_dim) != 0) return;

    int blocks = (total + 255) / 256;
    f32_to_f16_kernel<<<blocks, 256, 0, stream>>>(Q, g_fp16_ws.d_q, total);
    f32_to_f16_kernel<<<blocks, 256, 0, stream>>>(K, g_fp16_ws.d_k, total);
    f32_to_f16_kernel<<<blocks, 256, 0, stream>>>(V, g_fp16_ws.d_v, total);
    cudaMemsetAsync(Output, 0, (size_t)total * sizeof(float), stream);

    for (int h = 0; h < num_heads; h++) {
        fp16_one_head(
            g_fp16_ws.d_q,
            g_fp16_ws.d_k,
            g_fp16_ws.d_v,
            Output,
            seq_len,
            head_dim,
            hidden,
            h * head_dim,
            scale,
            stream);
    }
}
