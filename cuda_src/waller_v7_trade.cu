// attention-transformer-v2 — TRADE tiled Waller attention (online-softmax + cuBLAS)
// Algorithm port: tiled QK GEMM + streaming softmax/V update (O(N) score tile memory).

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <algorithm>

#define WALLER_V7_TILE 512

__global__ void v7_fused_softmax_v_update(
    const float* __restrict__ scores,
    const float* __restrict__ V_tile,
    float* __restrict__ O,
    float* __restrict__ m_global,
    float* __restrict__ l_global,
    int row_offset,
    int col_offset,
    int tile_rows,
    int tile_cols,
    int score_ld,
    int seq_len,
    int head_dim,
    float scale
) {
    int local_row = blockIdx.x;
    int row = row_offset + local_row;
    if (row >= seq_len || local_row >= tile_rows) return;

    int tid = threadIdx.x;
    int max_col_for_row = min(tile_cols, row - col_offset + 1);
    if (max_col_for_row <= 0) return;

    float m_old = m_global[row];
    float l_old = l_global[row];
    float m_tile = -INFINITY;
    for (int c = 0; c < max_col_for_row; c++) {
        float s = scores[local_row * score_ld + c] * scale;
        m_tile = fmaxf(m_tile, s);
    }
    float m_new = fmaxf(m_old, m_tile);
    float rescale = expf(m_old - m_new);
    float exp_sum = 0.0f;
    for (int c = 0; c < max_col_for_row; c++) {
        float s = scores[local_row * score_ld + c] * scale;
        exp_sum += expf(s - m_new);
    }
    float l_new = l_old * rescale + exp_sum;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = O[row * head_dim + d] * rescale;
        for (int c = 0; c < max_col_for_row; c++) {
            float s = scores[local_row * score_ld + c] * scale;
            float w = expf(s - m_new);
            acc += w * V_tile[c * head_dim + d];
        }
        O[row * head_dim + d] = acc;
    }
    if (tid == 0) {
        m_global[row] = m_new;
        l_global[row] = l_new;
    }
}

__global__ void v7_normalize_output(float* O, const float* l, int seq_len, int head_dim) {
    int row = blockIdx.x;
    if (row >= seq_len) return;
    float norm = l[row];
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        O[row * head_dim + d] /= norm;
    }
}

__global__ void v7_init_ml(float* m, float* l, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
    }
}

static cublasHandle_t v7_cublas_handle() {
    static cublasHandle_t h = nullptr;
    if (h == nullptr) {
        cublasCreate(&h);
    }
    return h;
}

extern "C" void launch_waller_v7_trade(
    const float* Q,
    const float* K,
    const float* V,
    float* Output,
    int seq_len,
    int head_dim,
    float scale,
    cudaStream_t stream
) {
    if (seq_len <= 0 || head_dim <= 0) return;

    cublasHandle_t handle = v7_cublas_handle();
    cublasSetStream(handle, stream);

    size_t mat_size = (size_t)seq_len * head_dim * sizeof(float);
    size_t score_tile_size = (size_t)WALLER_V7_TILE * WALLER_V7_TILE * sizeof(float);

    float *d_scores = nullptr, *d_m = nullptr, *d_l = nullptr;
    cudaMalloc(&d_scores, score_tile_size);
    cudaMalloc(&d_m, (size_t)seq_len * sizeof(float));
    cudaMalloc(&d_l, (size_t)seq_len * sizeof(float));

    cudaMemsetAsync(Output, 0, mat_size, stream);
    v7_init_ml<<<(seq_len + 255) / 256, 256, 0, stream>>>(d_m, d_l, seq_len);

    float alpha = 1.0f, beta = 0.0f;

    for (int row_start = 0; row_start < seq_len; row_start += WALLER_V7_TILE) {
        int tile_rows = min(WALLER_V7_TILE, seq_len - row_start);
        int max_col = row_start + tile_rows;
        for (int col_start = 0; col_start < max_col; col_start += WALLER_V7_TILE) {
            int tile_cols = min(WALLER_V7_TILE, seq_len - col_start);
            cublasSgemm(
                handle, CUBLAS_OP_T, CUBLAS_OP_N,
                tile_cols, tile_rows, head_dim,
                &alpha,
                K + col_start * head_dim, head_dim,
                Q + row_start * head_dim, head_dim,
                &beta,
                d_scores, WALLER_V7_TILE);
            v7_fused_softmax_v_update<<<tile_rows, 128, 0, stream>>>(
                d_scores, V + col_start * head_dim,
                Output, d_m, d_l,
                row_start, col_start, tile_rows, tile_cols, WALLER_V7_TILE,
                seq_len, head_dim, scale);
        }
    }
    v7_normalize_output<<<seq_len, 128, 0, stream>>>(Output, d_l, seq_len, head_dim);

    cudaFree(d_scores);
    cudaFree(d_m);
    cudaFree(d_l);
}