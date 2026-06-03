// attention-transformer-v2 — TRADE tiled Waller attention (online-softmax + cuBLAS)
// Multi-head causal attention; persistent device workspace (no per-call cudaMalloc).

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>

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

struct V7Workspace {
    float* d_scores = nullptr;
    float* d_m = nullptr;
    float* d_l = nullptr;
    size_t score_bytes = 0;
    int seq_cap = 0;
};

static V7Workspace g_v7_ws;

static int v7_env_on(const char* name) {
    const char* v = getenv(name);
    return v != nullptr && (v[0] == '1' || (v[0] == 't' && v[1] == 'r'));
}

static int v7_env_off(const char* name) {
    const char* v = getenv(name);
    return v != nullptr && (v[0] == '0' || (v[0] == 'f' && v[1] == 'a'));
}

// TRADE auto: tiled path when seq exceeds register-kernel sweet spot (override with LUXI_CUDA_V7).
extern "C" int waller_v7_should_use(int seq_len, int head_dim, int num_heads) {
    (void)head_dim;
    (void)num_heads;
    if (v7_env_off("LUXI_CUDA_V7")) return 0;
    if (v7_env_on("LUXI_CUDA_V7")) return 1;
    const char* auto_env = getenv("LUXI_CUDA_V7_AUTO");
    if (auto_env != nullptr && auto_env[0] == '0') return 0;
    return seq_len >= 2048 ? 1 : 0;
}

static int v7_ensure_workspace(int seq_len) {
    size_t score_bytes = (size_t)WALLER_V7_TILE * WALLER_V7_TILE * sizeof(float);
    size_t ml_bytes = (size_t)seq_len * sizeof(float);
    if (seq_len <= g_v7_ws.seq_cap && g_v7_ws.d_scores != nullptr) {
        return 0;
    }
    if (g_v7_ws.d_scores) cudaFree(g_v7_ws.d_scores);
    if (g_v7_ws.d_m) cudaFree(g_v7_ws.d_m);
    if (g_v7_ws.d_l) cudaFree(g_v7_ws.d_l);
    g_v7_ws.d_scores = nullptr;
    g_v7_ws.d_m = nullptr;
    g_v7_ws.d_l = nullptr;

    int new_cap = seq_len;
    int grow = g_v7_ws.seq_cap > 0 ? g_v7_ws.seq_cap : 512;
    while (grow < new_cap) grow *= 2;
    g_v7_ws.seq_cap = grow;

    ml_bytes = (size_t)g_v7_ws.seq_cap * sizeof(float);
    if (cudaMalloc(&g_v7_ws.d_scores, score_bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&g_v7_ws.d_m, ml_bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&g_v7_ws.d_l, ml_bytes) != cudaSuccess) return -1;
    g_v7_ws.score_bytes = score_bytes;
    return 0;
}

static void v7_one_head(
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
    if (v7_ensure_workspace(seq_len) != 0) return;

    cublasHandle_t handle = v7_cublas_handle();
    cublasSetStream(handle, stream);

    size_t mat_size = (size_t)seq_len * head_dim * sizeof(float);
    cudaMemsetAsync(Output, 0, mat_size, stream);
    v7_init_ml<<<(seq_len + 255) / 256, 256, 0, stream>>>(g_v7_ws.d_m, g_v7_ws.d_l, seq_len);

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
                g_v7_ws.d_scores, WALLER_V7_TILE);
            v7_fused_softmax_v_update<<<tile_rows, 128, 0, stream>>>(
                g_v7_ws.d_scores, V + col_start * head_dim,
                Output, g_v7_ws.d_m, g_v7_ws.d_l,
                row_start, col_start, tile_rows, tile_cols, WALLER_V7_TILE,
                seq_len, head_dim, scale);
        }
    }
    v7_normalize_output<<<seq_len, 128, 0, stream>>>(Output, g_v7_ws.d_l, seq_len, head_dim);
}

extern "C" void launch_waller_v7_trade(
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
    for (int h = 0; h < num_heads; h++) {
        const int off = h * head_dim;
        v7_one_head(
            Q + off, K + off, V + off, Output + off,
            seq_len, head_dim, scale, stream);
    }
}