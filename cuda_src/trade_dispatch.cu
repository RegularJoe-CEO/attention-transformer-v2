// trade_dispatch.cu — route TRADE attention (fp16 / v7 / waller) from env.
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>

extern "C" void launch_waller_operator(
    const float* Q,
    const float* K,
    const float* V,
    float* Output,
    int seq_len,
    int head_dim,
    int num_heads,
    float scale,
    cudaStream_t stream);

extern "C" void launch_waller_v7_trade(
    const float* Q,
    const float* K,
    const float* V,
    float* Output,
    int seq_len,
    int head_dim,
    int num_heads,
    float scale,
    cudaStream_t stream);

extern "C" void launch_trade_fp16_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* Output,
    int seq_len,
    int head_dim,
    int num_heads,
    float scale,
    cudaStream_t stream);

extern "C" int waller_v7_should_use(int seq_len, int head_dim, int num_heads);

static int env_on(const char* name) {
    const char* v = getenv(name);
    return v && (v[0] == '1' || (v[0] == 't' && v[1] == 'r'));
}

static int receipt_audit() {
    return env_on("LUXI_RECEIPT_AUDIT");
}

static const char* trade_attn_env() {
    const char* v = getenv("LUXI_TRADE_ATTN");
    return v ? v : "fp16";
}

extern "C" void launch_trade_attention(
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
    if (receipt_audit()) {
        launch_waller_operator(Q, K, V, Output, seq_len, head_dim, num_heads, scale, stream);
        return;
    }
    const char* mode = trade_attn_env();
    if (strcmp(mode, "waller") == 0 || strcmp(mode, "register") == 0 || strcmp(mode, "audit") == 0) {
        launch_waller_operator(Q, K, V, Output, seq_len, head_dim, num_heads, scale, stream);
        return;
    }
    if (strcmp(mode, "v7") == 0 || strcmp(mode, "tiled") == 0) {
        launch_waller_v7_trade(Q, K, V, Output, seq_len, head_dim, num_heads, scale, stream);
        return;
    }
    if (strcmp(mode, "flash") == 0 || strcmp(mode, "flash2") == 0 || strcmp(mode, "flash_attn") == 0) {
        // Rust flash-bridge (pyo3) handles Flash when LUXI_FLASH_BRIDGE=1; else register waller here.
        launch_waller_operator(Q, K, V, Output, seq_len, head_dim, num_heads, scale, stream);
        return;
    }
    if (strcmp(mode, "fp16") == 0 || strcmp(mode, "fp16_tiled") == 0 || strcmp(mode, "half") == 0) {
        // fp16 tiled is for long-seq only (short seq = thousands of tiny HGEMMs → ~60ms+).
        if (seq_len >= 2048) {
            launch_trade_fp16_attention(Q, K, V, Output, seq_len, head_dim, num_heads, scale, stream);
        } else {
            launch_waller_operator(Q, K, V, Output, seq_len, head_dim, num_heads, scale, stream);
        }
        return;
    }
    // Default TRADE v3 (unset env): fast register waller @ seq<2048, v7 @ seq>=2048 via waller_operator.
    launch_waller_operator(Q, K, V, Output, seq_len, head_dim, num_heads, scale, stream);
}
