//! LLaMA-class layer demo (small profile) — RoPE + RMSNorm + Waller.

use attention_transformer::{format_receipt, Llama7bProfile, LlamaAttentionLayer};

fn main() {
    let mut profile = Llama7bProfile::standard();
    profile.hidden_dim = 256;
    profile.num_heads = 8;
    profile.head_dim = 32;
    profile.max_seq_len = 128;
    let layer = LlamaAttentionLayer::new(profile);
    let seq_len = 16;
    let hidden = vec![0.001f32; seq_len * 256];
    let (out, receipt) = layer.forward_prefill(&hidden, seq_len);
    println!(
        "llama_demo seq={seq_len} out_len={} receipt={}",
        out.len(),
        format_receipt(&receipt)
    );
}