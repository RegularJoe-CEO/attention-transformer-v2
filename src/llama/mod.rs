//! LLaMA-class decoder block (AUDIT f32) — RoPE + RMSNorm + Waller attention.

use crate::config::Config;
use crate::rope_rmsnorm::{rmsnorm_f32, RopeFrequenciesF32};
use crate::waller_operator::waller_operator;
use crate::{sha256_of_f32_slice, WNSM_GAE_Layer};

/// LLaMA 7B-ish shape preset (inference-only stub weights).
#[derive(Clone, Debug)]
pub struct Llama7bProfile {
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub rope_base: f32,
}

impl Llama7bProfile {
    pub fn standard() -> Self {
        Self {
            hidden_dim: 4096,
            num_heads: 32,
            head_dim: 128,
            num_layers: 32,
            max_seq_len: 8192,
            rope_base: 10_000.0,
        }
    }

    pub fn to_config(&self) -> Config {
        Config::new(
            self.hidden_dim,
            self.num_heads,
            self.hidden_dim * 4,
            self.max_seq_len,
        )
    }
}

/// One LLaMA-style layer: pre-norm RMS → QKV → RoPE on Q/K → Waller → residual.
pub struct LlamaAttentionLayer {
    pub profile: Llama7bProfile,
    pub rope: RopeFrequenciesF32,
    pub gamma: Vec<f32>,
    /// Underlying WNSM/GAE layer for MLP + projections (reuses v2 stack).
    pub core: WNSM_GAE_Layer,
}

impl LlamaAttentionLayer {
    pub fn new(profile: Llama7bProfile) -> Self {
        let cfg = profile.to_config();
        let rope = RopeFrequenciesF32::new(profile.head_dim, profile.max_seq_len, profile.rope_base);
        let gamma = vec![1.0f32; profile.hidden_dim];
        Self {
            profile: profile.clone(),
            rope,
            gamma,
            core: WNSM_GAE_Layer::new(cfg),
        }
    }

    /// Forward hidden states `[seq_len * hidden_dim]` with Waller multi-head attention.
    pub fn forward_prefill(&self, hidden: &[f32], seq_len: usize) -> (Vec<f32>, [u8; 32]) {
        let hd = self.profile.head_dim;
        let nh = self.profile.num_heads;
        let h = self.profile.hidden_dim;
        let scale = 1.0 / (hd as f32).sqrt();

        let mut normed = vec![0.0f32; seq_len * h];
        for t in 0..seq_len {
            let row = &hidden[t * h..(t + 1) * h];
            let n = rmsnorm_f32(row, &self.gamma, 1e-5);
            normed[t * h..(t + 1) * h].copy_from_slice(&n);
        }
        let mut q = vec![0.0f32; seq_len * h];
        let mut k = vec![0.0f32; seq_len * h];
        let mut v = vec![0.0f32; seq_len * h];
        // Identity QKV stub until sharded weights wired (shape-correct path).
        q.copy_from_slice(&normed);
        k.copy_from_slice(&normed);
        v.copy_from_slice(&normed);

        for head in 0..nh {
            let off = head * hd;
            let mut q_h = vec![0.0f32; seq_len * hd];
            let mut k_h = vec![0.0f32; seq_len * hd];
            let mut v_h = vec![0.0f32; seq_len * hd];
            for t in 0..seq_len {
                let src = t * h + off;
                let dst = t * hd;
                q_h[dst..dst + hd].copy_from_slice(&q[src..src + hd]);
                k_h[dst..dst + hd].copy_from_slice(&k[src..src + hd]);
                v_h[dst..dst + hd].copy_from_slice(&v[src..src + hd]);
            }
            self.rope.apply_flat(&mut q_h, seq_len, 0);
            self.rope.apply_flat(&mut k_h, seq_len, 0);
            let attn_out = waller_operator(&q_h, &k_h, &v_h, seq_len, hd, scale);
            for t in 0..seq_len {
                let dst = t * h + off;
                let src = t * hd;
                q[dst..dst + hd].copy_from_slice(&attn_out[src..src + hd]);
            }
        }

        let mut out: Vec<f32> = hidden
            .iter()
            .zip(q.iter())
            .map(|(&h0, &a)| h0 + a)
            .collect();
        out = self.core.forward(&out, seq_len);
        let receipt = sha256_of_f32_slice(&out);
        (out, receipt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama_layer_small_seq_runs() {
        let mut profile = Llama7bProfile::standard();
        profile.hidden_dim = 128;
        profile.num_heads = 4;
        profile.head_dim = 32;
        profile.max_seq_len = 64;
        let h = profile.hidden_dim;
        let layer = LlamaAttentionLayer::new(profile);
        let seq_len = 8;
        let hidden = vec![0.01f32; seq_len * h];
        let (out, receipt) = layer.forward_prefill(&hidden, seq_len);
        assert_eq!(out.len(), hidden.len());
        assert_ne!(receipt, [0u8; 32]);
    }
}