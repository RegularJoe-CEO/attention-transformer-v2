//! LLaMA-class RoPE + RMSNorm (f32 inference path, ported from Luxi audit snapshot).

/// RMSNorm: x / sqrt(mean(x²) + eps) * gamma
pub fn rmsnorm_f32(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), gamma.len());
    let n = x.len() as f32;
    let sum_sq: f32 = x.iter().map(|xi| xi * xi).sum();
    let inv_rms = 1.0 / (sum_sq / n + eps).sqrt();
    x.iter()
        .zip(gamma.iter())
        .map(|(&xi, &gi)| xi * inv_rms * gi)
        .collect()
}

/// Precomputed RoPE tables for f32 inference.
pub struct RopeFrequenciesF32 {
    pub cos_cached: Vec<Vec<f32>>,
    pub sin_cached: Vec<Vec<f32>>,
    pub head_dim: usize,
}

impl RopeFrequenciesF32 {
    pub fn new(head_dim: usize, max_seq_len: usize, base: f32) -> Self {
        assert!(head_dim % 2 == 0, "head_dim must be even");
        let half_dim = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| base.powf(-2.0 * (i as f32) / (head_dim as f32)))
            .collect();
        let mut cos_cached = Vec::with_capacity(max_seq_len);
        let mut sin_cached = Vec::with_capacity(max_seq_len);
        for pos in 0..max_seq_len {
            let p = pos as f32;
            cos_cached.push(inv_freq.iter().map(|&f| (p * f).cos()).collect());
            sin_cached.push(inv_freq.iter().map(|&f| (p * f).sin()).collect());
        }
        Self {
            cos_cached,
            sin_cached,
            head_dim,
        }
    }

    /// Apply RoPE to flattened head rows `[seq_len][head_dim]`.
    pub fn apply_flat(&self, x: &mut [f32], seq_len: usize, start_pos: usize) {
        let hd = self.head_dim;
        let half = hd / 2;
        for i in 0..seq_len {
            let pos = start_pos + i;
            let cos = &self.cos_cached[pos];
            let sin = &self.sin_cached[pos];
            let base = i * hd;
            for j in 0..half {
                let x0 = x[base + 2 * j];
                let x1 = x[base + 2 * j + 1];
                let c = cos[j];
                let s = sin[j];
                x[base + 2 * j] = x0 * c - x1 * s;
                x[base + 2 * j + 1] = x0 * s + x1 * c;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_unit_scale() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let g = vec![1.0; 4];
        let y = rmsnorm_f32(&x, &g, 1e-6);
        let norm: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 2.0).abs() < 0.01);
    }
}