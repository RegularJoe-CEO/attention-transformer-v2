//! Standard O(N²) causal attention — reference baseline for correctness gates.
//!
//! Textbook implementation that materializes the full attention score matrix.
//! Used only to prove `waller_operator` matches dense softmax attention.

/// Standard causal self-attention (O(N²) memory).
pub fn standard_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut scores = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j <= i {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                scores[i * seq_len + j] = dot * scale;
            } else {
                scores[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }

    for i in 0..seq_len {
        let row = &mut scores[i * seq_len..(i + 1) * seq_len];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for val in row.iter_mut() {
            *val = (*val - max).exp();
            sum += *val;
        }
        for val in row.iter_mut() {
            *val /= sum;
        }
    }

    let mut output = vec![0.0f32; seq_len * head_dim];
    for i in 0..seq_len {
        for j in 0..=i {
            let w = scores[i * seq_len + j];
            for d in 0..head_dim {
                output[i * head_dim + d] += w * v[j * head_dim + d];
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::waller_operator::waller_operator;

    #[test]
    fn waller_matches_standard_small() {
        let seq_len = 16;
        let head_dim = 8;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let k: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| (i as f32 * 0.2).cos())
            .collect();
        let v: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| (i as f32 * 0.05).sin())
            .collect();
        let standard = standard_attention(&q, &k, &v, seq_len, head_dim, scale);
        let waller = waller_operator(&q, &k, &v, seq_len, head_dim, scale);
        for (a, b) in standard.iter().zip(waller.iter()) {
            assert!((a - b).abs() < 1e-4, "diff {} vs {}", a, b);
        }
    }
}