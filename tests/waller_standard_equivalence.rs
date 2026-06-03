//! Integration gate: Waller operator ≡ standard causal softmax attention.

use attention_transformer::standard_attention::standard_attention;
use attention_transformer::waller_operator::waller_operator;

fn assert_close(a: &[f32], b: &[f32], epsilon: f32) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < epsilon, "standard={x} waller={y}");
    }
}

#[test]
fn equivalence_seq16_head8() {
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
    assert_close(&standard, &waller, 1e-4);
}

#[test]
fn equivalence_seq512_head64() {
    let seq_len = 512;
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 97) as f32 * 0.017).sin())
        .collect();
    let k: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 53) as f32 * 0.023).cos())
        .collect();
    let v: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 71) as f32 * 0.011).sin())
        .collect();
    let standard = standard_attention(&q, &k, &v, seq_len, head_dim, scale);
    let waller = waller_operator(&q, &k, &v, seq_len, head_dim, scale);
    assert_close(&standard, &waller, 1e-4);
}