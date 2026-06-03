//! HyperWall bench — v2 CPU scaling + energy (GPU TRADE: cuda_bench on H100).
//!
//! Run: `cargo run --release --example hyperwall_bench`
//! Parallel: `cargo run --release --features rayon --example hyperwall_bench`

use attention_transformer::standard_attention::standard_attention;
use attention_transformer::waller_operator::{waller_operator, waller_operator_parallel};

use std::time::Instant;

fn waller_run(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    #[cfg(feature = "rayon")]
    {
        return waller_operator_parallel(q, k, v, seq_len, head_dim, scale);
    }
    #[cfg(not(feature = "rayon"))]
    {
        return waller_operator(q, k, v, seq_len, head_dim, scale);
    }
}

fn main() {
    println!("attention-transformer-v2 HyperWall bench (CPU AUDIT)");
    #[cfg(feature = "rayon")]
    println!("rayon: enabled");
    println!("seq_len\thead_dim\tstd_ms\twaller_ms\tmem_ratio\tmax_err");
    for &seq in &[64, 256, 1024, 4096] {
        let head_dim = 64;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let n = seq * head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.02).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.03).sin()).collect();

        let t0 = Instant::now();
        let _ = standard_attention(&q, &k, &v, seq, head_dim, scale);
        let std_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        let waller_out = waller_run(&q, &k, &v, seq, head_dim, scale);
        let waller_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let ref_out = standard_attention(&q, &k, &v, seq, head_dim, scale);
        let mut max_err = 0.0f32;
        for (a, b) in ref_out.iter().zip(waller_out.iter()) {
            max_err = max_err.max((a - b).abs());
        }

        let heads = 1u64;
        let layers = 1u64;
        let s_mem = 2.0 * (seq as f64).powi(2) * 4.0 * heads as f64 * layers as f64;
        let w_mem = 2.0 * (seq as f64) * (head_dim as f64) * 4.0 * heads as f64 * layers as f64;
        let mem_ratio = if w_mem > 0.0 { s_mem / w_mem } else { 0.0 };

        println!("{seq}\t{head_dim}\t{std_ms:.2}\t{waller_ms:.2}\t{mem_ratio:.1}\t{max_err:.2e}");
    }
    println!("\nGPU TRADE: cargo run --release --features cuda --example cuda_bench");
    println!("Commercial kit: bash scripts/commercial_proof.sh");
}