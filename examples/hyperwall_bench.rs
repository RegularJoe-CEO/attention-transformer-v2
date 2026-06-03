//! HyperWall bench — v2 CPU scaling + energy sweep (GPU TRADE on H100 via cuda_bench).
//!
//! Run: `cargo run --release --example hyperwall_bench`

use attention_transformer::standard_attention::standard_attention;
use attention_transformer::waller_operator::waller_operator;
use std::time::Instant;

fn bench_pair(seq_len: usize, head_dim: usize) -> (f64, f64, f32) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n = seq_len * head_dim;
    let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
    let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.02).cos()).collect();
    let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.03).sin()).collect();

    let t0 = Instant::now();
    let _ = standard_attention(&q, &k, &v, seq_len, head_dim, scale);
    let std_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let waller_out = waller_operator(&q, &k, &v, seq_len, head_dim, scale);
    let waller_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let ref_out = standard_attention(&q, &k, &v, seq_len, head_dim, scale);
    let mut max_err = 0.0f32;
    for (a, b) in ref_out.iter().zip(waller_out.iter()) {
        max_err = max_err.max((a - b).abs());
    }

    (std_ms, waller_ms, max_err)
}

fn main() {
    println!("attention-transformer-v2 HyperWall bench (CPU AUDIT)");
    println!("seq_len\thead_dim\tstd_ms\twaller_ms\tmax_err");
    for &seq in &[64, 256, 1024, 4096] {
        let (std_ms, waller_ms, err) = bench_pair(seq, 64);
        println!("{seq}\t64\t{std_ms:.2}\t{waller_ms:.2}\t{err:.2e}");
    }
    println!("\nFor GPU TRADE: cargo run --release --features cuda --example cuda_bench");
}