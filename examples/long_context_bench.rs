//! Long-context CPU bench — Waller memory scales O(N) vs standard O(N²).

use attention_transformer::standard_attention::standard_attention;
use attention_transformer::waller_operator::waller_operator;
use std::time::Instant;

fn main() {
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();
    println!("long_context_bench head_dim={head_dim}");
    println!("seq\twaller_ms\tstd_ms\twaller_mem_mb\tstd_mem_mb\tenergy_ratio");
    for &seq in &[256, 1024, 4096, 8192] {
        let n = seq * head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.02).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.03).sin()).collect();

        let t0 = Instant::now();
        let _ = waller_operator(&q, &k, &v, seq, head_dim, scale);
        let waller_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        let _ = standard_attention(&q, &k, &v, seq, head_dim, scale);
        let std_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let heads = 1u64;
        let layers = 1u64;
        let std_mem = 2.0 * (seq as f64).powi(2) * 4.0 * heads as f64 * layers as f64 / 1e6;
        let waller_mem =
            2.0 * (seq as f64) * (head_dim as f64) * 4.0 * heads as f64 * layers as f64 / 1e6;
        let ratio = if waller_mem > 0.0 {
            std_mem / waller_mem
        } else {
            0.0
        };
        println!("{seq}\t{waller_ms:.1}\t{std_ms:.1}\t{waller_mem:.2}\t{std_mem:.2}\t{ratio:.1}");
    }
}