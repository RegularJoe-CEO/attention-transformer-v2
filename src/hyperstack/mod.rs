//! HyperStack — v2 orchestration layer for the unified energy-efficient transformer.
//!
//! Composes AUDIT (CPU f32 golden), TRADE (CUDA when enabled), WNSM, and receipts.
//! Source audit repos under `~/Desktop/eRock/_audit_repos/` remain read-only; all v2
//! changes live only in this crate.

use crate::{sha256_of_f32_slice, WNSM_GAE_Decoder};

/// Operating lane for inference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferenceLane {
    /// CPU f32 — cryptographic golden path.
    Audit,
    /// CUDA fused TRADE path (requires `cuda` feature + nvcc at build time).
    Trade,
}

/// v2 engine facade over WNSM/GAE decoder blocks.
pub struct HyperWallerEngine {
    pub lane: InferenceLane,
    pub decoder: WNSM_GAE_Decoder,
}

impl HyperWallerEngine {
    pub fn new_audit(decoder: WNSM_GAE_Decoder) -> Self {
        Self {
            lane: InferenceLane::Audit,
            decoder,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn new_trade(decoder: WNSM_GAE_Decoder) -> Self {
        Self {
            lane: InferenceLane::Trade,
            decoder,
        }
    }

    pub fn layer_count(&self) -> usize {
        self.decoder.layers.len()
    }

    /// Forward hidden states through the full decoder (CPU AUDIT path; CUDA when built).
    pub fn forward_audit(&mut self, hidden: &[f32], seq_len: usize) -> (Vec<f32>, [u8; 32]) {
        let out = self.decoder.forward(hidden.to_vec(), seq_len);
        let receipt = sha256_of_f32_slice(&out);
        (out, receipt)
    }

}