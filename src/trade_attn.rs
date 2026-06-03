//! TRADE attention backend selection (v3).
//!
//! AUDIT keeps deterministic Waller f32. TRADE defaults to fast paths (fp16 tiled / Flash).

/// Which attention kernel TRADE uses on GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TradeAttnBackend {
    /// Deterministic register / fused Waller f32 — receipt lane only.
    Waller,
    /// Tiled cuBLAS online-softmax (f32), all seq lengths when TRADE default.
    V7,
    /// fp16 tiled attention (cuBLAS HGEMM) — competitive interim without libtorch.
    Fp16Tiled,
    /// Native Flash-Attn 2 when built with `FLASH_ATTN_ROOT`; else use Python `integrations/trade_geodesic_flash.py`.
    Flash,
}

/// `LUXI_RECEIPT_AUDIT=1` → Waller. Else `LUXI_TRADE_ATTN` (default `fp16`).
pub fn cuda_trade_attn_backend() -> TradeAttnBackend {
    #[cfg(feature = "cuda")]
    if crate::gpu::cuda::cuda_receipt_audit_mode() {
        return TradeAttnBackend::Waller;
    }
    match std::env::var("LUXI_TRADE_ATTN")
        .unwrap_or_else(|_| "fp16".into())
        .to_ascii_lowercase()
        .as_str()
    {
        "waller" | "register" | "audit" => TradeAttnBackend::Waller,
        "v7" | "tiled" => TradeAttnBackend::V7,
        "flash" | "flash2" | "flash_attn" => TradeAttnBackend::Flash,
        "fp16" | "fp16_tiled" | "half" => TradeAttnBackend::Fp16Tiled,
        _ => TradeAttnBackend::Fp16Tiled,
    }
}

pub fn trade_attn_backend_label(b: TradeAttnBackend) -> &'static str {
    match b {
        TradeAttnBackend::Waller => "waller_f32_audit",
        TradeAttnBackend::V7 => "v7_tiled_f32",
        TradeAttnBackend::Fp16Tiled => "fp16_tiled_trade",
        TradeAttnBackend::Flash => "flash_attn2",
    }
}
