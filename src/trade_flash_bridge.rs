//! Flash-Attn 2 via installed `flash_attn` + PyTorch (device pointer bridge).
//! Build: `cargo build --release --features cuda,flash-bridge`
//! Run: `LUXI_TRADE_ATTN=flash LUXI_FLASH_BRIDGE=1`

#[cfg(all(feature = "cuda", feature = "flash-bridge"))]
use pyo3::prelude::*;
#[cfg(all(feature = "cuda", feature = "flash-bridge"))]
use pyo3::types::PyModule;
#[cfg(all(feature = "cuda", feature = "flash-bridge"))]
use std::sync::OnceLock;

#[cfg(all(feature = "cuda", feature = "flash-bridge"))]
static FLASH_FN: OnceLock<Py<PyAny>> = OnceLock::new();

#[cfg(all(feature = "cuda", feature = "flash-bridge"))]
fn flash_fn(py: Python<'_>) -> PyResult<&Py<PyAny>> {
    if let Some(f) = FLASH_FN.get() {
        return Ok(f);
    }
    let code = r#"
import torch
from flash_attn.flash_attn_interface import flash_attn_func

def luxi_flash_attn_f32(q_ptr, k_ptr, v_ptr, out_ptr, seq, heads, head_dim):
    hidden = heads * head_dim
    nbytes = seq * hidden * 4

    def tensor_from_ptr(ptr):
        storage = torch.UntypedStorage.from_ptr(ptr, nbytes, device='cuda')
        return torch.as_tensor(storage, dtype=torch.float32, device='cuda')

    q = tensor_from_ptr(q_ptr).view(1, seq, heads, head_dim).to(torch.float16)
    k = tensor_from_ptr(k_ptr).view(1, seq, heads, head_dim).to(torch.float16)
    v = tensor_from_ptr(v_ptr).view(1, seq, heads, head_dim).to(torch.float16)
    o = flash_attn_func(q, k, v, causal=True)
    dst = tensor_from_ptr(out_ptr).view(1, seq, hidden)
    dst.copy_(o.reshape(1, seq, hidden).to(torch.float32))
"#;
    let module = PyModule::from_code_bound(py, code, "luxi_flash_bridge.py", "luxi")?;
    let f = module.getattr("luxi_flash_attn_f32")?.unbind();
    let _ = FLASH_FN.set(f);
    Ok(FLASH_FN.get().expect("flash fn set"))
}

#[cfg(all(feature = "cuda", feature = "flash-bridge"))]
pub fn flash_bridge_enabled() -> bool {
    !std::env::var("LUXI_FLASH_BRIDGE")
        .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
        .unwrap_or(false)
}

#[cfg(all(feature = "cuda", not(feature = "flash-bridge")))]
pub fn flash_bridge_enabled() -> bool {
    false
}

#[cfg(all(feature = "cuda", feature = "flash-bridge"))]
pub unsafe fn flash_attn_device_f32(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    out: *mut f32,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<(), String> {
    if q.is_null() || k.is_null() || v.is_null() || out.is_null() {
        return Err("flash_attn_device_f32: null device pointer".into());
    }
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| -> PyResult<()> {
        flash_fn(py)?.call1(
            py,
            (
                q as usize,
                k as usize,
                v as usize,
                out as usize,
                seq_len,
                num_heads,
                head_dim,
            ),
        )?;
        Ok(())
    })
    .map_err(|e| format!("flash_attn bridge: {e}"))
}

#[cfg(all(feature = "cuda", not(feature = "flash-bridge")))]
pub unsafe fn flash_attn_device_f32(
    _q: *const f32,
    _k: *const f32,
    _v: *const f32,
    _out: *mut f32,
    _seq_len: usize,
    _num_heads: usize,
    _head_dim: usize,
) -> Result<(), String> {
    Err("rebuild with --features cuda,flash-bridge for Flash TRADE".into())
}