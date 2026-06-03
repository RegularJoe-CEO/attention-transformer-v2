//! Flash-Attn 2 via installed `flash_attn` + PyTorch (D2D staging — no `from_ptr`).
//! Build: `cargo build --release --features cuda,flash-bridge`

#[cfg(all(feature = "cuda", feature = "flash-bridge"))]
use pyo3::prelude::*;
#[cfg(all(feature = "cuda", feature = "flash-bridge"))]
use pyo3::types::{PyDict, PyModule};
#[cfg(all(feature = "cuda", feature = "flash-bridge"))]
use std::ffi::c_void;
#[cfg(all(feature = "cuda", feature = "flash-bridge"))]
use std::sync::Mutex;

#[cfg(all(feature = "cuda", feature = "flash-bridge"))]
const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

#[cfg(all(feature = "cuda", feature = "flash-bridge"))]
extern "C" {
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

#[cfg(all(feature = "cuda", feature = "flash-bridge"))]
struct FlashStage {
    q: Py<PyAny>,
    k: Py<PyAny>,
    v: Py<PyAny>,
    out: Py<PyAny>,
    q_ptr: usize,
    k_ptr: usize,
    v_ptr: usize,
    out_ptr: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
}

#[cfg(all(feature = "cuda", feature = "flash-bridge"))]
static STAGE: Mutex<Option<FlashStage>> = Mutex::new(None);

#[cfg(all(feature = "cuda", feature = "flash-bridge"))]
static FLASH_FWD: std::sync::OnceLock<Py<PyAny>> = std::sync::OnceLock::new();

#[cfg(all(feature = "cuda", feature = "flash-bridge"))]
fn flash_fwd_fn(py: Python<'_>) -> PyResult<&Py<PyAny>> {
    if let Some(f) = FLASH_FWD.get() {
        return Ok(f);
    }
    let code = r#"
import torch
from flash_attn.flash_attn_interface import flash_attn_func

def luxi_flash_fwd(q, k, v, out, seq, heads, head_dim):
    qh = q.view(1, seq, heads, head_dim).to(torch.float16)
    kh = k.view(1, seq, heads, head_dim).to(torch.float16)
    vh = v.view(1, seq, heads, head_dim).to(torch.float16)
    o = flash_attn_func(qh, kh, vh, causal=True)
    out.copy_(o.reshape(1, seq, heads * head_dim).to(torch.float32).reshape(-1))
"#;
    let module = PyModule::from_code_bound(py, code, "luxi_flash_bridge.py", "luxi")?;
    let f = module.getattr("luxi_flash_fwd")?.unbind();
    let _ = FLASH_FWD.set(f);
    Ok(FLASH_FWD.get().expect("flash_fwd set"))
}

#[cfg(all(feature = "cuda", feature = "flash-bridge"))]
fn tensor_data_ptr(tensor: &Bound<'_, PyAny>) -> PyResult<usize> {
    let ptr: usize = tensor.getattr("data_ptr")?.call0()?.extract()?;
    Ok(ptr)
}

#[cfg(all(feature = "cuda", feature = "flash-bridge"))]
fn ensure_stage(
    py: Python<'_>,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> PyResult<()> {
    let hidden = num_heads * head_dim;
    let n = seq_len * hidden;
    let mut guard = STAGE.lock().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{e}")))?;
    let need_new = guard
        .as_ref()
        .map(|s| s.seq_len != seq_len || s.num_heads != num_heads || s.head_dim != head_dim)
        .unwrap_or(true);
    if !need_new {
        return Ok(());
    }
    let torch = PyModule::import_bound(py, "torch")?;
    let cuda = torch.getattr("device")?.call1(("cuda",))?;
    let f32 = torch.getattr("float32")?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("device", cuda)?;
    kwargs.set_item("dtype", f32)?;
    let empty = torch.getattr("empty")?;
    let q = empty.call((n,), Some(&kwargs))?.unbind();
    let k = empty.call((n,), Some(&kwargs))?.unbind();
    let v = empty.call((n,), Some(&kwargs))?.unbind();
    let out = empty.call((n,), Some(&kwargs))?.unbind();
    let q_b = q.bind(py);
    let k_b = k.bind(py);
    let v_b = v.bind(py);
    let out_b = out.bind(py);
    *guard = Some(FlashStage {
        q_ptr: tensor_data_ptr(q_b)?,
        k_ptr: tensor_data_ptr(k_b)?,
        v_ptr: tensor_data_ptr(v_b)?,
        out_ptr: tensor_data_ptr(out_b)?,
        q,
        k,
        v,
        out,
        seq_len,
        num_heads,
        head_dim,
    });
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "flash-bridge"))]
unsafe fn memcpy_d2d(dst: usize, src: usize, bytes: usize) -> Result<(), String> {
    if dst == 0 || src == 0 {
        return Err("flash bridge: null staging pointer".into());
    }
    let rc = cudaMemcpy(
        dst as *mut c_void,
        src as *const c_void,
        bytes,
        CUDA_MEMCPY_DEVICE_TO_DEVICE,
    );
    if rc != 0 {
        return Err(format!("flash bridge cudaMemcpy D2D failed: code {rc}"));
    }
    Ok(())
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
    let nbytes = seq_len * num_heads * head_dim * std::mem::size_of::<f32>();
    pyo3::prepare_freethreaded_python();
    let _ = unsafe { cudaDeviceSynchronize() };
    Python::with_gil(|py| -> PyResult<()> {
        ensure_stage(py, seq_len, num_heads, head_dim)?;
        let (q_ptr, k_ptr, v_ptr, out_ptr) = {
            let guard = STAGE
                .lock()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{e}")))?;
            let s = guard.as_ref().expect("stage");
            (s.q_ptr, s.k_ptr, s.v_ptr, s.out_ptr)
        };
        memcpy_d2d(q_ptr, q as usize, nbytes).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)
        })?;
        memcpy_d2d(k_ptr, k as usize, nbytes).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)
        })?;
        memcpy_d2d(v_ptr, v as usize, nbytes).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)
        })?;
        {
            let guard = STAGE
                .lock()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{e}")))?;
            let s = guard.as_ref().expect("stage");
            flash_fwd_fn(py)?.bind(py).call1((
                s.q.bind(py),
                s.k.bind(py),
                s.v.bind(py),
                s.out.bind(py),
                seq_len,
                num_heads,
                head_dim,
            ))?;
        }
        memcpy_d2d(out as usize, out_ptr, nbytes).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)
        })?;
        Ok(())
    })
    .map_err(|e| format!("flash_attn bridge: {e}"))?;
    let _ = unsafe { cudaDeviceSynchronize() };
    Ok(())
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