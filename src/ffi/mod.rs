//! C ABI for PyTorch / external integrations (receipt + Waller attention).

use crate::waller_operator::waller_operator;
use crate::wnsm_transformer::sha256_of_f32_slice;

/// ABI version — bump when breaking FFI.
pub const LUXIEDGE_FFI_VERSION: u32 = 2;

/// NUL-terminated version string.
#[no_mangle]
pub extern "C" fn luxiedge_version() -> *const u8 {
    c"luxiedge-2.0.0".as_ptr().cast()
}

/// Hash `len` f32 values at `ptr` (little-endian to_bits per element). Returns 0 on success.
///
/// # Safety
///
/// `ptr` must point to `len` valid `f32` values; `out32` must point to 32 writable bytes.
#[no_mangle]
pub unsafe extern "C" fn luxiedge_sha256_f32(
    ptr: *const f32,
    len: usize,
    out32: *mut u8,
) -> i32 {
    if ptr.is_null() || out32.is_null() {
        return -1;
    }
    let slice = std::slice::from_raw_parts(ptr, len);
    let hash = sha256_of_f32_slice(slice);
    std::ptr::copy_nonoverlapping(hash.as_ptr(), out32, 32);
    0
}

/// Max abs diff between two f32 buffers; returns -1 on null.
///
/// # Safety
///
/// `a` and `b` must each point to `len` valid `f32` values.
#[no_mangle]
pub unsafe extern "C" fn luxiedge_max_abs_diff_f32(
    a: *const f32,
    b: *const f32,
    len: usize,
) -> f32 {
    if a.is_null() || b.is_null() {
        return -1.0;
    }
    let sa = std::slice::from_raw_parts(a, len);
    let sb = std::slice::from_raw_parts(b, len);
    sa.iter()
        .zip(sb.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Waller causal attention (CPU AUDIT). Writes `seq_len * head_dim` floats to `out`.
///
/// # Safety
///
/// All pointers must be valid for the given lengths.
#[no_mangle]
pub unsafe extern "C" fn luxiedge_waller_attention_f32(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    out: *mut f32,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) -> i32 {
    if q.is_null() || k.is_null() || v.is_null() || out.is_null() || seq_len == 0 || head_dim == 0 {
        return -1;
    }
    let qs = std::slice::from_raw_parts(q, seq_len * head_dim);
    let ks = std::slice::from_raw_parts(k, seq_len * head_dim);
    let vs = std::slice::from_raw_parts(v, seq_len * head_dim);
    let result = waller_operator(qs, ks, vs, seq_len, head_dim, scale);
    std::ptr::copy_nonoverlapping(result.as_ptr(), out, result.len());
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffi_hash_matches_rust() {
        let data = [1.0f32, 2.0, 3.0];
        let rust = sha256_of_f32_slice(&data);
        let mut out = [0u8; 32];
        unsafe {
            assert_eq!(
                luxiedge_sha256_f32(data.as_ptr(), data.len(), out.as_mut_ptr()),
                0
            );
        }
        assert_eq!(rust, out);
    }
}