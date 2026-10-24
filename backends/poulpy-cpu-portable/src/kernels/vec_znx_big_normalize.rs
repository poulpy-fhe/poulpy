//! BIG normalization over [`ZnxNormalize::znx_normalize_big`], for either big word.

use crate::{
    kernels::znx_normalize::{Limbs, LimbsMut, ZnxNormalize},
    layouts::{HostDataMut, HostDataRef, VecZnxBackendMut, VecZnxBigBackendRef, VecZnxShape, ZnxView, ZnxViewMut},
};

#[allow(clippy::too_many_arguments)]
pub fn vec_znx_big_normalize_portable<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_base2k: usize,
    res_k: usize,
    res_offset: i64,
    res_col: usize,
    a: &VecZnxBigBackendRef<'a, BE>,
    a_base2k: usize,
    a_col: usize,
) where
    BE: ZnxNormalize,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    assert!(res_col < res.cols(), "operand column out of range");
    if res.size() > 0 {
        let _ = res.at_ptr(res_col, res.size() - 1);
    }
    let (n, shape) = (res.n(), res.shape());
    let ptr = res.base_mut_ptr();
    // SAFETY: `ptr` and `shape` come from `res`, bounds-checked above, and `&mut res` is exclusive.
    unsafe {
        vec_znx_big_normalize_range_raw_portable::<BE>(
            ptr, shape, res_base2k, res_k, res_offset, res_col, a, a_base2k, a_col, 0, n,
        )
    }
}

/// [`crate::kernels::vec_znx::vec_znx_normalize_range_raw_portable`] from a big source.
///
/// # Safety
///
/// As [`crate::kernels::vec_znx::vec_znx_normalize_range_raw_portable`].
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub unsafe fn vec_znx_big_normalize_range_raw_portable<'a, BE>(
    res_ptr: *mut i64,
    res_shape: VecZnxShape,
    res_base2k: usize,
    res_k: usize,
    res_offset: i64,
    res_col: usize,
    a: &VecZnxBigBackendRef<'a, BE>,
    a_base2k: usize,
    a_col: usize,
    coeff_start: usize,
    coeff_len: usize,
) where
    BE: ZnxNormalize,
    BE::BufRef<'a>: HostDataRef,
{
    assert_eq!(res_shape.n(), a.n(), "big normalize: degree mismatch");
    assert!(res_col < res_shape.cols() && a_col < a.cols(), "operand column out of range");
    assert!(
        coeff_start + coeff_len <= res_shape.n(),
        "big normalize: coefficient range out of bounds"
    );
    if a.size() > 0 {
        let _ = a.at_ptr(a_col, a.size() - 1);
    }
    // SAFETY: the caller's contract covers `res`; `a` is a checked view, bounds-checked above.
    unsafe {
        BE::znx_normalize_big(
            LimbsMut::new(res_ptr, res_shape, res_col, coeff_start),
            res_base2k,
            res_k,
            res_offset,
            Limbs::new(a.base_ptr(), a.shape(), a_col, coeff_start),
            a_base2k,
            coeff_len,
        )
    }
}
