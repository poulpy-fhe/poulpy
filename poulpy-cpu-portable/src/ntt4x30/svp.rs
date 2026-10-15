//! Scalar-vector product for [`NTT4x30Portable`](crate::NTT4x30Portable) on the packed layout.
//!
//! A prepared polynomial is one packed limb whose residues are multiplied by `2^32`.

use crate::kernels::ntt4x30::NttDFTExecute;
use crate::kernels::ntt4x30::{
    ntt::{NttTable, NttTableInv},
    primes::Primes30,
};
use bytemuck::{cast_slice, cast_slice_mut};
use poulpy_hal::layouts::Ring;
use poulpy_hal::layouts::{
    DataView, DataViewMut, Module, ScalarZnxBackendRef, SvpPPolBackendMut, SvpPPolBackendRef, VecZnxBackendRef,
    VecZnxDftBackendMut, VecZnxDftBackendRef, ZnxView, check_degree,
};

use super::packed::{limb_mont_mul, limb_mont_mul_assign};
use super::{
    NTT4x30Portable,
    vec_znx_dft::{dft_limb, dft_limb_scaled, packed_limb, packed_limb_mut},
};

pub(crate) fn svp_prepare<R: Ring>(
    module: &Module<NTT4x30Portable<R>>,
    res: &mut SvpPPolBackendMut<'_, NTT4x30Portable<R>>,
    res_col: usize,
    a: &ScalarZnxBackendRef<'_, NTT4x30Portable<R>>,
    a_col: usize,
) where
    NTT4x30Portable<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
{
    let n = res.n();
    check_degree::<NTT4x30Portable<R>>(module.n(), n);
    assert!(a.n() == n, "svp_prepare: a.n() != res.n()");
    let data: &mut [u32] = cast_slice_mut(res.data_mut());
    dft_limb_scaled(module, n, &mut data[4 * n * res_col..][..4 * n], a.at(a_col, 0), true);
}

pub(crate) fn svp_ppol_copy<R: Ring>(
    res: &mut SvpPPolBackendMut<'_, NTT4x30Portable<R>>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT4x30Portable<R>>,
    a_col: usize,
) {
    assert_eq!(res.n(), a.n(), "svp_ppol_copy: res.n() {} != a.n() {}", res.n(), a.n());
    assert_eq!(
        res.hint(),
        a.hint(),
        "svp_ppol_copy: res and a must carry the same PrepareHint ({:?} != {:?})",
        res.hint(),
        a.hint()
    );
    let n = res.n();
    let dst: &mut [u32] = cast_slice_mut(res.data_mut());
    let src: &[u32] = cast_slice(a.data());
    dst[4 * n * res_col..][..4 * n].copy_from_slice(&src[4 * n * a_col..][..4 * n]);
}

/// `res = a * dft(b)`: each limb of `b` is transformed into its destination, then multiplied there.
pub(crate) fn svp_apply_dft<R: Ring>(
    module: &Module<NTT4x30Portable<R>>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Portable<R>>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT4x30Portable<R>>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, NTT4x30Portable<R>>,
    b_col: usize,
) {
    poulpy_hal::layouts::assert_dense(b, "svp_apply_dft");
    let n = res.n();
    check_degree::<NTT4x30Portable<R>>(module.n(), n);
    assert_eq!(a.n(), n, "svp_apply_dft: a.n():{} != res.n():{n}", a.n());
    assert_eq!(b.n(), n, "svp_apply_dft: b.n():{} != res.n():{n}", b.n());
    let (res_cols, res_size, b_size) = (res.cols(), res.size(), b.size());
    let factor_data: &[u32] = cast_slice(a.data());
    let factor = &factor_data[4 * n * a_col..][..4 * n];
    let res_data: &mut [u32] = cast_slice_mut(res.data_mut());
    for limb in 0..res_size {
        let dst = packed_limb_mut(res_data, n, res_cols, res_col, limb);
        dft_limb(module, n, dst, (limb < b_size).then(|| b.at(b_col, limb)));
        if limb < b_size {
            limb_mont_mul_assign(n, dst, factor);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn svp_apply_dft_to_dft<R: Ring>(
    _module: &Module<NTT4x30Portable<R>>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Portable<R>>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT4x30Portable<R>>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'_, NTT4x30Portable<R>>,
    b_col: usize,
) {
    let n = res.n();
    assert_eq!(a.n(), n, "svp_apply_dft_to_dft: a.n():{} != res.n():{n}", a.n());
    assert_eq!(b.n(), n, "svp_apply_dft_to_dft: b.n():{} != res.n():{n}", b.n());
    let (res_cols, b_cols) = (res.cols(), b.cols());
    let min_size = res.size().min(b.size());
    let factor_data: &[u32] = cast_slice(a.data());
    let factor = &factor_data[4 * n * a_col..][..4 * n];
    let b_data: &[u32] = cast_slice(b.data());
    let res_size = res.size();
    let res_data: &mut [u32] = cast_slice_mut(res.data_mut());
    for limb in 0..res_size {
        let dst = packed_limb_mut(res_data, n, res_cols, res_col, limb);
        if limb < min_size {
            limb_mont_mul(n, dst, packed_limb(b_data, n, b_cols, b_col, limb), factor);
        } else {
            dst.fill(0);
        }
    }
}

pub(crate) fn svp_apply_dft_to_dft_assign<R: Ring>(
    _module: &Module<NTT4x30Portable<R>>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Portable<R>>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT4x30Portable<R>>,
    a_col: usize,
) {
    let n = res.n();
    assert_eq!(a.n(), n, "svp_apply_dft_to_dft_assign: a.n():{} != res.n():{n}", a.n());
    let cols = res.cols();
    let factor_data: &[u32] = cast_slice(a.data());
    let factor = &factor_data[4 * n * a_col..][..4 * n];
    let size = res.size();
    let data: &mut [u32] = cast_slice_mut(res.data_mut());
    for limb in 0..size {
        let dst = packed_limb_mut(data, n, cols, res_col, limb);
        limb_mont_mul_assign(n, dst, factor);
    }
}
