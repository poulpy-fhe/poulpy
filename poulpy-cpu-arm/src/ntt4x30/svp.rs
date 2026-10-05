//! Scalar-vector product for [`NTT4x30Neon`](crate::NTT4x30Neon) on the packed layout.
//!
//! A prepared polynomial is one packed limb whose residues are multiplied by `2^32`.

use bytemuck::{cast_slice, cast_slice_mut};
use poulpy_cpu_portable::kernels::ntt4x30::NttDFTExecute;
use poulpy_cpu_portable::kernels::ntt4x30::{
    ntt::{NttTable, NttTableInv},
    primes::Primes30,
};
use poulpy_hal::layouts::Ring;
use poulpy_hal::{
    api::VecZnxDftAlloc,
    layouts::{
        DataView, DataViewMut, Module, ScalarZnxBackendRef, SvpPPolBackendMut, SvpPPolBackendRef, VecZnxBackendRef,
        VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftReborrowBackendRef, VecZnxDftToBackendMut, ZnxView, check_degree,
    },
};

use super::{
    NTT4x30Neon,
    vec_znx_dft::{dft_limb_scaled, dft_tmp_len, packed_limb, packed_limb_mut},
};
use crate::neon::ntt4x30_packed::{OP_MONT_MUL, limb_op};

fn mul_packed_limb(n: usize, dst: &mut [u32], src: &[u32], factor: &[u32]) {
    assert!(dst.len() >= 4 * n && src.len() >= 4 * n && factor.len() >= 4 * n);
    unsafe { limb_op::<OP_MONT_MUL>(n, dst.as_mut_ptr(), src.as_ptr(), factor.as_ptr()) }
}

fn mul_packed_limb_assign(n: usize, dst: &mut [u32], factor: &[u32]) {
    assert!(dst.len() >= 4 * n && factor.len() >= 4 * n);
    let dst = dst.as_mut_ptr();
    unsafe { limb_op::<OP_MONT_MUL>(n, dst, dst, factor.as_ptr()) }
}

pub(crate) fn svp_prepare<R: Ring>(
    module: &Module<NTT4x30Neon<R>>,
    res: &mut SvpPPolBackendMut<'_, NTT4x30Neon<R>>,
    res_col: usize,
    a: &ScalarZnxBackendRef<'_, NTT4x30Neon<R>>,
    a_col: usize,
) where
    NTT4x30Neon<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
{
    let n = res.n();
    check_degree::<NTT4x30Neon<R>>(module.n(), n);
    assert!(a.n() == n, "svp_prepare: a.n() != res.n()");
    let mut tmp = vec![0u64; dft_tmp_len(module, n)];
    let data: &mut [u32] = cast_slice_mut(res.data_mut());
    dft_limb_scaled(
        module,
        n,
        &mut data[4 * n * res_col..][..4 * n],
        a.at(a_col, 0),
        true,
        &mut tmp,
    );
}

pub(crate) fn svp_ppol_copy<R: Ring>(
    res: &mut SvpPPolBackendMut<'_, NTT4x30Neon<R>>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT4x30Neon<R>>,
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

pub(crate) fn svp_apply_dft<R: Ring>(
    module: &Module<NTT4x30Neon<R>>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Neon<R>>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT4x30Neon<R>>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, NTT4x30Neon<R>>,
    b_col: usize,
) where
    NTT4x30Neon<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
{
    let mut b_dft_owned = module.vec_znx_dft_alloc(b.n(), 1, b.size());
    let mut b_dft = b_dft_owned.to_backend_mut();
    super::vec_znx_dft::vec_znx_dft_apply(module, 1, 0, &mut b_dft, 0, b, b_col);
    svp_apply_dft_to_dft(module, res, res_col, a, a_col, &b_dft.reborrow_backend_ref(), 0);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn svp_apply_dft_to_dft<R: Ring>(
    _module: &Module<NTT4x30Neon<R>>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Neon<R>>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT4x30Neon<R>>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'_, NTT4x30Neon<R>>,
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
            mul_packed_limb(n, dst, packed_limb(b_data, n, b_cols, b_col, limb), factor);
        } else {
            dst.fill(0);
        }
    }
}

pub(crate) fn svp_apply_dft_to_dft_assign<R: Ring>(
    _module: &Module<NTT4x30Neon<R>>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Neon<R>>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT4x30Neon<R>>,
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
        mul_packed_limb_assign(n, dst, factor);
    }
}
