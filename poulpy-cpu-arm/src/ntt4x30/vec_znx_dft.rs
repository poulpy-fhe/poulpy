//! Packed transform-domain drivers of [`NTT4x30Neon`](crate::NTT4x30Neon).
//!
//! A limb of a `VecZnxDft` is four planes of `n` canonical `u32` residues, see [`crate::neon::ntt4x30_packed`].

use bytemuck::{cast_slice, cast_slice_mut};
use poulpy_cpu_portable::kernels::ntt4x30::ntt::{NttTable, NttTableInv};
use poulpy_cpu_portable::kernels::ntt4x30::{
    NttDFTExecute, NttFromZnx64, NttToZnx128,
    primes::Primes30,
    vec_znx_dft::{NttAutomorphismPlan, NttModuleHandle},
};
use poulpy_hal::execution::TaskExecutor;
use poulpy_hal::layouts::Ring;
use poulpy_hal::layouts::{
    DataView, DataViewMut, Module, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut, VecZnxDftBackendRef, ZnxView,
    ZnxViewMut, check_degree,
};

use super::NTT4x30Neon;
use crate::neon::ntt4x30_ntt32::{MIN_N, Ntt32Table, intt32, ntt32};
use crate::neon::ntt4x30_packed::{OP_ADD, OP_NEG, OP_SUB, Q, limb_op, pack_limb, unpack_limb};

#[inline(always)]
pub(crate) fn packed_limb(data: &[u32], n: usize, cols: usize, col: usize, limb: usize) -> &[u32] {
    let start = 4 * n * (limb * cols + col);
    &data[start..start + 4 * n]
}

#[inline(always)]
pub(crate) fn packed_limb_mut(data: &mut [u32], n: usize, cols: usize, col: usize, limb: usize) -> &mut [u32] {
    let start = 4 * n * (limb * cols + col);
    &mut data[start..start + 4 * n]
}

/// Tables of the packed NTT of degree `n`, absent on the conjugate-invariant ring.
#[inline(always)]
fn packed_table<R: Ring>(module: &Module<NTT4x30Neon<R>>, n: usize) -> Option<&Ntt32Table> {
    // SAFETY: the handle is initialised before `Module::new` returns and lives as long as the module.
    unsafe { (*module.ptr()).packed_table(n) }
}

/// Length in `u64` of the scratch the forward transform of one limb of degree `n` needs on ring `R`.
///
/// The packed NTT of the standard ring works in place, the q120 fallback widens the limb.
#[inline(always)]
pub(crate) fn dft_tmp_words<R: Ring>(n: usize) -> usize {
    if R::CYCLOTOMIC_ORDER_FACTOR == 2 && n >= MIN_N {
        0
    } else {
        4 * n
    }
}

/// Length in `u64` of the scratch the inverse transform of one limb of degree `n` needs on ring `R`.
///
/// The packed NTT moves the planes to the scratch, the q120 fallback widens the limb.
#[inline(always)]
pub(crate) fn idft_tmp_words<R: Ring>(n: usize) -> usize {
    if dft_tmp_words::<R>(n) == 0 { 2 * n } else { 4 * n }
}

/// Length in `u64` of the scratch a prepare kernel needs per limb: one packed limb and the forward transform scratch.
#[inline(always)]
pub(crate) fn prepare_tmp_words<R: Ring>(n: usize) -> usize {
    2 * n + dft_tmp_words::<R>(n)
}

/// [`dft_tmp_words`] for the ring and degree of an operand of `module`.
#[inline(always)]
pub(crate) fn dft_tmp_len<R: Ring>(module: &Module<NTT4x30Neon<R>>, n: usize) -> usize {
    debug_assert_eq!(packed_table(module, n).is_some(), dft_tmp_words::<R>(n) == 0);
    dft_tmp_words::<R>(n)
}

/// Forward transform of `src` into one packed limb, multiplied by `2^32` when `prepared` is set.
///
/// `tmp` holds [`dft_tmp_len`] words.
pub(crate) fn dft_limb_scaled<R: Ring>(
    module: &Module<NTT4x30Neon<R>>,
    n: usize,
    dst: &mut [u32],
    src: &[i64],
    prepared: bool,
    tmp: &mut [u64],
) where
    NTT4x30Neon<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
{
    if let Some(table) = packed_table(module, n) {
        ntt32(table, dst, src, prepared);
    } else {
        NTT4x30Neon::<R>::ntt_from_znx64(tmp, src);
        NTT4x30Neon::<R>::ntt_dft_execute(module.get_ntt_table_for(n), tmp);
        pack_limb(n, dst, tmp, prepared);
    }
}

/// Forward transform into a packed limb, for drivers shared by the serial and Rayon backends.
pub(crate) trait PackedDft {
    /// See [`dft_limb_scaled`], `tmp` holding [`dft_tmp_words`] words.
    fn packed_dft_limb(&self, n: usize, dst: &mut [u32], src: &[i64], prepared: bool, tmp: &mut [u64]);
}

impl<R: Ring> PackedDft for Module<NTT4x30Neon<R>>
where
    NTT4x30Neon<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
{
    #[inline(always)]
    fn packed_dft_limb(&self, n: usize, dst: &mut [u32], src: &[i64], prepared: bool, tmp: &mut [u64]) {
        dft_limb_scaled(self, n, dst, src, prepared, tmp)
    }
}

pub(crate) fn dft_limb<R: Ring>(module: &Module<NTT4x30Neon<R>>, n: usize, dst: &mut [u32], src: Option<&[i64]>, tmp: &mut [u64])
where
    NTT4x30Neon<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
{
    match src {
        // A zero limb transforms to zero: the scan stops at the first nonzero coefficient.
        Some(src) if src[..n].iter().any(|&x| x != 0) => dft_limb_scaled(module, n, dst, src, false, tmp),
        _ => dst.fill(0),
    }
}

/// Inverse transform of one packed limb.
///
/// `tmp` holds [`idft_tmp_words`] words.
pub(crate) fn idft_limb<R: Ring>(module: &Module<NTT4x30Neon<R>>, n: usize, dst: &mut [i128], src: &[u32], tmp: &mut [u64])
where
    NTT4x30Neon<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
{
    assert!(dst.len() >= n && src.len() >= 4 * n);
    if let Some(table) = packed_table(module, n) {
        let work: &mut [u32] = cast_slice_mut(tmp);
        assert!(work.len() >= 4 * n);
        unsafe { intt32(table, dst.as_mut_ptr(), src.as_ptr(), work.as_mut_ptr()) };
    } else {
        unpack_limb(n, tmp, src);
        NTT4x30Neon::<R>::ntt_dft_execute(module.get_intt_table_for(n), tmp);
        NTT4x30Neon::<R>::ntt_to_znx128(dst, n, tmp);
    }
}

/// Inverse transform of one packed limb, which it overwrites.
pub(crate) fn idft_limb_tmpa<R: Ring>(module: &Module<NTT4x30Neon<R>>, n: usize, dst: &mut [i128], src: &mut [u32])
where
    NTT4x30Neon<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
{
    assert!(dst.len() >= n && src.len() >= 4 * n);
    if let Some(table) = packed_table(module, n) {
        let src = src.as_mut_ptr();
        unsafe { intt32(table, dst.as_mut_ptr(), src, src) };
    } else {
        let mut tmp = vec![0u64; 4 * n];
        idft_limb(module, n, dst, src, &mut tmp);
    }
}

fn packed_add(n: usize, dst: &mut [u32], a: &[u32], b: &[u32]) {
    assert!(dst.len() >= 4 * n && a.len() >= 4 * n && b.len() >= 4 * n);
    unsafe { limb_op::<OP_ADD>(n, dst.as_mut_ptr(), a.as_ptr(), b.as_ptr()) }
}

fn packed_add_assign(n: usize, dst: &mut [u32], a: &[u32]) {
    assert!(dst.len() >= 4 * n && a.len() >= 4 * n);
    let dst = dst.as_mut_ptr();
    unsafe { limb_op::<OP_ADD>(n, dst, dst, a.as_ptr()) }
}

fn packed_sub(n: usize, dst: &mut [u32], a: &[u32], b: &[u32]) {
    assert!(dst.len() >= 4 * n && a.len() >= 4 * n && b.len() >= 4 * n);
    unsafe { limb_op::<OP_SUB>(n, dst.as_mut_ptr(), a.as_ptr(), b.as_ptr()) }
}

fn packed_sub_assign(n: usize, dst: &mut [u32], a: &[u32]) {
    assert!(dst.len() >= 4 * n && a.len() >= 4 * n);
    let dst = dst.as_mut_ptr();
    unsafe { limb_op::<OP_SUB>(n, dst, dst, a.as_ptr()) }
}

fn packed_sub_negate_assign(n: usize, dst: &mut [u32], a: &[u32]) {
    assert!(dst.len() >= 4 * n && a.len() >= 4 * n);
    let dst = dst.as_mut_ptr();
    unsafe { limb_op::<OP_SUB>(n, dst, a.as_ptr(), dst) }
}

fn packed_negate_assign(n: usize, dst: &mut [u32]) {
    assert!(dst.len() >= 4 * n);
    let dst = dst.as_mut_ptr();
    unsafe { limb_op::<OP_NEG>(n, dst, dst, dst) }
}

pub(crate) fn vec_znx_dft_apply<R: Ring>(
    module: &Module<NTT4x30Neon<R>>,
    step: usize,
    offset: usize,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Neon<R>>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, NTT4x30Neon<R>>,
    a_col: usize,
) where
    NTT4x30Neon<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
{
    poulpy_hal::layouts::assert_dense(a, "vec_znx_dft_apply");
    assert!(step >= 1, "vec_znx_dft_apply: step must be >= 1");
    let n = res.n();
    check_degree::<NTT4x30Neon<R>>(module.n(), n);
    assert!(a.n() == n, "vec_znx_dft_apply: a.n() != res.n()");
    let cols = res.cols();
    let res_size = res.size();
    let a_size = a.size();
    let mut tmp = vec![0u64; dft_tmp_len(module, n)];
    let res_data: &mut [u32] = cast_slice_mut(res.data_mut());
    for limb in 0..res_size {
        let dst = packed_limb_mut(res_data, n, cols, res_col, limb);
        let src_limb = offset + limb * step;
        dft_limb(module, n, dst, (src_limb < a_size).then(|| a.at(a_col, src_limb)), &mut tmp);
    }
}

pub(crate) fn vec_znx_idft_apply_tmp_bytes<R: Ring>(n: usize) -> usize {
    idft_tmp_words::<R>(n) * size_of::<u64>()
}

pub(crate) fn vec_znx_idft_apply<R: Ring>(
    module: &Module<NTT4x30Neon<R>>,
    res: &mut VecZnxBigBackendMut<'_, NTT4x30Neon<R>>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Neon<R>>,
    a_col: usize,
    tmp: &mut [u64],
) where
    NTT4x30Neon<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
{
    poulpy_hal::layouts::assert_dense(res, "vec_znx_idft_apply");
    let n = res.n();
    check_degree::<NTT4x30Neon<R>>(module.n(), n);
    assert_eq!(a.n(), n, "vec_znx_idft_apply: a.n():{} != res.n():{n}", a.n());
    let min_size = res.size().min(a.size());
    let a_cols = a.cols();
    let a_data: &[u32] = cast_slice(a.data());
    for limb in 0..min_size {
        idft_limb(
            module,
            n,
            res.at_mut(res_col, limb),
            packed_limb(a_data, n, a_cols, a_col, limb),
            tmp,
        );
    }
    for limb in min_size..res.size() {
        res.at_mut(res_col, limb).fill(0);
    }
}

pub(crate) fn vec_znx_idft_apply_tmpa<R: Ring>(
    module: &Module<NTT4x30Neon<R>>,
    res: &mut VecZnxBigBackendMut<'_, NTT4x30Neon<R>>,
    res_col: usize,
    a: &mut VecZnxDftBackendMut<'_, NTT4x30Neon<R>>,
    a_col: usize,
) where
    NTT4x30Neon<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
{
    poulpy_hal::layouts::assert_dense(res, "vec_znx_idft_apply_tmpa");
    let n = res.n();
    check_degree::<NTT4x30Neon<R>>(module.n(), n);
    assert_eq!(a.n(), n, "vec_znx_idft_apply_tmpa: a.n():{} != res.n():{n}", a.n());
    let min_size = res.size().min(a.size());
    let a_cols = a.cols();
    let a_data: &mut [u32] = cast_slice_mut(a.data_mut());
    for limb in 0..min_size {
        idft_limb_tmpa(
            module,
            n,
            res.at_mut(res_col, limb),
            packed_limb_mut(a_data, n, a_cols, a_col, limb),
        );
    }
    for limb in min_size..res.size() {
        res.at_mut(res_col, limb).fill(0);
    }
}

pub(crate) fn idft_compact_in_place<R: Ring>(
    module: &Module<NTT4x30Neon<R>>,
    a: &mut VecZnxDftBackendMut<'_, NTT4x30Neon<R>>,
    a_col: usize,
    tmp: &mut [u64],
) where
    NTT4x30Neon<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
{
    let n = a.n();
    check_degree::<NTT4x30Neon<R>>(module.n(), n);
    let cols = a.cols();
    let size = a.size();
    let data: &mut [u32] = cast_slice_mut(a.data_mut());
    for limb in 0..size {
        let slot = packed_limb_mut(data, n, cols, a_col, limb);
        if let Some(table) = packed_table(module, n) {
            // The planes move to `tmp` during the transform, so the coefficients can overwrite the limb.
            let work: &mut [u32] = cast_slice_mut(tmp);
            assert!(work.len() >= 4 * n);
            let slot = slot.as_mut_ptr();
            unsafe { intt32(table, slot as *mut i128, slot, work.as_mut_ptr()) };
        } else {
            unpack_limb(n, tmp, slot);
            NTT4x30Neon::<R>::ntt_dft_execute(module.get_intt_table_for(n), tmp);
            let dst = unsafe { std::slice::from_raw_parts_mut(slot.as_mut_ptr() as *mut i128, n) };
            NTT4x30Neon::<R>::ntt_to_znx128(dst, n, tmp);
        }
    }
}

pub(crate) fn vec_znx_dft_add<R: Ring>(
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Neon<R>>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Neon<R>>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'_, NTT4x30Neon<R>>,
    b_col: usize,
) {
    let n = res.n();
    let (rc, ac, bc) = (res.cols(), a.cols(), b.cols());
    let (rs, asz, bsz) = (res.size(), a.size(), b.size());
    let (sum_size, copy_size, copy_b) = if asz <= bsz {
        (asz.min(rs), bsz.min(rs), true)
    } else {
        (bsz.min(rs), asz.min(rs), false)
    };
    let rp: &mut [u32] = cast_slice_mut(res.data_mut());
    let ap: &[u32] = cast_slice(a.data());
    let bp: &[u32] = cast_slice(b.data());
    for limb in 0..rs {
        let dst = packed_limb_mut(rp, n, rc, res_col, limb);
        if limb < sum_size {
            packed_add(
                n,
                dst,
                packed_limb(ap, n, ac, a_col, limb),
                packed_limb(bp, n, bc, b_col, limb),
            );
        } else if limb < copy_size {
            let src = if copy_b {
                packed_limb(bp, n, bc, b_col, limb)
            } else {
                packed_limb(ap, n, ac, a_col, limb)
            };
            dst.copy_from_slice(src);
        } else {
            dst.fill(0);
        }
    }
}

pub(crate) fn vec_znx_dft_add_assign<R: Ring>(
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Neon<R>>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Neon<R>>,
    a_col: usize,
) {
    let n = res.n();
    let (rc, ac) = (res.cols(), a.cols());
    let size = res.size().min(a.size());
    let rp: &mut [u32] = cast_slice_mut(res.data_mut());
    let ap: &[u32] = cast_slice(a.data());
    for limb in 0..size {
        packed_add_assign(
            n,
            packed_limb_mut(rp, n, rc, res_col, limb),
            packed_limb(ap, n, ac, a_col, limb),
        );
    }
}

pub(crate) fn vec_znx_dft_sub<R: Ring>(
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Neon<R>>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Neon<R>>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'_, NTT4x30Neon<R>>,
    b_col: usize,
) {
    let n = res.n();
    let (rc, ac, bc) = (res.cols(), a.cols(), b.cols());
    let (rs, asz, bsz) = (res.size(), a.size(), b.size());
    let (sub_size, copy_size, negate_b) = if asz <= bsz {
        (asz.min(rs), bsz.min(rs), true)
    } else {
        (bsz.min(rs), asz.min(rs), false)
    };
    let rp: &mut [u32] = cast_slice_mut(res.data_mut());
    let ap: &[u32] = cast_slice(a.data());
    let bp: &[u32] = cast_slice(b.data());
    for limb in 0..rs {
        let dst = packed_limb_mut(rp, n, rc, res_col, limb);
        if limb < sub_size {
            packed_sub(
                n,
                dst,
                packed_limb(ap, n, ac, a_col, limb),
                packed_limb(bp, n, bc, b_col, limb),
            );
        } else if limb < copy_size {
            if negate_b {
                dst.copy_from_slice(packed_limb(bp, n, bc, b_col, limb));
                packed_negate_assign(n, dst);
            } else {
                dst.copy_from_slice(packed_limb(ap, n, ac, a_col, limb));
            }
        } else {
            dst.fill(0);
        }
    }
}

pub(crate) fn vec_znx_dft_sub_assign<R: Ring>(
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Neon<R>>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Neon<R>>,
    a_col: usize,
) {
    let n = res.n();
    let (rc, ac) = (res.cols(), a.cols());
    let size = res.size().min(a.size());
    let rp: &mut [u32] = cast_slice_mut(res.data_mut());
    let ap: &[u32] = cast_slice(a.data());
    for limb in 0..size {
        packed_sub_assign(
            n,
            packed_limb_mut(rp, n, rc, res_col, limb),
            packed_limb(ap, n, ac, a_col, limb),
        );
    }
}

pub(crate) fn vec_znx_dft_sub_negate_assign<R: Ring>(
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Neon<R>>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Neon<R>>,
    a_col: usize,
) {
    let n = res.n();
    let (rc, ac) = (res.cols(), a.cols());
    let rs = res.size();
    let size = rs.min(a.size());
    let rp: &mut [u32] = cast_slice_mut(res.data_mut());
    let ap: &[u32] = cast_slice(a.data());
    for limb in 0..rs {
        let dst = packed_limb_mut(rp, n, rc, res_col, limb);
        if limb < size {
            packed_sub_negate_assign(n, dst, packed_limb(ap, n, ac, a_col, limb));
        } else {
            packed_negate_assign(n, dst);
        }
    }
}

pub(crate) fn vec_znx_dft_copy<R: Ring>(
    step: usize,
    offset: usize,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Neon<R>>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Neon<R>>,
    a_col: usize,
) {
    assert!(step >= 1, "vec_znx_dft_copy: step must be >= 1");
    let n = res.n();
    let (rc, ac) = (res.cols(), a.cols());
    let size = res.size();
    let rp: &mut [u32] = cast_slice_mut(res.data_mut());
    let ap: &[u32] = cast_slice(a.data());
    for limb in 0..size {
        let dst = packed_limb_mut(rp, n, rc, res_col, limb);
        let src_limb = offset + limb * step;
        if src_limb < a.size() {
            dst.copy_from_slice(packed_limb(ap, n, ac, a_col, src_limb));
        } else {
            dst.fill(0);
        }
    }
}

pub(crate) fn vec_znx_dft_zero<R: Ring>(res: &mut VecZnxDftBackendMut<'_, NTT4x30Neon<R>>, res_col: usize) {
    let n = res.n();
    let cols = res.cols();
    let size = res.size();
    let data: &mut [u32] = cast_slice_mut(res.data_mut());
    for limb in 0..size {
        packed_limb_mut(data, n, cols, res_col, limb).fill(0);
    }
}

pub(crate) fn vec_znx_dft_automorphism<R: Ring>(
    plan: &NttAutomorphismPlan,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Neon<R>>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Neon<R>>,
    a_col: usize,
) {
    {
        assert_eq!(a.n(), res.n());
        assert_eq!(plan.perm.len(), res.n());
    }

    let n = res.n();
    let (rc, ac) = (res.cols(), a.cols());
    let size = res.size().min(a.size());
    let rs = res.size();
    let rp: &mut [u32] = cast_slice_mut(res.data_mut());
    let ap: &[u32] = cast_slice(a.data());
    for limb in 0..rs {
        let dst = packed_limb_mut(rp, n, rc, res_col, limb);
        if limb < size {
            let src = packed_limb(ap, n, ac, a_col, limb);
            for (dst, src) in dst.chunks_exact_mut(n).zip(src.chunks_exact(n)) {
                for (d, &p) in dst.iter_mut().zip(plan.perm.iter()) {
                    *d = src[p as usize];
                }
            }
        } else {
            dst.fill(0);
        }
    }
}

pub(crate) fn vec_znx_dft_automorphism_add<R: Ring, E: TaskExecutor>(
    plan: &NttAutomorphismPlan,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Neon<R>>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Neon<R>>,
    a_col: usize,
) {
    {
        assert_eq!(a.n(), res.n());
        assert_eq!(plan.perm.len(), res.n());
    }

    let n = res.n();
    let (rc, ac) = (res.cols(), a.cols());
    let size = res.size().min(a.size());
    let res_ptr = cast_slice_mut::<_, u32>(res.data_mut()).as_mut_ptr() as usize;
    let ap: &[u32] = cast_slice(a.data());
    E::for_each(size, |limb| {
        let start = 4 * n * (limb * rc + res_col);
        let dst = unsafe { std::slice::from_raw_parts_mut((res_ptr as *mut u32).add(start), 4 * n) };
        let src = packed_limb(ap, n, ac, a_col, limb);
        for (prime, (dst, src)) in dst.chunks_exact_mut(n).zip(src.chunks_exact(n)).enumerate() {
            let q = Q[prime];
            for (d, &p) in dst.iter_mut().zip(plan.perm.iter()) {
                let sum = *d + src[p as usize];
                *d = sum.min(sum.wrapping_sub(q));
            }
        }
    });
}
