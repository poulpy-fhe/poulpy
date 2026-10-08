//! Packed transform-domain drivers of [`NTT4x30Portable`](crate::NTT4x30Portable).
//!
//! A limb of a `VecZnxDft` is four planes of `n` canonical `u32` residues, see [`super::packed`].

use crate::kernels::ntt4x30::ntt::{NttTable, NttTableInv};
use crate::kernels::ntt4x30::{NttDFTExecute, primes::Primes30, vec_znx_dft::NttAutomorphismPlan};
use bytemuck::{cast_slice, cast_slice_mut};
use poulpy_hal::execution::TaskExecutor;
use poulpy_hal::layouts::Ring;
use poulpy_hal::layouts::{
    DataView, DataViewMut, Module, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut, VecZnxDftBackendRef, ZnxView,
    ZnxViewMut, check_degree,
};

use super::NTT4x30Portable;
use super::ntt32::{Ntt32Table, intt32, intt32_assign, intt32_compact, ntt32};
use super::packed::{
    Q, SendPtr, add_mod, limb_add, limb_add_assign, limb_negate_assign, limb_sub, limb_sub_assign, limb_sub_negate_assign,
};

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

/// Tables of the packed NTT of degree `n`.
#[inline(always)]
pub(crate) fn packed_table<R: Ring>(module: &Module<NTT4x30Portable<R>>, n: usize) -> &Ntt32Table {
    // SAFETY: the handle is initialised before `Module::new` returns and lives as long as the module.
    unsafe { (*module.ptr()).packed_table(n) }
}

/// Length in `u64` of the scratch the inverse transform of one limb of degree `n` needs: the planes move to it.
#[inline(always)]
pub fn idft_tmp_words(n: usize) -> usize {
    2 * n
}

/// Length in `u64` of the scratch a prepare kernel needs per limb: one packed limb.
#[inline(always)]
pub(crate) fn prepare_tmp_words(n: usize) -> usize {
    2 * n
}

/// Forward transform of `src` into one packed limb, multiplied by `2^32` when `prepared` is set.
pub fn dft_limb_scaled<R: Ring>(module: &Module<NTT4x30Portable<R>>, n: usize, dst: &mut [u32], src: &[i64], prepared: bool) {
    ntt32(packed_table(module, n), dst, src, prepared);
}

/// Forward transform of `src` into one packed limb, or zeros when `src` is `None`.
pub fn dft_limb<R: Ring>(module: &Module<NTT4x30Portable<R>>, n: usize, dst: &mut [u32], src: Option<&[i64]>) {
    match src {
        // A zero limb transforms to zero: the scan stops at the first nonzero coefficient.
        Some(src) if src[..n].iter().any(|&x| x != 0) => dft_limb_scaled(module, n, dst, src, false),
        _ => dst.fill(0),
    }
}

/// Inverse transform of one packed limb.
///
/// `tmp` holds [`idft_tmp_words`] words.
pub fn idft_limb<R: Ring>(module: &Module<NTT4x30Portable<R>>, n: usize, dst: &mut [i128], src: &[u32], tmp: &mut [u64]) {
    intt32(packed_table(module, n), dst, src, cast_slice_mut(tmp));
}

/// Inverse transform of one packed limb, which it overwrites.
pub fn idft_limb_tmpa<R: Ring>(module: &Module<NTT4x30Portable<R>>, n: usize, dst: &mut [i128], src: &mut [u32]) {
    intt32_assign(packed_table(module, n), dst, src);
}

/// Inverse transform of the packed limb `slot` into `n` coefficients that overwrite it.
///
/// The planes move to `tmp`, which holds [`idft_tmp_words`] words, so the coefficients can take the place of the limb.
pub fn idft_limb_compact<R: Ring>(module: &Module<NTT4x30Portable<R>>, n: usize, slot: &mut [u32], tmp: &mut [u64]) {
    intt32_compact(packed_table(module, n), slot, cast_slice_mut(tmp));
}

pub(crate) fn vec_znx_dft_apply<R: Ring>(
    module: &Module<NTT4x30Portable<R>>,
    step: usize,
    offset: usize,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Portable<R>>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, NTT4x30Portable<R>>,
    a_col: usize,
) where
    NTT4x30Portable<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
{
    poulpy_hal::layouts::assert_dense(a, "vec_znx_dft_apply");
    assert!(step >= 1, "vec_znx_dft_apply: step must be >= 1");
    let n = res.n();
    check_degree::<NTT4x30Portable<R>>(module.n(), n);
    assert!(a.n() == n, "vec_znx_dft_apply: a.n() != res.n()");
    let cols = res.cols();
    let res_size = res.size();
    let a_size = a.size();
    let res_data: &mut [u32] = cast_slice_mut(res.data_mut());
    for limb in 0..res_size {
        let dst = packed_limb_mut(res_data, n, cols, res_col, limb);
        let src_limb = offset + limb * step;
        dft_limb(module, n, dst, (src_limb < a_size).then(|| a.at(a_col, src_limb)));
    }
}

pub(crate) fn vec_znx_idft_apply_tmp_bytes(n: usize) -> usize {
    idft_tmp_words(n) * size_of::<u64>()
}

pub(crate) fn vec_znx_idft_apply<R: Ring>(
    module: &Module<NTT4x30Portable<R>>,
    res: &mut VecZnxBigBackendMut<'_, NTT4x30Portable<R>>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Portable<R>>,
    a_col: usize,
    tmp: &mut [u64],
) where
    NTT4x30Portable<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
{
    poulpy_hal::layouts::assert_dense(res, "vec_znx_idft_apply");
    let n = res.n();
    check_degree::<NTT4x30Portable<R>>(module.n(), n);
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
    module: &Module<NTT4x30Portable<R>>,
    res: &mut VecZnxBigBackendMut<'_, NTT4x30Portable<R>>,
    res_col: usize,
    a: &mut VecZnxDftBackendMut<'_, NTT4x30Portable<R>>,
    a_col: usize,
) where
    NTT4x30Portable<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
{
    poulpy_hal::layouts::assert_dense(res, "vec_znx_idft_apply_tmpa");
    let n = res.n();
    check_degree::<NTT4x30Portable<R>>(module.n(), n);
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
    module: &Module<NTT4x30Portable<R>>,
    a: &mut VecZnxDftBackendMut<'_, NTT4x30Portable<R>>,
    a_col: usize,
    tmp: &mut [u64],
) where
    NTT4x30Portable<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
{
    let n = a.n();
    check_degree::<NTT4x30Portable<R>>(module.n(), n);
    let cols = a.cols();
    let size = a.size();
    let data: &mut [u32] = cast_slice_mut(a.data_mut());
    for limb in 0..size {
        idft_limb_compact(module, n, packed_limb_mut(data, n, cols, a_col, limb), tmp);
    }
}

/// Runs `task` on limb `0..count` of column `col` of `data`, as tasks of the executor `E`.
fn for_each_limb<E: TaskExecutor>(
    data: &mut [u32],
    n: usize,
    cols: usize,
    col: usize,
    count: usize,
    task: impl Fn(usize, &mut [u32]) + Send + Sync,
) {
    if count == 0 {
        return;
    }
    assert!(col < cols && data.len() >= 4 * n * cols * count);
    let ptr = SendPtr(data.as_mut_ptr());
    E::for_each(count, |limb| {
        // Limbs are disjoint, one task each.
        task(limb, unsafe {
            std::slice::from_raw_parts_mut(ptr.get().add(4 * n * (limb * cols + col)), 4 * n)
        })
    });
}

pub fn vec_znx_dft_add<R: Ring, E: TaskExecutor>(
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Portable<R>>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Portable<R>>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'_, NTT4x30Portable<R>>,
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
    for_each_limb::<E>(rp, n, rc, res_col, rs, |limb, dst| {
        if limb < sum_size {
            limb_add(
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
    });
}

pub fn vec_znx_dft_add_assign<R: Ring, E: TaskExecutor>(
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Portable<R>>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Portable<R>>,
    a_col: usize,
) {
    let n = res.n();
    let (rc, ac) = (res.cols(), a.cols());
    let size = res.size().min(a.size());
    let rp: &mut [u32] = cast_slice_mut(res.data_mut());
    let ap: &[u32] = cast_slice(a.data());
    for_each_limb::<E>(rp, n, rc, res_col, size, |limb, dst| {
        limb_add_assign(n, dst, packed_limb(ap, n, ac, a_col, limb));
    });
}

pub fn vec_znx_dft_sub<R: Ring, E: TaskExecutor>(
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Portable<R>>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Portable<R>>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'_, NTT4x30Portable<R>>,
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
    for_each_limb::<E>(rp, n, rc, res_col, rs, |limb, dst| {
        if limb < sub_size {
            limb_sub(
                n,
                dst,
                packed_limb(ap, n, ac, a_col, limb),
                packed_limb(bp, n, bc, b_col, limb),
            );
        } else if limb < copy_size {
            if negate_b {
                dst.copy_from_slice(packed_limb(bp, n, bc, b_col, limb));
                limb_negate_assign(n, dst);
            } else {
                dst.copy_from_slice(packed_limb(ap, n, ac, a_col, limb));
            }
        } else {
            dst.fill(0);
        }
    });
}

pub fn vec_znx_dft_sub_assign<R: Ring, E: TaskExecutor>(
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Portable<R>>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Portable<R>>,
    a_col: usize,
) {
    let n = res.n();
    let (rc, ac) = (res.cols(), a.cols());
    let size = res.size().min(a.size());
    let rp: &mut [u32] = cast_slice_mut(res.data_mut());
    let ap: &[u32] = cast_slice(a.data());
    for_each_limb::<E>(rp, n, rc, res_col, size, |limb, dst| {
        limb_sub_assign(n, dst, packed_limb(ap, n, ac, a_col, limb));
    });
}

pub fn vec_znx_dft_sub_negate_assign<R: Ring, E: TaskExecutor>(
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Portable<R>>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Portable<R>>,
    a_col: usize,
) {
    let n = res.n();
    let (rc, ac) = (res.cols(), a.cols());
    let rs = res.size();
    let size = rs.min(a.size());
    let rp: &mut [u32] = cast_slice_mut(res.data_mut());
    let ap: &[u32] = cast_slice(a.data());
    for_each_limb::<E>(rp, n, rc, res_col, rs, |limb, dst| {
        if limb < size {
            limb_sub_negate_assign(n, dst, packed_limb(ap, n, ac, a_col, limb));
        } else {
            limb_negate_assign(n, dst);
        }
    });
}

pub fn vec_znx_dft_copy<R: Ring, E: TaskExecutor>(
    step: usize,
    offset: usize,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Portable<R>>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Portable<R>>,
    a_col: usize,
) {
    assert!(step >= 1, "vec_znx_dft_copy: step must be >= 1");
    let n = res.n();
    let (rc, ac) = (res.cols(), a.cols());
    let size = res.size();
    let a_size = a.size();
    let rp: &mut [u32] = cast_slice_mut(res.data_mut());
    let ap: &[u32] = cast_slice(a.data());
    for_each_limb::<E>(rp, n, rc, res_col, size, |limb, dst| {
        let src_limb = offset + limb * step;
        if src_limb < a_size {
            dst.copy_from_slice(packed_limb(ap, n, ac, a_col, src_limb));
        } else {
            dst.fill(0);
        }
    });
}

pub(crate) fn vec_znx_dft_zero<R: Ring>(res: &mut VecZnxDftBackendMut<'_, NTT4x30Portable<R>>, res_col: usize) {
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
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Portable<R>>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Portable<R>>,
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

pub fn vec_znx_dft_automorphism_add<R: Ring, E: TaskExecutor>(
    plan: &NttAutomorphismPlan,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Portable<R>>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT4x30Portable<R>>,
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
    // One task per plane: a vector has few limbs at a large `base2k`.
    E::for_each(4 * size, |task| {
        let (limb, prime) = (task / 4, task % 4);
        let start = 4 * n * (limb * rc + res_col) + prime * n;
        let dst = unsafe { std::slice::from_raw_parts_mut((res_ptr as *mut u32).add(start), n) };
        let src = &packed_limb(ap, n, ac, a_col, limb)[prime * n..][..n];
        for (d, &p) in dst.iter_mut().zip(plan.perm.iter()) {
            *d = add_mod(*d, src[p as usize], Q[prime]);
        }
    });
}
