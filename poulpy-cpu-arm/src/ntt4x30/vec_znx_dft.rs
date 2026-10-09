//! Packed transform-domain drivers of [`NTT4x30Neon`](crate::NTT4x30Neon).
//!
//! A limb of a `VecZnxDft` is four planes of `n` canonical `u32` residues, see [`crate::neon::ntt4x30_packed`].

use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::aarch64::{vaddq_u32, vdupq_n_u32, vld1q_u32, vminq_u32, vst1q_u32, vsubq_u32};
use poulpy_cpu_portable::kernels::ntt4x30::ntt::{NttTable, NttTableInv};
use poulpy_cpu_portable::kernels::ntt4x30::{NttDFTExecute, primes::Primes30, vec_znx_dft::NttAutomorphismPlan};
use poulpy_hal::execution::{SerialTaskExecutor, TaskExecutor};
use poulpy_hal::layouts::Ring;
use poulpy_hal::layouts::{
    DataView, DataViewMut, Module, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut, VecZnxDftBackendRef, ZnxView,
    ZnxViewMut, check_degree,
};

use super::NTT4x30Neon;
use super::convolution::SendPtr;
use crate::neon::ntt4x30_ntt32::{Ntt32Table, intt32, intt32_crt, intt32_plane, ntt32, ntt32_plane};
use crate::neon::ntt4x30_packed::{OP_ADD, OP_NEG, OP_SUB, Q, limb_op};

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
pub(crate) fn packed_table<R: Ring>(module: &Module<NTT4x30Neon<R>>, n: usize) -> &Ntt32Table {
    // SAFETY: the handle is initialised before `Module::new` returns and lives as long as the module.
    unsafe { (*module.ptr()).packed_table(n) }
}

/// Length in `u64` of the scratch the inverse transform of one limb of degree `n` needs: the planes move to it.
#[inline(always)]
pub(crate) fn idft_tmp_words(n: usize) -> usize {
    2 * n
}

/// Length in `u64` of the scratch a prepare kernel needs per limb: one packed limb.
#[inline(always)]
pub(crate) fn prepare_tmp_words(n: usize) -> usize {
    2 * n
}

/// Ring degree from which the four planes of a limb are transformed as separate tasks of a parallel executor.
///
/// A ciphertext has few limbs at a large `base2k`, fewer than a pool has threads.
const PLANE_TASKS_MIN_N: usize = 1 << 13;

#[inline]
fn plane_tasks<E: TaskExecutor>(n: usize) -> bool {
    E::is_parallel() && n >= PLANE_TASKS_MIN_N && E::max_parallelism() > 1
}

/// Runs `task` on `0..4` as four tasks of `E`.
#[inline]
fn join4<E: TaskExecutor>(task: impl Fn(usize) + Sync) {
    E::join(|| E::join(|| task(0), || task(1)), || E::join(|| task(2), || task(3)));
}

/// Forward transform of `src` into one packed limb, multiplied by `2^32` when `prepared` is set.
///
/// A parallel executor `E` transforms the four planes as separate tasks at large degrees, here and in the inverse transforms.
pub(crate) fn dft_limb_scaled<R: Ring, E: TaskExecutor>(
    module: &Module<NTT4x30Neon<R>>,
    n: usize,
    dst: &mut [u32],
    src: &[i64],
    prepared: bool,
) {
    let table = packed_table(module, n);
    if !plane_tasks::<E>(n) {
        return ntt32(table, dst, src, prepared);
    }
    assert!(dst.len() >= 4 * n);
    let dst = SendPtr(dst.as_mut_ptr());
    join4::<E>(|p| {
        // Tasks take distinct planes.
        let plane = unsafe { std::slice::from_raw_parts_mut(dst.get().add(p * n), n) };
        ntt32_plane(table, p, plane, src, prepared)
    });
}

/// Forward transform of `src` into one packed limb, or zeros when `src` is `None`.
pub(crate) fn dft_limb<R: Ring, E: TaskExecutor>(
    module: &Module<NTT4x30Neon<R>>,
    n: usize,
    dst: &mut [u32],
    src: Option<&[i64]>,
) {
    match src {
        // A zero limb transforms to zero: the scan stops at the first nonzero coefficient.
        Some(src) if src[..n].iter().any(|&x| x != 0) => dft_limb_scaled::<R, E>(module, n, dst, src, false),
        _ => dst.fill(0),
    }
}

/// Inverse transform of one packed limb, its planes and its reconstruction split into tasks of `E` at large degrees.
///
/// # Safety
/// Same contract as [`intt32`].
unsafe fn idft_limb_planes<R: Ring, E: TaskExecutor>(
    module: &Module<NTT4x30Neon<R>>,
    n: usize,
    dst: *mut i128,
    src: *const u32,
    work: *mut u32,
) {
    let table = packed_table(module, n);
    if !plane_tasks::<E>(n) {
        return unsafe { intt32(table, dst, src, work) };
    }
    let (dst, src, work) = (SendPtr(dst), SendPtr(src as *mut u32), SendPtr(work));
    join4::<E>(|p| unsafe { intt32_plane(table, p, src.get(), work.get()) });
    let quarter = n / 4;
    join4::<E>(|k| unsafe { intt32_crt(table, dst.get(), work.get(), k * quarter, (k + 1) * quarter) });
}

/// Inverse transform of one packed limb.
///
/// `tmp` holds [`idft_tmp_words`] words.
pub(crate) fn idft_limb<R: Ring, E: TaskExecutor>(
    module: &Module<NTT4x30Neon<R>>,
    n: usize,
    dst: &mut [i128],
    src: &[u32],
    tmp: &mut [u64],
) {
    assert!(dst.len() >= n && src.len() >= 4 * n);
    let work: &mut [u32] = cast_slice_mut(tmp);
    assert!(work.len() >= 4 * n);
    unsafe { idft_limb_planes::<R, E>(module, n, dst.as_mut_ptr(), src.as_ptr(), work.as_mut_ptr()) };
}

/// Inverse transform of one packed limb, which it overwrites.
pub(crate) fn idft_limb_tmpa<R: Ring, E: TaskExecutor>(
    module: &Module<NTT4x30Neon<R>>,
    n: usize,
    dst: &mut [i128],
    src: &mut [u32],
) {
    assert!(dst.len() >= n && src.len() >= 4 * n);
    let src = src.as_mut_ptr();
    unsafe { idft_limb_planes::<R, E>(module, n, dst.as_mut_ptr(), src, src) };
}

/// Inverse transform of the packed limb `slot` into `n` coefficients that overwrite it.
///
/// The planes move to `tmp`, which holds [`idft_tmp_words`] words, so the coefficients can take the place of the limb.
pub(crate) fn idft_limb_compact<R: Ring, E: TaskExecutor>(
    module: &Module<NTT4x30Neon<R>>,
    n: usize,
    slot: &mut [u32],
    tmp: &mut [u64],
) {
    assert!(slot.len() >= 4 * n);
    let work: &mut [u32] = cast_slice_mut(tmp);
    assert!(work.len() >= 4 * n);
    let slot = slot.as_mut_ptr();
    unsafe { idft_limb_planes::<R, E>(module, n, slot as *mut i128, slot, work.as_mut_ptr()) };
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
    let res_data: &mut [u32] = cast_slice_mut(res.data_mut());
    for limb in 0..res_size {
        let dst = packed_limb_mut(res_data, n, cols, res_col, limb);
        let src_limb = offset + limb * step;
        dft_limb::<R, SerialTaskExecutor>(module, n, dst, (src_limb < a_size).then(|| a.at(a_col, src_limb)));
    }
}

pub(crate) fn vec_znx_idft_apply_tmp_bytes(n: usize) -> usize {
    idft_tmp_words(n) * size_of::<u64>()
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
        idft_limb::<R, SerialTaskExecutor>(
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
        idft_limb_tmpa::<R, SerialTaskExecutor>(
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
        idft_limb_compact::<R, SerialTaskExecutor>(module, n, packed_limb_mut(data, n, cols, a_col, limb), tmp);
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

pub(crate) fn vec_znx_dft_add<R: Ring, E: TaskExecutor>(
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
    for_each_limb::<E>(rp, n, rc, res_col, rs, |limb, dst| {
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
    });
}

pub(crate) fn vec_znx_dft_add_assign<R: Ring, E: TaskExecutor>(
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
    for_each_limb::<E>(rp, n, rc, res_col, size, |limb, dst| {
        packed_add_assign(n, dst, packed_limb(ap, n, ac, a_col, limb));
    });
}

pub(crate) fn vec_znx_dft_sub<R: Ring, E: TaskExecutor>(
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
    for_each_limb::<E>(rp, n, rc, res_col, rs, |limb, dst| {
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
    });
}

pub(crate) fn vec_znx_dft_sub_assign<R: Ring, E: TaskExecutor>(
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
    for_each_limb::<E>(rp, n, rc, res_col, size, |limb, dst| {
        packed_sub_assign(n, dst, packed_limb(ap, n, ac, a_col, limb));
    });
}

pub(crate) fn vec_znx_dft_sub_negate_assign<R: Ring, E: TaskExecutor>(
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
    for_each_limb::<E>(rp, n, rc, res_col, rs, |limb, dst| {
        if limb < size {
            packed_sub_negate_assign(n, dst, packed_limb(ap, n, ac, a_col, limb));
        } else {
            packed_negate_assign(n, dst);
        }
    });
}

pub(crate) fn vec_znx_dft_copy<R: Ring, E: TaskExecutor>(
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
        assert!(res.n().is_multiple_of(4));
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
        // The gather stays scalar and checked, the modular add runs on four lanes.
        let q = unsafe { vdupq_n_u32(Q[prime]) };
        for (d, p) in dst.chunks_exact_mut(4).zip(plan.perm.chunks_exact(4)) {
            let gathered = [src[p[0] as usize], src[p[1] as usize], src[p[2] as usize], src[p[3] as usize]];
            unsafe {
                let sum = vaddq_u32(vld1q_u32(d.as_ptr()), vld1q_u32(gathered.as_ptr()));
                vst1q_u32(d.as_mut_ptr(), vminq_u32(sum, vsubq_u32(sum, q)));
            }
        }
    });
}
