//! Convolution for the packed NTT4x30 layout.
//!
//! A prepared column is ordered `block -> limb`, where a block is four consecutive coefficients.
//! One limb of a block is 16 `u32`: four lanes for each of the four primes.
//! The left operand holds canonical residues.
//! The right operand holds residues multiplied by `2^32`, with its limbs in reverse order, so both operands of one output limb are read forward.

use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::aarch64::{vld1q_dup_u32, vld1q_u32, vst1q_u32, vzip1q_u32, vzip2q_u32};
use poulpy_cpu_portable::kernels::ntt4x30::{NttDFTExecute, NttFromZnx64, primes::Primes30, vec_znx_dft::NttModuleHandle};
use poulpy_cpu_portable::kernels::sparse_log_gap_portable;
use poulpy_hal::execution::TaskExecutor;
#[cfg(feature = "enable-rayon")]
use poulpy_hal::layouts::CnvDftAccTerm;
use poulpy_hal::layouts::{
    Backend, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef, CrtWord, HostDataMut, HostDataRef,
    Module, VecZnxBackendRef, VecZnxDftBackendMut, ZnxView, ZnxViewMut, check_degree,
};
use std::mem::size_of;

use crate::neon::ntt4x30_packed::{DOT_CHUNK, Plane, add_mod, dot_rows, dot_rows_pairwise, pack_limb, planes};

/// `u32` per limb of a block.
const ROW: usize = 16;
/// Blocks per parallel task.
const TASK_BLOCKS: usize = 64;

#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);

unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    fn get(self) -> *mut T {
        self.0
    }
}

#[inline(always)]
fn packed_row_offset(size: usize, limb: usize, blk: usize) -> usize {
    (blk * size + limb) * ROW
}

#[inline(always)]
fn col_slice(raw: &[u32], n: usize, size: usize, col: usize) -> &[u32] {
    let stride = 4 * n * size;
    &raw[col * stride..(col + 1) * stride]
}

fn zero_res_limb<BE>(res: &mut VecZnxDftBackendMut<'_, BE>, col: usize, limb: usize)
where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    cast_slice_mut::<_, u32>(res.at_mut(col, limb)).fill(0);
}

/// Expands `rows` rows of a sparse right operand to the four slots of block `blk`.
///
/// Slot `s` of the dense operand reads slot `s >> log_gap` of the sparse one (spec 4.5).
/// `b` is the sparse column, already offset to its first row, with rows `b_size` limbs apart in its own block order.
#[inline(always)]
unsafe fn expand_sparse_rows(dst: *mut u32, b: *const u32, b_size: usize, blk: usize, log_gap: usize, rows: usize) {
    unsafe {
        let slot = (4 * blk) >> log_gap;
        let base = b.add((slot / 4) * b_size * ROW);
        let lane = slot % 4;
        for row in 0..rows {
            for p in 0..4 {
                let src = base.add(row * ROW + 4 * p);
                let v = if log_gap >= 2 {
                    vld1q_dup_u32(src.add(lane))
                } else {
                    let v = vld1q_u32(src);
                    if lane == 0 { vzip1q_u32(v, v) } else { vzip2q_u32(v, v) }
                };
                vst1q_u32(dst.add(row * ROW + 4 * p), v);
            }
        }
    }
}

/// Inner product of `rows` rows of block `blk`, with a possibly sparse right operand.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn conv_dot<const PAIRWISE: bool>(
    c: &[Plane; 4],
    blk: usize,
    a0: *const u32,
    a1: *const u32,
    b0: *const u32,
    b1: *const u32,
    b_size: usize,
    b_log_gap: usize,
    rows: usize,
) -> [core::arch::aarch64::uint32x4_t; 4] {
    unsafe {
        if b_log_gap == 0 {
            let (b0, b1) = (b0.add(blk * b_size * ROW), b1.add(blk * b_size * ROW));
            return if PAIRWISE {
                dot_rows_pairwise(a0, a1, b0, b1, rows, c)
            } else {
                dot_rows(a0, b0, rows, c)
            };
        }
        let mut buf0 = [0u32; ROW * DOT_CHUNK];
        let mut buf1 = [0u32; ROW * DOT_CHUNK];
        let mut out = [core::arch::aarch64::vdupq_n_u32(0); 4];
        let mut done = 0;
        while done < rows {
            let len = (rows - done).min(DOT_CHUNK);
            expand_sparse_rows(buf0.as_mut_ptr(), b0.add(done * ROW), b_size, blk, b_log_gap, len);
            let r = if PAIRWISE {
                expand_sparse_rows(buf1.as_mut_ptr(), b1.add(done * ROW), b_size, blk, b_log_gap, len);
                dot_rows_pairwise(a0.add(done * ROW), a1.add(done * ROW), buf0.as_ptr(), buf1.as_ptr(), len, c)
            } else {
                dot_rows(a0.add(done * ROW), buf0.as_ptr(), len, c)
            };
            for p in 0..4 {
                out[p] = if done == 0 { r[p] } else { add_mod(out[p], r[p], c[p].q) };
            }
            done += len;
        }
        out
    }
}

/// Computes block `blk` of every output limb.
///
/// `a0`, `a1` are left columns, `b0`, `b1` right columns, each from its first block.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn conv_block<const ACC: bool, const PAIRWISE: bool>(
    c: &[Plane; 4],
    res: *mut u32,
    res_col: usize,
    n: usize,
    res_cols: usize,
    min_size: usize,
    offset: usize,
    blk: usize,
    a0: *const u32,
    a1: *const u32,
    a_size: usize,
    b0: *const u32,
    b1: *const u32,
    b_size: usize,
    b_log_gap: usize,
) {
    unsafe {
        for k in 0..min_size {
            let k_abs = k + offset;
            let j_min = k_abs.saturating_sub(a_size - 1);
            let j_max = (k_abs + 1).min(b_size);
            let a_off = packed_row_offset(a_size, k_abs + 1 - j_max, blk);
            let b_off = (b_size - j_max) * ROW;
            let r = conv_dot::<PAIRWISE>(
                c,
                blk,
                a0.add(a_off),
                a1.add(a_off),
                b0.add(b_off),
                b1.add(b_off),
                b_size,
                b_log_gap,
                j_max - j_min,
            );
            let dst = res.add((k * res_cols + res_col) * 4 * n + 4 * blk);
            for p in 0..4 {
                let d = dst.add(p * n);
                if ACC {
                    vst1q_u32(d, add_mod(vld1q_u32(d), r[p], c[p].q));
                } else {
                    vst1q_u32(d, r[p]);
                }
            }
        }
    }
}

/// Scratch for one transform and one packed limb.
pub(crate) fn cnv_prepare_tmp_bytes(n: usize) -> usize {
    6 * n * size_of::<u64>()
}

/// Scatters one packed limb into row `limb` of every block of a prepared column.
fn scatter_prepared_limb(dst: &mut [u32], src: &[u32], n: usize, size: usize, limb: usize) {
    for blk in 0..n / 4 {
        let off = packed_row_offset(size, limb, blk);
        for p in 0..4 {
            dst[off + 4 * p..off + 4 * p + 4].copy_from_slice(&src[p * n + 4 * blk..p * n + 4 * blk + 4]);
        }
    }
}

fn zero_prepared_limb(dst: &mut [u32], n: usize, size: usize, limb: usize) {
    for blk in 0..n / 4 {
        let off = packed_row_offset(size, limb, blk);
        dst[off..off + ROW].fill(0);
    }
}

fn prepare<BE, E: TaskExecutor>(
    module: &Module<BE>,
    left: Option<&mut CnvPVecLBackendMut<'_, BE>>,
    right: Option<&mut CnvPVecRBackendMut<'_, BE>>,
    a: &VecZnxBackendRef<'_, BE>,
    tmp: &mut [u64],
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>
        + NttDFTExecute<poulpy_cpu_portable::kernels::ntt4x30::ntt::NttTable<Primes30, <Module<BE> as NttModuleHandle>::Ring>>
        + NttFromZnx64,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: NttModuleHandle,
{
    poulpy_hal::layouts::assert_dense(a, "prepare");
    let (n, cols, size) = if let Some(res) = left.as_ref() {
        (res.n(), res.cols(), res.size())
    } else {
        let res = right.as_ref().unwrap();
        (res.n(), res.cols(), res.size())
    };
    check_degree::<BE>(module.n(), n);
    assert_eq!(a.n(), n, "prepare: a.n():{} != res.n():{n}", a.n());
    assert_eq!(a.cols(), cols, "a.cols():{} != res.cols():{cols}", a.cols());
    assert!(n.is_multiple_of(4));
    if let (Some(l), Some(r)) = (left.as_ref(), right.as_ref()) {
        assert_eq!(r.n(), n, "prepare: right.n():{} != left.n():{n}", r.n());
        assert_eq!(r.cols(), l.cols(), "right.cols():{} != left.cols():{}", r.cols(), l.cols());
        assert_eq!(r.size(), l.size(), "right.size():{} != left.size():{}", r.size(), l.size());
    }
    let table = module.get_ntt_table_for(n);
    let min_size = size.min(a.size());
    let stride = 4 * n * size;
    let left_ptr = left.map(|res| SendPtr(cast_slice_mut::<_, u32>(res.raw_mut()).as_mut_ptr()));
    let right_ptr = right.map(|res| SendPtr(cast_slice_mut::<_, u32>(res.raw_mut()).as_mut_ptr()));
    // Tasks write distinct limbs of distinct columns.
    E::for_each_chunked(cols * size, tmp, 6 * n, |tmp, task| {
        let col = task / size;
        let limb = task % size;
        let dst_l = left_ptr.map(|ptr| unsafe { std::slice::from_raw_parts_mut(ptr.get().add(col * stride), stride) });
        let dst_r = right_ptr.map(|ptr| unsafe { std::slice::from_raw_parts_mut(ptr.get().add(col * stride), stride) });
        if limb < min_size {
            let (tmp_b, tmp_packed) = tmp.split_at_mut(4 * n);
            let tmp_packed: &mut [u32] = &mut cast_slice_mut(tmp_packed)[..4 * n];
            BE::ntt_from_znx64(tmp_b, a.at(col, limb));
            BE::ntt_dft_execute(table, tmp_b);
            if let Some(dst) = dst_l {
                pack_limb(n, tmp_packed, tmp_b, false);
                scatter_prepared_limb(dst, tmp_packed, n, size, limb);
            }
            if let Some(dst) = dst_r {
                pack_limb(n, tmp_packed, tmp_b, true);
                scatter_prepared_limb(dst, tmp_packed, n, size, size - 1 - limb);
            }
        } else {
            if let Some(dst) = dst_l {
                zero_prepared_limb(dst, n, size, limb);
            }
            if let Some(dst) = dst_r {
                zero_prepared_limb(dst, n, size, size - 1 - limb);
            }
        }
    });
}

pub(crate) fn cnv_prepare_left<BE, E: TaskExecutor>(
    module: &Module<BE>,
    res: &mut CnvPVecLBackendMut<'_, BE>,
    a: &VecZnxBackendRef<'_, BE>,
    tmp: &mut [u64],
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>
        + NttDFTExecute<poulpy_cpu_portable::kernels::ntt4x30::ntt::NttTable<Primes30, <Module<BE> as NttModuleHandle>::Ring>>
        + NttFromZnx64,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: NttModuleHandle,
{
    prepare::<BE, E>(module, Some(res), None, a, tmp);
}

pub(crate) fn cnv_prepare_right<BE, E: TaskExecutor>(
    module: &Module<BE>,
    res: &mut CnvPVecRBackendMut<'_, BE>,
    a: &VecZnxBackendRef<'_, BE>,
    tmp: &mut [u64],
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>
        + NttDFTExecute<poulpy_cpu_portable::kernels::ntt4x30::ntt::NttTable<Primes30, <Module<BE> as NttModuleHandle>::Ring>>
        + NttFromZnx64,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: NttModuleHandle,
{
    prepare::<BE, E>(module, None, Some(res), a, tmp);
}

pub(crate) fn cnv_prepare_self<BE, E: TaskExecutor>(
    module: &Module<BE>,
    left: &mut CnvPVecLBackendMut<'_, BE>,
    right: &mut CnvPVecRBackendMut<'_, BE>,
    a: &VecZnxBackendRef<'_, BE>,
    tmp: &mut [u64],
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>
        + NttDFTExecute<poulpy_cpu_portable::kernels::ntt4x30::ntt::NttTable<Primes30, <Module<BE> as NttModuleHandle>::Ring>>
        + NttFromZnx64,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: NttModuleHandle,
{
    prepare::<BE, E>(module, Some(left), Some(right), a, tmp);
}

pub(crate) fn cnv_apply_dft_tmp_bytes(_res_size: usize, _a_size: usize, _b_size: usize) -> usize {
    0
}

#[allow(clippy::too_many_arguments)]
unsafe fn apply<BE, E: TaskExecutor, const ACC: bool, const PAIRWISE: bool>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, BE>,
    a0_col: usize,
    a1_col: usize,
    b: &CnvPVecRBackendRef<'_, BE>,
    b0_col: usize,
    b1_col: usize,
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: NttModuleHandle,
{
    let (n, res_size, a_size, b_size) = (res.n(), res.size(), a.size(), b.size());
    check_degree::<BE>(module.n(), n);
    assert_eq!(a.n(), n, "a.n():{} != res.n():{n}", a.n());
    assert!(n.is_multiple_of(4));
    let b_log_gap = sparse_log_gap_portable(n, b.n());
    if res_size == 0 || a_size == 0 || b_size == 0 {
        if !ACC {
            for limb in 0..res_size {
                zero_res_limb(res, res_col, limb);
            }
        }
        return;
    }
    let bound = a_size + b_size - 1;
    let offset = cnv_offset.min(bound);
    let min_size = res_size.min((bound + 1).saturating_sub(offset));
    let a_raw: &[u32] = cast_slice(a.raw());
    let b_raw: &[u32] = cast_slice(b.raw());
    let a0 = col_slice(a_raw, n, a_size, a0_col);
    let a1 = col_slice(a_raw, n, a_size, a1_col);
    let b0 = col_slice(b_raw, b.n(), b_size, b0_col);
    let b1 = col_slice(b_raw, b.n(), b_size, b1_col);
    let res_cols = res.cols();
    let res_u32 = cast_slice_mut::<_, u32>(res.raw_mut());
    assert!(res_u32.len() >= min_size * res_cols * 4 * n);
    let res_ptr = SendPtr(res_u32.as_mut_ptr());
    let n_blocks = n / 4;
    E::for_each(n_blocks.div_ceil(TASK_BLOCKS), |task| unsafe {
        let c = planes();
        for blk in task * TASK_BLOCKS..((task + 1) * TASK_BLOCKS).min(n_blocks) {
            conv_block::<ACC, PAIRWISE>(
                &c,
                res_ptr.get(),
                res_col,
                n,
                res_cols,
                min_size,
                offset,
                blk,
                a0.as_ptr(),
                a1.as_ptr(),
                a_size,
                b0.as_ptr(),
                b1.as_ptr(),
                b_size,
                b_log_gap,
            )
        }
    });
    if !ACC {
        for limb in min_size..res_size {
            zero_res_limb(res, res_col, limb);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn cnv_apply_dft<BE, E: TaskExecutor>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, BE>,
    a_col: usize,
    b: &CnvPVecRBackendRef<'_, BE>,
    b_col: usize,
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: NttModuleHandle,
{
    unsafe { apply::<BE, E, false, false>(module, cnv_offset, res, res_col, a, a_col, a_col, b, b_col, b_col) };
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn cnv_apply_dft_add<BE, E: TaskExecutor>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, BE>,
    a_col: usize,
    b: &CnvPVecRBackendRef<'_, BE>,
    b_col: usize,
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: NttModuleHandle,
{
    unsafe { apply::<BE, E, true, false>(module, cnv_offset, res, res_col, a, a_col, a_col, b, b_col, b_col) };
}

#[cfg(feature = "enable-rayon")]
pub(crate) fn cnv_apply_dft_sum_neon_tmp_bytes(_res_size: usize) -> usize {
    0
}

#[cfg(feature = "enable-rayon")]
pub(crate) unsafe fn cnv_apply_dft_sum_neon<BE, E: TaskExecutor>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    terms: &[CnvDftAccTerm<'_, BE>],
    _tmp: &mut [u8],
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: NttModuleHandle,
{
    if terms.is_empty() {
        for limb in 0..res.size() {
            zero_res_limb(res, res_col, limb);
        }
        return;
    }

    for (index, term) in terms.iter().enumerate() {
        if index == 0 {
            unsafe {
                apply::<BE, E, false, false>(
                    module, cnv_offset, res, res_col, &term.a, term.a_col, term.a_col, &term.b, term.b_col, term.b_col,
                )
            };
        } else {
            unsafe {
                apply::<BE, E, true, false>(
                    module, cnv_offset, res, res_col, &term.a, term.a_col, term.a_col, &term.b, term.b_col, term.b_col,
                )
            };
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn cnv_pairwise_apply_dft<BE, E: TaskExecutor>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, BE>,
    b: &CnvPVecRBackendRef<'_, BE>,
    i: usize,
    j: usize,
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: NttModuleHandle,
{
    if i == j {
        unsafe { apply::<BE, E, false, false>(module, cnv_offset, res, res_col, a, i, i, b, i, i) };
    } else {
        unsafe { apply::<BE, E, false, true>(module, cnv_offset, res, res_col, a, i, j, b, i, j) };
    }
}

pub(crate) fn cnv_by_const_apply_tmp_bytes(_res_size: usize, _a_size: usize, _b_size: usize) -> usize {
    0
}
