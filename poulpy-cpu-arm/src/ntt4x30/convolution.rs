//! Convolution for the packed NTT4x30 layout.
//!
//! A prepared column is ordered `block -> limb`, where a block is four consecutive coefficients.
//! One limb of a block is 16 `u32`: four lanes for each of the four primes.
//! Both operands hold residues centered around zero, which lets twice as many products share one Montgomery step.
//! The right operand is also multiplied by `2^32`, and has its limbs in reverse order, so both operands of one output limb are read forward.

use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::aarch64::{
    int64x2_t, uint32x4_t, vaddq_u32, vdupq_n_s64, vdupq_n_u32, vld1q_dup_u32, vld1q_u32, vst1q_u32, vzip1q_u32, vzip2q_u32,
};
use poulpy_cpu_portable::kernels::ntt4x30::{primes::Primes30, vec_znx_dft::NttModuleHandle};
use poulpy_cpu_portable::kernels::sparse_log_gap_portable;
use poulpy_hal::execution::TaskExecutor;
use poulpy_hal::layouts::CnvDftAccTerm;
use poulpy_hal::layouts::{
    Backend, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef, CrtWord, HostDataMut, HostDataRef,
    Module, VecZnxBackendRef, VecZnxDftBackendMut, ZnxView, ZnxViewMut, check_degree,
};
use std::mem::size_of;

use super::vec_znx_dft::{PackedDft, prepare_tmp_words};
use crate::neon::ntt4x30_packed::{
    DOT_CHUNK, DOT_SHORT, Plane, add_mod, center, limb_to_prepared, mla_centered, planes, redc_acc,
};

/// `u32` per limb of a block.
const ROW: usize = 16;
/// Blocks per parallel task.
const TASK_BLOCKS: usize = 64;

#[derive(Clone, Copy)]
pub(super) struct SendPtr<T>(pub(super) *mut T);

unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    pub(super) fn get(self) -> *mut T {
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

/// Right operand read as stored.
const DENSE: u8 = 0;
/// Right operand slot shared by the four slots of the block: one lane, repeated.
const SPARSE_ONE: u8 = 1;
/// Right operand slots shared by pairs of slots of the block: the two low lanes, each repeated.
const SPARSE_LOW: u8 = 2;
/// As [`SPARSE_LOW`], on the two high lanes.
const SPARSE_HIGH: u8 = 3;

#[inline(always)]
unsafe fn load_right<const MODE: u8>(p: *const u32) -> uint32x4_t {
    unsafe {
        match MODE {
            DENSE => vld1q_u32(p),
            SPARSE_ONE => vld1q_dup_u32(p),
            SPARSE_LOW => {
                let v = vld1q_u32(p);
                vzip1q_u32(v, v)
            }
            _ => {
                let v = vld1q_u32(p);
                vzip2q_u32(v, v)
            }
        }
    }
}

/// Running inner product of one output limb of one block, fed one run of rows at a time.
///
/// The state is passed and returned by value so that it stays in registers.
#[derive(Clone, Copy)]
struct Acc {
    lo: [int64x2_t; 4],
    hi: [int64x2_t; 4],
    out: [uint32x4_t; 4],
    count: usize,
    flushed: bool,
}

impl Acc {
    #[inline(always)]
    unsafe fn new() -> Self {
        unsafe {
            let zero = vdupq_n_s64(0);
            Self {
                lo: [zero; 4],
                hi: [zero; 4],
                out: [vdupq_n_u32(0); 4],
                count: 0,
                flushed: false,
            }
        }
    }

    #[inline(always)]
    unsafe fn flush(self, c: &[Plane; 4]) -> Self {
        unsafe {
            // `count` is in units of products of centered residues: a pairwise product weighs four.
            let short = self.count <= DOT_SHORT;
            let mut out = self.out;
            for (p, c) in c.iter().enumerate() {
                let r = redc_acc(self.lo[p], self.hi[p], c, short);
                out[p] = if self.flushed { add_mod(out[p], r, c.q) } else { r };
            }
            let zero = vdupq_n_s64(0);
            Self {
                lo: [zero; 4],
                hi: [zero; 4],
                out,
                count: 0,
                flushed: true,
            }
        }
    }

    /// Adds the inner product of `rows` rows, the right operand read through `MODE`.
    #[inline(always)]
    unsafe fn push_rows<const PAIRWISE: bool, const MODE: u8>(
        mut self,
        c: &[Plane; 4],
        a0: *const u32,
        a1: *const u32,
        b0: *const u32,
        b1: *const u32,
        rows: usize,
    ) -> Self {
        unsafe {
            let mut row = 0;
            while row < rows {
                // The sum of two centered residues is twice as large, so a pairwise product weighs four plain ones.
                let weight = if PAIRWISE { 4 } else { 1 };
                if self.count + weight > DOT_CHUNK {
                    self = self.flush(c);
                }
                let end = row + (rows - row).min((DOT_CHUNK - self.count) / weight);
                self.count += weight * (end - row);
                let (mut lo, mut hi) = (self.lo, self.hi);
                while row < end {
                    for p in 0..4 {
                        let o = ROW * row + 4 * p;
                        let mut xv = vld1q_u32(a0.add(o));
                        let mut mv = load_right::<MODE>(b0.add(o));
                        if PAIRWISE {
                            xv = vaddq_u32(xv, vld1q_u32(a1.add(o)));
                            mv = vaddq_u32(mv, load_right::<MODE>(b1.add(o)));
                        }
                        mla_centered(&mut lo[p], &mut hi[p], xv, mv);
                    }
                    row += 1;
                }
                self.lo = lo;
                self.hi = hi;
            }
            self
        }
    }

    /// Adds `rows` rows of block `blk`, with a possibly sparse right operand.
    ///
    /// Slot `s` of the dense operand reads slot `s >> b_log_gap` of the sparse one (spec 4.5).
    /// `b0` and `b1` are right columns offset to their first row, with rows `b_size` limbs apart in their own block order.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn push_block<const PAIRWISE: bool>(
        self,
        c: &[Plane; 4],
        blk: usize,
        a0: *const u32,
        a1: *const u32,
        b0: *const u32,
        b1: *const u32,
        b_size: usize,
        b_log_gap: usize,
        rows: usize,
    ) -> Self {
        unsafe {
            let slot = (4 * blk) >> b_log_gap;
            let block = (slot / 4) * b_size * ROW;
            let (b0, b1) = (b0.add(block), b1.add(block));
            match b_log_gap {
                0 => self.push_rows::<PAIRWISE, DENSE>(c, a0, a1, b0, b1, rows),
                1 if slot.is_multiple_of(4) => self.push_rows::<PAIRWISE, SPARSE_LOW>(c, a0, a1, b0, b1, rows),
                1 => self.push_rows::<PAIRWISE, SPARSE_HIGH>(c, a0, a1, b0, b1, rows),
                _ => self.push_rows::<PAIRWISE, SPARSE_ONE>(c, a0, a1, b0.add(slot % 4), b1.add(slot % 4), rows),
            }
        }
    }

    /// One canonical vector per prime.
    #[inline(always)]
    unsafe fn finish(self, c: &[Plane; 4]) -> [uint32x4_t; 4] {
        unsafe { if self.count != 0 { self.flush(c).out } else { self.out } }
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
            let acc = Acc::new().push_block::<PAIRWISE>(
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
            let r = acc.finish(c);
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

/// Scratch for one packed limb, per worker.
pub(crate) fn cnv_prepare_tmp_bytes(n: usize) -> usize {
    prepare_tmp_words(n) * size_of::<u64>()
}

/// Scatters one packed limb of canonical residues into row `limb` of every block of a prepared column, centered.
///
/// `src` keeps its canonical residues.
fn scatter_centered_limb(dst: &mut [u32], src: &[u32], n: usize, size: usize, limb: usize) {
    assert!(src.len() >= 4 * n);
    assert!(dst.len() >= packed_row_offset(size, limb, n / 4 - 1) + ROW);
    unsafe {
        let c = planes();
        for blk in 0..n / 4 {
            let row = dst.as_mut_ptr().add(packed_row_offset(size, limb, blk));
            for (p, c) in c.iter().enumerate() {
                vst1q_u32(row.add(4 * p), center(vld1q_u32(src.as_ptr().add(p * n + 4 * blk)), c));
            }
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
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: NttModuleHandle + PackedDft,
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
    let min_size = size.min(a.size());
    let stride = 4 * n * size;
    let left_ptr = left.map(|res| SendPtr(cast_slice_mut::<_, u32>(res.raw_mut()).as_mut_ptr()));
    let right_ptr = right.map(|res| SendPtr(cast_slice_mut::<_, u32>(res.raw_mut()).as_mut_ptr()));
    // Tasks write distinct limbs of distinct columns.
    E::for_each_chunked(cols * size, tmp, prepare_tmp_words(n), |tmp, task| {
        let col = task / size;
        let limb = task % size;
        let dst_l = left_ptr.map(|ptr| unsafe { std::slice::from_raw_parts_mut(ptr.get().add(col * stride), stride) });
        let dst_r = right_ptr.map(|ptr| unsafe { std::slice::from_raw_parts_mut(ptr.get().add(col * stride), stride) });
        if limb < min_size {
            let tmp_packed: &mut [u32] = &mut cast_slice_mut(tmp)[..4 * n];
            module.packed_dft_limb(n, tmp_packed, a.at(col, limb), dst_l.is_none());
            if let Some(dst) = dst_l {
                scatter_centered_limb(dst, tmp_packed, n, size, limb);
                if dst_r.is_some() {
                    limb_to_prepared(n, tmp_packed);
                }
            }
            if let Some(dst) = dst_r {
                scatter_centered_limb(dst, tmp_packed, n, size, size - 1 - limb);
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
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: NttModuleHandle + PackedDft,
{
    prepare::<BE, E>(module, Some(res), None, a, tmp);
}

pub(crate) fn cnv_prepare_right<BE, E: TaskExecutor>(
    module: &Module<BE>,
    res: &mut CnvPVecRBackendMut<'_, BE>,
    a: &VecZnxBackendRef<'_, BE>,
    tmp: &mut [u64],
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: NttModuleHandle + PackedDft,
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
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: NttModuleHandle + PackedDft,
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

pub(crate) fn cnv_apply_dft_sum_neon_tmp_bytes(_res_size: usize) -> usize {
    0
}

/// Terms fused in one pass of [`cnv_apply_dft_sum_neon`].
const SUM_TERMS: usize = 16;

/// One term of a fused sum of convolutions.
#[derive(Clone, Copy)]
struct SumTerm {
    /// Left and right columns, from their first block.
    a: *const u32,
    b: *const u32,
    a_size: usize,
    b_size: usize,
    b_log_gap: usize,
    offset: usize,
    /// Output limbs the term contributes to.
    limbs: usize,
}

// The columns are only read, and outlive the tasks that share them.
unsafe impl Send for SumTerm {}
unsafe impl Sync for SumTerm {}

/// `res[res_col] = sum_t a_t (x) b_t`, the terms of each group of `SUM_TERMS` accumulated in one pass.
///
/// An output limb is reduced and stored once per group, where the per-term fallback reduces, reads and writes it for every term.
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
    let (n, res_size, res_cols) = (res.n(), res.size(), res.cols());
    check_degree::<BE>(module.n(), n);
    assert!(n.is_multiple_of(4));
    let n_blocks = n / 4;
    // Limbs written so far: a group overwrites the limbs no earlier group reached and adds to the others.
    let mut written = 0;
    for group in terms.chunks(SUM_TERMS) {
        let mut sum_terms = [SumTerm {
            a: std::ptr::null(),
            b: std::ptr::null(),
            a_size: 0,
            b_size: 0,
            b_log_gap: 0,
            offset: 0,
            limbs: 0,
        }; SUM_TERMS];
        let mut limbs = 0;
        for (dst, term) in sum_terms.iter_mut().zip(group) {
            let (a_size, b_size) = (term.a.size(), term.b.size());
            assert_eq!(term.a.n(), n, "a.n():{} != res.n():{n}", term.a.n());
            let b_log_gap = sparse_log_gap_portable(n, term.b.n());
            if a_size == 0 || b_size == 0 {
                continue;
            }
            let bound = a_size + b_size - 1;
            let offset = cnv_offset.min(bound);
            let a_raw: &[u32] = cast_slice(term.a.raw());
            let b_raw: &[u32] = cast_slice(term.b.raw());
            *dst = SumTerm {
                a: col_slice(a_raw, n, a_size, term.a_col).as_ptr(),
                b: col_slice(b_raw, term.b.n(), b_size, term.b_col).as_ptr(),
                a_size,
                b_size,
                b_log_gap,
                offset,
                limbs: res_size.min((bound + 1).saturating_sub(offset)),
            };
            limbs = limbs.max(dst.limbs);
        }
        let sum_terms = &sum_terms[..group.len()];
        let res_u32 = cast_slice_mut::<_, u32>(res.raw_mut());
        assert!(res_u32.len() >= res_size * res_cols * 4 * n);
        let res_ptr = SendPtr(res_u32.as_mut_ptr());
        E::for_each(n_blocks.div_ceil(TASK_BLOCKS), |task| unsafe {
            let c = planes();
            for blk in task * TASK_BLOCKS..((task + 1) * TASK_BLOCKS).min(n_blocks) {
                for k in 0..limbs {
                    let mut acc = Acc::new();
                    for term in sum_terms {
                        if k < term.limbs {
                            let k_abs = k + term.offset;
                            let j_min = k_abs.saturating_sub(term.a_size - 1);
                            let j_max = (k_abs + 1).min(term.b_size);
                            let a = term.a.add(packed_row_offset(term.a_size, k_abs + 1 - j_max, blk));
                            let b = term.b.add((term.b_size - j_max) * ROW);
                            acc = acc.push_block::<false>(&c, blk, a, a, b, b, term.b_size, term.b_log_gap, j_max - j_min);
                        }
                    }
                    let r = acc.finish(&c);
                    let dst = res_ptr.get().add((k * res_cols + res_col) * 4 * n + 4 * blk);
                    for (p, &r) in r.iter().enumerate() {
                        let d = dst.add(p * n);
                        if k < written {
                            vst1q_u32(d, add_mod(vld1q_u32(d), r, c[p].q));
                        } else {
                            vst1q_u32(d, r);
                        }
                    }
                }
            }
        });
        written = written.max(limbs);
    }
    for limb in written..res_size {
        zero_res_limb(res, res_col, limb);
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
