//! Convolution for the packed NTT4x30 layout.
//!
//! A prepared column is ordered `block -> limb`, where a block is four consecutive coefficients.
//! One limb of a block is 16 `u32`: four lanes for each of the four primes.
//! Both operands hold residues centered around zero, which lets twice as many products share one Montgomery step.
//! The right operand is also multiplied by `2^32`, and has its limbs in reverse order, so both operands of one output limb are read forward.

use bytemuck::{cast_slice, cast_slice_mut};
use poulpy_hal::execution::TaskExecutor;
use poulpy_hal::layouts::CnvDftAccTerm;
use poulpy_hal::layouts::{
    Backend, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef, CrtWord, HostDataMut, HostDataRef,
    Module, VecZnxBackendRef, VecZnxDftBackendMut, ZnxView, ZnxViewMut, check_degree,
};
use std::mem::size_of;

use super::packed::{DotState, ROW, SendPtr, limb_to_prepared, scatter_centered_limb, stage_flush, stage_store};
use super::vec_znx_dft::prepare_tmp_words;
use crate::kernels::ntt4x30::primes::Primes30;
use crate::kernels::sparse_log_gap_portable;

/// Blocks per parallel task.
const TASK_BLOCKS: usize = 64;

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

/// One convolution `a (x) b` of a sum, or the pairwise product `(a0 + a1) (x) (b0 + b1)`.
#[derive(Clone, Copy)]
struct Term<'a> {
    /// Left and right columns. `a1` and `b1` are only read by a pairwise product.
    a0: &'a [u32],
    a1: &'a [u32],
    b0: &'a [u32],
    b1: &'a [u32],
    a_size: usize,
    b_size: usize,
    /// Slot `s` of the left operand reads slot `s >> b_log_gap` of the right one.
    b_log_gap: usize,
    offset: usize,
    /// Output limbs the term contributes to.
    limbs: usize,
}

impl<'a> Term<'a> {
    const EMPTY: Self = Self {
        a0: &[],
        a1: &[],
        b0: &[],
        b1: &[],
        a_size: 0,
        b_size: 0,
        b_log_gap: 0,
        offset: 0,
        limbs: 0,
    };

    /// The term over `n` coefficients for a result of `res_size` limbs, or the empty term when an operand has no limb.
    #[allow(clippy::too_many_arguments)]
    fn new<BE>(
        n: usize,
        res_size: usize,
        cnv_offset: usize,
        a: &'a CnvPVecLBackendRef<'_, BE>,
        a_cols: (usize, usize),
        b: &'a CnvPVecRBackendRef<'_, BE>,
        b_cols: (usize, usize),
    ) -> Self
    where
        BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        let (a_size, b_size) = (a.size(), b.size());
        assert_eq!(a.n(), n, "a.n():{} != res.n():{n}", a.n());
        let b_log_gap = sparse_log_gap_portable(n, b.n());
        if a_size == 0 || b_size == 0 {
            return Self::EMPTY;
        }
        let bound = a_size + b_size - 1;
        let offset = cnv_offset.min(bound);
        let a_raw: &[u32] = cast_slice(a.raw());
        let b_raw: &[u32] = cast_slice(b.raw());
        Self {
            a0: col_slice(a_raw, n, a_size, a_cols.0),
            a1: col_slice(a_raw, n, a_size, a_cols.1),
            b0: col_slice(b_raw, b.n(), b_size, b_cols.0),
            b1: col_slice(b_raw, b.n(), b_size, b_cols.1),
            a_size,
            b_size,
            b_log_gap,
            offset,
            limbs: res_size.min((bound + 1).saturating_sub(offset)),
        }
    }

    /// Adds the products of output limb `k` of block `blk` to `state`.
    #[inline(always)]
    fn push<const PAIRWISE: bool>(&self, state: DotState, blk: usize, k: usize) -> DotState {
        let k_abs = k + self.offset;
        let j_min = k_abs.saturating_sub(self.a_size - 1);
        let j_max = (k_abs + 1).min(self.b_size);
        let rows = j_max - j_min;
        let a_off = packed_row_offset(self.a_size, k_abs + 1 - j_max, blk);
        let b_row = self.b_size - j_max;
        if !PAIRWISE && self.b_log_gap == 0 {
            let b_off = packed_row_offset(self.b_size, b_row, blk);
            return state.push_rows(&self.a0[a_off..][..rows * ROW], &self.b0[b_off..]);
        }
        // Lane `l` reads the lane of the right operand its slot maps to, in the block order of that operand.
        let lane: [usize; 4] = std::array::from_fn(|l| {
            let slot = (4 * blk + l) >> self.b_log_gap;
            packed_row_offset(self.b_size, b_row, slot / 4) + slot % 4
        });
        let (a0, a1) = (&self.a0[a_off..][..rows * ROW], &self.a1[if PAIRWISE { a_off } else { 0 }..]);
        // The sum of two centered residues is twice as large, so a pairwise product weighs four plain ones.
        state.push_with(rows, if PAIRWISE { 4 } else { 1 }, |i| {
            let mut x: [u32; ROW] = a0[i * ROW..][..ROW].try_into().unwrap();
            let mut m = [0u32; ROW];
            for p in 0..4 {
                for l in 0..4 {
                    m[4 * p + l] = self.b0[lane[l] + i * ROW + 4 * p];
                }
            }
            if PAIRWISE {
                for (x, &y) in x.iter_mut().zip(&a1[i * ROW..][..ROW]) {
                    *x = x.wrapping_add(y);
                }
                for p in 0..4 {
                    for l in 0..4 {
                        m[4 * p + l] = m[4 * p + l].wrapping_add(self.b1[lane[l] + i * ROW + 4 * p]);
                    }
                }
            }
            (x, m)
        })
    }
}

/// Blocks whose outputs are staged together before they reach the result.
///
/// The planes of the output limbs are `4 n` bytes apart, so their write streams fall in the same cache sets.
/// Storing one block at a time evicts every line before its four blocks are written.
/// A run fills whole lines, which are then written once.
const RUN: usize = 32;

/// Output limbs staged together, [`RUN`] blocks of four planes each.
const STAGE_LIMBS: usize = 32;

/// Words of the stage of one output limb: plane `p` holds its run at `p * RUN * 4`.
const STAGE_LIMB: usize = ROW * RUN;

/// Words of scratch of the apply kernels, per worker: the stage.
pub fn apply_tmp_words(res_size: usize) -> usize {
    res_size.clamp(1, STAGE_LIMBS) * STAGE_LIMB
}

/// `res[res_col]` receives the sum of `terms` over its first `limbs` limbs.
///
/// Limbs below `add_below` receive the sum of their content and the terms, the others are overwritten.
#[allow(clippy::too_many_arguments)]
fn apply_terms<BE, E: TaskExecutor, const PAIRWISE: bool>(
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    terms: &[Term<'_>],
    limbs: usize,
    add_below: usize,
    tmp: &mut [u32],
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    if limbs == 0 {
        return;
    }
    let (n, res_cols) = (res.n(), res.cols());
    let n_blocks = n / 4;
    let res_u32 = cast_slice_mut::<_, u32>(res.raw_mut());
    assert!(res_u32.len() >= limbs * res_cols * 4 * n);
    let res_ptr = SendPtr(res_u32.as_mut_ptr());
    let per_worker = apply_tmp_words(limbs);
    E::for_each_chunked(n_blocks.div_ceil(TASK_BLOCKS), tmp, per_worker, |stage, task| {
        let end = ((task + 1) * TASK_BLOCKS).min(n_blocks);
        for blk in (task * TASK_BLOCKS..end).step_by(RUN) {
            let len = RUN.min(end - blk);
            for first in (0..limbs).step_by(STAGE_LIMBS) {
                let count = (limbs - first).min(STAGE_LIMBS);
                for b in 0..len {
                    for (k, stage) in stage.chunks_exact_mut(STAGE_LIMB).take(count).enumerate() {
                        let mut state = DotState::new();
                        for term in terms {
                            if first + k < term.limbs {
                                state = term.push::<PAIRWISE>(state, blk + b, first + k);
                            }
                        }
                        stage_store(stage, RUN, b, &state.finish());
                    }
                }
                for (k, stage) in stage.chunks_exact(STAGE_LIMB).take(count).enumerate() {
                    let limb = ((first + k) * res_cols + res_col) * 4 * n;
                    // SAFETY: the limb lies within `res_u32`, and this task is the only one on these blocks.
                    unsafe { stage_flush(n, res_ptr.get().add(limb), stage, RUN, blk, len, first + k < add_below) };
                }
            }
        }
    });
}

/// Scratch for one packed limb, per worker.
pub fn cnv_prepare_tmp_bytes(n: usize) -> usize {
    prepare_tmp_words(n) * size_of::<u64>()
}

fn zero_prepared_limb(dst: &mut [u32], n: usize, size: usize, limb: usize) {
    for blk in 0..n / 4 {
        let off = packed_row_offset(size, limb, blk);
        dst[off..off + ROW].fill(0);
    }
}

/// Prepares `a` into `left`, `right`, or both.
///
/// `dft(n, dst, src, prepared)` is the forward transform of `src` into the packed limb `dst`,
/// multiplied by `2^32` when `prepared` is set.
fn prepare<BE, E: TaskExecutor>(
    module: &Module<BE>,
    left: Option<&mut CnvPVecLBackendMut<'_, BE>>,
    right: Option<&mut CnvPVecRBackendMut<'_, BE>>,
    a: &VecZnxBackendRef<'_, BE>,
    tmp: &mut [u64],
    dft: impl Fn(usize, &mut [u32], &[i64], bool) + Send + Sync,
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
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
            dft(n, tmp_packed, a.at(col, limb), dst_l.is_none());
            if let Some(dst) = dst_l {
                scatter_centered_limb(n, dst, tmp_packed, |blk| packed_row_offset(size, limb, blk));
                if dst_r.is_some() {
                    limb_to_prepared(n, tmp_packed);
                }
            }
            if let Some(dst) = dst_r {
                scatter_centered_limb(n, dst, tmp_packed, |blk| packed_row_offset(size, size - 1 - limb, blk));
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

pub fn cnv_prepare_left<BE, E: TaskExecutor>(
    module: &Module<BE>,
    res: &mut CnvPVecLBackendMut<'_, BE>,
    a: &VecZnxBackendRef<'_, BE>,
    tmp: &mut [u64],
    dft: impl Fn(usize, &mut [u32], &[i64], bool) + Send + Sync,
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    prepare::<BE, E>(module, Some(res), None, a, tmp, dft);
}

pub fn cnv_prepare_right<BE, E: TaskExecutor>(
    module: &Module<BE>,
    res: &mut CnvPVecRBackendMut<'_, BE>,
    a: &VecZnxBackendRef<'_, BE>,
    tmp: &mut [u64],
    dft: impl Fn(usize, &mut [u32], &[i64], bool) + Send + Sync,
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    prepare::<BE, E>(module, None, Some(res), a, tmp, dft);
}

pub fn cnv_prepare_self<BE, E: TaskExecutor>(
    module: &Module<BE>,
    left: &mut CnvPVecLBackendMut<'_, BE>,
    right: &mut CnvPVecRBackendMut<'_, BE>,
    a: &VecZnxBackendRef<'_, BE>,
    tmp: &mut [u64],
    dft: impl Fn(usize, &mut [u32], &[i64], bool) + Send + Sync,
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    prepare::<BE, E>(module, Some(left), Some(right), a, tmp, dft);
}

/// Scratch space (in bytes) of the apply kernels, per worker.
pub(crate) fn cnv_apply_dft_tmp_bytes(res_size: usize) -> usize {
    apply_tmp_words(res_size) * size_of::<u32>()
}

#[allow(clippy::too_many_arguments)]
fn apply<BE, E: TaskExecutor, const ACC: bool, const PAIRWISE: bool>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, BE>,
    a_cols: (usize, usize),
    b: &CnvPVecRBackendRef<'_, BE>,
    b_cols: (usize, usize),
    tmp: &mut [u32],
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let (n, res_size) = (res.n(), res.size());
    check_degree::<BE>(module.n(), n);
    let term = Term::new(n, res_size, cnv_offset, a, a_cols, b, b_cols);
    apply_terms::<BE, E, PAIRWISE>(res, res_col, &[term], term.limbs, if ACC { term.limbs } else { 0 }, tmp);
    if !ACC {
        for limb in term.limbs..res_size {
            zero_res_limb(res, res_col, limb);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn cnv_apply_dft<BE, E: TaskExecutor>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, BE>,
    a_col: usize,
    b: &CnvPVecRBackendRef<'_, BE>,
    b_col: usize,
    tmp: &mut [u32],
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    apply::<BE, E, false, false>(module, cnv_offset, res, res_col, a, (a_col, a_col), b, (b_col, b_col), tmp);
}

#[allow(clippy::too_many_arguments)]
pub fn cnv_apply_dft_add<BE, E: TaskExecutor>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, BE>,
    a_col: usize,
    b: &CnvPVecRBackendRef<'_, BE>,
    b_col: usize,
    tmp: &mut [u32],
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    apply::<BE, E, true, false>(module, cnv_offset, res, res_col, a, (a_col, a_col), b, (b_col, b_col), tmp);
}

/// Terms fused in one pass of [`cnv_apply_dft_sum`].
const SUM_TERMS: usize = 16;

/// `res[res_col] = sum_t a_t (x) b_t`, the terms of each group of `SUM_TERMS` accumulated in one pass.
///
/// An output limb is reduced and stored once per group, where a per-term loop reduces, reads and writes it for every term.
pub fn cnv_apply_dft_sum<BE, E: TaskExecutor>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    terms: &[CnvDftAccTerm<'_, BE>],
    tmp: &mut [u32],
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let (n, res_size) = (res.n(), res.size());
    check_degree::<BE>(module.n(), n);
    // Limbs written so far: a group overwrites the limbs no earlier group reached and adds to the others.
    let mut written = 0;
    for group in terms.chunks(SUM_TERMS) {
        let mut sum_terms = [Term::EMPTY; SUM_TERMS];
        let mut limbs = 0;
        for (dst, term) in sum_terms.iter_mut().zip(group) {
            *dst = Term::new(
                n,
                res_size,
                cnv_offset,
                &term.a,
                (term.a_col, term.a_col),
                &term.b,
                (term.b_col, term.b_col),
            );
            limbs = limbs.max(dst.limbs);
        }
        apply_terms::<BE, E, false>(res, res_col, &sum_terms[..group.len()], limbs, written, tmp);
        written = written.max(limbs);
    }
    for limb in written..res_size {
        zero_res_limb(res, res_col, limb);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn cnv_pairwise_apply_dft<BE, E: TaskExecutor>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, BE>,
    b: &CnvPVecRBackendRef<'_, BE>,
    i: usize,
    j: usize,
    tmp: &mut [u32],
) where
    BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    if i == j {
        apply::<BE, E, false, false>(module, cnv_offset, res, res_col, a, (i, i), b, (i, i), tmp);
    } else {
        apply::<BE, E, false, true>(module, cnv_offset, res, res_col, a, (i, j), b, (i, j), tmp);
    }
}
