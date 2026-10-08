//! Vector-matrix product for [`NTT4x30Neon`](crate::NTT4x30Neon) on the packed layout.
//!
//! The prepared matrix is ordered `block -> output column -> input row`, where a block is four consecutive coefficients.
//! One row of a block is 16 `u32`: four lanes for each of the four primes, multiplied by `2^32` and centered around zero.
//! The apply path therefore reads the matrix as one contiguous stream, in the order of its inner loop.

use std::mem::size_of;

use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::aarch64::{uint32x4_t, vld1q_u32, vst1q_u32};
use poulpy_cpu_portable::kernels::vmp_select::assert_extractable_portable;

use poulpy_cpu_portable::kernels::ntt4x30::{NttDFTExecute, primes::Primes30};
use poulpy_hal::{
    execution::TaskExecutor,
    layouts::{
        DataView, DataViewMut, MatZnxBackendRef, Module, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendMut,
        VmpPMatBackendRef, ZnxView, ZnxViewMut, check_degree,
    },
};

use crate::NTT4x30Neon;
use crate::neon::ntt4x30_packed::{DotState, Plane, add_mod, center, dot_rows, limb_center, planes};
use crate::ntt4x30::vec_znx_dft::{dft_limb_scaled, prepare_tmp_words};
use poulpy_core::oep::gglwe_product_digit_output_size;
use poulpy_cpu_portable::kernels::ntt4x30::ntt::{NttTable, NttTableInv};
use poulpy_hal::layouts::Ring;

/// `u32` per row of a block.
const ROW: usize = 16;

#[derive(Clone, Copy)]
struct SendU32Ptr(*mut u32);

// Each task writes a distinct block and joins before reuse.
unsafe impl Send for SendU32Ptr {}
unsafe impl Sync for SendU32Ptr {}

impl SendU32Ptr {
    #[inline(always)]
    fn get(&self) -> *mut u32 {
        self.0
    }
}

/// Scratch space (in bytes) required by the VMP prepare kernel.
///
/// Holds one packed limb.
pub(crate) fn vmp_prepare_tmp_bytes_neon(n: usize) -> usize {
    prepare_tmp_words(n) * size_of::<u64>()
}

/// VMP prepare into the block-major prepared layout.
pub(crate) fn vmp_prepare_neon_pm<R: Ring>(
    module: &Module<NTT4x30Neon<R>>,
    res: &mut VmpPMatBackendMut<'_, NTT4x30Neon<R>>,
    a: &MatZnxBackendRef<'_, NTT4x30Neon<R>>,
    tmp: &mut [u64],
) where
    NTT4x30Neon<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
{
    let n = res.n();
    check_degree::<NTT4x30Neon<R>>(module.n(), n);

    assert_eq!(a.n(), n);
    assert_eq!(res.cols_in(), a.cols_in());
    assert_eq!(res.rows(), a.rows());
    assert_eq!(res.cols_out(), a.cols_out());
    assert_eq!(res.size(), a.size());
    assert!(std::mem::size_of_val(tmp) >= vmp_prepare_tmp_bytes_neon(n));
    assert!(n.is_multiple_of(4));

    let nrows = a.cols_in() * a.rows();
    let ncols = a.cols_out() * a.size();
    let n_blocks = n / 4;

    let tmp_packed: &mut [u32] = &mut cast_slice_mut(tmp)[..4 * n];
    let mat_i64: &[i64] = a.raw();
    let pmat: &mut [u32] = cast_slice_mut(res.data_mut());

    for row_i in 0..nrows {
        for col_i in 0..ncols {
            let pos = n * (row_i * ncols + col_i);

            dft_limb_scaled(module, n, tmp_packed, &mat_i64[pos..pos + n], true);
            limb_center(n, tmp_packed);

            for blk in 0..n_blocks {
                let dst = ((blk * ncols + col_i) * nrows + row_i) * ROW;
                for p in 0..4 {
                    pmat[dst + 4 * p..dst + 4 * p + 4].copy_from_slice(&tmp_packed[p * n + 4 * blk..p * n + 4 * blk + 4]);
                }
            }
        }
    }
}

/// Copies rows `first_row + i * row_step` of `a`, truncated to `res.size()` limbs, into rows `i` of `res`.
pub(crate) fn vmp_extract_selected_rows_neon_pm<R: Ring>(
    res: &mut VmpPMatBackendMut<'_, NTT4x30Neon<R>>,
    a: &VmpPMatBackendRef<'_, NTT4x30Neon<R>>,
    first_row: usize,
    row_step: usize,
) {
    assert_extractable_portable(res, a, first_row, row_step);
    let n: usize = a.n();

    let cols_in: usize = a.cols_in();
    let (res_rows, res_nrows, res_ncols) = (res.rows(), res.rows() * cols_in, res.cols_out() * res.size());
    let (a_nrows, a_ncols) = (a.rows() * cols_in, a.cols_out() * a.size());
    // One selected row spans `cols_in` consecutive input rows.
    let span: usize = cols_in * ROW;

    let src: &[u32] = cast_slice(a.raw());
    let dst: &mut [u32] = cast_slice_mut(res.data_mut());
    for blk in 0..n / 4 {
        for col in 0..res_ncols {
            let dst_base: usize = (blk * res_ncols + col) * res_nrows * ROW;
            let src_base: usize = (blk * a_ncols + col) * a_nrows * ROW;
            for i in 0..res_rows {
                let (d, s) = (dst_base + i * span, src_base + (first_row + i * row_step) * span);
                dst[d..d + span].copy_from_slice(&src[s..s + span]);
            }
        }
    }
}

/// Blocks processed together.
///
/// Inputs are gathered and outputs staged in runs of this many blocks.
/// The planes of a vector are `4 n` bytes apart, so their streams fall in the same cache sets.
/// One block at a time touches every plane for 16 bytes and evicts each line before its four blocks are done.
/// A run reads and writes whole lines once.
const GROUP: usize = 32;

/// Outputs staged together, one output being a limb of a column: [`GROUP`] blocks of four planes each.
const STAGE_OUTPUTS: usize = 64;

/// Stage of [`STAGE_OUTPUTS`] outputs over a group of blocks: plane `p` of output `o` holds its run at `(o * 4 + p) * GROUP * 4`.
type Stage = std::mem::MaybeUninit<[u32; STAGE_OUTPUTS * ROW * GROUP]>;

/// Stores the canonical vectors of block `b` of the group for staged output `o`.
#[inline(always)]
unsafe fn stage_store(stage: *mut u32, o: usize, b: usize, r: &[uint32x4_t; 4]) {
    unsafe {
        for (p, &r) in r.iter().enumerate() {
            vst1q_u32(stage.add(((o * 4 + p) * GROUP + b) * 4), r);
        }
    }
}

/// Moves `count` staged outputs of a group of `len` blocks from block `blk` to the outputs `first..` of `res`.
///
/// The outputs are overwritten when `OVERWRITE` is set and accumulated otherwise.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn stage_flush<const OVERWRITE: bool>(
    c: &[Plane; 4],
    stage: *const u32,
    res: *mut u32,
    n: usize,
    first: usize,
    count: usize,
    blk: usize,
    len: usize,
) {
    unsafe {
        for o in 0..count {
            let dst = res.add((first + o) * 4 * n + 4 * blk);
            for (p, c) in c.iter().enumerate() {
                let (d, s) = (dst.add(p * n), stage.add((o * 4 + p) * GROUP * 4));
                if OVERWRITE {
                    std::ptr::copy_nonoverlapping(s, d, 4 * len);
                } else {
                    for b in 0..len {
                        vst1q_u32(d.add(4 * b), add_mod(vld1q_u32(d.add(4 * b)), vld1q_u32(s.add(4 * b)), c.q));
                    }
                }
            }
        }
    }
}

/// Scratch space (in bytes) required by the VMP apply kernels, per worker.
///
/// Holds the input rows of one group of blocks.
pub(crate) fn vmp_apply_tmp_bytes_neon(a_size: usize, b_rows: usize, b_cols_in: usize) -> usize {
    let row_max = a_size.min(b_rows) * b_cols_in;
    ROW * GROUP * row_max.max(1) * size_of::<u32>()
}

/// Gathers `len` blocks of one input limb, from block `blk`, as row `row` of each block, centered.
///
/// Block `b` of the group has its `rows` rows at `x + b * rows * ROW`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn gather_limb(n: usize, blk: usize, len: usize, limb: *const u32, x: *mut u32, row: usize, rows: usize, c: &[Plane; 4]) {
    unsafe {
        for (p, c) in c.iter().enumerate() {
            let src = limb.add(p * n + 4 * blk);
            let dst = x.add(row * ROW + 4 * p);
            for b in 0..len {
                vst1q_u32(dst.add(b * rows * ROW), center(vld1q_u32(src.add(4 * b)), c));
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn vmp_apply_core_neon_pm<const OVERWRITE: bool, E: TaskExecutor>(
    n: usize,
    res_u32: &mut [u32],
    a_u32: &[u32],
    pmat_u32: &[u32],
    limb_offset: usize,
    nrows: usize,
    ncols: usize,
    tmp: &mut [u32],
) {
    assert!(n >= 4);
    assert!(n.is_power_of_two());

    let a_size = a_u32.len() / (4 * n);
    let res_size = res_u32.len() / (4 * n);
    let n_blocks = n / 4;

    let row_end = nrows.min(a_size);
    let row_start = a_u32
        .chunks_exact(4 * n)
        .take(row_end)
        .take_while(|row| row.iter().all(|&x| x == 0))
        .count();
    let row_max = row_end - row_start;
    let col_max = ncols.min(res_size + limb_offset);

    if limb_offset >= col_max || row_max == 0 {
        if OVERWRITE {
            res_u32.fill(0);
        }
        return;
    }

    assert!(pmat_u32.len() >= n_blocks * ncols * nrows * ROW);
    assert!(a_u32.len() >= row_end * 4 * n);
    assert!(res_u32.len() >= (col_max - limb_offset) * 4 * n);
    let cols = col_max - limb_offset;
    let per_worker = ROW * GROUP * row_max;
    assert!(tmp.len() >= per_worker);

    let a_ptr = a_u32[row_start * 4 * n..].as_ptr() as usize;
    let pmat_ptr = pmat_u32[row_start * ROW..].as_ptr() as usize;
    let res_ptr = SendU32Ptr(res_u32.as_mut_ptr());
    let len = GROUP.min(n_blocks);

    E::for_each_chunked(n_blocks / len, tmp, per_worker, |buf, group| unsafe {
        let c = planes();
        let (a, pmat) = (a_ptr as *const u32, pmat_ptr as *const u32);
        let x = buf.as_mut_ptr();
        let blk = group * len;
        for row in 0..row_max {
            gather_limb(n, blk, len, a.add(row * 4 * n), x, row, row_max, &c);
        }
        let mut stage = Stage::uninit();
        let sp = stage.as_mut_ptr() as *mut u32;
        let mut first = 0;
        while first < cols {
            let count = (cols - first).min(STAGE_OUTPUTS);
            for b in 0..len {
                let xb = x.add(b * row_max * ROW);
                let block = pmat.add(((blk + b) * ncols + limb_offset + first) * nrows * ROW);
                for o in 0..count {
                    stage_store(sp, o, b, &dot_rows(xb, block.add(o * nrows * ROW), row_max, &c));
                }
            }
            stage_flush::<OVERWRITE>(&c, sp, res_ptr.get(), n, first, count, blk, len);
            first += count;
        }
    });

    if OVERWRITE {
        let active_cols = col_max - limb_offset;
        for col in active_cols..res_size {
            res_u32[col * 4 * n..(col + 1) * 4 * n].fill(0);
        }
    }
}

pub(crate) fn vmp_apply_dft_to_dft_neon<R: Ring, E: TaskExecutor>(
    _module: &Module<NTT4x30Neon<R>>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Neon<R>>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Neon<R>>,
    pmat: &VmpPMatBackendRef<'_, NTT4x30Neon<R>>,
    limb_offset: usize,
    tmp: &mut [u64],
) {
    assert_eq!(res.n(), pmat.n());
    assert_eq!(a.n(), pmat.n());
    assert_eq!(res.cols(), pmat.cols_out());
    assert_eq!(a.cols(), pmat.cols_in());
    let n = res.n();
    let nrows = pmat.cols_in() * pmat.rows();
    let ncols = pmat.cols_out() * pmat.size();

    let res_u32: &mut [u32] = cast_slice_mut(res.raw_mut());
    let a_u32: &[u32] = cast_slice(a.raw());
    let pmat_u32: &[u32] = cast_slice(pmat.data());

    unsafe {
        vmp_apply_core_neon_pm::<true, E>(
            n,
            res_u32,
            a_u32,
            pmat_u32,
            limb_offset * pmat.cols_out(),
            nrows,
            ncols,
            cast_slice_mut(tmp),
        );
    }
}

pub(crate) fn vmp_apply_dft_to_dft_add_neon<R: Ring, E: TaskExecutor>(
    _module: &Module<NTT4x30Neon<R>>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Neon<R>>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Neon<R>>,
    pmat: &VmpPMatBackendRef<'_, NTT4x30Neon<R>>,
    limb_offset: usize,
    tmp: &mut [u64],
) {
    assert_eq!(res.n(), pmat.n());
    assert_eq!(a.n(), pmat.n());
    assert_eq!(res.cols(), pmat.cols_out());
    assert_eq!(a.cols(), pmat.cols_in());
    let n = res.n();
    let nrows = pmat.cols_in() * pmat.rows();
    let ncols = pmat.cols_out() * pmat.size();

    let res_u32: &mut [u32] = cast_slice_mut(res.raw_mut());
    let a_u32: &[u32] = cast_slice(a.raw());
    let pmat_u32: &[u32] = cast_slice(pmat.data());

    unsafe {
        vmp_apply_core_neon_pm::<false, E>(
            n,
            res_u32,
            a_u32,
            pmat_u32,
            limb_offset * pmat.cols_out(),
            nrows,
            ncols,
            cast_slice_mut(tmp),
        );
    }
}

/// Largest `dsize` the fused interleaved-digit product handles.
pub(crate) const STRIDED_MAX_DSIZE: usize = 16;

/// Scratch space (in bytes) of the fused interleaved-digit product, per worker.
///
/// Holds the input rows of one group of blocks for every digit: the digits partition the input limbs.
pub(crate) fn vmp_apply_digits_strided_tmp_bytes_neon(a_cols: usize, a_size: usize) -> usize {
    ROW * GROUP * (a_size * a_cols).max(1) * size_of::<u32>()
}

/// One gadget digit of the interleaved-digit product.
#[derive(Clone, Copy, Default)]
struct Digit {
    /// First input limb, the next ones follow every `dsize` limbs.
    first_limb: usize,
    /// Leading input rows skipped because their limbs are zero.
    skip: usize,
    /// Remaining input rows, and their offset in the gathered block.
    rows: usize,
    x_off: usize,
    /// Output limbs this digit contributes to.
    out_limbs: usize,
}

/// Interleaved-digit GGLWE product in one pass over the prepared matrix.
///
/// Digit `di` gathers the input limbs congruent to `dsize - 1 - di` modulo `dsize` and reads the matrix `di` limbs ahead.
/// Returns the residues of `gglwe_product_digits_strided_reference`.
pub(crate) fn vmp_apply_dft_to_dft_digits_strided_neon<R: Ring, E: TaskExecutor>(
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Neon<R>>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Neon<R>>,
    dsize: usize,
    product_limbs: usize,
    pmat: &VmpPMatBackendRef<'_, NTT4x30Neon<R>>,
    tmp: &mut [u64],
) {
    assert_eq!(res.n(), pmat.n());
    assert_eq!(a.n(), pmat.n());
    assert_eq!(res.cols(), pmat.cols_out());
    assert_eq!(a.cols(), pmat.cols_in());
    assert!((1..=STRIDED_MAX_DSIZE).contains(&dsize));
    let n = res.n();
    assert!(n >= 4 && n.is_power_of_two());
    let (cols_in, cols_out) = (pmat.cols_in(), pmat.cols_out());
    let (dnum, key_size) = (pmat.rows(), pmat.size());
    let nrows = cols_in * dnum;
    let ncols = cols_out * key_size;
    let (a_size, res_size) = (a.size(), res.size());

    let res_u32: &mut [u32] = cast_slice_mut(res.raw_mut());
    let a_u32: &[u32] = cast_slice(a.raw());
    let pmat_u32: &[u32] = cast_slice(pmat.data());
    assert!(a_u32.len() >= a_size * cols_in * 4 * n);

    // Leading input limbs that are zero in every column contribute nothing: their rows are skipped.
    // A ciphertext raised to a larger modulus has most of its limbs in this case.
    let zero_limbs = a_u32
        .chunks_exact(cols_in * 4 * n)
        .take(a_size)
        .take_while(|limb| limb.iter().all(|&x| x == 0))
        .count();

    let mut digits = [Digit::default(); STRIDED_MAX_DSIZE];
    let mut total_rows = 0;
    let mut active_limbs = 0;
    for (di, digit) in digits[..dsize].iter_mut().enumerate() {
        let first_limb = dsize - di - 1;
        let all_rows = ((a_size + di) / dsize).min(dnum) * cols_in;
        // Row `j * cols_in + col` reads limb `first_limb + j * dsize`.
        let skip = (zero_limbs.saturating_sub(first_limb).div_ceil(dsize) * cols_in).min(all_rows);
        let rows = all_rows - skip;
        // The first digit overwrites every limb the key covers, the next ones read the key `di` limbs ahead.
        let out_limbs = if di == 0 {
            res_size.min(key_size)
        } else {
            gglwe_product_digit_output_size(res_size, key_size, dsize, di, product_limbs).min(key_size.saturating_sub(di))
        };
        *digit = Digit {
            first_limb,
            skip,
            rows,
            x_off: total_rows,
            out_limbs,
        };
        total_rows += rows;
        if rows != 0 {
            active_limbs = active_limbs.max(out_limbs);
        }
    }
    let digits = &digits[..dsize];

    let n_blocks = n / 4;
    assert!(pmat_u32.len() >= n_blocks * ncols * nrows * ROW);
    assert!(res_u32.len() >= res_size * cols_out * 4 * n);
    let per_worker = ROW * GROUP * total_rows.max(1);
    let tmp: &mut [u32] = cast_slice_mut(tmp);
    assert!(tmp.len() >= per_worker);

    let a_ptr = a_u32.as_ptr() as usize;
    let pmat_ptr = pmat_u32.as_ptr() as usize;
    let res_ptr = SendU32Ptr(res_u32.as_mut_ptr());
    let len = GROUP.min(n_blocks);

    if active_limbs != 0 {
        E::for_each_chunked(n_blocks / len, tmp, per_worker, |buf, group| unsafe {
            let c = planes();
            let (a, pmat) = (a_ptr as *const u32, pmat_ptr as *const u32);
            let x = buf.as_mut_ptr();
            let blk = group * len;
            for digit in digits {
                for row in 0..digit.rows {
                    let source = digit.skip + row;
                    let flat = (digit.first_limb + (source / cols_in) * dsize) * cols_in + source % cols_in;
                    gather_limb(n, blk, len, a.add(flat * 4 * n), x, digit.x_off + row, total_rows, &c);
                }
            }
            let outputs = active_limbs * cols_out;
            let mut stage = Stage::uninit();
            let sp = stage.as_mut_ptr() as *mut u32;
            let mut first = 0;
            while first < outputs {
                let count = (outputs - first).min(STAGE_OUTPUTS);
                for b in 0..len {
                    let xb = x.add(b * total_rows * ROW);
                    let block = pmat.add((blk + b) * ncols * nrows * ROW);
                    for o in 0..count {
                        let (limb, col) = ((first + o) / cols_out, (first + o) % cols_out);
                        let mut state = DotState::new();
                        for (di, digit) in digits.iter().enumerate() {
                            if limb < digit.out_limbs {
                                let m = block.add((((limb + di) * cols_out + col) * nrows + digit.skip) * ROW);
                                state = state.push_rows(xb.add(digit.x_off * ROW), m, digit.rows, &c);
                            }
                        }
                        stage_store(sp, o, b, &state.finish(&c));
                    }
                }
                stage_flush::<true>(&c, sp, res_ptr.get(), n, first, count, blk, len);
                first += count;
            }
        });
    }
    res_u32[active_limbs * cols_out * 4 * n..res_size * cols_out * 4 * n].fill(0);
}
