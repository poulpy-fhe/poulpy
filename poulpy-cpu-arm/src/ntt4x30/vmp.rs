//! Vector-matrix product for [`NTT4x30Neon`](crate::NTT4x30Neon) on the packed layout.
//!
//! The prepared matrix is ordered `block -> output column -> input row`, where a block is four consecutive coefficients.
//! One row of a block is 16 `u32`: four lanes for each of the four primes, multiplied by `2^32`.
//! The apply path therefore reads the matrix as one contiguous stream, in the order of its inner loop.

use std::mem::size_of;

use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::aarch64::{vld1q_u32, vst1q_u32};
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
use crate::neon::ntt4x30_packed::{DotState, add_mod, dot_rows, planes};
use crate::ntt4x30::vec_znx_dft::{dft_limb_scaled, dft_tmp_words, prepare_tmp_words};
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
/// Holds one packed limb and the forward transform scratch.
pub(crate) fn vmp_prepare_tmp_bytes_neon<R: Ring>(n: usize) -> usize {
    prepare_tmp_words::<R>(n) * size_of::<u64>()
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
    assert!(std::mem::size_of_val(tmp) >= vmp_prepare_tmp_bytes_neon::<R>(n));
    assert!(n.is_multiple_of(4));

    let nrows = a.cols_in() * a.rows();
    let ncols = a.cols_out() * a.size();
    let n_blocks = n / 4;

    let (tmp_b, tmp_packed) = tmp.split_at_mut(dft_tmp_words::<R>(n));
    let tmp_packed: &mut [u32] = &mut cast_slice_mut(tmp_packed)[..4 * n];
    let mat_i64: &[i64] = a.raw();
    let pmat: &mut [u32] = cast_slice_mut(res.data_mut());

    for row_i in 0..nrows {
        for col_i in 0..ncols {
            let pos = n * (row_i * ncols + col_i);

            dft_limb_scaled(module, n, tmp_packed, &mat_i64[pos..pos + n], true, tmp_b);

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

/// Scratch space (in bytes) required by the VMP apply kernels, per worker.
///
/// Holds the input rows of one block.
pub(crate) fn vmp_apply_tmp_bytes_neon(a_size: usize, b_rows: usize, b_cols_in: usize) -> usize {
    let row_max = a_size.min(b_rows) * b_cols_in;
    ROW * row_max.max(1) * size_of::<u32>()
}

/// Gathers block `blk` of `row_max` input limbs into rows of 16 `u32`.
#[inline(always)]
unsafe fn extract_block(n: usize, row_max: usize, blk: usize, a: *const u32, x: *mut u32) {
    unsafe {
        for row in 0..row_max {
            let limb = a.add(row * 4 * n + 4 * blk);
            let dst = x.add(row * ROW);
            for p in 0..4 {
                vst1q_u32(dst.add(4 * p), vld1q_u32(limb.add(p * n)));
            }
        }
    }
}

/// Computes block `blk` of every active output column.
///
/// # Safety
/// `res` addresses the output limbs, `a` the `row_max` active input limbs and `pmat` the first active row of the matrix.
/// `x` addresses `16 * row_max` `u32` of scratch.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn apply_block<const OVERWRITE: bool>(
    n: usize,
    blk: usize,
    res: *mut u32,
    a: *const u32,
    pmat: *const u32,
    nrows: usize,
    ncols: usize,
    row_max: usize,
    limb_offset: usize,
    col_max: usize,
    x: *mut u32,
) {
    unsafe {
        let c = planes();
        extract_block(n, row_max, blk, a, x);
        for col_pmat in limb_offset..col_max {
            let m = pmat.add((blk * ncols + col_pmat) * nrows * ROW);
            let r = dot_rows(x, m, row_max, &c);
            let dst = res.add((col_pmat - limb_offset) * 4 * n + 4 * blk);
            for p in 0..4 {
                let d = dst.add(p * n);
                if OVERWRITE {
                    vst1q_u32(d, r[p]);
                } else {
                    vst1q_u32(d, add_mod(vld1q_u32(d), r[p], c[p].q));
                }
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
    let per_worker = ROW * row_max;
    assert!(tmp.len() >= per_worker);

    let a_ptr = a_u32[row_start * 4 * n..].as_ptr() as usize;
    let pmat_ptr = pmat_u32[row_start * ROW..].as_ptr() as usize;
    let res_ptr = SendU32Ptr(res_u32.as_mut_ptr());

    E::for_each_chunked(n_blocks, tmp, per_worker, |x, blk| unsafe {
        apply_block::<OVERWRITE>(
            n,
            blk,
            res_ptr.get(),
            a_ptr as *const u32,
            pmat_ptr as *const u32,
            nrows,
            ncols,
            row_max,
            limb_offset,
            col_max,
            x.as_mut_ptr(),
        )
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
/// Holds the input rows of one block for every digit: the digits partition the input limbs.
pub(crate) fn vmp_apply_digits_strided_tmp_bytes_neon(a_cols: usize, a_size: usize) -> usize {
    ROW * (a_size * a_cols).max(1) * size_of::<u32>()
}

/// One gadget digit of the interleaved-digit product.
#[derive(Clone, Copy, Default)]
struct Digit {
    /// First input limb, the next ones follow every `dsize` limbs.
    first_limb: usize,
    /// Input rows, and their offset in the gathered block.
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

    let mut digits = [Digit::default(); STRIDED_MAX_DSIZE];
    let mut total_rows = 0;
    let mut active_limbs = 0;
    for (di, digit) in digits[..dsize].iter_mut().enumerate() {
        let rows = ((a_size + di) / dsize).min(dnum) * cols_in;
        // The first digit overwrites every limb the key covers, the next ones read the key `di` limbs ahead.
        let out_limbs = if di == 0 {
            res_size.min(key_size)
        } else {
            gglwe_product_digit_output_size(res_size, key_size, dsize, di, product_limbs).min(key_size.saturating_sub(di))
        };
        *digit = Digit {
            first_limb: dsize - di - 1,
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

    let res_u32: &mut [u32] = cast_slice_mut(res.raw_mut());
    let a_u32: &[u32] = cast_slice(a.raw());
    let pmat_u32: &[u32] = cast_slice(pmat.data());
    let n_blocks = n / 4;
    assert!(pmat_u32.len() >= n_blocks * ncols * nrows * ROW);
    assert!(a_u32.len() >= a_size * cols_in * 4 * n);
    assert!(res_u32.len() >= res_size * cols_out * 4 * n);
    let per_worker = ROW * total_rows.max(1);
    let tmp: &mut [u32] = cast_slice_mut(tmp);
    assert!(tmp.len() >= per_worker);

    let a_ptr = a_u32.as_ptr() as usize;
    let pmat_ptr = pmat_u32.as_ptr() as usize;
    let res_ptr = SendU32Ptr(res_u32.as_mut_ptr());

    if active_limbs != 0 {
        E::for_each_chunked(n_blocks, tmp, per_worker, |x, blk| unsafe {
            let c = planes();
            let (a, pmat, x) = (a_ptr as *const u32, pmat_ptr as *const u32, x.as_mut_ptr());
            for digit in digits {
                for row in 0..digit.rows {
                    let flat = (digit.first_limb + (row / cols_in) * dsize) * cols_in + row % cols_in;
                    let limb = a.add(flat * 4 * n + 4 * blk);
                    let dst = x.add((digit.x_off + row) * ROW);
                    for p in 0..4 {
                        vst1q_u32(dst.add(4 * p), vld1q_u32(limb.add(p * n)));
                    }
                }
            }
            let block = pmat.add(blk * ncols * nrows * ROW);
            for limb in 0..active_limbs {
                for col in 0..cols_out {
                    let mut state = DotState::new();
                    for (di, digit) in digits.iter().enumerate() {
                        if limb < digit.out_limbs {
                            let m = block.add(((limb + di) * cols_out + col) * nrows * ROW);
                            state.push_rows(x.add(digit.x_off * ROW), m, digit.rows, &c);
                        }
                    }
                    let r = state.finish(&c);
                    let dst = res_ptr.get().add((limb * cols_out + col) * 4 * n + 4 * blk);
                    for (p, &r) in r.iter().enumerate() {
                        vst1q_u32(dst.add(p * n), r);
                    }
                }
            }
        });
    }
    res_u32[active_limbs * cols_out * 4 * n..res_size * cols_out * 4 * n].fill(0);
}
