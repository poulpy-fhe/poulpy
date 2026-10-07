//! Interleaved-digit GGLWE product for [`NTT4x30Portable`](crate::NTT4x30Portable), fused in one pass over the prepared matrix.

use std::mem::size_of;

use bytemuck::{cast_slice, cast_slice_mut};
use poulpy_core::oep::gglwe_product_digit_output_size;
use poulpy_hal::{
    execution::TaskExecutor,
    layouts::{DataView, Ring, ScratchArena, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendRef, ZnxView, ZnxViewMut},
};

use super::NTT4x30Portable;
use super::hal_impl::take_host_typed;
use super::packed::{DotState, ROW, SendPtr, gather_limb, stage_flush, stage_store};
use super::vmp::{GROUP, STAGE_OUTPUT, STAGE_OUTPUTS, apply_tmp_words};

/// Largest `dsize` the fused interleaved-digit product handles.
pub(crate) const STRIDED_MAX_DSIZE: usize = 16;

/// Scratch space (in bytes) of the fused interleaved-digit product, per worker.
///
/// Holds the input rows of one group of blocks for every digit, and the stage: the digits partition the input limbs.
pub(crate) fn gglwe_product_digits_strided_tmp_bytes(a_cols: usize, a_size: usize) -> usize {
    apply_tmp_words(a_size * a_cols) * size_of::<u32>()
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
pub(crate) fn gglwe_product_digits_strided<R: Ring, E: TaskExecutor>(
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Portable<R>>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Portable<R>>,
    dsize: usize,
    product_limbs: usize,
    pmat: &VmpPMatBackendRef<'_, NTT4x30Portable<R>>,
    scratch: &mut ScratchArena<'_, NTT4x30Portable<R>>,
) {
    assert_eq!(res.n(), pmat.n());
    assert_eq!(a.n(), pmat.n());
    assert_eq!(res.cols(), pmat.cols_out());
    assert_eq!(a.cols(), pmat.cols_in());
    assert!((1..=STRIDED_MAX_DSIZE).contains(&dsize));
    let n = res.n();
    let (cols_in, cols_out) = (pmat.cols_in(), pmat.cols_out());
    let (dnum, key_size) = (pmat.rows(), pmat.size());
    let nrows = cols_in * dnum;
    let ncols = cols_out * key_size;
    let (a_size, res_size) = (a.size(), res.size());

    let bytes = gglwe_product_digits_strided_tmp_bytes(a.cols(), a.size());
    let (tmp, _) = take_host_typed::<NTT4x30Portable<R>, u32>(scratch.borrow(), bytes / size_of::<u32>());
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
    let res_ptr = SendPtr(res_u32.as_mut_ptr());
    let len = GROUP.min(n_blocks);
    let outputs = active_limbs * cols_out;

    if outputs != 0 {
        E::for_each_chunked(
            n_blocks / len,
            cast_slice_mut(tmp),
            apply_tmp_words(total_rows),
            |buf: &mut [u32], group| {
                let (x, stage) = buf.split_at_mut(ROW * GROUP * total_rows);
                let blk = group * len;
                for digit in digits {
                    for row in 0..digit.rows {
                        let source = digit.skip + row;
                        let flat = (digit.first_limb + (source / cols_in) * dsize) * cols_in + source % cols_in;
                        gather_limb(n, blk, len, &a_u32[flat * 4 * n..], x, digit.x_off + row, total_rows);
                    }
                }
                for first in (0..outputs).step_by(STAGE_OUTPUTS) {
                    let count = (outputs - first).min(STAGE_OUTPUTS);
                    for (b, xb) in x.chunks_exact(total_rows * ROW).take(len).enumerate() {
                        let block = &pmat_u32[(blk + b) * ncols * nrows * ROW..];
                        for (o, stage) in stage.chunks_exact_mut(STAGE_OUTPUT).take(count).enumerate() {
                            let (limb, col) = ((first + o) / cols_out, (first + o) % cols_out);
                            let mut state = DotState::new();
                            for (di, digit) in digits.iter().enumerate() {
                                if limb < digit.out_limbs {
                                    let m = &block[(((limb + di) * cols_out + col) * nrows + digit.skip) * ROW..];
                                    state = state.push_rows(&xb[digit.x_off * ROW..][..digit.rows * ROW], m);
                                }
                            }
                            stage_store(stage, GROUP, b, &state.finish());
                        }
                    }
                    for (o, stage) in stage.chunks_exact(STAGE_OUTPUT).take(count).enumerate() {
                        // SAFETY: the output limb lies within `res_u32`, and this task is the only one on these blocks.
                        unsafe { stage_flush(n, res_ptr.get().add((first + o) * 4 * n), stage, GROUP, blk, len, false) };
                    }
                }
            },
        );
    }
    res_u32[outputs * 4 * n..res_size * cols_out * 4 * n].fill(0);
}
