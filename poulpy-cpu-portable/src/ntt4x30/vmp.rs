//! Vector-matrix product for [`NTT4x30Portable`](crate::NTT4x30Portable) on the packed layout.
//!
//! The prepared matrix is ordered `block -> output column -> input row`, where a block is four consecutive coefficients.
//! One row of a block is 16 `u32`: four lanes for each of the four primes, multiplied by `2^32` and centered around zero.
//! The apply path therefore reads the matrix as one contiguous stream, in the order of its inner loop.

use std::mem::size_of;

use bytemuck::{cast_slice, cast_slice_mut};
use poulpy_hal::{
    execution::TaskExecutor,
    layouts::{
        DataView, DataViewMut, MatZnxBackendRef, Module, Ring, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendMut,
        VmpPMatBackendRef, ZnxView, ZnxViewMut, check_degree,
    },
};

use super::NTT4x30Portable;
use super::packed::{DotState, ROW, SendPtr, gather_limb, scatter_centered_limb, stage_flush, stage_store};
use super::vec_znx_dft::{dft_limb_scaled, prepare_tmp_words};
use crate::kernels::vmp_select::assert_extractable_portable;

/// Scratch space (in bytes) required by the VMP prepare kernel.
///
/// Holds one packed limb.
pub(crate) fn vmp_prepare_tmp_bytes(n: usize) -> usize {
    prepare_tmp_words(n) * size_of::<u64>()
}

/// VMP prepare into the block-major prepared layout.
pub(crate) fn vmp_prepare<R: Ring>(
    module: &Module<NTT4x30Portable<R>>,
    res: &mut VmpPMatBackendMut<'_, NTT4x30Portable<R>>,
    a: &MatZnxBackendRef<'_, NTT4x30Portable<R>>,
    tmp: &mut [u64],
) {
    let n = res.n();
    check_degree::<NTT4x30Portable<R>>(module.n(), n);

    assert_eq!(a.n(), n);
    assert_eq!(res.cols_in(), a.cols_in());
    assert_eq!(res.rows(), a.rows());
    assert_eq!(res.cols_out(), a.cols_out());
    assert_eq!(res.size(), a.size());
    assert!(std::mem::size_of_val(tmp) >= vmp_prepare_tmp_bytes(n));

    let nrows = a.cols_in() * a.rows();
    let ncols = a.cols_out() * a.size();

    let tmp_packed: &mut [u32] = &mut cast_slice_mut(tmp)[..4 * n];
    let mat_i64: &[i64] = a.raw();
    let pmat: &mut [u32] = cast_slice_mut(res.data_mut());

    for row_i in 0..nrows {
        for col_i in 0..ncols {
            let pos = n * (row_i * ncols + col_i);
            dft_limb_scaled::<_, poulpy_hal::execution::SerialTaskExecutor>(module, n, tmp_packed, &mat_i64[pos..pos + n], true);
            scatter_centered_limb(n, pmat, tmp_packed, |blk| ((blk * ncols + col_i) * nrows + row_i) * ROW);
        }
    }
}

/// Copies rows `first_row + i * row_step` of `a`, truncated to `res.size()` limbs, into rows `i` of `res`.
pub(crate) fn vmp_extract_selected_rows<R: Ring>(
    res: &mut VmpPMatBackendMut<'_, NTT4x30Portable<R>>,
    a: &VmpPMatBackendRef<'_, NTT4x30Portable<R>>,
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
pub(super) const GROUP: usize = 32;

/// Outputs staged together, one output being a limb of a column: [`GROUP`] blocks of four planes each.
pub(super) const STAGE_OUTPUTS: usize = 64;

/// Words of the stage of one output: plane `p` holds its run at `p * GROUP * 4`.
pub(super) const STAGE_OUTPUT: usize = ROW * GROUP;

/// Words of scratch per worker for `rows` gathered input rows: the rows of one group of blocks, then the stage.
pub(super) fn apply_tmp_words(rows: usize) -> usize {
    ROW * GROUP * rows.max(1) + STAGE_OUTPUTS * STAGE_OUTPUT
}

/// Scratch space (in bytes) required by the VMP apply kernels, per worker.
pub fn vmp_apply_tmp_bytes(a_size: usize, b_rows: usize, b_cols_in: usize) -> usize {
    apply_tmp_words(a_size.min(b_rows) * b_cols_in) * size_of::<u32>()
}

#[allow(clippy::too_many_arguments)]
fn vmp_apply_core<const OVERWRITE: bool, E: TaskExecutor>(
    n: usize,
    res_u32: &mut [u32],
    a_u32: &[u32],
    pmat_u32: &[u32],
    limb_offset: usize,
    nrows: usize,
    ncols: usize,
    tmp: &mut [u32],
) {
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
    let cols = col_max - limb_offset;
    assert!(res_u32.len() >= cols * 4 * n);
    let a_u32 = &a_u32[row_start * 4 * n..row_end * 4 * n];
    let res_ptr = SendPtr(res_u32.as_mut_ptr());
    let len = GROUP.min(n_blocks);

    E::for_each_chunked(n_blocks / len, tmp, apply_tmp_words(row_max), |buf, group| {
        let (x, stage) = buf.split_at_mut(ROW * GROUP * row_max);
        let blk = group * len;
        for (row, limb) in a_u32.chunks_exact(4 * n).enumerate() {
            gather_limb(n, blk, len, limb, x, row, row_max);
        }
        for first in (0..cols).step_by(STAGE_OUTPUTS) {
            let count = (cols - first).min(STAGE_OUTPUTS);
            for (b, xb) in x.chunks_exact(row_max * ROW).take(len).enumerate() {
                let block = &pmat_u32[((blk + b) * ncols + limb_offset + first) * nrows * ROW..];
                for (o, stage) in stage.chunks_exact_mut(STAGE_OUTPUT).take(count).enumerate() {
                    let m = &block[(o * nrows + row_start) * ROW..][..row_max * ROW];
                    stage_store(stage, GROUP, b, &DotState::new().push_rows(xb, m).finish());
                }
            }
            for (o, stage) in stage.chunks_exact(STAGE_OUTPUT).take(count).enumerate() {
                // SAFETY: the output limb lies within `res_u32`, and this task is the only one on these blocks.
                unsafe { stage_flush(n, res_ptr.get().add((first + o) * 4 * n), stage, GROUP, blk, len, !OVERWRITE) };
            }
        }
    });

    if OVERWRITE {
        res_u32[cols * 4 * n..].fill(0);
    }
}

fn vmp_apply<const OVERWRITE: bool, R: Ring, E: TaskExecutor>(
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Portable<R>>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Portable<R>>,
    pmat: &VmpPMatBackendRef<'_, NTT4x30Portable<R>>,
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
    vmp_apply_core::<OVERWRITE, E>(
        n,
        cast_slice_mut(res.raw_mut()),
        cast_slice(a.raw()),
        cast_slice(pmat.data()),
        limb_offset * pmat.cols_out(),
        nrows,
        ncols,
        cast_slice_mut(tmp),
    );
}

pub fn vmp_apply_dft_to_dft<R: Ring, E: TaskExecutor>(
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Portable<R>>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Portable<R>>,
    pmat: &VmpPMatBackendRef<'_, NTT4x30Portable<R>>,
    limb_offset: usize,
    tmp: &mut [u64],
) {
    vmp_apply::<true, R, E>(res, a, pmat, limb_offset, tmp);
}

pub fn vmp_apply_dft_to_dft_add<R: Ring, E: TaskExecutor>(
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Portable<R>>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Portable<R>>,
    pmat: &VmpPMatBackendRef<'_, NTT4x30Portable<R>>,
    limb_offset: usize,
    tmp: &mut [u64],
) {
    vmp_apply::<false, R, E>(res, a, pmat, limb_offset, tmp);
}
