//! Selected-row extraction shared by the block-interleaved prepared layouts.
//!
//! `fft64` and `ntt4x30` scatter a prepared matrix the same way and differ only
//! in the word type and the cell width: cell `(row, col)` of block `blk` lives
//! at `blk * nrows * ncols * cell + cell_offset(..)`, paired columns sharing a
//! `2 * cell` slot and a trailing odd column using a `cell`-wide one.

use crate::layouts::{Backend, VmpPMatBackendMut, VmpPMatBackendRef};

/// Rejects a selection the kernel must not be handed: mismatched prepared
/// shapes, a truncation that widens, or a last row that is outside `a` or
/// whose index overflows.
///
/// Every kernel entry point calls it first, so the kernel may index without
/// bounds checks in release.
pub fn assert_extractable<BE: Backend>(
    res: &VmpPMatBackendMut<'_, BE>,
    a: &VmpPMatBackendRef<'_, BE>,
    first_row: usize,
    row_step: usize,
) {
    assert!(row_step > 0, "row_step must be positive");
    assert_eq!(res.n(), a.n(), "res.n(): {} != a.n(): {}", res.n(), a.n());
    assert_eq!(
        res.cols_in(),
        a.cols_in(),
        "res.cols_in(): {} != a.cols_in(): {}",
        res.cols_in(),
        a.cols_in()
    );
    assert_eq!(
        res.cols_out(),
        a.cols_out(),
        "res.cols_out(): {} != a.cols_out(): {}",
        res.cols_out(),
        a.cols_out()
    );
    assert!(res.size() <= a.size(), "res.size(): {} > a.size(): {}", res.size(), a.size());
    let Some(rows) = res.rows().checked_sub(1) else {
        return;
    };
    let last_row: Option<usize> = rows.checked_mul(row_step).and_then(|o| o.checked_add(first_row));
    assert!(
        last_row.is_some_and(|last| last < a.rows()),
        "selected rows {first_row}..={:?} step {row_step} exceed a.rows(): {}",
        last_row,
        a.rows()
    );
}

/// Word offset of cell `(row, col)` within a block.
#[inline]
fn cell_offset(nrows: usize, ncols: usize, row: usize, col: usize, cell: usize) -> usize {
    if col == ncols - 1 && !ncols.is_multiple_of(2) {
        col * nrows * cell + row * cell
    } else {
        (col / 2) * (nrows * 2 * cell) + row * 2 * cell + (col % 2) * cell
    }
}

/// Copies rows `first_row + i * row_step` of `a`, truncated to `res_ncols`
/// columns, into rows `i` of `res`.
///
/// Flat rows group the `cols_in` columns of one gadget row, so a gadget row
/// step of `s` is a flat row step of `s * cols_in`. Reads touch only the
/// selected cells.
#[allow(clippy::too_many_arguments)]
pub fn vmp_extract_selected_rows_core<T: Copy>(
    res: &mut [T],
    res_rows: usize,
    res_ncols: usize,
    a: &[T],
    a_rows: usize,
    a_ncols: usize,
    cols_in: usize,
    blocks: usize,
    cell: usize,
    first_row: usize,
    row_step: usize,
) {
    assert!(row_step > 0);
    assert!(res_ncols <= a_ncols);
    assert!(res_rows == 0 || first_row + (res_rows - 1) * row_step < a_rows);
    assert_eq!(res.len(), blocks * res_rows * cols_in * res_ncols * cell);
    assert_eq!(a.len(), blocks * a_rows * cols_in * a_ncols * cell);

    let (res_nrows, a_nrows) = (res_rows * cols_in, a_rows * cols_in);
    for blk in 0..blocks {
        let (res_blk, a_blk) = (blk * res_nrows * res_ncols * cell, blk * a_nrows * a_ncols * cell);
        for i in 0..res_rows {
            for c in 0..cols_in {
                let (res_row, a_row) = (i * cols_in + c, (first_row + i * row_step) * cols_in + c);
                for col in 0..res_ncols {
                    let dst: usize = res_blk + cell_offset(res_nrows, res_ncols, res_row, col, cell);
                    let src: usize = a_blk + cell_offset(a_nrows, a_ncols, a_row, col, cell);
                    res[dst..dst + cell].copy_from_slice(&a[src..src + cell]);
                }
            }
        }
    }
}
