//! Lines of a grid's matrices, which repacking reads and writes as vectors of the blocks that hold them.

use super::{BlasGridInfos, BlasVectorLayout, BlasVectorShape};

/// A row or a column of one matrix of a grid's batch, read
/// ([`blas_grid_to_vector`](crate::api::BlasMatVecOps::blas_grid_to_vector)) or written
/// ([`blas_vector_to_grid`](crate::api::BlasMatVecOps::blas_vector_to_grid)) as a vector whose entry `e` is the line's
/// `e`-th element, `d` entries per block for blocks of dimension `d`.
///
/// Entry `e` of row `row` sits in block `(row / d, e / d)` at slot `mat_slot d^2 + (row mod d) d + e mod d`; entry `e`
/// of column `col` in block `(e / d, col / d)` at slot `mat_slot d^2 + (e mod d) d + col mod d`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlasGridLine {
    /// Row `row` of matrix `mat_slot`.
    Row { mat_slot: usize, row: usize },
    /// Column `col` of matrix `mat_slot`.
    Col { mat_slot: usize, col: usize },
}

impl BlasGridLine {
    /// The line of a grid of `grid`'s dimensions as a vector shape: a row's `cols d` entries at stride 1, a column's
    /// `rows d` entries at stride `d`, `d` per block.
    ///
    /// # Panics
    ///
    /// When the line's length or slots overflow a `usize`.
    pub fn shape(&self, grid: &impl BlasGridInfos) -> BlasVectorShape {
        let shape = u32::try_from(grid.log_d())
            .ok()
            .and_then(|log_d| 1usize.checked_shl(log_d))
            .and_then(|d| {
                let area = d.checked_mul(d)?;
                // `(row mod d) d < d^2`, which fits.
                let (lines, at, stride) = match *self {
                    Self::Row { row, .. } => (grid.cols(), (row % d) * d, 1),
                    Self::Col { col, .. } => (grid.rows(), col % d, d),
                };
                let offset = self.mat_slot().checked_mul(area)?.checked_add(at)?;
                Some(BlasVectorShape::new(lines.checked_mul(d)?, d, offset, stride))
            });
        shape.expect("the line's length or slots overflow a usize")
    }

    /// The line of `grid` as a vector layout: [`Self::shape`] on the grid's block layout. The planning queries of
    /// [`blas_matvec_pt`](crate::api::BlasMatVecOps::blas_matvec_pt) read it as the vector the repacking reads or
    /// writes.
    ///
    /// # Panics
    ///
    /// As [`Self::shape`].
    pub fn vector_layout(&self, grid: &impl BlasGridInfos) -> BlasVectorLayout {
        BlasVectorLayout::new(grid.block_layout(), self.shape(grid))
    }

    /// The matrix of the batch the line belongs to.
    pub(crate) fn mat_slot(&self) -> usize {
        match *self {
            Self::Row { mat_slot, .. } | Self::Col { mat_slot, .. } => mat_slot,
        }
    }

    /// The line's index and the number of lines of its kind in a grid of `grid`'s dimensions, `None` on overflow: a
    /// row's and `rows d`, a column's and `cols d`.
    pub(crate) fn index(&self, grid: &impl BlasGridInfos) -> (usize, Option<usize>) {
        let d = 1usize << grid.log_d();
        match *self {
            Self::Row { row, .. } => (row, grid.rows().checked_mul(d)),
            Self::Col { col, .. } => (col, grid.cols().checked_mul(d)),
        }
    }

    /// The row-major index of the block holding entries `t d .. (t + 1) d` of the line.
    pub(crate) fn block(&self, grid: &impl BlasGridInfos, t: usize) -> usize {
        let (d, cols) = (1usize << grid.log_d(), grid.cols());
        match *self {
            Self::Row { row, .. } => (row / d) * cols + t,
            Self::Col { col, .. } => t * cols + col / d,
        }
    }
}

#[cfg(test)]
mod tests {
    use poulpy_ckks::{CKKSLayout, GLWEPlaintextMeta};
    use poulpy_core::layouts::{Base2K, Degree, GLWELayout, Rank, TorusPrecision};

    use super::*;
    use crate::layouts::{BlasLayout, grid::split_into_blocks};

    /// Every entry of the line read through its shape and blocks is the matrix's, for rows and columns past the first
    /// block and a matrix slot past the first.
    #[test]
    fn line_shapes_read_the_matrix() {
        let (log_d, rows, cols, count) = (2, 3, 2, 2);
        let d = 1usize << log_d;
        let slots = count * d * d;
        let entry = |s: usize, r: usize, c: usize| (s * 1000 + r * 10 + c) as f64;
        let blocks = split_into_blocks(rows, cols, log_d, slots, count, entry);
        let block = CKKSLayout {
            glwe_layout: GLWELayout {
                n: Degree(64),
                base2k: Base2K(19),
                k: TorusPrecision(38),
                rank: Rank(1),
            },
            meta: GLWEPlaintextMeta::default(),
        };
        let grid = BlasLayout::new(block, log_d, rows, cols);
        let lines = [
            BlasGridLine::Row { mat_slot: 1, row: 5 },
            BlasGridLine::Row { mat_slot: 0, row: 0 },
            BlasGridLine::Col { mat_slot: 1, col: 6 },
            BlasGridLine::Col { mat_slot: 0, col: 3 },
        ];
        for line in lines {
            let shape = line.shape(&grid);
            assert_eq!(shape.count() * shape.chunk, shape.len, "{line:?}");
            for e in 0..shape.len {
                let (t, c) = (e / shape.chunk, e % shape.chunk);
                let have = blocks[line.block(&grid, t)][shape.offset + shape.stride * c];
                let want = match line {
                    BlasGridLine::Row { mat_slot, row } => entry(mat_slot, row, e),
                    BlasGridLine::Col { mat_slot, col } => entry(mat_slot, e, col),
                };
                assert_eq!(have, want, "{line:?}: entry {e}");
            }
        }
    }
}
