//! A grid of CKKS ciphertexts, each holding one `d x d` block of every matrix of a batch.

use anyhow::{Result, ensure};
use poulpy_ckks::{
    CKKSInfos, CKKSLayout,
    layouts::{CKKSCiphertext, CKKSModuleAlloc},
};
use poulpy_core::layouts::{Base2K, GLWEInfos, LWEInfos, TorusPrecision};
use poulpy_hal::layouts::{Backend, Data, Module, Ring, ZnxWord};

/// An `rows x cols` grid of `2^log_d x 2^log_d` blocks: block `(i, j)` is one CKKS ciphertext whose matrix slot `s`
/// holds block `(i, j)` of the `s`-th matrix of the batch.
pub struct BlasCiphertext<D: Data, W: ZnxWord, R: Ring> {
    blocks: Vec<CKKSCiphertext<D, W, R>>,
    log_d: usize,
    rows: usize,
    cols: usize,
}

/// Backend-owned grid.
pub type BlasCiphertextOwned<BE> = BlasCiphertext<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord, <BE as Backend>::Ring>;

/// Dimensions of a grid and the layout every one of its blocks shares.
pub trait BlasGridInfos {
    /// `log2` of the block dimension `d`.
    fn log_d(&self) -> usize;

    /// Number of block rows.
    fn rows(&self) -> usize;

    /// Number of block columns.
    fn cols(&self) -> usize;

    /// Layout of every block.
    fn block_layout(&self) -> CKKSLayout;
}

/// A grid's dimensions and block layout, without ciphertexts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlasLayout {
    pub block: CKKSLayout,
    pub log_d: usize,
    pub rows: usize,
    pub cols: usize,
}

impl BlasGridInfos for BlasLayout {
    fn log_d(&self) -> usize {
        self.log_d
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn block_layout(&self) -> CKKSLayout {
        self.block
    }
}

/// The CKKS layout of one block.
pub(crate) fn layout_of<B: GLWEInfos + CKKSInfos>(block: &B) -> CKKSLayout {
    CKKSLayout {
        glwe_layout: block.glwe_layout(),
        meta: block.meta(),
    }
}

/// Matrix slots of `2^log_d x 2^log_d` blocks in `slots` slots at sparsity `log_sparsity`, `0` when one block does not
/// fit, including when a shift passes the word width.
pub(crate) fn mat_slots_in(slots: usize, log_d: usize, log_sparsity: usize) -> usize {
    let shr = |x: usize, shift: usize| u32::try_from(shift).ok().and_then(|shift| x.checked_shr(shift));
    shr(slots, log_sparsity)
        .zip(log_d.checked_mul(2))
        .and_then(|(slots, shift)| shr(slots, shift))
        .unwrap_or(0)
}

impl<D: Data, W: ZnxWord, R: Ring> BlasCiphertext<D, W, R> {
    /// Reads `blocks`, row-major, as a `rows x cols` grid of `2^log_d x 2^log_d` blocks.
    pub fn from_blocks(blocks: Vec<CKKSCiphertext<D, W, R>>, log_d: usize, rows: usize, cols: usize) -> Result<Self> {
        ensure!(rows > 0 && cols > 0, "a grid has at least one block");
        ensure!(
            rows.checked_mul(cols) == Some(blocks.len()),
            "{} blocks do not form a {rows} x {cols} grid",
            blocks.len()
        );
        Ok(Self {
            blocks,
            log_d,
            rows,
            cols,
        })
    }

    fn index(&self, i: usize, j: usize) -> usize {
        assert!(
            i < self.rows && j < self.cols,
            "block ({i}, {j}) is outside a {} x {} grid",
            self.rows,
            self.cols
        );
        i * self.cols + j
    }

    /// Block `(i, j)`.
    pub fn block(&self, i: usize, j: usize) -> &CKKSCiphertext<D, W, R> {
        &self.blocks[self.index(i, j)]
    }

    /// Block `(i, j)`, mutably.
    pub fn block_mut(&mut self, i: usize, j: usize) -> &mut CKKSCiphertext<D, W, R> {
        let t = self.index(i, j);
        &mut self.blocks[t]
    }

    /// Block `(i, j)` without the bounds assertion, for the references' loops after validation.
    pub(crate) fn at(&self, i: usize, j: usize) -> &CKKSCiphertext<D, W, R> {
        &self.blocks[i * self.cols + j]
    }

    /// Block `(i, j)`, mutably, without the bounds assertion.
    pub(crate) fn at_mut(&mut self, i: usize, j: usize) -> &mut CKKSCiphertext<D, W, R> {
        &mut self.blocks[i * self.cols + j]
    }

    /// Every block, row-major.
    pub fn blocks(&self) -> &[CKKSCiphertext<D, W, R>] {
        &self.blocks
    }

    /// Every block, row-major, mutably.
    pub fn blocks_mut(&mut self) -> &mut [CKKSCiphertext<D, W, R>] {
        &mut self.blocks
    }

    /// Returns the blocks, row-major.
    pub fn into_blocks(self) -> Vec<CKKSCiphertext<D, W, R>> {
        self.blocks
    }

    /// Number of matrices in the batch, `0` when one block does not fit the slots of block `(0, 0)`.
    pub fn mat_slots(&self) -> usize {
        let block = &self.blocks[0];
        let max_slots = block.n().as_usize() * R::CYCLOTOMIC_ORDER_FACTOR as usize / 4;
        mat_slots_in(max_slots, self.log_d, block.log_sparsity())
    }

    /// Moves the first `at` block columns into the first grid and the others into the second; panics unless
    /// `0 < at < cols`.
    pub fn split_cols(self, at: usize) -> (Self, Self) {
        let (rows, cols, log_d) = (self.rows, self.cols, self.log_d);
        assert!(0 < at && at < cols, "split_cols: {at} is not strictly inside 0..{cols}");
        let (mut left, mut right) = (Vec::with_capacity(rows * at), Vec::with_capacity(rows * (cols - at)));
        for (t, block) in self.blocks.into_iter().enumerate() {
            if t % cols < at {
                left.push(block);
            } else {
                right.push(block);
            }
        }
        let grid = |blocks, cols| Self {
            blocks,
            log_d,
            rows,
            cols,
        };
        (grid(left, at), grid(right, cols - at))
    }

    /// Moves the first `at` block rows into the first grid and the others into the second; panics unless
    /// `0 < at < rows`.
    pub fn split_rows(self, at: usize) -> (Self, Self) {
        let (rows, cols, log_d) = (self.rows, self.cols, self.log_d);
        assert!(0 < at && at < rows, "split_rows: {at} is not strictly inside 0..{rows}");
        let mut top = self.blocks;
        let bottom = top.split_off(at * cols);
        let grid = |blocks, rows| Self {
            blocks,
            log_d,
            rows,
            cols,
        };
        (grid(top, at), grid(bottom, rows - at))
    }

    /// Moves the blocks of `left` then `right` into one grid, the inverse of [`Self::split_cols`]; rejects grids of
    /// different `log_d` or row counts.
    pub fn concat_cols(left: Self, right: Self) -> Result<Self> {
        ensure!(
            (left.log_d, left.rows) == (right.log_d, right.rows),
            "concat_cols: a 2^{} x {}-row grid against a 2^{} x {}-row grid",
            left.log_d,
            left.rows,
            right.log_d,
            right.rows
        );
        let (rows, lc, rc) = (left.rows, left.cols, right.cols);
        let mut blocks = Vec::with_capacity(rows * (lc + rc));
        let (mut l, mut r) = (left.blocks.into_iter(), right.blocks.into_iter());
        for _ in 0..rows {
            blocks.extend(l.by_ref().take(lc));
            blocks.extend(r.by_ref().take(rc));
        }
        Self::from_blocks(blocks, left.log_d, rows, lc + rc)
    }

    /// Moves the blocks of `top` then `bottom` into one grid, the inverse of [`Self::split_rows`]; rejects grids of
    /// different `log_d` or column counts.
    pub fn concat_rows(top: Self, bottom: Self) -> Result<Self> {
        ensure!(
            (top.log_d, top.cols) == (bottom.log_d, bottom.cols),
            "concat_rows: a 2^{} x {}-column grid against a 2^{} x {}-column grid",
            top.log_d,
            top.cols,
            bottom.log_d,
            bottom.cols
        );
        let (log_d, rows, cols) = (top.log_d, top.rows + bottom.rows, top.cols);
        let mut blocks = top.blocks;
        blocks.extend(bottom.blocks);
        Self::from_blocks(blocks, log_d, rows, cols)
    }
}

impl<D: Data, W: ZnxWord, R: Ring> Clone for BlasCiphertext<D, W, R>
where
    CKKSCiphertext<D, W, R>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            blocks: self.blocks.clone(),
            log_d: self.log_d,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl<D: Data, W: ZnxWord, R: Ring> BlasGridInfos for BlasCiphertext<D, W, R> {
    fn log_d(&self) -> usize {
        self.log_d
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn block_layout(&self) -> CKKSLayout {
        layout_of(&self.blocks[0])
    }
}

/// The left operand of [`crate::api::BlasMatMulOps::blas_matmul_prepared`]: sigma of every block, filled by
/// [`crate::api::BlasMatMulOps::blas_matmul_prepare_left`].
///
/// Preparing both operands then calling `blas_matmul_prepared` equals `blas_matmul` bit for bit only when this grid is
/// allocated with the input's block layouts, for example `blas_ciphertext_alloc_from_infos(&a)`.
pub struct BlasMatMulLeft<D: Data, W: ZnxWord, R: Ring>(BlasCiphertext<D, W, R>);

impl<D: Data, W: ZnxWord, R: Ring> BlasMatMulLeft<D, W, R> {
    /// Wraps a caller-allocated grid shaped like the left operand.
    pub fn from_grid(grid: BlasCiphertext<D, W, R>) -> Self {
        Self(grid)
    }

    /// The transformed grid.
    pub fn as_grid(&self) -> &BlasCiphertext<D, W, R> {
        &self.0
    }

    /// The transformed grid, mutably.
    pub fn as_grid_mut(&mut self) -> &mut BlasCiphertext<D, W, R> {
        &mut self.0
    }

    /// Returns the wrapped grid.
    pub fn into_grid(self) -> BlasCiphertext<D, W, R> {
        self.0
    }
}

/// The right operand of [`crate::api::BlasMatMulOps::blas_matmul_prepared`]: tau of every block, filled by
/// [`crate::api::BlasMatMulOps::blas_matmul_prepare_right`].
///
/// Preparing both operands then calling `blas_matmul_prepared` equals `blas_matmul` bit for bit only when this grid is
/// allocated with the input's block layouts, for example `blas_ciphertext_alloc_from_infos(&b)`.
pub struct BlasMatMulRight<D: Data, W: ZnxWord, R: Ring>(BlasCiphertext<D, W, R>);

impl<D: Data, W: ZnxWord, R: Ring> BlasMatMulRight<D, W, R> {
    /// Wraps a caller-allocated grid shaped like the right operand.
    pub fn from_grid(grid: BlasCiphertext<D, W, R>) -> Self {
        Self(grid)
    }

    /// The transformed grid.
    pub fn as_grid(&self) -> &BlasCiphertext<D, W, R> {
        &self.0
    }

    /// The transformed grid, mutably.
    pub fn as_grid_mut(&mut self) -> &mut BlasCiphertext<D, W, R> {
        &mut self.0
    }

    /// Returns the wrapped grid.
    pub fn into_grid(self) -> BlasCiphertext<D, W, R> {
        self.0
    }
}

/// Allocation of grids.
pub trait BlasModuleAlloc<BE: Backend>: CKKSModuleAlloc<BE> {
    /// A grid of `rows x cols` rank-1 blocks of dimension `2^log_d` at torus width `k`.
    ///
    /// # Panics
    ///
    /// Panics if `rows` or `cols` is zero, or if `rows * cols` overflows `usize`.
    fn blas_ciphertext_alloc(
        &self,
        log_d: usize,
        rows: usize,
        cols: usize,
        base2k: Base2K,
        k: TorusPrecision,
    ) -> BlasCiphertextOwned<BE> {
        let blocks = (0..rows.checked_mul(cols).expect("rows * cols overflows usize"))
            .map(|_| self.ckks_ciphertext_alloc(base2k, k))
            .collect();
        BlasCiphertext::from_blocks(blocks, log_d, rows, cols).expect("a grid has at least one block")
    }

    /// A grid with the dimensions of `infos` and its block layout.
    ///
    /// # Panics
    ///
    /// Panics if the `rows` or `cols` of `infos` is zero, or if their product overflows `usize`.
    fn blas_ciphertext_alloc_from_infos<G: BlasGridInfos>(&self, infos: &G) -> BlasCiphertextOwned<BE> {
        let layout = infos.block_layout();
        let blocks = (0..infos.rows().checked_mul(infos.cols()).expect("rows * cols overflows usize"))
            .map(|_| self.ckks_ciphertext_alloc_from_infos(&layout))
            .collect();
        BlasCiphertext::from_blocks(blocks, infos.log_d(), infos.rows(), infos.cols()).expect("a grid has at least one block")
    }
}

impl<BE: Backend> BlasModuleAlloc<BE> for Module<BE> where Module<BE>: CKKSModuleAlloc<BE> {}

#[cfg(test)]
mod tests {
    use super::*;
    use poulpy_ckks::SetCKKSInfos;
    use poulpy_cpu_oracle::FFT64Oracle;

    fn module() -> Module<FFT64Oracle> {
        Module::<FFT64Oracle>::new(256)
    }

    #[test]
    fn mat_slots_follow_degree_sparsity_and_dimension() {
        let module = module();
        let mut ct = module.blas_ciphertext_alloc(2, 1, 1, 19usize.into(), 60usize.into());
        assert_eq!(ct.log_d(), 2);
        assert_eq!(ct.mat_slots(), 128 / 16);
        ct.block_mut(0, 0).set_log_sparsity(1);
        assert_eq!(ct.mat_slots(), 64 / 16);
        let too_big = module.blas_ciphertext_alloc(4, 1, 1, 19usize.into(), 60usize.into());
        assert_eq!(too_big.mat_slots(), 0);
        // A shift past the word width reads as no matrix, not an overflow.
        let wide = BlasCiphertext::from_blocks(too_big.into_blocks(), 40, 1, 1).unwrap();
        assert_eq!(wide.mat_slots(), 0);
    }

    #[test]
    fn grids_check_their_shape_and_split_and_concat_by_moving_blocks() {
        let module = module();
        let blocks = || (0..6usize).map(|t| module.ckks_ciphertext_alloc(19usize.into(), (40 + t).into()));
        assert!(BlasCiphertext::from_blocks(blocks().take(5).collect(), 2, 2, 3).is_err());
        assert!(BlasCiphertext::from_blocks(blocks().take(0).collect(), 2, 0, 3).is_err());
        let grid = BlasCiphertext::from_blocks(blocks().collect(), 2, 2, 3).unwrap();
        assert_eq!((grid.rows(), grid.cols(), grid.block(1, 2).k().as_usize()), (2, 3, 45));
        // Each block carries its own width, so the widths show where every block moved.
        let widths = |g: &BlasCiphertextOwned<FFT64Oracle>| g.blocks().iter().map(|b| b.k().as_usize()).collect::<Vec<_>>();
        let (left, right) = grid.clone().split_cols(1);
        assert_eq!((left.rows(), left.cols(), widths(&left)), (2, 1, vec![40, 43]));
        assert_eq!((right.rows(), right.cols(), widths(&right)), (2, 2, vec![41, 42, 44, 45]));
        let (top, bottom) = grid.clone().split_rows(1);
        assert_eq!((top.rows(), top.cols(), widths(&top)), (1, 3, vec![40, 41, 42]));
        assert_eq!((bottom.rows(), bottom.cols(), widths(&bottom)), (1, 3, vec![43, 44, 45]));

        // Concatenation moves the blocks back, and rejects grids whose dimension or shared extent differ.
        let joined = BlasCiphertext::concat_cols(left.clone(), right.clone()).unwrap();
        assert_eq!((joined.rows(), joined.cols(), widths(&joined)), (2, 3, widths(&grid)));
        let joined = BlasCiphertext::concat_rows(top.clone(), bottom.clone()).unwrap();
        assert_eq!((joined.rows(), joined.cols(), widths(&joined)), (2, 3, widths(&grid)));
        assert!(BlasCiphertext::concat_cols(left.clone(), top.clone()).is_err());
        assert!(BlasCiphertext::concat_rows(top.clone(), left.clone()).is_err());
        let other_d = BlasCiphertext::from_blocks(right.clone().into_blocks(), 3, 2, 2).unwrap();
        assert!(BlasCiphertext::concat_cols(left, other_d).is_err());
        let other_d = BlasCiphertext::from_blocks(bottom.into_blocks(), 3, 1, 3).unwrap();
        assert!(BlasCiphertext::concat_rows(top, other_d).is_err());
    }

    #[test]
    fn grids_allocate_from_infos() {
        let module = module();
        let a = module.blas_ciphertext_alloc(2, 2, 3, 19usize.into(), 60usize.into());
        let layout = BlasLayout {
            block: a.block_layout(),
            log_d: 2,
            rows: 3,
            cols: 2,
        };
        let t = module.blas_ciphertext_alloc_from_infos(&layout);
        assert_eq!((t.log_d(), t.rows(), t.cols(), t.block_layout()), (2, 3, 2, a.block_layout()));
    }
}
