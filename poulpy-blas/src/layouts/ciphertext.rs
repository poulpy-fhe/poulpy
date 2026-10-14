//! A grid of CKKS ciphertexts, each holding one `d x d` block of every matrix of a batch.

use anyhow::{Result, ensure};
use poulpy_ckks::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, GLWEPlaintextInfos, SetCKKSInfos,
    layouts::{CKKSCiphertext, CKKSModuleAlloc},
};
use poulpy_core::layouts::{Base2K, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, TorusPrecision};
use poulpy_hal::layouts::{Backend, Data, Module, PrepareHint, Ring, ZnxWord};

/// An `rows x cols` grid of `2^log_d x 2^log_d` blocks: block `(i, j)` is one CKKS ciphertext whose matrix slot `s`
/// holds block `(i, j)` of the `s`-th matrix of the batch.
///
/// The grid records its batch, [`BlasGridInfos::mat_slots`]: the number of matrices its blocks hold, repeated over
/// their slots. [`Self::from_blocks`] records the batch the blocks' label proves, and every operation records one on
/// its result. Every operation also rejects a grid whose label does not allow its record: with `S` slots, degree `n`
/// and label `s`, the record is a power of two, `s <= log2 n`, and `(S >> s) / d^2 <= mat_slots <= S / d^2`.
pub struct BlasCiphertext<D: Data, W: ZnxWord, R: Ring> {
    blocks: Vec<CKKSCiphertext<D, W, R>>,
    log_d: usize,
    rows: usize,
    cols: usize,
    /// The batch record; the delegates stamp it on `res` after `Ok`.
    pub(crate) mat_slots: usize,
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

    /// The recorded batch, the number of matrices each block holds; `None` when only the blocks' label tells it, as
    /// for a [`BlasLayout`].
    fn mat_slots(&self) -> Option<usize> {
        None
    }
}

/// A grid the operations read: its blocks, row-major, shareable across the tasks of a parallel implementation.
pub trait BlasGrid<BE: Backend>: BlasGridInfos {
    /// One block ciphertext.
    type Block: GLWEToBackendRef<BE> + CKKSCtBounds + Sync;

    /// Every block, row-major: block `(i, j)` at `i * cols + j`, so exactly `rows * cols` of them, at least one. Every
    /// operation rejects a grid that breaks this.
    fn blocks(&self) -> &[Self::Block];
}

/// A grid the operations write: each block can move into its own task.
pub trait BlasGridMut<BE: Backend>: BlasGrid<BE, Block: GLWEToBackendMut<BE> + SetCKKSInfos + Send> {
    /// Every block, row-major, mutably.
    fn blocks_mut(&mut self) -> &mut [Self::Block];

    /// Records the batch, [`BlasGridInfos::mat_slots`]. Only the API calls it, on `res` after `Ok`: implementations
    /// write blocks only.
    fn set_mat_slots(&mut self, mat_slots: usize);
}

/// A grid's dimensions and block layout, without ciphertexts; its batch is the one its blocks' label proves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct BlasLayout {
    pub block: CKKSLayout,
    pub log_d: usize,
    pub rows: usize,
    pub cols: usize,
}

impl BlasLayout {
    /// A `rows x cols` grid of `2^log_d x 2^log_d` blocks of layout `block`.
    pub fn new(block: CKKSLayout, log_d: usize, rows: usize, cols: usize) -> Self {
        Self {
            block,
            log_d,
            rows,
            cols,
        }
    }
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

/// A prepared operand of [`crate::api::BlasMatMulOps::blas_matmul_prepared`], beyond its [`BlasGridInfos`]: the hint
/// it was prepared under and the block layout of the grid it was prepared from. A hot operand, `OneShot`, reports that
/// layout as its block layout too, since the context of the product it enters defines its terms; a cold one, `Reuse`,
/// reports the layout of its terms.
pub trait BlasMatMulOperandInfos: BlasGridInfos {
    /// `OneShot` for a hot operand, prepared for one product; `Reuse` for a cold one, prepared for many.
    fn hint(&self) -> PrepareHint;

    /// The block layout of the grid the operand was prepared from.
    fn source_block_layout(&self) -> CKKSLayout;
}

/// The infos a preparation of a grid under a hint reports, before preparing it, as
/// [`blas_matmul_operand_layout`](crate::reference::blas_matmul_operand_layout) gives them, for the prepared product's
/// query. Like a [`BlasLayout`], it carries no batch record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct BlasMatMulOperandLayout {
    /// The grid's dimensions and the block layout the operand reports: the source's when hot, its terms' when cold.
    pub grid: BlasLayout,
    /// The block layout of the source grid.
    pub source: CKKSLayout,
    /// The hint of the preparation.
    pub hint: PrepareHint,
}

impl BlasGridInfos for BlasMatMulOperandLayout {
    fn log_d(&self) -> usize {
        self.grid.log_d
    }

    fn rows(&self) -> usize {
        self.grid.rows
    }

    fn cols(&self) -> usize {
        self.grid.cols
    }

    fn block_layout(&self) -> CKKSLayout {
        self.grid.block
    }
}

impl BlasMatMulOperandInfos for BlasMatMulOperandLayout {
    fn hint(&self) -> PrepareHint {
        self.hint
    }

    fn source_block_layout(&self) -> CKKSLayout {
        self.source
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

/// Slots of a block of degree `n` on ring `R`.
fn slots_of<R: Ring>(n: usize) -> usize {
    n * R::CYCLOTOMIC_ORDER_FACTOR as usize / 4
}

/// The batch of `g`, the number of matrices its blocks hold, on ring `R` (`BE::Ring`): its record
/// ([`BlasGridInfos::mat_slots`]), or for a grid without one the batch its blocks' label proves. Every operation reads an
/// input's batch so, and a preparation records it on its operand.
pub fn grid_batch<R: Ring>(g: &impl BlasGridInfos) -> usize {
    g.mat_slots().unwrap_or_else(|| {
        let block = g.block_layout();
        mat_slots_in(slots_of::<R>(block.n().as_usize()), g.log_d(), block.log_sparsity())
    })
}

/// Whether the label of the blocks of `g`, on ring `R`, allows its batch ([`grid_batch`]): with `S` slots, label `s` and
/// degree `n`, a power of two with `s <= log2 n` and `(S >> s) / d^2 <= mat_slots <= S / d^2`. The lower bound covers
/// whole periods of the slots, so rotations modulo the batch and sums over it keep their meaning.
pub(crate) fn batch_fits<R: Ring>(g: &impl BlasGridInfos) -> bool {
    let block = g.block_layout();
    let (n, s) = (block.n().as_usize(), block.log_sparsity());
    let (slots, mat_slots) = (slots_of::<R>(n), grid_batch::<R>(g));
    mat_slots.is_power_of_two()
        && s <= n.ilog2() as usize
        && mat_slots_in(slots, g.log_d(), s) <= mat_slots
        && mat_slots <= mat_slots_in(slots, g.log_d(), 0)
}

impl<D: Data, W: ZnxWord, R: Ring> BlasCiphertext<D, W, R> {
    /// Reads `blocks`, row-major, as a `rows x cols` grid of `2^log_d x 2^log_d` blocks, recording the batch the label
    /// of block `(0, 0)` proves: with `S` slots and label `s`, `(S >> s) / d^2` matrices, `0` when one block does not
    /// fit (every operation then rejects the grid).
    ///
    /// A label `s` claims the plaintext lies in `Z[X^(2^s)]`, so the slots repeat with period `S >> s`. Encryption
    /// derives it from the plaintext degree: a plaintext of degree `n_pt` labels its ciphertext `log2(N / n_pt)`,
    /// whatever label the plaintext carries. So encrypt a batch of `count` matrices from a compact plaintext,
    /// `ckks_pt_vec_alloc_compact(count * d^2)`; a dense plaintext records the full batch `S / d^2`, its matrices
    /// repeated.
    ///
    /// Compact degrees are raised to the backend's `MIN_DEGREE` (8 by default, 16 on AVX-512 FFT64), and a raised
    /// degree proves a larger batch than `count`. So a compact batch records `count` on every backend only when its
    /// degree, `2 * count * d^2` on the standard ring and `count * d^2` on the conjugate-invariant one, reaches every
    /// target's `MIN_DEGREE` (16 in this workspace); otherwise encrypt it dense.
    ///
    /// Plaintext operations keep the smaller of the ciphertext's label and the plaintext's: adding a dense plaintext
    /// to the blocks of a sparse batch drops their label to 0, so the next operation rejects the grid.
    pub fn from_blocks(blocks: Vec<CKKSCiphertext<D, W, R>>, log_d: usize, rows: usize, cols: usize) -> Result<Self> {
        ensure!(rows > 0 && cols > 0, "a grid has at least one block");
        ensure!(
            rows.checked_mul(cols) == Some(blocks.len()),
            "{} blocks do not form a {rows} x {cols} grid",
            blocks.len()
        );
        let block = &blocks[0];
        let mat_slots = mat_slots_in(slots_of::<R>(block.n().as_usize()), log_d, block.log_sparsity());
        Ok(Self {
            blocks,
            log_d,
            rows,
            cols,
            mat_slots,
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

    /// Moves the first `at` block columns into the first grid and the others into the second; panics unless
    /// `0 < at < cols`.
    pub fn split_cols(self, at: usize) -> (Self, Self) {
        let (rows, cols, log_d, mat_slots) = (self.rows, self.cols, self.log_d, self.mat_slots);
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
            mat_slots,
        };
        (grid(left, at), grid(right, cols - at))
    }

    /// Moves the first `at` block rows into the first grid and the others into the second; panics unless
    /// `0 < at < rows`.
    pub fn split_rows(self, at: usize) -> (Self, Self) {
        let (rows, cols, log_d, mat_slots) = (self.rows, self.cols, self.log_d, self.mat_slots);
        assert!(0 < at && at < rows, "split_rows: {at} is not strictly inside 0..{rows}");
        let mut top = self.blocks;
        let bottom = top.split_off(at * cols);
        let grid = |blocks, rows| Self {
            blocks,
            log_d,
            rows,
            cols,
            mat_slots,
        };
        (grid(top, at), grid(bottom, rows - at))
    }

    /// Moves the blocks of `left` then `right` into one grid, the inverse of [`Self::split_cols`]; rejects grids of
    /// different `log_d`, row counts or batch records.
    pub fn concat_cols(left: Self, right: Self) -> Result<Self> {
        ensure!(
            (left.log_d, left.rows, left.mat_slots) == (right.log_d, right.rows, right.mat_slots),
            "concat_cols: a 2^{} x {}-row grid of {} matrices against a 2^{} x {}-row grid of {}",
            left.log_d,
            left.rows,
            left.mat_slots,
            right.log_d,
            right.rows,
            right.mat_slots
        );
        let (rows, lc, rc, mat_slots) = (left.rows, left.cols, right.cols, left.mat_slots);
        let mut blocks = Vec::with_capacity(rows * (lc + rc));
        let (mut l, mut r) = (left.blocks.into_iter(), right.blocks.into_iter());
        for _ in 0..rows {
            blocks.extend(l.by_ref().take(lc));
            blocks.extend(r.by_ref().take(rc));
        }
        Ok(Self {
            mat_slots,
            ..Self::from_blocks(blocks, left.log_d, rows, lc + rc)?
        })
    }

    /// Moves the blocks of `top` then `bottom` into one grid, the inverse of [`Self::split_rows`]; rejects grids of
    /// different `log_d`, column counts or batch records.
    pub fn concat_rows(top: Self, bottom: Self) -> Result<Self> {
        ensure!(
            (top.log_d, top.cols, top.mat_slots) == (bottom.log_d, bottom.cols, bottom.mat_slots),
            "concat_rows: a 2^{} x {}-column grid of {} matrices against a 2^{} x {}-column grid of {}",
            top.log_d,
            top.cols,
            top.mat_slots,
            bottom.log_d,
            bottom.cols,
            bottom.mat_slots
        );
        let (log_d, rows, cols, mat_slots) = (top.log_d, top.rows + bottom.rows, top.cols, top.mat_slots);
        let mut blocks = top.blocks;
        blocks.extend(bottom.blocks);
        Ok(Self {
            mat_slots,
            ..Self::from_blocks(blocks, log_d, rows, cols)?
        })
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
            mat_slots: self.mat_slots,
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

    fn mat_slots(&self) -> Option<usize> {
        Some(self.mat_slots)
    }
}

impl<BE: Backend, D: Data + Sync> BlasGrid<BE> for BlasCiphertext<D, BE::ZnxWord, BE::Ring>
where
    CKKSCiphertext<D, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
{
    type Block = CKKSCiphertext<D, BE::ZnxWord, BE::Ring>;

    fn blocks(&self) -> &[Self::Block] {
        &self.blocks
    }
}

impl<BE: Backend, D: Data + Send + Sync> BlasGridMut<BE> for BlasCiphertext<D, BE::ZnxWord, BE::Ring>
where
    CKKSCiphertext<D, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
{
    fn blocks_mut(&mut self) -> &mut [Self::Block] {
        &mut self.blocks
    }

    fn set_mat_slots(&mut self, mat_slots: usize) {
        self.mat_slots = mat_slots;
    }
}

/// The left operand of [`crate::api::BlasMatMulOps::blas_matmul_prepared`], hot or cold, in `BE`'s representation,
/// which only [`crate::api::BlasMatMulOps::blas_matmul_prepare_left`] makes.
pub type BlasMatMulLeft<BE> = <BE as crate::oep::BlasMatMulImpl>::MatMulLeft;

/// The right operand of [`crate::api::BlasMatMulOps::blas_matmul_prepared`], hot or cold, in `BE`'s representation,
/// which only [`crate::api::BlasMatMulOps::blas_matmul_prepare_right`] makes.
pub type BlasMatMulRight<BE> = <BE as crate::oep::BlasMatMulImpl>::MatMulRight;

/// Allocation of grids.
pub trait BlasModuleAlloc<BE: Backend>: CKKSModuleAlloc<BE> {
    /// A grid of `rows x cols` rank-1 blocks of dimension `2^log_d` at torus width `k`, recording the dense batch.
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

    /// A grid with the dimensions of `infos`, its block layout and its batch record when it has one; otherwise the
    /// batch the layout's label proves.
    ///
    /// # Panics
    ///
    /// Panics if the `rows` or `cols` of `infos` is zero, or if their product overflows `usize`.
    fn blas_ciphertext_alloc_from_infos<G: BlasGridInfos>(&self, infos: &G) -> BlasCiphertextOwned<BE> {
        let layout = infos.block_layout();
        let blocks = (0..infos.rows().checked_mul(infos.cols()).expect("rows * cols overflows usize"))
            .map(|_| self.ckks_ciphertext_alloc_from_infos(&layout))
            .collect();
        let grid = BlasCiphertext::from_blocks(blocks, infos.log_d(), infos.rows(), infos.cols())
            .expect("a grid has at least one block");
        BlasCiphertext {
            mat_slots: infos.mat_slots().unwrap_or(grid.mat_slots),
            ..grid
        }
    }
}

impl<BE: Backend> BlasModuleAlloc<BE> for Module<BE> where Module<BE>: CKKSModuleAlloc<BE> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        layouts::{BlasContext, BlasPtOperand},
        reference::BlasMatMulOperand,
    };
    use poulpy_ckks::{
        CKKSError, SetGLWEPlaintextInfos,
        layouts::{CKKSPreparedLeft, CKKSPreparedRight},
    };
    use poulpy_core::layouts::prepared::GLWETensorKeyPrepared;
    use poulpy_hal::layouts::{HostBytesBackend, Standard};

    type Grid = BlasCiphertextOwned<HostBytesBackend>;

    /// Grids over owned storage are inputs and outputs of every operation on every backend, with no unsafe impl, and
    /// contexts, public operands, the reference's prepared operands and the prepared terms the JKLS kernel shares across
    /// one block's tasks, poulpy-core's prepared tensor key and errors cross threads: they meet the traits' `Send` and
    /// `Sync` bounds as they are. Borrowed views are not grids, their blocks not being `GLWEToBackendRef`; they are as
    /// thread-safe as their buffers: `BufRef` views `Sync`, `BufMut` views only `Send`.
    fn assert_thread_safe<'a, BE: Backend + 'a>() {
        fn output<BE: Backend, G: BlasGridMut<BE> + Send + Sync>() {}
        fn sync<T: Sync>() {}
        fn send<T: Send>() {}
        fn operand<T: Send + Sync>() {}
        output::<BE, BlasCiphertextOwned<BE>>();
        sync::<BlasContext<BE>>();
        sync::<BlasPtOperand<BE>>();
        operand::<BlasMatMulOperand<BE, CKKSPreparedLeft<BE>>>();
        operand::<BlasMatMulOperand<BE, CKKSPreparedRight<BE>>>();
        sync::<CKKSPreparedLeft<BE>>();
        sync::<CKKSPreparedRight<BE>>();
        sync::<GLWETensorKeyPrepared<BE::OwnedBuf, BE>>();
        send::<CKKSError>();
        sync::<BlasCiphertext<BE::BufRef<'a>, BE::ZnxWord, BE::Ring>>();
        send::<BlasCiphertext<BE::BufMut<'a>, BE::ZnxWord, BE::Ring>>();
    }

    #[test]
    fn grids_are_send_and_sync() {
        assert_thread_safe::<HostBytesBackend>();
    }

    fn module() -> Module<HostBytesBackend> {
        Module::<HostBytesBackend>::new(256)
    }

    /// `grid`'s blocks labelled `log_sparsity`, read again at `log_d`.
    fn relabelled(grid: Grid, log_d: usize, log_sparsity: usize) -> Grid {
        let (rows, cols) = (grid.rows(), grid.cols());
        let mut blocks = grid.into_blocks();
        blocks.iter_mut().for_each(|block| block.set_log_sparsity(log_sparsity));
        BlasCiphertext::from_blocks(blocks, log_d, rows, cols).unwrap()
    }

    #[test]
    fn grids_record_the_batch_their_labels_prove() {
        // 128 slots: 8 matrices of 4 x 4 blocks.
        let module = module();
        let ct = module.blas_ciphertext_alloc(2, 1, 1, 19usize.into(), 60usize.into());
        assert_eq!((ct.log_d(), ct.mat_slots()), (2, Some(8)));
        for (log_sparsity, mat_slots) in [(1, 4), (3, 1), (4, 0), (64, 0), (65, 0)] {
            let ct = relabelled(ct.clone(), 2, log_sparsity);
            assert_eq!(ct.mat_slots(), Some(mat_slots), "log_sparsity={log_sparsity}");
        }
        let too_big = module.blas_ciphertext_alloc(4, 1, 1, 19usize.into(), 60usize.into());
        assert_eq!(too_big.mat_slots(), Some(0));
        // A shift past the word width reads as no matrix, not an overflow.
        let wide = BlasCiphertext::from_blocks(too_big.into_blocks(), 40, 1, 1).unwrap();
        assert_eq!(wide.mat_slots(), Some(0));
    }

    #[test]
    fn records_fit_labels_from_the_proven_batch_to_the_dense_one() {
        // Degree 256, 128 slots: labels up to 8, records from `(128 >> s) / 16` to 8.
        let mut ct = module().blas_ciphertext_alloc(2, 1, 1, 19usize.into(), 60usize.into());
        for (log_sparsity, mat_slots, fits) in [
            (0, 8, true),
            (0, 4, false),
            (1, 4, true),
            (1, 8, true),
            (1, 2, false),
            (1, 6, false),
            (1, 16, false),
            (3, 1, true),
            (8, 1, true),
            (9, 1, false),
            (64, 1, false),
            (0, 0, false),
            (4, 0, false),
        ] {
            ct.block_mut(0, 0).set_log_sparsity(log_sparsity);
            ct.mat_slots = mat_slots;
            assert_eq!(
                batch_fits::<Standard>(&ct),
                fits,
                "log_sparsity={log_sparsity}, mat_slots={mat_slots}"
            );
        }
        // 16 x 16 blocks do not fit 128 slots, whatever the record.
        let mut ct = relabelled(ct, 4, 0);
        ct.mat_slots = 1;
        assert!(!batch_fits::<Standard>(&ct));
    }

    #[test]
    fn grids_check_their_shape_and_split_and_concat_by_moving_blocks() {
        let module = module();
        let blocks = || (0..6usize).map(|t| module.ckks_ciphertext_alloc(19usize.into(), (40 + t).into()));
        assert!(BlasCiphertext::from_blocks(blocks().take(5).collect(), 2, 2, 3).is_err());
        assert!(BlasCiphertext::from_blocks(blocks().take(0).collect(), 2, 0, 3).is_err());
        let mut grid = BlasCiphertext::from_blocks(blocks().collect(), 2, 2, 3).unwrap();
        assert_eq!((grid.rows(), grid.cols(), grid.block(1, 2).k().as_usize()), (2, 3, 45));
        // A record other than the labels' 8, so that every grid below shows it carried.
        grid.mat_slots = 4;
        assert_eq!(grid.clone().mat_slots(), Some(4));
        // Each block carries its own width, so the widths show where every block moved.
        let parts = |g: &Grid| {
            (
                g.rows(),
                g.cols(),
                g.mat_slots,
                g.blocks().iter().map(|b| b.k().as_usize()).collect(),
            )
        };
        let (left, right) = grid.clone().split_cols(1);
        assert_eq!(parts(&left), (2, 1, 4, vec![40, 43]));
        assert_eq!(parts(&right), (2, 2, 4, vec![41, 42, 44, 45]));
        let (top, bottom) = grid.clone().split_rows(1);
        assert_eq!(parts(&top), (1, 3, 4, vec![40, 41, 42]));
        assert_eq!(parts(&bottom), (1, 3, 4, vec![43, 44, 45]));

        // Concatenation moves the blocks back, and rejects grids whose dimension, shared extent or record differ.
        let joined = BlasCiphertext::concat_cols(left.clone(), right.clone()).unwrap();
        assert_eq!(parts(&joined), parts(&grid));
        let joined = BlasCiphertext::concat_rows(top.clone(), bottom.clone()).unwrap();
        assert_eq!(parts(&joined), parts(&grid));
        assert!(BlasCiphertext::concat_cols(left.clone(), top.clone()).is_err());
        assert!(BlasCiphertext::concat_rows(top.clone(), left.clone()).is_err());
        let other_d = BlasCiphertext::from_blocks(right.clone().into_blocks(), 3, 2, 2).unwrap();
        assert!(BlasCiphertext::concat_cols(left.clone(), other_d).is_err());
        let other_d = BlasCiphertext::from_blocks(bottom.clone().into_blocks(), 3, 1, 3).unwrap();
        assert!(BlasCiphertext::concat_rows(top.clone(), other_d).is_err());
        let (mut right, mut bottom) = (right, bottom);
        (right.mat_slots, bottom.mat_slots) = (2, 2);
        assert!(BlasCiphertext::concat_cols(left, right).is_err());
        assert!(BlasCiphertext::concat_rows(top, bottom).is_err());
    }

    #[test]
    fn grids_allocate_from_infos() {
        let module = module();
        let mut a = module.blas_ciphertext_alloc(2, 2, 3, 19usize.into(), 60usize.into());
        let layout = BlasLayout::new(a.block_layout(), 2, 3, 2);
        let t = module.blas_ciphertext_alloc_from_infos(&layout);
        assert_eq!(
            (t.log_d(), t.rows(), t.cols(), t.block_layout(), t.mat_slots()),
            (2, 3, 2, a.block_layout(), Some(8))
        );
        // A layout reads its batch from its label, a grid copies its record.
        let mut sparse = layout;
        sparse.block.meta.log_sparsity = 1;
        assert_eq!(module.blas_ciphertext_alloc_from_infos(&sparse).mat_slots(), Some(4));
        a.mat_slots = 2;
        assert_eq!(module.blas_ciphertext_alloc_from_infos(&a).mat_slots(), Some(2));
    }
}
