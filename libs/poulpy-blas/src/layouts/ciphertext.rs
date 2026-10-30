//! A grid of CKKS ciphertexts, each holding one `d x d` block of every matrix of a batch.

use anyhow::{Result, ensure};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use poulpy_ckks::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, CoeffsMeta, GLWEPlaintextInfos, SetCKKSInfos,
    layouts::{CKKSCiphertext, CKKSModuleAlloc},
};
use poulpy_core::{
    GLWEBytesOf,
    layouts::{Base2K, GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetDegree, LWEInfos, TorusPrecision},
};
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef, Module, PrepareHint, ReaderFrom, Ring, WriterTo, ZnxWord};

use super::{
    BlasContextInfos, BlasContextUnprepared, BlasMatVecOperandInfos, BlasMatVecOperandLayout, BlasMatVecOperandUnprepared,
    BlasPlan, BlasPtOperandInfos, BlasPtOperandLayout, BlasPtOperandUnprepared, BlasPtSide, BlasVector, BlasVectorInfos,
    BlasVectorOwned, BlasVectorShape, context, matvec_operand, pt_operand, vector,
};

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
    /// The batch record; implementations stamp it on `res` after `Ok`.
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

    /// The grid's dimensions and block layout, without its batch record.
    fn blas_layout(&self) -> BlasLayout {
        BlasLayout::new(self.block_layout(), self.log_d(), self.rows(), self.cols())
    }
}

impl<T: BlasGridInfos + ?Sized> BlasGridInfos for &T {
    fn log_d(&self) -> usize {
        (**self).log_d()
    }

    fn rows(&self) -> usize {
        (**self).rows()
    }

    fn cols(&self) -> usize {
        (**self).cols()
    }

    fn block_layout(&self) -> CKKSLayout {
        (**self).block_layout()
    }

    fn mat_slots(&self) -> Option<usize> {
        (**self).mat_slots()
    }
}

impl<T: BlasGridInfos + ?Sized> BlasGridInfos for &mut T {
    fn log_d(&self) -> usize {
        (**self).log_d()
    }

    fn rows(&self) -> usize {
        (**self).rows()
    }

    fn cols(&self) -> usize {
        (**self).cols()
    }

    fn block_layout(&self) -> CKKSLayout {
        (**self).block_layout()
    }

    fn mat_slots(&self) -> Option<usize> {
        (**self).mat_slots()
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

    /// Records the batch, [`BlasGridInfos::mat_slots`]. Only implementations call it, on `res` after `Ok`, with the
    /// batch their validator returns.
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

/// A prepared operand of [`blas_matmul_prepared`](crate::api::BlasMatMulOps::blas_matmul_prepared), beyond its
/// [`BlasGridInfos`]: the hint it was prepared under and the block layout of the grid it was prepared from. A hot
/// operand, `OneShot`, reports that layout as its block layout too, since the context of the product it enters defines
/// its terms; a cold one, `Reuse`, reports the layout of its terms.
pub trait BlasMatMulOperandInfos: BlasGridInfos {
    /// `OneShot` for a hot operand, prepared for one product; `Reuse` for a cold one, prepared for many.
    fn hint(&self) -> PrepareHint;

    /// The block layout of the grid the operand was prepared from.
    fn source_block_layout(&self) -> CKKSLayout;
}

/// The infos a preparation of a grid under a hint reports, before preparing it, as
/// [`blas_matmul_left_operand_layout`](crate::layouts::blas_matmul_left_operand_layout) and
/// [`blas_matmul_right_operand_layout`](crate::layouts::blas_matmul_right_operand_layout) give them, for the prepared
/// product's query. Like a [`BlasLayout`], it carries no batch record.
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
pub(crate) fn slots_of<R: Ring>(n: usize) -> usize {
    n * R::CYCLOTOMIC_ORDER_FACTOR as usize / 4
}

/// The batch of `g`, the number of matrices its blocks hold, on ring `R` (`BE::Ring`): its record
/// ([`BlasGridInfos::mat_slots`]), or for a grid without one the batch its blocks' label proves. Every operation reads an
/// input's batch so, and a preparation records it on its operand.
pub fn grid_batch<R: Ring>(g: &impl BlasGridInfos) -> usize {
    grid_batch_at(slots_of::<R>(g.block_layout().n().as_usize()), g)
}

/// [`grid_batch`] at `slots` slots, which the caller supplies.
pub(crate) fn grid_batch_at(slots: usize, g: &impl BlasGridInfos) -> usize {
    g.mat_slots()
        .unwrap_or_else(|| mat_slots_in(slots, g.log_d(), g.block_layout().log_sparsity()))
}

/// Whether every matrix slot of `g`, on ring `R`, holds the same matrix, as its label proves: the slots repeat with a
/// period of at most `d^2`, so a rotation by a multiple of `d` wraps inside a block. It is meaningful once `g` passes
/// the batch check every operation runs first: a label proving no matrix makes it vacuously true.
pub fn replicated<R: Ring>(g: &impl BlasGridInfos) -> bool {
    replicated_at(slots_of::<R>(g.block_layout().n().as_usize()), g)
}

/// [`replicated`] at `slots` slots, which the caller supplies: the context's or the operand's slot count. The validators
/// reject a grid of another degree.
pub(crate) fn replicated_at(slots: usize, g: &impl BlasGridInfos) -> bool {
    mat_slots_in(slots, g.log_d(), g.block_layout().log_sparsity()) <= 1
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
    /// whatever label the plaintext carries. So encrypt a batch of `count` matrices from a plaintext of `count * d^2`
    /// slots, of degree `2 * count * d^2` on the standard ring and `count * d^2` on the conjugate-invariant one; a
    /// plaintext of degree `N` records the full batch `S / d^2`, its matrices repeated.
    ///
    /// A plaintext's degree is at least the backend's `MIN_DEGREE`, and a degree raised to it proves a larger batch
    /// than `count`. So such a batch records `count` only when its degree reaches the `MIN_DEGREE` of every backend it
    /// runs on; otherwise encrypt it at degree `N`.
    ///
    /// Plaintext operations keep the smaller of the ciphertext's label and the plaintext's: adding a degree-`N` plaintext
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

impl<D: Data, W: ZnxWord, R: Ring> BlasCiphertext<D, W, R> {
    /// The grid's header: `log_d` (u8), its rows and columns (u64), then the blocks' `n` (u64), `base2k`, `k` (u32) and
    /// rank (u64). Blocks that differ in `n`, `base2k`, `k` or rank, which it cannot describe, are
    /// [`std::io::ErrorKind::InvalidInput`].
    fn header(&self) -> std::io::Result<Vec<u8>> {
        let first = self.blocks[0].glwe_layout();
        if self.blocks.iter().any(|block| block.glwe_layout() != first) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "the blocks of the grid do not share one layout",
            ));
        }
        let log_d = u8::try_from(self.log_d)
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "the block dimension exceeds 2^255"))?;
        let mut header = vec![log_d];
        header.write_u64::<LittleEndian>(self.rows as u64)?;
        header.write_u64::<LittleEndian>(self.cols as u64)?;
        let block = self.blocks[0].glwe_layout();
        header.write_u64::<LittleEndian>(block.n.as_usize() as u64)?;
        header.write_u32::<LittleEndian>(block.base2k.as_u32())?;
        header.write_u32::<LittleEndian>(block.k.as_u32())?;
        header.write_u64::<LittleEndian>(block.rank.as_usize() as u64)?;
        Ok(header)
    }
}

impl<D: HostDataRef, W: ZnxWord, R: Ring> WriterTo for BlasCiphertext<D, W, R> {
    /// Serialises the grid in little-endian binary format: its header, its batch record (u64), then every block,
    /// row-major, as a [`CKKSCiphertext`] of its first `ceil(k / base2k)` limbs, so the stream follows the header's `k`,
    /// not the allocation. Blocks that differ in `n`, `base2k`, `k` or rank, which the header cannot describe, and a
    /// flag-clear block, fail with [`std::io::ErrorKind::InvalidInput`], writing nothing: normalize it first.
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        let header = self.header()?;
        if self.blocks.iter().any(|block| !GLWE::is_canonical(block)) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "a block is not canonical: normalize it before serializing",
            ));
        }
        writer.write_all(&header)?;
        writer.write_u64::<LittleEndian>(self.mat_slots as u64)?;
        self.blocks.iter().try_for_each(|block| block.write_live_to(writer))
    }
}

impl<D: HostDataMut, W: ZnxWord, R: Ring> ReaderFrom for BlasCiphertext<D, W, R> {
    /// Deserialises into a grid allocated from the writer's layout, or one of more limbs per block, which then holds the
    /// written limbs: a receiver whose blocks differ in layout fails with [`std::io::ErrorKind::InvalidInput`], another
    /// `log_d`, row or column count, or block `n`, `base2k`, `k` or rank with [`std::io::ErrorKind::InvalidData`], before
    /// any block is read. The grid then records the written batch.
    fn read_from<Rd: std::io::Read>(&mut self, reader: &mut Rd) -> std::io::Result<()> {
        let header = self.header()?;
        let mut read = vec![0u8; header.len()];
        reader.read_exact(&mut read)?;
        if read != header {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "the serialized layout differs from the receiver's",
            ));
        }
        let mat_slots = usize::try_from(reader.read_u64::<LittleEndian>()?)
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "the batch record exceeds a usize"))?;
        self.blocks.iter_mut().try_for_each(|block| block.read_from(reader))?;
        self.mat_slots = mat_slots;
        Ok(())
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

/// Allocation of grids.
pub trait BlasModuleAlloc<BE: Backend>: CKKSModuleAlloc<BE> + GetDegree + Sized {
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

    /// A vector of `shape` on rank-1 ciphertexts at torus width `k`, labelled dense.
    ///
    /// # Panics
    ///
    /// Panics if `shape` counts no ciphertext: no entry, or no entry per ciphertext.
    fn blas_vector_alloc(&self, shape: BlasVectorShape, base2k: Base2K, k: TorusPrecision) -> BlasVectorOwned<BE> {
        let cts = (0..shape.count()).map(|_| self.ckks_ciphertext_alloc(base2k, k)).collect();
        BlasVector::from_cts(cts, shape).expect("a vector has at least one entry and one entry per ciphertext")
    }

    /// A vector of the shape and ciphertext layout of `infos`.
    ///
    /// # Panics
    ///
    /// As [`Self::blas_vector_alloc`].
    fn blas_vector_alloc_from_infos<V: BlasVectorInfos>(&self, infos: &V) -> BlasVectorOwned<BE> {
        vector::alloc(self, &infos.ct_layout(), infos.vector_shape())
    }

    /// A public operand of `rows x cols` blocks of `count` public matrices per batch on side `S`, on this module's
    /// slots, every diagonal zero, which
    /// [`BlasEncodingOps::blas_pt_operand_encode`](crate::api::BlasEncodingOps::blas_pt_operand_encode)
    /// fills.
    ///
    /// # Panics
    ///
    /// Panics if the blocks of `plan` are smaller than `2 x 2` or do not fit the module's slots, if the grid has no block
    /// or more than a `usize` counts, if `count` does not divide the matrix slots, if `coeffs_meta.k` is `0`, or if
    /// `base2k` is outside `1..=63`.
    fn blas_pt_operand_alloc<S: BlasPtSide>(
        &self,
        plan: BlasPlan,
        rows: usize,
        cols: usize,
        count: usize,
        base2k: Base2K,
        coeffs_meta: CoeffsMeta,
    ) -> BlasPtOperandUnprepared<BE, S> {
        let layout = BlasPtOperandLayout::<S>::new(plan, self.ckks_max_slots(), rows, cols, count, base2k, coeffs_meta);
        self.blas_pt_operand_alloc_from_infos(&layout)
    }

    /// A public operand of `infos`, which must be of this module's slots.
    ///
    /// # Panics
    ///
    /// As [`Self::blas_pt_operand_alloc`], and if the slots of `infos` are not the module's.
    fn blas_pt_operand_alloc_from_infos<A: BlasPtOperandInfos>(&self, infos: &A) -> BlasPtOperandUnprepared<BE, A::Side> {
        pt_operand::alloc(self, &infos.blas_pt_operand_layout())
    }

    /// A public operand of products of a vector by a matrix from vectors of shape `input` repeating every
    /// `input_period` slots into vectors of shape `output` repeating every `output_period` slots, on this module's
    /// slots and the `Direct` schedule, every diagonal zero, which
    /// [`BlasEncodingOps::blas_matvec_operand_encode`](crate::api::BlasEncodingOps::blas_matvec_operand_encode) fills.
    /// Another schedule is a [`BlasMatVecOperandLayout`] built with
    /// [`with_strategy`](BlasMatVecOperandLayout::with_strategy) or
    /// [`with_optimal_bsgs`](BlasMatVecOperandLayout::with_optimal_bsgs), allocated by
    /// [`Self::blas_matvec_operand_alloc_from_infos`].
    ///
    /// # Panics
    ///
    /// Panics unless each shape holds an entry, an entry per ciphertext and a positive stride, and its entries fit its
    /// period, which divides the slots, if the operand has more blocks than a `usize` counts, if `coeffs_meta.k` is `0`,
    /// if `base2k` is outside `1..=63`, or, from infos, for a schedule other than `Direct` and `Bsgs` of a positive
    /// giant step.
    fn blas_matvec_operand_alloc(
        &self,
        input: BlasVectorShape,
        input_period: usize,
        output: BlasVectorShape,
        output_period: usize,
        base2k: Base2K,
        coeffs_meta: CoeffsMeta,
    ) -> BlasMatVecOperandUnprepared<BE> {
        let layout = BlasMatVecOperandLayout::new(
            self.ckks_max_slots(),
            input,
            input_period,
            output,
            output_period,
            base2k,
            coeffs_meta,
        );
        self.blas_matvec_operand_alloc_from_infos(&layout)
    }

    /// A public operand of products of a vector by a matrix of `infos`, which must be of this module's slots.
    ///
    /// # Panics
    ///
    /// As [`Self::blas_matvec_operand_alloc`], and if the slots of `infos` are not the module's.
    fn blas_matvec_operand_alloc_from_infos<A: BlasMatVecOperandInfos>(&self, infos: &A) -> BlasMatVecOperandUnprepared<BE> {
        matvec_operand::alloc(self, &infos.blas_matvec_operand_layout())
    }

    /// A context of `plan` on this module's ring, every diagonal zero, which
    /// [`BlasEncodingOps::blas_context_encode`](crate::api::BlasEncodingOps::blas_context_encode) fills.
    ///
    /// # Panics
    ///
    /// Panics if the blocks of `plan` are smaller than `2 x 2` or do not fit the module's slots, if `coeffs_meta.k` is
    /// `0` or if `base2k` is outside `1..=63`.
    fn blas_context_alloc(&self, plan: BlasPlan, base2k: Base2K, coeffs_meta: CoeffsMeta) -> BlasContextUnprepared<BE> {
        self.blas_context_alloc_from_infos(&context::module_layout(self, plan, base2k, coeffs_meta))
    }

    /// A context of `infos`, which must be of this module's ring.
    ///
    /// # Panics
    ///
    /// As [`Self::blas_context_alloc`], and if the degree or the slots of `infos` are not the module's.
    fn blas_context_alloc_from_infos<A: BlasContextInfos>(&self, infos: &A) -> BlasContextUnprepared<BE> {
        context::alloc(self, &infos.blas_context_layout())
    }
}

impl<BE: Backend> BlasModuleAlloc<BE> for Module<BE> where Module<BE>: CKKSModuleAlloc<BE> {}

/// Byte sizes of grids, routed through the backend.
pub trait BlasBytesOf<BE: Backend>: GLWEBytesOf<BE> + CKKSModuleAlloc<BE> + GetDegree + Sized {
    /// Bytes of [`BlasModuleAlloc::blas_ciphertext_alloc_from_infos`] of `infos`.
    ///
    /// # Panics
    ///
    /// Panics if the bytes overflow `usize`.
    fn blas_ciphertext_bytes_of_from_infos<G: BlasGridInfos>(&self, infos: &G) -> usize {
        let block = self.glwe_bytes_of_from_infos(&infos.block_layout());
        infos
            .rows()
            .checked_mul(infos.cols())
            .and_then(|blocks| blocks.checked_mul(block))
            .expect("the grid's bytes overflow usize")
    }

    /// Bytes of [`BlasModuleAlloc::blas_vector_alloc_from_infos`] of `infos`.
    ///
    /// # Panics
    ///
    /// Panics if the bytes overflow `usize`.
    fn blas_vector_bytes_of_from_infos<V: BlasVectorInfos>(&self, infos: &V) -> usize {
        let ct = self.glwe_bytes_of_from_infos(&infos.ct_layout());
        infos
            .vector_shape()
            .count()
            .checked_mul(ct)
            .expect("the vector's bytes overflow usize")
    }
}

impl<BE: Backend> BlasBytesOf<BE> for Module<BE> where Module<BE>: CKKSModuleAlloc<BE> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layouts::{BlasContext, BlasMatMulOperand, BlasMatVecOperand, BlasPtOperand, BlasVectorMut};
    use poulpy_ckks::{
        CKKSError, SetGLWEPlaintextInfos,
        layouts::{CKKSPreparedLeft, CKKSPreparedRight},
    };
    use poulpy_core::layouts::{SetK, prepared::GLWETensorKeyPrepared};
    use poulpy_hal::layouts::{HostBytesBackend, Standard};
    use poulpy_hal::layouts::{ReaderFrom, WriterTo, ZnxView, ZnxViewMut};

    type Grid = BlasCiphertextOwned<HostBytesBackend>;

    /// Grids and vectors over owned storage are inputs and outputs of every operation on every backend, with no unsafe
    /// impl, and contexts, public operands of either kind, the reference's prepared operands and the prepared terms the
    /// JKLS kernel shares across one block's tasks, poulpy-core's prepared tensor key and errors cross threads: they meet
    /// the traits' `Send` and `Sync` bounds as they are. Borrowed views are not grids, their blocks not being `GLWEToBackendRef`; they are as
    /// thread-safe as their buffers: `BufRef` views `Sync`, `BufMut` views only `Send`.
    fn assert_thread_safe<'a, BE: Backend + 'a>() {
        fn output<BE: Backend, G: BlasGridMut<BE> + Send + Sync>() {}
        fn vector<BE: Backend, V: BlasVectorMut<BE> + Send + Sync>() {}
        fn sync<T: Sync>() {}
        fn send<T: Send>() {}
        fn operand<T: Send + Sync>() {}
        output::<BE, BlasCiphertextOwned<BE>>();
        vector::<BE, BlasVectorOwned<BE>>();
        sync::<BlasMatVecOperand<BE>>();
        sync::<BlasMatVecOperandUnprepared<BE>>();
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
    fn grids_check_their_shape() {
        let module = module();
        let blocks = || (0..6usize).map(|t| module.ckks_ciphertext_alloc(19usize.into(), (40 + t).into()));
        assert!(BlasCiphertext::from_blocks(blocks().take(5).collect(), 2, 2, 3).is_err());
        assert!(BlasCiphertext::from_blocks(blocks().take(0).collect(), 2, 0, 3).is_err());
        let mut grid = BlasCiphertext::from_blocks(blocks().collect(), 2, 2, 3).unwrap();
        assert_eq!((grid.rows(), grid.cols(), grid.block(1, 2).k().as_usize()), (2, 3, 45));
        // A clone carries the record, not the labels' 8.
        grid.mat_slots = 4;
        assert_eq!(grid.clone().mat_slots(), Some(4));
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

    #[test]
    fn grid_bytes_match_their_allocation() {
        let module = module();
        let grid = module.blas_ciphertext_alloc(2, 2, 3, 19usize.into(), 60usize.into());
        let held: usize = grid
            .blocks()
            .iter()
            .map(|block| <HostBytesBackend as Backend>::len_bytes(block.data().data()))
            .sum();
        assert_eq!(module.blas_ciphertext_bytes_of_from_infos(&grid), held);
        assert_eq!(module.blas_ciphertext_bytes_of_from_infos(&grid.blas_layout()), held);
        assert_eq!(module.blas_ciphertext_bytes_of_from_infos(&&grid), held);
        assert_eq!(grid.blas_layout(), BlasLayout::new(grid.block_layout(), 2, 2, 3));
    }

    #[test]
    fn grids_round_trip_and_reject_other_layouts() {
        let module = module();
        let mut grid = module.blas_ciphertext_alloc(2, 2, 3, 19usize.into(), 60usize.into());
        for (t, block) in grid.blocks_mut().iter_mut().enumerate() {
            block.data_mut().raw_mut().iter_mut().for_each(|x| *x = t as i64 - 2);
        }
        let mut bytes = Vec::new();
        grid.write_to(&mut bytes).unwrap();
        let mut back = module.blas_ciphertext_alloc(2, 2, 3, 19usize.into(), 60usize.into());
        back.read_from(&mut bytes.as_slice()).unwrap();
        let mut again = Vec::new();
        back.write_to(&mut again).unwrap();
        assert!(again == bytes, "a read grid serializes as the original");
        assert_eq!(back.mat_slots(), grid.mat_slots());

        // The record travels, though the label proves fewer matrices.
        let mut sparse = relabelled(grid.clone(), 2, 1);
        sparse.mat_slots = 8;
        let mut sparse_bytes = Vec::new();
        sparse.write_to(&mut sparse_bytes).unwrap();
        back.mat_slots = 1;
        back.read_from(&mut sparse_bytes.as_slice()).unwrap();
        assert_eq!((back.mat_slots(), back.block_layout().log_sparsity()), (Some(8), 1));

        // Another shape is rejected before any block is read, the receiver left bit-equal.
        let mut other = module.blas_ciphertext_alloc(2, 3, 2, 19usize.into(), 60usize.into());
        other.blocks_mut()[0].data_mut().raw_mut()[0] = 7;
        let mut before = Vec::new();
        other.write_to(&mut before).unwrap();
        let err = other.read_from(&mut bytes.as_slice()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let mut after = Vec::new();
        other.write_to(&mut after).unwrap();
        assert!(after == before, "a rejected read changed the grid");

        // Another radix at the same limb count, and another precision, are rejected the same way.
        for (base2k, k) in [(20usize, 60usize), (19, 57)] {
            let mut other = module.blas_ciphertext_alloc(2, 2, 3, base2k.into(), k.into());
            other.blocks_mut()[0].data_mut().raw_mut()[0] = 7;
            let mut before = Vec::new();
            other.write_to(&mut before).unwrap();
            let err = other.read_from(&mut bytes.as_slice()).unwrap_err();
            assert_eq!(err.kind(), std::io::ErrorKind::InvalidData, "{base2k} {k}");
            let mut after = Vec::new();
            other.write_to(&mut after).unwrap();
            assert!(after == before, "a rejected read changed the grid: {base2k} {k}");
        }
    }

    #[test]
    fn grids_narrowed_below_their_allocation_read_into_their_layout() {
        let module = module();
        // Four limbs allocated, three live after `k` drops to 50, as an operation writing a narrower result leaves them.
        let narrowed = |fill: bool| {
            let mut grid = module.blas_ciphertext_alloc(2, 2, 1, 19usize.into(), 76usize.into());
            for (t, block) in grid.blocks_mut().iter_mut().enumerate() {
                if fill {
                    for limb in 0..3 {
                        block.data_mut().at_mut(0, limb).fill(t as i64 - limb as i64);
                    }
                }
                block.set_k(50usize.into());
                GLWE::set_canonical(block, true);
            }
            grid
        };
        let grid = narrowed(true);
        let mut bytes = Vec::new();
        grid.write_to(&mut bytes).unwrap();
        for (label, mut back) in [
            ("allocated from its layout", module.blas_ciphertext_alloc_from_infos(&grid)),
            ("allocated as the writer", narrowed(false)),
        ] {
            back.read_from(&mut bytes.as_slice()).unwrap();
            assert_eq!(back.blas_layout(), grid.blas_layout(), "{label}");
            for (have, want) in back.blocks().iter().zip(grid.blocks()) {
                assert_eq!(have.data().size(), 3, "{label}: the live limbs");
                for limb in 0..3 {
                    assert_eq!(have.data().at(0, limb), want.data().at(0, limb), "{label}");
                }
            }
            let mut again = Vec::new();
            back.write_to(&mut again).unwrap();
            assert!(again == bytes, "{label}: a read grid serializes as the original");
        }
    }

    #[test]
    fn grid_write_rejects_a_flag_clear_block_writing_nothing() {
        let mut grid = module().blas_ciphertext_alloc(2, 1, 2, 19usize.into(), 60usize.into());
        GLWE::set_canonical(grid.block_mut(0, 1), false);
        let mut bytes = Vec::new();
        assert_eq!(
            grid.write_to(&mut bytes).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );
        assert!(bytes.is_empty());
    }

    #[test]
    fn grids_of_blocks_of_different_layouts_neither_write_nor_read() {
        let module = module();
        let blocks = (0..2usize)
            .map(|t| module.ckks_ciphertext_alloc(19usize.into(), (40 + 19 * t).into()))
            .collect();
        let mut grid = BlasCiphertext::from_blocks(blocks, 2, 1, 2).unwrap();
        let mut bytes = Vec::new();
        assert_eq!(
            grid.write_to(&mut bytes).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );
        assert!(bytes.is_empty());

        // A stream whose header matches the first block is rejected before any block is read.
        let mut uniform = module.blas_ciphertext_alloc(2, 1, 2, 19usize.into(), 40usize.into());
        uniform.blocks_mut()[1].data_mut().raw_mut()[0] = 7;
        uniform.write_to(&mut bytes).unwrap();
        assert_eq!(
            grid.read_from(&mut bytes.as_slice()).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );
        assert_eq!(grid.block(0, 1).k().as_usize(), 59);
        assert!(grid.blocks().iter().all(|block| block.data().raw().iter().all(|&x| x == 0)));
    }
}
