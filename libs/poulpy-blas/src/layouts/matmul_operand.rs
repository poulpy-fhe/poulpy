//! The prepared operands of a product: their storage, their allocation, and the layouts of the terms they hold.

use poulpy_ckks::{
    CKKSLayout,
    api::{CKKSMulOps, LinearTransformation, LtDiagonalMeta},
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPreparedLeft, CKKSPreparedRight},
};
use poulpy_core::layouts::GLWELayout;
use poulpy_hal::layouts::{Backend, Module, PrepareHint};

use super::{
    BlasBytesOf, BlasContextInfos, BlasGridInfos, BlasLayout, BlasMatMulOperandInfos, BlasMatMulOperandLayout, BlasModuleAlloc,
    ciphertext::replicated_at,
    context::{ContextMaps, context_shapes},
    encoding::mapped_block,
};

/// The left operand of [`blas_matmul_prepared`](crate::api::BlasMatMulOps::blas_matmul_prepared), hot or cold, which
/// [`BlasMatMulOperandPreparedFactory::blas_matmul_left_alloc_from_infos`] allocates and
/// [`blas_matmul_prepare_left`](crate::api::BlasMatMulOps::blas_matmul_prepare_left) fills.
pub type BlasMatMulLeft<BE> = BlasMatMulOperand<BE, CKKSPreparedLeft<BE>>;

/// The right operand of [`blas_matmul_prepared`](crate::api::BlasMatMulOps::blas_matmul_prepared), hot or cold, which
/// [`BlasMatMulOperandPreparedFactory::blas_matmul_right_alloc_from_infos`] allocates and
/// [`blas_matmul_prepare_right`](crate::api::BlasMatMulOps::blas_matmul_prepare_right) fills.
pub type BlasMatMulRight<BE> = BlasMatMulOperand<BE, CKKSPreparedRight<BE>>;

/// A prepared operand of a product, of terms `P`: [`CKKSPreparedLeft`] on the left, [`BlasMatMulLeft`], and
/// [`CKKSPreparedRight`] on the right, [`BlasMatMulRight`], so that swapping the operands of a product does not
/// compile. Its storage is `BE`'s, poulpy-ckks ciphertexts and prepared terms. An allocation from
/// [`BlasMatMulOperandInfos`] makes all its storage, after their hint, and a preparation fills it:
/// - hot, `OneShot`, prepared for one product: a block of the source's layout per block of its source, row-major, into
///   which the preparation copies the source's block. The product evaluates their terms under its own context;
/// - cold, `Reuse`, prepared for many: the `d` terms of every block of its source, all of the infos' block layout, in
///   the transform domain, each converted once. On the left, term `0` of block `(i, l)` is sigma of the source block,
///   narrowed exactly to the terms' width, and term `k` its phi^k; on the right, tau and psi^k, or for a replicated
///   source [`BlasContext::tau_rep`](crate::layouts::BlasContext::tau_rep) and
///   [`BlasContext::rotate_rows_rep`](crate::layouts::BlasContext::rotate_rows_rep)
///   ([`blas_matmul_right_term_layout`]).
///
/// Its [`BlasMatMulOperandInfos`] report the infos it was allocated from and the batch `0`, then those of its latest
/// preparation, which [`blas_matmul_left_operand_layout`] and [`blas_matmul_right_operand_layout`] give for its source,
/// and its source's batch. The preparation's validator checks that the source's infos have the operand's shape. An
/// implementation of the preparations and of the prepared product reads and fills the storage through [`Self::blocks`]
/// and [`Self::terms`], and records a preparation with [`Self::set_prepared`].
pub struct BlasMatMulOperand<BE: Backend, P> {
    pub(crate) blocks: Vec<CKKSCiphertextOwned<BE>>,
    pub(crate) terms: Vec<P>,
    pub(crate) layout: BlasMatMulOperandLayout,
    pub(crate) mat_slots: usize,
}

impl<BE: Backend, P> BlasMatMulOperand<BE, P> {
    /// A hot operand's blocks, row-major; empty when cold.
    pub fn blocks(&self) -> &[CKKSCiphertextOwned<BE>] {
        &self.blocks
    }

    /// The blocks a hot preparation fills.
    pub fn blocks_mut(&mut self) -> &mut [CKKSCiphertextOwned<BE>] {
        &mut self.blocks
    }

    /// A cold operand's terms, block `(i, j)`'s term `k` at `(i * cols + j) * d + k`; empty when hot.
    pub fn terms(&self) -> &[P] {
        &self.terms
    }

    /// The terms a cold preparation fills.
    pub fn terms_mut(&mut self) -> &mut [P] {
        &mut self.terms
    }

    /// Records a preparation: the operand then reports `layout` and the batch `mat_slots`.
    ///
    /// # Panics
    ///
    /// Panics unless `layout` has the operand's shape: its hint, dimensions and the GLWE layouts of its source's and
    /// reported blocks.
    pub fn set_prepared(&mut self, layout: BlasMatMulOperandLayout, mat_slots: usize) {
        assert!(
            same_shape(&self.layout, &layout),
            "blas_matmul_operand: layout of another shape"
        );
        self.layout = layout;
        self.mat_slots = mat_slots;
    }
}

/// The operand's infos and the count of what it holds; its digits may live on a device.
impl<BE: Backend, P> std::fmt::Debug for BlasMatMulOperand<BE, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlasMatMulOperand")
            .field("layout", &self.layout)
            .field("mat_slots", &self.mat_slots)
            .field("held", &(self.blocks.len() + self.terms.len()))
            .finish()
    }
}

impl<BE: Backend, P> BlasGridInfos for BlasMatMulOperand<BE, P> {
    fn log_d(&self) -> usize {
        self.layout.grid.log_d
    }

    fn rows(&self) -> usize {
        self.layout.grid.rows
    }

    fn cols(&self) -> usize {
        self.layout.grid.cols
    }

    fn block_layout(&self) -> CKKSLayout {
        self.layout.grid.block
    }

    fn mat_slots(&self) -> Option<usize> {
        Some(self.mat_slots)
    }
}

impl<BE: Backend, P> BlasMatMulOperandInfos for BlasMatMulOperand<BE, P> {
    fn hint(&self) -> PrepareHint {
        self.layout.hint
    }

    fn source_block_layout(&self) -> CKKSLayout {
        self.layout.source
    }
}

/// Allocation and sizing of the prepared operands of a product.
pub trait BlasMatMulOperandPreparedFactory<BE: Backend>: BlasModuleAlloc<BE> + BlasBytesOf<BE> + CKKSMulOps<BE> {
    /// Bytes a left operand of `infos` holds, as [`Self::blas_matmul_left_alloc_from_infos`] allocates it.
    fn blas_matmul_left_bytes_of_from_infos<A: BlasMatMulOperandInfos>(&self, infos: &A) -> usize {
        operand_bytes(self, infos, |term| {
            self.ckks_prepared_left_bytes_of_from_infos(term, PrepareHint::Reuse)
        })
    }

    /// A left operand of `infos`, holding all its storage, which
    /// [`blas_matmul_prepare_left`](crate::api::BlasMatMulOps::blas_matmul_prepare_left) fills: the infos
    /// [`blas_matmul_left_operand_layout`] gives for a grid, a context and a hint. Until then it reports `infos` and the
    /// batch `0`, which the prepared product rejects.
    ///
    /// # Panics
    ///
    /// May panic on infos of no block, or whose bytes overflow `usize`.
    fn blas_matmul_left_alloc_from_infos<A: BlasMatMulOperandInfos>(&self, infos: &A) -> BlasMatMulLeft<BE> {
        alloc_operand(self, infos, |term| {
            self.ckks_prepared_left_alloc_from_infos(term, PrepareHint::Reuse)
        })
    }

    /// As [`Self::blas_matmul_left_bytes_of_from_infos`], for a right operand.
    fn blas_matmul_right_bytes_of_from_infos<A: BlasMatMulOperandInfos>(&self, infos: &A) -> usize {
        operand_bytes(self, infos, |term| {
            self.ckks_prepared_right_bytes_of_from_infos(term, PrepareHint::Reuse)
        })
    }

    /// As [`Self::blas_matmul_left_alloc_from_infos`], a right operand, which
    /// [`blas_matmul_prepare_right`](crate::api::BlasMatMulOps::blas_matmul_prepare_right) fills from the infos
    /// [`blas_matmul_right_operand_layout`] gives.
    ///
    /// # Panics
    ///
    /// As [`Self::blas_matmul_left_alloc_from_infos`].
    fn blas_matmul_right_alloc_from_infos<A: BlasMatMulOperandInfos>(&self, infos: &A) -> BlasMatMulRight<BE> {
        alloc_operand(self, infos, |term| {
            self.ckks_prepared_right_alloc_from_infos(term, PrepareHint::Reuse)
        })
    }
}

impl<BE: Backend> BlasMatMulOperandPreparedFactory<BE> for Module<BE> where Module<BE>: CKKSModuleAlloc<BE> + CKKSMulOps<BE> {}

/// An operand of `infos` holding all its storage, the batch `0`: hot, a grid of the source's block layout; cold, `d`
/// terms of `term` per source block.
fn alloc_operand<BE: Backend, P>(
    module: &impl BlasModuleAlloc<BE>,
    infos: &impl BlasMatMulOperandInfos,
    term: impl Fn(&CKKSLayout) -> P,
) -> BlasMatMulOperand<BE, P> {
    let layout = BlasMatMulOperandLayout {
        grid: infos.blas_layout(),
        source: infos.source_block_layout(),
        hint: infos.hint(),
    };
    let (blocks, terms) = match layout.hint {
        PrepareHint::OneShot => (
            module.blas_ciphertext_alloc_from_infos(&hot_layout(infos)).into_blocks(),
            Vec::new(),
        ),
        PrepareHint::Reuse => {
            let count = (layout.grid.rows * layout.grid.cols) << layout.grid.log_d;
            (Vec::new(), (0..count).map(|_| term(&layout.grid.block)).collect())
        }
    };
    BlasMatMulOperand {
        blocks,
        terms,
        layout,
        mat_slots: 0,
    }
}

/// The bytes [`alloc_operand`] allocates for `infos`, `term` bytes per term.
fn operand_bytes<BE: Backend>(
    module: &impl BlasBytesOf<BE>,
    infos: &impl BlasMatMulOperandInfos,
    term: impl Fn(&CKKSLayout) -> usize,
) -> usize {
    match infos.hint() {
        PrepareHint::OneShot => module.blas_ciphertext_bytes_of_from_infos(&hot_layout(infos)),
        PrepareHint::Reuse => ((infos.rows() * infos.cols()) << infos.log_d()) * term(&infos.block_layout()),
    }
}

/// The grid of `infos`'s dimensions at its source's block layout, which a hot operand holds.
fn hot_layout(infos: &impl BlasMatMulOperandInfos) -> BlasLayout {
    BlasLayout::new(infos.source_block_layout(), infos.log_d(), infos.rows(), infos.cols())
}

/// The layout of the left terms of `a` under `ctx`, sigma then phi^k: `a`'s at `a.k - 2p` for the maps' scale `p`, its
/// slots joined with theirs; no batch record. A cold left operand of `a` under `ctx` reports it. `ctx` is a context or
/// its layout: the query reads the shape of its maps.
///
/// # Panics
///
/// Unless the blocks of `ctx`, at least `2 x 2`, fit its slots, and its maps have a torus width and a radix in
/// `1..=63`.
pub fn blas_matmul_left_term_layout(a: &impl BlasGridInfos, ctx: &impl BlasContextInfos) -> BlasLayout {
    left_term_layout(a, &context_shapes(ctx))
}

/// The layout of the right terms of `b` under `ctx`: tau then psi^k at `b.k - 2p`, or for a replicated `b` the
/// rotations at no scale, at `b.k - p`; no batch record. A cold right operand of `b` under `ctx` reports it. `ctx` is a
/// context or its layout: the query reads the shape of its maps.
///
/// # Panics
///
/// As [`blas_matmul_left_term_layout`].
pub fn blas_matmul_right_term_layout(b: &impl BlasGridInfos, ctx: &impl BlasContextInfos) -> BlasLayout {
    right_term_layout(b, &context_shapes(ctx))
}

/// The infos a left preparation of `a` under `ctx` and `hint` reports before preparing it: `hint`, `a`'s dimensions and
/// block layout as the source's, and as the block layout `a`'s when hot, [`blas_matmul_left_term_layout`] when cold.
/// `ctx` is a context or its layout: the query reads the shape of its maps.
///
/// # Panics
///
/// As [`blas_matmul_left_term_layout`].
pub fn blas_matmul_left_operand_layout(
    a: &impl BlasGridInfos,
    ctx: &impl BlasContextInfos,
    hint: PrepareHint,
) -> BlasMatMulOperandLayout {
    operand_layout(a, hint, || left_term_layout(a, &context_shapes(ctx)))
}

/// As [`blas_matmul_left_operand_layout`] for a right preparation, with [`blas_matmul_right_term_layout`].
///
/// # Panics
///
/// As [`blas_matmul_left_term_layout`].
pub fn blas_matmul_right_operand_layout(
    b: &impl BlasGridInfos,
    ctx: &impl BlasContextInfos,
    hint: PrepareHint,
) -> BlasMatMulOperandLayout {
    operand_layout(b, hint, || right_term_layout(b, &context_shapes(ctx)))
}

/// The maps of the right factor's terms under `ctx`: tau and psi^k, or, when the right factor is replicated
/// ([`crate::layouts::replicated`]), [`crate::layouts::BlasContext::tau_rep`] and
/// [`crate::layouts::BlasContext::rotate_rows_rep`], which spend no scale on psi^k.
pub(crate) fn right_maps<'c, T>(maps: &'c ContextMaps<T>, rep: bool) -> (&'c T, impl Fn(usize) -> &'c T + Copy + 'c) {
    let first = if rep { maps.tau_rep() } else { maps.tau() };
    (
        first,
        move |k| if rep { maps.rotate_rows_rep(k) } else { maps.rotate_rows(k) },
    )
}

/// The layout of the terms of `a` through `first` then a shift `shift`: `a`'s dimensions, its block layout at the
/// width both maps leave, its slots joined with theirs; no batch record.
fn term_layout<P: LtDiagonalMeta>(
    a: &impl BlasGridInfos,
    first: &LinearTransformation<P>,
    shift: &LinearTransformation<P>,
) -> BlasLayout {
    BlasLayout::new(
        mapped_block(mapped_block(a.block_layout(), first), shift),
        a.log_d(),
        a.rows(),
        a.cols(),
    )
}

/// [`blas_matmul_left_term_layout`] from the maps.
pub(crate) fn left_term_layout<P: LtDiagonalMeta>(
    a: &impl BlasGridInfos,
    maps: &ContextMaps<LinearTransformation<P>>,
) -> BlasLayout {
    term_layout(a, maps.sigma(), maps.rotate_cols(1))
}

/// [`blas_matmul_right_term_layout`] from the maps.
pub(crate) fn right_term_layout<P: LtDiagonalMeta>(
    b: &impl BlasGridInfos,
    maps: &ContextMaps<LinearTransformation<P>>,
) -> BlasLayout {
    let (first, shift) = right_maps(maps, replicated_at(maps.slots(), b));
    term_layout(b, first, shift(1))
}

/// [`blas_matmul_left_operand_layout`] from the maps.
pub(crate) fn left_operand_layout<P: LtDiagonalMeta>(
    a: &impl BlasGridInfos,
    maps: &ContextMaps<LinearTransformation<P>>,
    hint: PrepareHint,
) -> BlasMatMulOperandLayout {
    operand_layout(a, hint, || left_term_layout(a, maps))
}

/// [`blas_matmul_right_operand_layout`] from the maps.
pub(crate) fn right_operand_layout<P: LtDiagonalMeta>(
    b: &impl BlasGridInfos,
    maps: &ContextMaps<LinearTransformation<P>>,
    hint: PrepareHint,
) -> BlasMatMulOperandLayout {
    operand_layout(b, hint, || right_term_layout(b, maps))
}

pub(crate) fn operand_layout(
    a: &impl BlasGridInfos,
    hint: PrepareHint,
    terms: impl FnOnce() -> BlasLayout,
) -> BlasMatMulOperandLayout {
    let source = a.block_layout();
    let grid = match hint {
        PrepareHint::OneShot => BlasLayout::new(source, a.log_d(), a.rows(), a.cols()),
        PrepareHint::Reuse => terms(),
    };
    BlasMatMulOperandLayout { grid, source, hint }
}

/// Whether `a` and `b` have one shape: the hint, the dimensions and the GLWE layouts of the source's and of the
/// reported blocks, whatever their metadata.
pub(crate) fn same_shape(a: &impl BlasMatMulOperandInfos, b: &impl BlasMatMulOperandInfos) -> bool {
    fn shape(x: &impl BlasMatMulOperandInfos) -> (PrepareHint, usize, usize, usize, GLWELayout, GLWELayout) {
        let grid = x.blas_layout();
        (
            x.hint(),
            grid.log_d,
            grid.rows,
            grid.cols,
            grid.block.glwe_layout,
            x.source_block_layout().glwe_layout,
        )
    }
    shape(a) == shape(b)
}

/// The layout of the terms a product multiplies on the side of `x`: a cold operand's own, a hot one's `hot()`.
pub(crate) fn product_terms(x: &impl BlasMatMulOperandInfos, hot: impl FnOnce() -> BlasLayout) -> CKKSLayout {
    match x.hint() {
        PrepareHint::OneShot => hot().block,
        PrepareHint::Reuse => x.block_layout(),
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use poulpy_ckks::layouts::CKKSModuleAlloc;
    use poulpy_hal::layouts::HostBytesBackend;

    use super::*;
    use crate::layouts::ciphertext::layout_of;

    #[test]
    fn set_prepared_records_infos_of_the_operand_shape_only() {
        let module = Module::<HostBytesBackend>::new(256);
        let block = layout_of(&module.ckks_ciphertext_alloc(19usize.into(), 60usize.into()));
        let layout = BlasMatMulOperandLayout {
            grid: BlasLayout::new(block, 2, 2, 1),
            source: block,
            hint: PrepareHint::Reuse,
        };
        let mut operand = BlasMatMulOperand::<HostBytesBackend, ()> {
            blocks: Vec::new(),
            terms: Vec::new(),
            layout,
            mat_slots: 0,
        };
        operand.set_prepared(layout, 4);
        assert_eq!(operand.mat_slots(), Some(4));
        let mut taller = layout;
        taller.grid.rows = 3;
        assert!(catch_unwind(AssertUnwindSafe(|| operand.set_prepared(taller, 4))).is_err());
        assert_eq!(operand.rows(), 2);
    }
}
