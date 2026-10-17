//! The prepared operands of a product: their storage, and the layouts of the terms they hold.

use poulpy_ckks::{
    CKKSLayout,
    api::{LinearTransformation, LtDiagonalMeta},
    layouts::{CKKSCiphertextOwned, CKKSPreparedLeft, CKKSPreparedRight},
};
use poulpy_hal::layouts::{Backend, PrepareHint};

use super::{
    BlasGridInfos, BlasLayout, BlasMatMulOperandInfos, BlasMatMulOperandLayout, ciphertext::replicated_at, context::ContextMaps,
    encoding::mapped_block,
};

/// The left operand of `api::BlasMatMulOps::blas_matmul_prepared`, hot or cold, which
/// `api::BlasMatMulOps::blas_matmul_left_alloc_from_infos` allocates and
/// `api::BlasMatMulOps::blas_matmul_prepare_left` fills.
pub type BlasMatMulLeft<BE> = BlasMatMulOperand<BE, CKKSPreparedLeft<BE>>;

/// The right operand of `api::BlasMatMulOps::blas_matmul_prepared`, hot or cold, which
/// `api::BlasMatMulOps::blas_matmul_right_alloc_from_infos` allocates and
/// `api::BlasMatMulOps::blas_matmul_prepare_right` fills.
pub type BlasMatMulRight<BE> = BlasMatMulOperand<BE, CKKSPreparedRight<BE>>;

/// A prepared operand of a product, of terms `P`: [`CKKSPreparedLeft`] on the left, [`BlasMatMulLeft`], and
/// [`CKKSPreparedRight`] on the right, [`BlasMatMulRight`], so that swapping the operands of a product does not compile.
/// Its storage is `BE`'s, poulpy-ckks ciphertexts and prepared terms. An allocation from [`BlasMatMulOperandInfos`]
/// makes all its storage, after their hint, and a preparation fills it:
/// - hot, `OneShot`: a block of the source's layout per block of its source, row-major, into which the preparation
///   copies the source's block. The product evaluates their terms under its own context;
/// - cold, `Reuse`: the `d` terms of every block of its source, all of the infos' block layout, in the transform domain,
///   each converted once. On the left, term `0` of block `(i, l)` is sigma of the source block, narrowed exactly to the
///   terms' width, and term `k` its phi^k; on the right, tau and psi^k, or for a replicated source
///   [`BlasContext::tau_rep`](crate::layouts::BlasContext::tau_rep) and
///   [`BlasContext::rotate_rows_rep`](crate::layouts::BlasContext::rotate_rows_rep)
///   (`blas_matmul_right_term_layout`).
///
/// Its [`BlasMatMulOperandInfos`] report the infos it was allocated from and the batch `0`, then those of its latest
/// preparation, which `blas_matmul_left_operand_layout` and
/// `blas_matmul_right_operand_layout` give for its source, and its
/// source's batch. The preparation's validator checks that the source's infos have the operand's shape.
pub struct BlasMatMulOperand<BE: Backend, P> {
    /// Hot, `OneShot`: the source's blocks, row-major; empty when cold.
    pub(crate) blocks: Vec<CKKSCiphertextOwned<BE>>,
    /// Cold, `Reuse`: block `(i, j)`'s term `k` at `(i * cols + j) * d + k`, converted once; empty when hot.
    pub(crate) terms: Vec<P>,
    pub(crate) layout: BlasMatMulOperandLayout,
    pub(crate) mat_slots: usize,
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

/// `reference::blas_matmul_left_term_layout` from the maps.
pub(crate) fn left_term_layout<P: LtDiagonalMeta>(
    a: &impl BlasGridInfos,
    maps: &ContextMaps<LinearTransformation<P>>,
) -> BlasLayout {
    term_layout(a, maps.sigma(), maps.rotate_cols(1))
}

/// `reference::blas_matmul_right_term_layout` from the maps.
pub(crate) fn right_term_layout<P: LtDiagonalMeta>(
    b: &impl BlasGridInfos,
    maps: &ContextMaps<LinearTransformation<P>>,
) -> BlasLayout {
    let (first, shift) = right_maps(maps, replicated_at(maps.slots(), b));
    term_layout(b, first, shift(1))
}

/// `reference::blas_matmul_left_operand_layout` from the maps.
pub(crate) fn left_operand_layout<P: LtDiagonalMeta>(
    a: &impl BlasGridInfos,
    maps: &ContextMaps<LinearTransformation<P>>,
    hint: PrepareHint,
) -> BlasMatMulOperandLayout {
    operand_layout(a, hint, || left_term_layout(a, maps))
}

/// `reference::blas_matmul_right_operand_layout` from the maps.
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

/// The layout of the terms a product multiplies on the side of `x`: a cold operand's own, a hot one's `hot()`.
pub(crate) fn product_terms(x: &impl BlasMatMulOperandInfos, hot: impl FnOnce() -> BlasLayout) -> CKKSLayout {
    match x.hint() {
        PrepareHint::OneShot => hot().block,
        PrepareHint::Reuse => x.block_layout(),
    }
}
