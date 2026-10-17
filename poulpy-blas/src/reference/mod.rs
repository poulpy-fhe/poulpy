//! Reference compositions over poulpy-ckks, the contract every override reproduces, and the planning queries: the keys
//! an operation looks up, the layout it stamps on `res`, and the largest radix of a product.

mod inter;
mod intra;
mod matmul;

pub use inter::{
    BlasMatInterReference, blas_rotate_key_uses, blas_rotate_output_layout, blas_sum_key_uses, blas_sum_output_layout,
};
pub use intra::{
    BlasMatIntraReference, blas_rotate_cols_key_uses, blas_rotate_cols_output_layout, blas_rotate_rows_key_uses,
    blas_rotate_rows_output_layout, blas_transpose_key_uses, blas_transpose_output_layout,
};
pub use matmul::{
    BlasMatMulReference, blas_matmul_key_uses, blas_matmul_left_operand_layout, blas_matmul_left_term_layout,
    blas_matmul_max_base2k, blas_matmul_output_layout, blas_matmul_prepare_left_key_uses, blas_matmul_prepare_right_key_uses,
    blas_matmul_prepared_key_uses, blas_matmul_prepared_output_layout, blas_matmul_pt_left_key_uses,
    blas_matmul_pt_left_max_base2k, blas_matmul_pt_left_output_layout, blas_matmul_pt_right_key_uses,
    blas_matmul_pt_right_max_base2k, blas_matmul_pt_right_output_layout, blas_matmul_right_operand_layout,
    blas_matmul_right_term_layout,
};

use poulpy_ckks::{
    CKKSCtBounds, CKKSLayout, CKKSResult as Result, GLWEPlaintextInfos, SlotsKind,
    api::{CKKSLinearTransformationOps, CKKSModuleInfos, LinearTransformation, LtDiagonalMeta, PreparedDiagonal},
};
use poulpy_core::layouts::{GGLWEInfos, LWEInfos, TorusPrecision};
use poulpy_hal::layouts::{Backend, Data, Module, Ring};

use crate::layouts::{
    BlasContextInfos, BlasGridInfos, BlasPtOperandInfos, BlasPtSide,
    ciphertext::{mat_slots_in, slots_of},
    context::{ContextMaps, assert_fits, context_map},
    encoding::{MapLayout, skeleton},
    pt_operand::{OperandMaps, layouts_of},
    validate::UseSink,
};

/// The keys an operation looks up, by the widths it looks them up at: the Galois elements that rotating by each
/// rotation of its linear transformations, rotations and sums takes, with the width of the input it key-switches, the
/// identity (element 1) which poulpy-ckks applies without a key left out, and the width of the tensor key where it
/// relinearizes. These are the very look-ups its validator makes, in sets: pairs sorted by element then width, with no
/// repeats.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BlasKeyUses {
    /// `(Galois element, width)` of every rotation key looked up.
    pub rotations: Vec<(i64, TorusPrecision)>,
    /// The width of the tensor key looked up, if the operation relinearizes.
    pub tensor: Option<TorusPrecision>,
}

impl BlasKeyUses {
    /// Collects the look-ups an enumerator makes, and the tensor key's width.
    pub(crate) fn collect<BE: Backend>(
        module: &Module<BE>,
        tensor: Option<TorusPrecision>,
        uses: impl FnOnce(UseSink<'_>) -> Result<()>,
    ) -> Self
    where
        Module<BE>: CKKSModuleInfos,
    {
        let mut rotations = Vec::new();
        let _ = uses(&mut |u| {
            rotations.push((module.ckks_galois_element(u.rotation), u.k));
            Ok(())
        });
        rotations.retain(|&(p, _)| p != 1);
        rotations.sort_unstable_by_key(|&(p, k)| (p, k.as_usize()));
        rotations.dedup();
        Self { rotations, tensor }
    }
}

/// The label of a product of `a`, on ring `R`, by `count` public matrices, with `S` slots per block:
/// `min(a.log_sparsity, log2(S / (count d^2)))`, the smaller of `a`'s label and the largest one `count` distinct
/// matrices allow, as poulpy-ckks's plaintext product keeps the smaller of its operands' labels; `0` when `count`
/// matrices do not fit.
pub(crate) fn pt_label<R: Ring>(a: &impl BlasGridInfos, count: usize) -> usize {
    let block = a.block_layout();
    mat_slots_in(slots_of::<R>(block.n().as_usize()), a.log_d(), 0)
        .checked_div(count)
        .and_then(usize::checked_ilog2)
        .map_or(0, |log| block.log_sparsity().min(log as usize))
}

/// `block` copied into a ciphertext of width `dst_k`: at the narrower of the two, as `ckks_copy` leaves it.
pub(crate) fn copied_block(mut block: CKKSLayout, dst_k: TorusPrecision) -> CKKSLayout {
    block.glwe_layout.k = block.k().min(dst_k);
    block
}

/// Scratch of one linear transformation from a block of layout `a` into a block of layout `res`.
pub(crate) fn lt_tmp_bytes<BE, M, K>(module: &M, res: &impl CKKSCtBounds, a: &impl CKKSCtBounds, key: &K) -> usize
where
    BE: Backend,
    M: CKKSLinearTransformationOps<BE> + ?Sized,
    K: GGLWEInfos,
{
    module
        .ckks_prepare_linear_transformation_baby_steps_tmp_bytes(a, key)
        .max(module.ckks_eval_linear_transformation_tmp_bytes(a, key))
        .max(module.ckks_eval_linear_transformation_tmp_bytes(res, key))
}

/// What the queries read of a map's diagonals: their width, scale and slot kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DiagonalShape {
    k: TorusPrecision,
    log_scale: usize,
    slots: SlotsKind,
}

impl LtDiagonalMeta for DiagonalShape {
    fn lt_log_scale(&self) -> usize {
        self.log_scale
    }

    fn lt_slots(&self) -> SlotsKind {
        self.slots
    }
}

/// A map's diagonal as the queries read it: its scale and slot kind, and its width.
pub(crate) trait MapDiagonal: LtDiagonalMeta {
    fn width(&self) -> TorusPrecision;
}

impl MapDiagonal for DiagonalShape {
    fn width(&self) -> TorusPrecision {
        self.k
    }
}

impl<D: Data, BE: Backend> MapDiagonal for PreparedDiagonal<D, BE> {
    fn width(&self) -> TorusPrecision {
        self.k()
    }
}

/// `map` as the queries read it: the schedule its prepared form holds, every diagonal of its width, scale and slot kind.
fn shape(map: &MapLayout) -> LinearTransformation<DiagonalShape> {
    let slots = if map.meta.meta.slots.is_real() {
        SlotsKind::Real
    } else {
        SlotsKind::Complex
    };
    let diagonal = DiagonalShape {
        k: map.meta.k,
        log_scale: map.meta.log_delta(),
        slots,
    };
    skeleton(&map.prepared_index(), diagonal)
}

/// The maps of a context of `infos` as the queries read them, without encoding any.
///
/// # Panics
///
/// Panics unless the blocks of `infos`, at least `2 x 2`, fit its slots, and its maps have a torus width.
pub(crate) fn context_shapes(infos: &impl BlasContextInfos) -> ContextMaps<LinearTransformation<DiagonalShape>> {
    let layout = infos.blas_context_layout();
    assert_fits(&layout);
    ContextMaps::build(layout.plan.log_d(), layout.slots, |id| {
        shape(&context_map::<f64>(&layout, id).layout())
    })
}

/// The maps of a public operand of `infos` as the queries read them, without encoding any.
///
/// # Panics
///
/// As [`crate::layouts::BlasModuleAlloc::blas_pt_operand_alloc`], for a layout that does not fit.
pub(crate) fn operand_shapes<S: BlasPtSide>(
    infos: &impl BlasPtOperandInfos<Side = S>,
) -> OperandMaps<LinearTransformation<DiagonalShape>> {
    layouts_of(&infos.blas_pt_operand_layout()).map(shape)
}
