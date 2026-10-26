//! Planning queries from layouts alone: the keys an operation looks up ([`BlasKeyUses`]) and the layout it stamps on
//! `res`, here for the axis reductions.

use poulpy_ckks::{
    CKKSResult as Result,
    api::{CKKSModuleInfos, LinearTransformation, LtDiagonalMeta},
};
use poulpy_core::layouts::{LWEInfos, TorusPrecision};
use poulpy_hal::layouts::{Backend, Module};

use super::{
    BlasContextInfos, BlasGridInfos, BlasLayout,
    ciphertext::replicated_at,
    context::{ContextMaps, context_shapes, reduction_projection},
    encoding::{copied_block, mapped_block_into},
    validate::{UseSink, reduction_uses},
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
    ) -> Self {
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

/// The layout of an axis reduction along the rows (`rows`) or the columns: `a`'s block copied into `res`'s width without
/// a projection ([`reduction_projection`]), else through it into `res`'s width.
fn reduction_output<P: LtDiagonalMeta>(
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    rows: bool,
    maps: &ContextMaps<LinearTransformation<P>>,
) -> BlasLayout {
    let (block, res_k) = (a.block_layout(), res.block_layout().k());
    let block = match reduction_projection(maps, rows, replicated_at(maps.slots(), a)) {
        None => copied_block(block, res_k),
        Some(projection) => mapped_block_into(block, projection, res_k),
    };
    BlasLayout::new(block, a.log_d(), a.rows(), a.cols())
}

/// The layout [`crate::api::BlasMatIntraOps::blas_sum_rows`] stamps on `res` for `a`. `ctx` is a context or its layout:
/// the query reads the shape of its maps.
pub fn blas_sum_rows_output_layout(res: &impl BlasGridInfos, a: &impl BlasGridInfos, ctx: &impl BlasContextInfos) -> BlasLayout {
    reduction_output(res, a, true, &context_shapes(ctx))
}

/// [`blas_sum_rows_output_layout`] for [`crate::api::BlasMatIntraOps::blas_sum_cols`].
pub fn blas_sum_cols_output_layout(res: &impl BlasGridInfos, a: &impl BlasGridInfos, ctx: &impl BlasContextInfos) -> BlasLayout {
    reduction_output(res, a, false, &context_shapes(ctx))
}

/// The keys [`crate::api::BlasMatIntraOps::blas_sum_rows`] looks up for `a` into `res`; see [`BlasKeyUses`]. `ctx` is a
/// context or its layout: the query reads the shape of its maps.
pub fn blas_sum_rows_key_uses<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    ctx: &impl BlasContextInfos,
) -> BlasKeyUses {
    let maps = context_shapes(ctx);
    BlasKeyUses::collect(module, None, |f| reduction_uses(res, a, true, &maps, f))
}

/// [`blas_sum_rows_key_uses`] for [`crate::api::BlasMatIntraOps::blas_sum_cols`].
pub fn blas_sum_cols_key_uses<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    ctx: &impl BlasContextInfos,
) -> BlasKeyUses {
    let maps = context_shapes(ctx);
    BlasKeyUses::collect(module, None, |f| reduction_uses(res, a, false, &maps, f))
}
