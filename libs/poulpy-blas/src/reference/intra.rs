use poulpy_ckks::{
    CKKSCtBounds, CKKSResult as Result, SetCKKSInfos,
    api::{
        CKKSLinearTransformationOps, CKKSModuleInfos, LinearTransformation, LinearTransformationBabySteps,
        LinearTransformationPrepared, LtDiagonalMeta,
    },
};
use poulpy_core::layouts::{GGLWEInfos, GGLWELayout, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos};
use poulpy_hal::{
    api::{CnvPVecAlloc, ModuleN},
    layouts::{Backend, Module, ScratchArena},
};

use super::{BlasKeyUses, context_shapes, lt_tmp_bytes};
use crate::layouts::{
    BlasContextInfos, BlasGridInfos, BlasLayout,
    ciphertext::{layout_of, replicated_at},
    context::{ContextMaps, reduction_projection, shift_maps, shift_rotates},
    encoding::{copied_block, mapped_block_into},
    validate::{reduction_uses, shift_uses, transpose_uses},
};
use crate::oep::derived::assert_maps_scratch;

/// The layout [`crate::api::BlasMatIntraOps::blas_transpose`] stamps on `res` for `a`: `a`'s block through the transpose
/// map into `res`'s width, `a.cols x a.rows` blocks. The record is `a`'s. `ctx` is a context or its layout: the
/// query reads the shape of its maps.
pub fn blas_transpose_output_layout(res: &impl BlasGridInfos, a: &impl BlasGridInfos, ctx: &impl BlasContextInfos) -> BlasLayout {
    let block = mapped_block_into(a.block_layout(), context_shapes(ctx).transpose(), res.block_layout().k());
    BlasLayout::new(block, a.log_d(), a.cols(), a.rows())
}

/// The layout of a row (`rows`) or column shift by `k`: a copy of `a`'s block for a shift by a multiple of `d` or one
/// that rotates ([`shift_rotates`]), else its block through the shift's maps ([`shift_maps`], of one scale) into
/// `res`'s width.
fn shift_output<P: LtDiagonalMeta>(
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    k: usize,
    rows: bool,
    maps: &ContextMaps<LinearTransformation<P>>,
) -> BlasLayout {
    let (block, res_k) = (a.block_layout(), res.block_layout().k());
    let extent = if rows { a.rows() } else { a.cols() };
    let lts = if shift_rotates(rows, extent, replicated_at(maps.slots(), a)) {
        Vec::new()
    } else {
        shift_maps(maps, k % (1 << maps.log_d()), rows, extent)
    };
    let block = match lts.first() {
        None => copied_block(block, res_k),
        Some(lt) => mapped_block_into(block, lt, res_k),
    };
    BlasLayout::new(block, a.log_d(), a.rows(), a.cols())
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

/// The layout [`crate::api::BlasMatIntraOps::blas_rotate_rows`] by `k` stamps on `res` for `a`; see
/// [`blas_transpose_output_layout`].
pub fn blas_rotate_rows_output_layout(
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    k: usize,
    ctx: &impl BlasContextInfos,
) -> BlasLayout {
    shift_output(res, a, k, true, &context_shapes(ctx))
}

/// [`blas_rotate_rows_output_layout`] for [`crate::api::BlasMatIntraOps::blas_rotate_cols`].
pub fn blas_rotate_cols_output_layout(
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    k: usize,
    ctx: &impl BlasContextInfos,
) -> BlasLayout {
    shift_output(res, a, k, false, &context_shapes(ctx))
}

/// The layout [`crate::api::BlasMatIntraOps::blas_sum_rows`] stamps on `res` for `a`; see
/// [`blas_transpose_output_layout`].
pub fn blas_sum_rows_output_layout(res: &impl BlasGridInfos, a: &impl BlasGridInfos, ctx: &impl BlasContextInfos) -> BlasLayout {
    reduction_output(res, a, true, &context_shapes(ctx))
}

/// [`blas_sum_rows_output_layout`] for [`crate::api::BlasMatIntraOps::blas_sum_cols`].
pub fn blas_sum_cols_output_layout(res: &impl BlasGridInfos, a: &impl BlasGridInfos, ctx: &impl BlasContextInfos) -> BlasLayout {
    reduction_output(res, a, false, &context_shapes(ctx))
}

/// The keys [`crate::api::BlasMatIntraOps::blas_transpose`] looks up for `a` into `res`; see [`BlasKeyUses`]. `ctx`
/// is a context or its layout: the query reads the shape of its maps.
pub fn blas_transpose_key_uses<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    ctx: &impl BlasContextInfos,
) -> BlasKeyUses {
    let maps = context_shapes(ctx);
    BlasKeyUses::collect(module, None, |f| transpose_uses(res, a, &maps, f))
}

/// The keys [`crate::api::BlasMatIntraOps::blas_rotate_rows`] by `k` looks up for `a` into `res`; see [`BlasKeyUses`].
/// The shifts of two values of `k` look up different keys: a client rotating by several takes the union. `ctx` is a
/// context or its layout: the query reads the shape of its maps.
pub fn blas_rotate_rows_key_uses<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    k: usize,
    ctx: &impl BlasContextInfos,
) -> BlasKeyUses {
    let maps = context_shapes(ctx);
    BlasKeyUses::collect(module, None, |f| shift_uses(res, a, k, true, &maps, f))
}

/// [`blas_rotate_rows_key_uses`] for [`crate::api::BlasMatIntraOps::blas_rotate_cols`].
pub fn blas_rotate_cols_key_uses<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    k: usize,
    ctx: &impl BlasContextInfos,
) -> BlasKeyUses {
    let maps = context_shapes(ctx);
    BlasKeyUses::collect(module, None, |f| shift_uses(res, a, k, false, &maps, f))
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

/// Reference of the map hook of [`crate::oep::BlasMatIntraImpl`], under which the transpose, the shifts and the axis
/// reductions are provided, and through which the reference cold preparations of products run, and the hot sides of the
/// reference products, and the first map of the reference JKLS products by public matrices. It runs on the inputs of an
/// operation its validator accepted ([`crate::layouts::validate_blas_transpose`],
/// [`crate::layouts::validate_blas_rotate_rows`], [`crate::layouts::validate_blas_rotate_cols`],
/// [`crate::layouts::validate_blas_sum_rows`], [`crate::layouts::validate_blas_sum_cols`],
/// [`crate::layouts::validate_blas_matmul_prepare_left`], [`crate::layouts::validate_blas_matmul_prepare_right`],
/// [`crate::layouts::validate_blas_matmul_prepared`], [`crate::layouts::validate_blas_matmul`],
/// [`crate::layouts::validate_blas_matmul_pt_right`], [`crate::layouts::validate_blas_matmul_pt_left`]): it checks
/// nothing but its scratch, and returns poulpy-ckks's budget errors before any write. It runs source after source on
/// one scratch and one baby-step cache, and the first error stops it.
pub trait BlasMatIntraReference<BE: Backend> {
    /// Filling a baby-step cache from a source of layout `src`, and the evaluation of a map from it into a block of
    /// layout `dst`.
    fn blas_map_blocks_tmp_bytes_reference<D, S, K>(
        &self,
        dst: &D,
        src: &S,
        _sources: usize,
        _lts: &[&LinearTransformationPrepared<BE>],
        atk: &K,
    ) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        D: CKKSCtBounds,
        S: CKKSCtBounds,
        K: GGLWEInfos,
    {
        lt_tmp_bytes(self, dst, src, atk, false)
    }

    /// `dst[t * lts.len() + m] = lts[m](src[t])`, stamped with the label of `src[t]`. Per source, one baby-step cache
    /// covers the baby steps of every map, then each map is evaluated from it with `ckks_eval_linear_transformation_into`.
    ///
    /// # Panics
    ///
    /// Panics unless `lts` is non-empty and `dst` holds `src.len() * lts.len()` blocks, or if `scratch` holds less than
    /// the query reports for the keys it looks up.
    fn blas_map_blocks_reference<D, S, K>(
        &self,
        dst: &mut [D],
        src: &[&S],
        lts: &[&LinearTransformationPrepared<BE>],
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE> + CKKSModuleInfos + ModuleN + CnvPVecAlloc<BE> + Sized,
        D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + Send,
        S: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let op = "blas_map_blocks";
        assert!(
            !lts.is_empty() && dst.len() == src.len() * lts.len(),
            "{op}: {} blocks for {} sources and {} maps",
            dst.len(),
            src.len(),
            lts.len()
        );
        let (Some(&first), Some(first_dst)) = (src.first(), dst.first()) else {
            return Ok(());
        };
        // At the blocks' widths, as the callers' queries read them, not their allocations.
        let (dst_layout, src_layout) = (layout_of(first_dst), layout_of(first));
        let bytes = |key: &GGLWELayout| self.blas_map_blocks_tmp_bytes_reference(&dst_layout, &src_layout, src.len(), lts, key);
        assert_maps_scratch(op, self, lts, atk, first.k(), first_dst.k(), first.rank(), bytes, scratch);
        let rotations: Vec<i64> = lts.iter().flat_map(|lt| lt.baby_steps().iter().copied()).collect();
        let mut babies = LinearTransformationBabySteps::alloc(self, &rotations, first);
        for (&src, dst) in src.iter().zip(dst.chunks_mut(lts.len())) {
            self.ckks_prepare_linear_transformation_baby_steps(&mut babies, src, atk, scratch)?;
            for (dst, lt) in dst.iter_mut().zip(lts) {
                self.ckks_eval_linear_transformation_into(dst, src, &babies, lt, atk, scratch)?;
                dst.set_log_sparsity(src.log_sparsity());
            }
        }
        Ok(())
    }
}

impl<BE: Backend> BlasMatIntraReference<BE> for Module<BE> {}
