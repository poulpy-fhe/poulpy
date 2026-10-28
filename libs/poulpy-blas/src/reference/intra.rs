use poulpy_ckks::{
    CKKSCtBounds, CKKSLayout, CKKSResult as Result, SetCKKSInfos,
    api::{
        CKKSAddManyOps, CKKSAddOps, CKKSCopyOps, CKKSInnerSumOps, CKKSLinearTransformationOps, CKKSModuleInfos, CKKSRotateOps,
        LinearTransformationBabySteps, LinearTransformationPrepared,
    },
    layouts::{CKKSModuleAlloc, ScratchArenaTakeCKKS},
};
use poulpy_core::{
    GLWEBytesOf, GLWENormalize,
    layouts::{
        GGLWEInfos, GGLWELayout, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos, max_lazy_terms,
    },
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ModuleN},
    layouts::{Backend, Module, ScratchArena},
};

use super::lt_tmp_bytes;
use crate::{
    layouts::{
        BlasContext, BlasContextInfos, BlasGrid, BlasGridInfos, BlasGridMut,
        ciphertext::{layout_of, replicated_at},
        context::{reduction_projection, shift_maps, shift_rotates},
        encoding::{copied_block, mapped_block_into},
        validate::{KeyLayouts, assert_maps_scratch, assert_scratch, ensure_rotation, reduction_uses, shift_uses},
        validate_blas_rotate_cols, validate_blas_rotate_rows, validate_blas_sum_cols, validate_blas_sum_rows,
    },
    oep::BlasMatIntraImpl,
};

/// References of [`crate::oep::BlasMatIntraImpl`], which
/// [`impl_blas_mat_intra_reference_full!`](crate::impl_blas_mat_intra_reference_full) forwards to: the shifts and the
/// axis reductions, which run their validators before any write, so their errors leave `res` and its batch record
/// untouched, and the map hook, under which the transpose is provided and through which the shifts, the axis
/// reductions, the reference cold preparations of products, the hot sides of the reference products and the first map
/// of the reference JKLS products by public matrices run.
///
/// The hook runs on the inputs of an operation its validator accepted ([`crate::layouts::validate_blas_transpose`],
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
        _sources: usize,
        dst: &D,
        src: &S,
        _lts: &[&LinearTransformationPrepared<BE>],
        keys: &K,
    ) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        D: CKKSCtBounds,
        S: CKKSCtBounds,
        K: GGLWEInfos,
    {
        lt_tmp_bytes(self, dst, src, keys, false)
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
        keys: &K,
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
        let bytes = |key: &GGLWELayout| self.blas_map_blocks_tmp_bytes_reference(src.len(), &dst_layout, &src_layout, lts, key);
        assert_maps_scratch(op, self, lts, keys, first.k(), first_dst.k(), first.rank(), bytes, scratch);
        let rotations: Vec<i64> = lts.iter().flat_map(|lt| lt.baby_steps().iter().copied()).collect();
        let mut babies = LinearTransformationBabySteps::alloc(self, &rotations, first);
        for (&src, dst) in src.iter().zip(dst.chunks_mut(lts.len())) {
            self.ckks_prepare_linear_transformation_baby_steps(&mut babies, src, keys, scratch)?;
            for (dst, lt) in dst.iter_mut().zip(lts) {
                self.ckks_eval_linear_transformation_into(
                    dst,
                    src,
                    &babies,
                    lt,
                    &poulpy_core::layouts::AutomorphismKeysOnly(keys),
                    scratch,
                )?;
                dst.set_log_sparsity(src.log_sparsity());
            }
        }
        Ok(())
    }

    /// The scratch of a row shift by any `k`: the copy for `k' = 0`, otherwise the rotation where the shift rotates, or
    /// else for every `k'` the hook with psi^k' along one block, or with both half maps plus the addition along
    /// several.
    fn blas_rotate_rows_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContext<BE>, keys: &K) -> usize
    where
        BE: BlasMatIntraImpl,
        Self: CKKSCopyOps<BE> + CKKSRotateOps<BE> + CKKSAddOps<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// Shifts every matrix by `k mod m d = q d + k'` rows, along the block axis of extent `m`, after
    /// [`crate::layouts::validate_blas_rotate_rows`]. With `src(t, o)` the block `o` steps past block `t` along that
    /// axis: `k' = 0` copies `src(t, q)` into block `t`, then clears its noise tag; where the shift rotates, block `t`
    /// is `a`'s rotated by `k' d` slots; `m = 1` is otherwise one hook call with psi^k'; otherwise one hook call
    /// evaluates both half maps of every block of `a` into a heap grid `h` of `res`'s block layout, and block `t` is
    /// `h[first, src(t, q)] + h[second, src(t, q + 1)]`. On `Ok`, `res` records the batch the validator returns, `a`'s.
    fn blas_rotate_rows_reference<R, A, K>(
        &self,
        k: usize,
        res: &mut R,
        a: &A,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasMatIntraImpl,
        Self: CKKSCopyOps<BE> + CKKSRotateOps<BE> + CKKSAddOps<BE>,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;

    /// [`Self::blas_rotate_rows_tmp_bytes_reference`] for a column shift, phi^k' along one block.
    fn blas_rotate_cols_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContext<BE>, keys: &K) -> usize
    where
        BE: BlasMatIntraImpl,
        Self: CKKSCopyOps<BE> + CKKSRotateOps<BE> + CKKSAddOps<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// [`Self::blas_rotate_rows_reference`] by `k` columns, after [`crate::layouts::validate_blas_rotate_cols`], with
    /// phi^k' along one block.
    fn blas_rotate_cols_reference<R, A, K>(
        &self,
        k: usize,
        res: &mut R,
        a: &A,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasMatIntraImpl,
        Self: CKKSCopyOps<BE> + CKKSRotateOps<BE> + CKKSAddOps<BE>,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;

    /// The scratch of a row sum: a block `u` of `a`'s layout, beside a line's sum into it, with a second block where
    /// the line sum normalizes flag-clear blocks, then the inner sum of `u` into the line's first block of `res`, or
    /// its inner sum in place, the hook with the projection and the replicate on a block of the layout the projection
    /// leaves, then the copies of that block.
    fn blas_sum_rows_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContext<BE>, keys: &K) -> usize
    where
        BE: BlasMatIntraImpl,
        Self: CKKSCopyOps<BE> + CKKSAddOps<BE> + CKKSAddManyOps<BE> + CKKSInnerSumOps<BE> + GLWENormalize<BE> + GLWEBytesOf<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// Sums every matrix along its rows, one block column of `m` blocks at a time, after
    /// [`crate::layouts::validate_blas_sum_rows`]: a scratch block `u` of `a`'s layout sums the line's blocks, the
    /// line's first block is the inner sum of `u` over `d` batches without a projection, else the projection of that
    /// inner sum, then its replicate over `d` batches, and is copied into the others. On `Ok`, `res` records the batch
    /// the validator returns, `a`'s.
    fn blas_sum_rows_reference<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasMatIntraImpl,
        Self: CKKSCopyOps<BE> + CKKSAddOps<BE> + CKKSAddManyOps<BE> + CKKSInnerSumOps<BE> + GLWENormalize<BE> + GLWEBytesOf<BE>,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;

    /// [`Self::blas_sum_rows_tmp_bytes_reference`] along the rows of blocks.
    fn blas_sum_cols_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContext<BE>, keys: &K) -> usize
    where
        BE: BlasMatIntraImpl,
        Self: CKKSCopyOps<BE> + CKKSAddOps<BE> + CKKSAddManyOps<BE> + CKKSInnerSumOps<BE> + GLWENormalize<BE> + GLWEBytesOf<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// [`Self::blas_sum_rows_reference`] along the columns, one block row at a time, after
    /// [`crate::layouts::validate_blas_sum_cols`].
    fn blas_sum_cols_reference<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasMatIntraImpl,
        Self: CKKSCopyOps<BE> + CKKSAddOps<BE> + CKKSAddManyOps<BE> + CKKSInnerSumOps<BE> + GLWENormalize<BE> + GLWEBytesOf<BE>,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;
}

impl<BE: Backend> BlasMatIntraReference<BE> for Module<BE> {
    fn blas_rotate_rows_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContext<BE>, keys: &K) -> usize
    where
        BE: BlasMatIntraImpl,
        Self: CKKSCopyOps<BE> + CKKSRotateOps<BE> + CKKSAddOps<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        shift_tmp_bytes(self, res, a, true, ctx, keys)
    }

    fn blas_rotate_rows_reference<R, A, K>(
        &self,
        k: usize,
        res: &mut R,
        a: &A,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasMatIntraImpl,
        Self: CKKSCopyOps<BE> + CKKSRotateOps<BE> + CKKSAddOps<BE>,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        shift(self, "blas_rotate_rows", k, res, a, true, ctx, keys, scratch)
    }

    fn blas_rotate_cols_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContext<BE>, keys: &K) -> usize
    where
        BE: BlasMatIntraImpl,
        Self: CKKSCopyOps<BE> + CKKSRotateOps<BE> + CKKSAddOps<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        shift_tmp_bytes(self, res, a, false, ctx, keys)
    }

    fn blas_rotate_cols_reference<R, A, K>(
        &self,
        k: usize,
        res: &mut R,
        a: &A,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasMatIntraImpl,
        Self: CKKSCopyOps<BE> + CKKSRotateOps<BE> + CKKSAddOps<BE>,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        shift(self, "blas_rotate_cols", k, res, a, false, ctx, keys, scratch)
    }

    fn blas_sum_rows_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContext<BE>, keys: &K) -> usize
    where
        BE: BlasMatIntraImpl,
        Self: CKKSCopyOps<BE> + CKKSAddOps<BE> + CKKSAddManyOps<BE> + CKKSInnerSumOps<BE> + GLWENormalize<BE> + GLWEBytesOf<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        reduction_tmp_bytes(self, res, a, true, ctx, keys)
    }

    fn blas_sum_rows_reference<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasMatIntraImpl,
        Self: CKKSCopyOps<BE> + CKKSAddOps<BE> + CKKSAddManyOps<BE> + CKKSInnerSumOps<BE> + GLWENormalize<BE> + GLWEBytesOf<BE>,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        reduction(self, res, a, true, ctx, keys, scratch)
    }

    fn blas_sum_cols_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContext<BE>, keys: &K) -> usize
    where
        BE: BlasMatIntraImpl,
        Self: CKKSCopyOps<BE> + CKKSAddOps<BE> + CKKSAddManyOps<BE> + CKKSInnerSumOps<BE> + GLWENormalize<BE> + GLWEBytesOf<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        reduction_tmp_bytes(self, res, a, false, ctx, keys)
    }

    fn blas_sum_cols_reference<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasMatIntraImpl,
        Self: CKKSCopyOps<BE> + CKKSAddOps<BE> + CKKSAddManyOps<BE> + CKKSInnerSumOps<BE> + GLWENormalize<BE> + GLWEBytesOf<BE>,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        reduction(self, res, a, false, ctx, keys, scratch)
    }
}

/// [`BlasMatIntraReference::blas_rotate_rows_tmp_bytes_reference`] (`rows`) or its column form.
fn shift_tmp_bytes<BE, R, A, K>(module: &Module<BE>, res: &R, a: &A, rows: bool, ctx: &BlasContext<BE>, keys: &K) -> usize
where
    BE: BlasMatIntraImpl,
    Module<BE>: CKKSCopyOps<BE> + CKKSRotateOps<BE> + CKKSAddOps<BE>,
    R: BlasGridInfos,
    A: BlasGridInfos,
    K: GGLWEInfos,
{
    let extent = if rows { a.rows() } else { a.cols() };
    let rotates = shift_rotates(rows, extent, replicated_at(ctx.slots(), a));
    let (sources, res, a) = (a.rows() * a.cols(), res.block_layout(), a.block_layout());
    let copy = module.ckks_copy_tmp_bytes(&res, &a);
    if rotates {
        return copy
            .max(module.ckks_rotate_tmp_bytes(&res, keys))
            .max(module.ckks_rotate_tmp_bytes(&a, keys));
    }
    (1..1 << ctx.log_d())
        .map(|k1| {
            let lts = shift_maps(&ctx.maps, k1, rows, extent);
            let hook = BE::blas_map_blocks_tmp_bytes(module, sources, &res, &a, &lts, keys);
            if lts.len() == 2 {
                hook.max(module.ckks_add_tmp_bytes(res.size()))
            } else {
                hook
            }
        })
        .fold(copy, usize::max)
}

/// [`BlasMatIntraReference::blas_rotate_rows_reference`] (`rows`) or its column form.
#[allow(clippy::too_many_arguments)]
fn shift<BE, R, A, K>(
    module: &Module<BE>,
    op: &'static str,
    k: usize,
    res: &mut R,
    a: &A,
    rows: bool,
    ctx: &BlasContext<BE>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasMatIntraImpl,
    Module<BE>: CKKSCopyOps<BE> + CKKSRotateOps<BE> + CKKSAddOps<BE>,
    R: BlasGridMut<BE>,
    A: BlasGrid<BE>,
    K: GetAutomorphismKey<BE> + Sync,
{
    let batch = if rows {
        validate_blas_rotate_rows(module, k, &*res, a, ctx, keys)?
    } else {
        validate_blas_rotate_cols(module, k, &*res, a, ctx, keys)?
    };
    shift_blocks(module, op, k, res, a, rows, ctx, keys, scratch)?;
    res.set_mat_slots(batch);
    Ok(())
}

/// The blocks of [`shift`], on inputs its validator accepted.
#[allow(clippy::too_many_arguments)]
fn shift_blocks<BE, R, A, K>(
    module: &Module<BE>,
    op: &'static str,
    k: usize,
    res: &mut R,
    a: &A,
    rows: bool,
    ctx: &BlasContext<BE>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasMatIntraImpl,
    Module<BE>: CKKSCopyOps<BE> + CKKSRotateOps<BE> + CKKSAddOps<BE>,
    R: BlasGridMut<BE>,
    A: BlasGrid<BE>,
    K: GetAutomorphismKey<BE> + Sync,
{
    let d = 1usize << ctx.log_d();
    let (a_rows, a_cols) = (a.rows(), a.cols());
    let extent = if rows { a_rows } else { a_cols };
    // `k mod extent d = q d + k1`, without forming `extent d`.
    let (q, k1) = (k / d % extent, k % d);
    // `res` has `a`'s shape, so block `t` of both sits at row `t / a_cols`, column `t % a_cols`.
    let src = |t: usize, o: usize| {
        let (i, j) = (t / a_cols, t % a_cols);
        if rows {
            (i + o) % a_rows * a_cols + j
        } else {
            i * a_cols + (j + o) % a_cols
        }
    };
    if k1 == 0 {
        for (t, dst) in res.blocks_mut().iter_mut().enumerate() {
            module.ckks_copy(dst, &a.blocks()[src(t, q)], scratch)?;
            dst.set_noise(None);
        }
        return Ok(());
    }
    let bytes = |key: &GGLWELayout| shift_tmp_bytes(module, &*res, a, rows, ctx, key);
    if shift_rotates(rows, extent, replicated_at(ctx.slots(), a)) {
        let used = KeyLayouts::new(keys);
        let _ = shift_uses(k, &*res, a, rows, &ctx.maps, &mut |u| ensure_rotation(op, module, &used, u));
        assert_scratch(op, used.max_bytes(bytes), scratch);
        let rotation = (k1 << ctx.log_d()) as i64;
        for (dst, src) in res.blocks_mut().iter_mut().zip(a.blocks()) {
            module.ckks_rotate_into(rotation, dst, src, keys, scratch)?;
        }
        return Ok(());
    }
    let lts = shift_maps(&ctx.maps, k1, rows, extent);
    let (src_k, dst_k, rank) = (a.block_layout().k(), res.block_layout().k(), a.block_layout().rank());
    assert_maps_scratch(op, module, &lts, keys, src_k, dst_k, rank, bytes, scratch);
    let sources: Vec<_> = a.blocks().iter().collect();
    if lts.len() == 1 {
        // One block along the axis: `q = 0`, so block `t` maps into block `t`.
        return BE::blas_map_blocks(module, res.blocks_mut(), &sources, &lts, keys, scratch);
    }
    // ponytail: one hook call over the whole heap grid, two blocks per block of `a`; an override chunks it.
    let layout = res.block_layout();
    let mut halves: Vec<_> = (0..2 * sources.len())
        .map(|_| module.ckks_ciphertext_alloc_from_infos(&layout))
        .collect();
    BE::blas_map_blocks(module, &mut halves, &sources, &lts, keys, scratch)?;
    for (t, dst) in res.blocks_mut().iter_mut().enumerate() {
        module.ckks_add_into(dst, &halves[2 * src(t, q)], &halves[2 * src(t, q + 1) + 1], scratch)?;
    }
    Ok(())
}

/// [`BlasMatIntraReference::blas_sum_rows_tmp_bytes_reference`] (`rows`) or its column form.
fn reduction_tmp_bytes<BE, R, A, K>(module: &Module<BE>, res: &R, a: &A, rows: bool, ctx: &BlasContext<BE>, keys: &K) -> usize
where
    BE: BlasMatIntraImpl,
    Module<BE>: CKKSCopyOps<BE> + CKKSAddOps<BE> + CKKSAddManyOps<BE> + CKKSInnerSumOps<BE> + GLWENormalize<BE> + GLWEBytesOf<BE>,
    R: BlasGridInfos,
    A: BlasGridInfos,
    K: GGLWEInfos,
{
    let projection = reduction_projection(&ctx.maps, rows, replicated_at(ctx.slots(), a));
    let (res, a) = (res.block_layout(), a.block_layout());
    let sum = sum_line_tmp_bytes(module, &a, lazy_terms(&a));
    let (first, head) = match projection {
        None => (module.ckks_inner_sum_tmp_bytes(&res, &a, keys), copied_block(a, res.k())),
        Some(projection) => {
            let projected = mapped_block_into(a, projection, res.k());
            let first = module
                .ckks_inner_sum_tmp_bytes(&a, &a, keys)
                .max(BE::blas_map_blocks_tmp_bytes(module, 1, &res, &a, &[projection], keys))
                .max(module.ckks_inner_sum_tmp_bytes(&projected, &projected, keys));
            (first, projected)
        }
    };
    BE::scratch_aligned(module.glwe_bytes_of_from_infos(&a)) + sum.max(first).max(module.ckks_copy_tmp_bytes(&res, &head))
}

/// Canonical digits per coefficient a flag-clear input block holds at most, a caller obligation: the sum of two
/// canonical values.
const FLAG_CLEAR_TERMS: usize = 2;

/// The terms a lazy sum of ciphertexts of `layout` holds: poulpy-core's `max_lazy_terms` over their coefficients.
fn lazy_terms(layout: &impl GLWEInfos) -> usize {
    max_lazy_terms(
        layout.base2k(),
        layout.n().as_usize() * (layout.rank().as_usize() + 1) * layout.size(),
    )
}

/// The additions after which a sum of inputs of `weight` terms each normalizes, so that it holds `terms` at most: the
/// first `n + 1` inputs, and a normalized sum plus `n` of them; 0 where two inputs already exceed `terms`.
fn lazy_normalize_after(terms: usize, weight: usize) -> usize {
    (terms / weight).saturating_sub(1)
}

/// Scratch of [`sum_line`] of blocks of `layout` at `terms`, whatever their flags.
pub(crate) fn sum_line_tmp_bytes<BE: Backend>(module: &Module<BE>, layout: &CKKSLayout, terms: usize) -> usize
where
    Module<BE>: CKKSAddManyOps<BE> + GLWEBytesOf<BE>,
{
    let sum = module.ckks_add_many_tmp_bytes(layout, layout);
    if lazy_normalize_after(terms, FLAG_CLEAR_TERMS) == 0 {
        BE::scratch_aligned(module.glwe_bytes_of_from_infos(layout)) + sum
    } else {
        sum
    }
}

/// `u` receives the sum of `blocks`, holding `terms` canonical digits per coefficient at most, a flag-clear block
/// counting [`FLAG_CLEAR_TERMS`]: by `ckks_add_many`, or, where two blocks exceed `terms`, block by block, each
/// flag-clear block normalized into a scratch block like it and `u` normalized before each addition.
pub(crate) fn sum_line<BE, U, B>(
    module: &Module<BE>,
    u: &mut U,
    blocks: &[&B],
    terms: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSCopyOps<BE> + CKKSAddOps<BE> + CKKSAddManyOps<BE> + GLWENormalize<BE>,
    U: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    B: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    let weight = if blocks.iter().all(|block| block.is_canonical()) {
        1
    } else {
        FLAG_CLEAR_TERMS
    };
    let normalize_after = lazy_normalize_after(terms, weight);
    if normalize_after > 0 {
        return module.ckks_add_many(normalize_after, u, blocks, scratch);
    }
    module.ckks_copy(u, blocks[0], scratch)?;
    for &block in &blocks[1..] {
        if !u.is_canonical() {
            module.glwe_normalize_assign(u, scratch);
        }
        if block.is_canonical() {
            module.ckks_add_assign(u, block, scratch)?;
        } else {
            let (mut v, mut s) = scratch.borrow().take_ckks_ciphertext_like_scratch(block);
            module.glwe_normalize(&mut v, block, &mut s);
            module.ckks_add_assign(u, &v, &mut s)?;
        }
    }
    Ok(())
}

/// [`BlasMatIntraReference::blas_sum_rows_reference`] (`rows`) or its column form, each line summed by [`sum_line`].
fn reduction<BE, R, A, K>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    rows: bool,
    ctx: &BlasContext<BE>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasMatIntraImpl,
    Module<BE>: CKKSCopyOps<BE> + CKKSAddOps<BE> + CKKSAddManyOps<BE> + CKKSInnerSumOps<BE> + GLWENormalize<BE> + GLWEBytesOf<BE>,
    R: BlasGridMut<BE>,
    A: BlasGrid<BE>,
    K: GetAutomorphismKey<BE> + Sync,
{
    let (op, batch) = if rows {
        ("blas_sum_rows", validate_blas_sum_rows(module, &*res, a, ctx, keys)?)
    } else {
        ("blas_sum_cols", validate_blas_sum_cols(module, &*res, a, ctx, keys)?)
    };
    let used = KeyLayouts::new(keys);
    let _ = reduction_uses(&*res, a, rows, &ctx.maps, &mut |u| ensure_rotation(op, module, &used, u));
    let bytes = used.max_bytes(|key| reduction_tmp_bytes(module, &*res, a, rows, ctx, key));
    assert_scratch(op, bytes, scratch);
    let a_cols = a.cols();
    let (lines, extent) = if rows { (a_cols, a.rows()) } else { (a.rows(), a_cols) };
    let at = |line: usize, t: usize| if rows { t * a_cols + line } else { line * a_cols + t };
    let (log_d, log_batch) = (a.log_d(), if rows { a.log_d() } else { 0 });
    let projection = reduction_projection(&ctx.maps, rows, replicated_at(ctx.slots(), a));
    let terms = lazy_terms(&a.block_layout());
    let (mut u, mut s) = scratch.borrow().take_ckks_ciphertext_like_scratch(&a.block_layout());
    let mut blocks = Vec::with_capacity(extent);
    for line in 0..lines {
        blocks.clear();
        blocks.extend((0..extent).map(|t| &a.blocks()[at(line, t)]));
        sum_line(module, &mut u, &blocks, terms, &mut s)?;
        // The line's first block has the lowest index of its blocks.
        let first = at(line, 0);
        let (head, tail) = res.blocks_mut().split_at_mut(first + 1);
        let head = &mut head[first];
        match projection {
            None => module.ckks_inner_sum(log_batch, log_d, false, head, &u, keys, &mut s)?,
            Some(projection) => {
                module.ckks_inner_sum_assign(log_batch, log_d, false, &mut u, keys, &mut s)?;
                let sources = [&u.to_backend_view_ref()];
                BE::blas_map_blocks(module, std::slice::from_mut(head), &sources, &[projection], keys, &mut s)?;
                module.ckks_replicate_assign(log_batch, log_d, head, keys, &mut s)?;
            }
        }
        head.set_noise(None);
        for t in 1..extent {
            let dst = &mut tail[at(line, t) - first - 1];
            module.ckks_copy(dst, &*head, &mut s)?;
            dst.set_noise(None);
        }
    }
    res.set_mat_slots(batch);
    Ok(())
}

#[cfg(test)]
mod tests {
    use poulpy_core::layouts::{Base2K, Degree, GLWELayout, Rank, TorusPrecision};

    use super::{FLAG_CLEAR_TERMS, lazy_normalize_after, lazy_terms};

    /// The line sum's cadence where the bound binds, over the coefficients of a degree-64 rank-1 block of four limbs: at
    /// radix 62 two flag-clear blocks already exceed it, so the sum goes block by block, at 61 it normalizes after every
    /// addition of flag-clear blocks and every third of canonical ones, and at 52 a line of a thousand flag-clear blocks
    /// stays unnormalized.
    #[test]
    fn line_sum_cadence() {
        let after = |base2k: u32, weight: usize| {
            let layout = GLWELayout {
                n: Degree(64),
                base2k: Base2K(base2k),
                k: TorusPrecision(4 * base2k),
                rank: Rank(1),
            };
            lazy_normalize_after(lazy_terms(&layout), weight)
        };
        assert_eq!((after(62, FLAG_CLEAR_TERMS), after(62, 1)), (0, 1));
        assert_eq!((after(61, FLAG_CLEAR_TERMS), after(61, 1)), (1, 3));
        assert!(after(52, FLAG_CLEAR_TERMS) > 1000);
    }
}
