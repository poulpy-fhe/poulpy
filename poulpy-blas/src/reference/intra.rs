use poulpy_ckks::{
    CKKSCtBounds, CKKSResult as Result, SetCKKSInfos,
    api::{CKKSLinearTransformationOps, CKKSModuleInfos, LinearTransformationBabySteps, LinearTransformationPrepared},
};
use poulpy_core::layouts::{GGLWEInfos, GGLWELayout, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos};
use poulpy_hal::{
    api::{CnvPVecAlloc, ModuleN},
    execution::worker_count,
    layouts::{Backend, Module, ScratchArena},
};

use super::{
    WORKERS, assert_maps_scratch, ensure_grid, ensure_layouts, ensure_lt_keys, ensure_maps, ensure_shape, lt_tmp_bytes,
    run_tasks, tasks_tmp_bytes,
};
use crate::layouts::{BlasContext, BlasGrid, batch_of};

/// Checks a [`crate::api::BlasMatIntraOps::blas_transpose`] call from metadata alone, before any implementation runs,
/// and returns the batch `res` records, `a`'s: `a` and `res` are grids at the context's `log_d` whose blocks tile their
/// shape and share one layout whose label allows their batch record ([`BlasGrid::blocks`],
/// [`crate::layouts::BlasGridInfos::mat_slots`]), `res` has the transposed shape of `a` and its blocks `a`'s degree,
/// radix and rank, `ctx` was encoded on a module of `module`'s slot count at `a`'s radix, and `atk` holds every key of
/// the transpose at the width it is looked up at.
///
/// # Errors
///
/// `InvalidOperand` for a grid, shape, layout or context it rejects, then `MissingAutomorphismKey`. It checks no
/// budget: poulpy-ckks rejects those inside the implementation, before any write.
pub fn validate_blas_transpose<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGrid<BE>,
    a: &impl BlasGrid<BE>,
    ctx: &BlasContext<BE>,
    atk: &impl GetAutomorphismKey<BE>,
) -> Result<usize> {
    let op = "blas_transpose";
    ensure_mapped(op, module, res, a, ctx, a.cols(), a.rows())?;
    ensure_lt_keys(op, module, ctx.transpose(), atk, a.block_layout().k(), res.block_layout().k())?;
    Ok(batch_of::<BE::Ring>(a))
}

/// Checks a [`crate::api::BlasMatIntraOps::blas_rotate_rows`] call as [`validate_blas_transpose`] does, `res` of `a`'s
/// shape and the keys of the maps the shift by `k` applies, and returns the batch `res` records, `a`'s.
///
/// # Errors
///
/// As [`validate_blas_transpose`].
pub fn validate_blas_rotate_rows<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGrid<BE>,
    a: &impl BlasGrid<BE>,
    k: usize,
    ctx: &BlasContext<BE>,
    atk: &impl GetAutomorphismKey<BE>,
) -> Result<usize> {
    validate_shift(module, res, a, k, true, ctx, atk)
}

/// [`validate_blas_rotate_rows`] for [`crate::api::BlasMatIntraOps::blas_rotate_cols`].
///
/// # Errors
///
/// As [`validate_blas_transpose`].
pub fn validate_blas_rotate_cols<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGrid<BE>,
    a: &impl BlasGrid<BE>,
    k: usize,
    ctx: &BlasContext<BE>,
    atk: &impl GetAutomorphismKey<BE>,
) -> Result<usize> {
    validate_shift(module, res, a, k, false, ctx, atk)
}

/// `a` and `res` are grids at the context's `log_d`, `res` of `rows x cols` blocks of `a`'s degree, radix and rank, and
/// the context's maps read `a`.
fn ensure_mapped<BE: Backend>(
    op: &'static str,
    module: &Module<BE>,
    res: &impl BlasGrid<BE>,
    a: &impl BlasGrid<BE>,
    ctx: &BlasContext<BE>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    ensure_grid::<BE::Ring>(op, ctx.log_d(), a, a.blocks())?;
    ensure_grid::<BE::Ring>(op, ctx.log_d(), res, res.blocks())?;
    ensure_shape(op, res, rows, cols)?;
    ensure_layouts(op, module.n(), res, &[a])?;
    ensure_maps(op, module, ctx.slots(), ctx.base2k(), &[a])
}

/// The checks of a row (`rows`) or column shift by `k`.
fn validate_shift<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGrid<BE>,
    a: &impl BlasGrid<BE>,
    k: usize,
    rows: bool,
    ctx: &BlasContext<BE>,
    atk: &impl GetAutomorphismKey<BE>,
) -> Result<usize> {
    let op = if rows { "blas_rotate_rows" } else { "blas_rotate_cols" };
    ensure_mapped(op, module, res, a, ctx, a.rows(), a.cols())?;
    let extent = if rows { a.rows() } else { a.cols() };
    for lt in shift_maps(ctx, k % (1 << ctx.log_d()), rows, extent) {
        ensure_lt_keys(op, module, lt, atk, a.block_layout().k(), res.block_layout().k())?;
    }
    Ok(batch_of::<BE::Ring>(a))
}

/// The maps a row (`rows`) or column shift by `k1 = k mod d` applies along a block axis of extent `extent`: none for
/// `k1 = 0`, a copy; psi^k1 (phi^k1) on one block; otherwise both half maps, the first of block `q`, the second of
/// block `q + 1`.
pub(crate) fn shift_maps<BE: Backend>(
    ctx: &BlasContext<BE>,
    k1: usize,
    rows: bool,
    extent: usize,
) -> Vec<&LinearTransformationPrepared<BE>> {
    match (k1, extent, rows) {
        (0, ..) => Vec::new(),
        (_, 1, true) => vec![ctx.rotate_rows(k1)],
        (_, 1, false) => vec![ctx.rotate_cols(k1)],
        (_, _, true) => vec![ctx.rows_upper(k1), ctx.rows_lower(k1)],
        (_, _, false) => vec![ctx.cols_left(k1), ctx.cols_right(k1)],
    }
}

/// Reference of the map hook of [`crate::oep::BlasMatIntraImpl`], under which the transpose and the shifts are
/// provided. It runs on the inputs of an operation its validator accepted ([`validate_blas_transpose`],
/// [`validate_blas_rotate_rows`], [`validate_blas_rotate_cols`]): it checks nothing but its scratch, and returns
/// poulpy-ckks's budget errors before any write. It splits the sources over up to four workers of `BE`'s task executor,
/// each with its own scratch slice and baby-step cache, and returns the error of the lowest failing source, as a serial
/// loop would.
pub trait BlasMatIntraReference<BE: Backend> {
    /// Filling a baby-step cache from a source of layout `src`, and the evaluation of a map from it into a block of
    /// layout `dst`, on each worker's slice.
    fn blas_map_blocks_tmp_bytes_reference<D, S, K>(
        &self,
        dst: &D,
        src: &S,
        sources: usize,
        _lts: &[&LinearTransformationPrepared<BE>],
        atk: &K,
    ) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        D: CKKSCtBounds,
        S: CKKSCtBounds,
        K: GGLWEInfos,
    {
        tasks_tmp_bytes::<BE>(sources, lt_tmp_bytes(self, dst, src, atk))
    }

    /// `dst[t * lts.len() + m] = lts[m](src[t])`, stamped with the label of `src[t]`. Per source, one baby-step cache
    /// covers the baby steps of every map, then each map is evaluated from it with `ckks_eval_linear_transformation_into`;
    /// each worker takes a run of consecutive sources.
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
        Self: CKKSLinearTransformationOps<BE> + CKKSModuleInfos + ModuleN + CnvPVecAlloc<BE> + Sized + Sync,
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
        let bytes = |key: &GGLWELayout| self.blas_map_blocks_tmp_bytes_reference(first_dst, first, src.len(), lts, key);
        assert_maps_scratch(op, self, lts, atk, first.k(), first_dst.k(), bytes, scratch);
        let rotations: Vec<i64> = lts.iter().flat_map(|lt| lt.baby_steps().iter().copied()).collect();
        let run = src.len().div_ceil(worker_count::<BE::TaskExecutor>(WORKERS, src.len()));
        let mut runs: Vec<_> = src.chunks(run).zip(dst.chunks_mut(run * lts.len())).collect();
        run_tasks::<BE::TaskExecutor, _, _>(&mut runs, scratch, |(src, dst), scratch| {
            let mut babies = LinearTransformationBabySteps::alloc(self, &rotations, first);
            for (&src, dst) in src.iter().zip(dst.chunks_mut(lts.len())) {
                self.ckks_prepare_linear_transformation_baby_steps(&mut babies, src, atk, scratch)?;
                for (dst, lt) in dst.iter_mut().zip(lts) {
                    self.ckks_eval_linear_transformation_into(dst, src, &babies, lt, atk, scratch)?;
                    dst.set_log_sparsity(src.log_sparsity());
                }
            }
            Ok(())
        })
    }
}

impl<BE: Backend> BlasMatIntraReference<BE> for Module<BE> {}
