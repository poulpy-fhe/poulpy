use poulpy_ckks::{
    CKKSResult as Result,
    api::{CKKSAddOps, CKKSCopyOps, CKKSLinearTransformationOps, CKKSModuleInfos, LinearTransformationPrepared},
    layouts::ScratchArenaTakeCKKS,
};
use poulpy_core::{
    GLWEBytesOf,
    layouts::{GGLWEInfos, GLWEToBackendMut, GetAutomorphismKey, LWEInfos},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use super::{
    KeyLayouts, assert_scratch, ensure_grid, ensure_layouts, ensure_lt_keys, ensure_maps, ensure_shape, lt_tmp_bytes,
    stamp_sparsity,
};
use crate::layouts::{BlasContext, BlasGrid, BlasGridInfos, BlasGridMut, batch_of};

/// Checks a [`crate::api::BlasMatIntraOps::blas_transpose`] call from metadata alone, before any implementation runs,
/// and returns the batch `res` records, `a`'s: `a` and `res` are grids at the context's `log_d` whose blocks tile their
/// shape and share one layout whose label allows their batch record ([`BlasGrid::blocks`],
/// [`BlasGridInfos::mat_slots`]), `res` has the transposed shape of `a` and its blocks `a`'s degree, radix and rank,
/// `ctx` was encoded on a module of `module`'s slot count at `a`'s radix, and `atk` holds every key of the transpose at
/// the width it is looked up at.
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
    ensure_layouts(op, res, &[a])?;
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
fn shift_maps<BE: Backend>(
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

/// Reference compositions of [`crate::api::BlasMatIntraOps`], on inputs [`validate_blas_transpose`],
/// [`validate_blas_rotate_rows`] and [`validate_blas_rotate_cols`] accepted: they check nothing but their scratch, and
/// return poulpy-ckks's budget errors before any write.
pub trait BlasMatIntraReference<BE: Backend> {
    fn blas_transpose_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, _ctx: &BlasContext<BE>, atk: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        lt_tmp_bytes(self, &res.block_layout(), &a.block_layout(), atk)
    }

    /// `res_ji = transpose(a_ij)`, one linear transformation per block.
    fn blas_transpose_reference<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE> + CKKSModuleInfos,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let op = "blas_transpose";
        let used = KeyLayouts::new(atk);
        let _ = ensure_lt_keys(op, self, ctx.transpose(), &used, a.block_layout().k(), res.block_layout().k());
        let bytes = used.max_bytes(|key| self.blas_transpose_tmp_bytes_reference(&*res, a, ctx, key));
        assert_scratch(op, bytes, scratch);
        let (rows, cols) = (a.rows(), a.cols());
        let (dst, src) = (res.blocks_mut(), a.blocks());
        for i in 0..rows {
            for j in 0..cols {
                self.ckks_eval_linear_transformation_self_into(
                    &mut dst[j * rows + i],
                    &src[i * cols + j],
                    ctx.transpose(),
                    atk,
                    scratch,
                )?;
            }
        }
        stamp_sparsity(res.blocks_mut(), a);
        Ok(())
    }

    /// One block-shaped temporary on top of the linear transformation, the addition and the copy.
    fn blas_rotate_rows_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, _ctx: &BlasContext<BE>, atk: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE> + CKKSAddOps<BE> + CKKSCopyOps<BE> + GLWEBytesOf<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        rotate_tmp_bytes(self, res, a, atk)
    }

    /// With `k mod rows d = q d + k'`: block row `i + q` copied for `k' = 0`, psi^k' per block on one block row,
    /// otherwise `rows_upper(k')` of block row `i + q` plus `rows_lower(k')` of block row `i + q + 1`.
    fn blas_rotate_rows_reference<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        k: usize,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE> + CKKSAddOps<BE> + CKKSCopyOps<BE> + CKKSModuleInfos + GLWEBytesOf<BE>,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        rotate_blocks(self, "blas_rotate_rows", res, a, k, true, ctx, atk, scratch)
    }

    /// One block-shaped temporary on top of the linear transformation, the addition and the copy.
    fn blas_rotate_cols_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, _ctx: &BlasContext<BE>, atk: &K) -> usize
    where
        Self: CKKSLinearTransformationOps<BE> + CKKSAddOps<BE> + CKKSCopyOps<BE> + GLWEBytesOf<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        rotate_tmp_bytes(self, res, a, atk)
    }

    /// With `k mod cols d = q d + k'`: block column `j + q` copied for `k' = 0`, phi^k' per block on one block column,
    /// otherwise `cols_left(k')` of block column `j + q` plus `cols_right(k')` of block column `j + q + 1`.
    fn blas_rotate_cols_reference<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        k: usize,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSLinearTransformationOps<BE> + CKKSAddOps<BE> + CKKSCopyOps<BE> + CKKSModuleInfos + GLWEBytesOf<BE>,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        rotate_blocks(self, "blas_rotate_cols", res, a, k, false, ctx, atk, scratch)
    }
}

impl<BE: Backend> BlasMatIntraReference<BE> for Module<BE> {}

fn rotate_tmp_bytes<BE, M, R, A, K>(module: &M, res: &R, a: &A, atk: &K) -> usize
where
    BE: Backend,
    M: CKKSLinearTransformationOps<BE> + CKKSAddOps<BE> + CKKSCopyOps<BE> + GLWEBytesOf<BE> + ?Sized,
    R: BlasGridInfos,
    A: BlasGridInfos,
    K: GGLWEInfos,
{
    let (res, a) = (res.block_layout(), a.block_layout());
    BE::scratch_aligned(module.glwe_bytes_of_from_infos(&res))
        + lt_tmp_bytes(module, &res, &a, atk)
            .max(module.ckks_add_tmp_bytes(res.size()))
            .max(module.ckks_copy_tmp_bytes(&res, &a))
}

/// Shifts every matrix by `k mod m d = q d + k'` rows (`rows`) or columns, along the block axis of extent `m`. With
/// `src(o)` the block `o` steps further along that axis: `k' = 0` copies `src(q)`; `m = 1` applies psi^k' (phi^k')
/// to the block; otherwise `res_ij` receives the first half map of `src(q)`, then the second half map of `src(q + 1)`
/// is evaluated into a temporary of `res`'s block layout and added.
#[allow(clippy::too_many_arguments)]
fn rotate_blocks<BE, M, R, A, K>(
    module: &M,
    op: &'static str,
    res: &mut R,
    a: &A,
    k: usize,
    rows: bool,
    ctx: &BlasContext<BE>,
    atk: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    M: CKKSLinearTransformationOps<BE> + CKKSAddOps<BE> + CKKSCopyOps<BE> + CKKSModuleInfos + GLWEBytesOf<BE> + ?Sized,
    R: BlasGridMut<BE>,
    A: BlasGrid<BE>,
    K: GetAutomorphismKey<BE> + Sync,
{
    let d = 1usize << ctx.log_d();
    let (a_rows, a_cols) = (a.rows(), a.cols());
    let extent = if rows { a_rows } else { a_cols };
    // `k mod extent d = q d + k1`, without forming `extent d`.
    let (q, k1) = (k / d % extent, k % d);
    let src = |i: usize, j: usize, o: usize| {
        if rows {
            &a.blocks()[(i + o) % a_rows * a_cols + j]
        } else {
            &a.blocks()[i * a_cols + (j + o) % a_cols]
        }
    };
    // `res` has `a`'s shape, so block `(i, j)` of both sits at `i * a_cols + j`.
    let blocks = (0..a_rows).flat_map(|i| (0..a_cols).map(move |j| (i, j)));
    if k1 == 0 {
        for (i, j) in blocks {
            let dst = &mut res.blocks_mut()[i * a_cols + j];
            module.ckks_copy(dst, src(i, j, q), scratch)?;
            dst.set_noise(None);
        }
        return Ok(());
    }
    let maps = shift_maps(ctx, k1, rows, extent);
    let used = KeyLayouts::new(atk);
    for lt in &maps {
        let _ = ensure_lt_keys(op, module, lt, &used, a.block_layout().k(), res.block_layout().k());
    }
    assert_scratch(op, used.max_bytes(|key| rotate_tmp_bytes(module, &*res, a, key)), scratch);
    let [first, second] = maps[..] else {
        for (dst, src) in res.blocks_mut().iter_mut().zip(a.blocks()) {
            module.ckks_eval_linear_transformation_self_into(dst, src, maps[0], atk, scratch)?;
        }
        stamp_sparsity(res.blocks_mut(), a);
        return Ok(());
    };
    let res_layout = res.block_layout();
    for (i, j) in blocks {
        let dst = &mut res.blocks_mut()[i * a_cols + j];
        module.ckks_eval_linear_transformation_self_into(dst, src(i, j, q), first, atk, scratch)?;
        scratch.scope(|s| -> Result<()> {
            let (mut tmp, mut s) = s.take_ckks_ciphertext_like_scratch(&res_layout);
            module.ckks_eval_linear_transformation_self_into(&mut tmp, src(i, j, q + 1), second, atk, &mut s)?;
            module.ckks_add_assign(dst, &tmp, &mut s)
        })?;
    }
    stamp_sparsity(res.blocks_mut(), a);
    Ok(())
}
