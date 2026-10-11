use poulpy_ckks::{
    CKKSResult as Result,
    api::{CKKSAddOps, CKKSCopyOps, CKKSLinearTransformationOps, CKKSModuleInfos},
    layouts::ScratchArenaTakeCKKS,
};
use poulpy_core::{
    GLWEBytesOf,
    layouts::{GGLWEInfos, GLWEToBackendMut, GetAutomorphismKey, LWEInfos},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use super::{KeyLayouts, assert_scratch, ensure_grid, ensure_lt_keys, ensure_maps, ensure_shape, lt_tmp_bytes, stamp_sparsity};
use crate::layouts::{BlasContext, BlasGrid, BlasGridInfos, BlasGridMut};

/// Reference compositions of [`crate::api::BlasMatIntraOps`].
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
        ensure_grid::<BE::Ring>(op, ctx.log_d(), a, a.blocks())?;
        ensure_grid::<BE::Ring>(op, ctx.log_d(), &*res, res.blocks())?;
        ensure_shape(op, res, a.cols(), a.rows())?;
        ensure_maps(op, self, ctx.slots(), ctx.base2k(), &[a])?;
        let used = KeyLayouts::new(atk);
        ensure_lt_keys(op, self, ctx.transpose(), &used, a.block_layout().k(), res.block_layout().k())?;
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
    ensure_grid::<BE::Ring>(op, ctx.log_d(), a, a.blocks())?;
    ensure_grid::<BE::Ring>(op, ctx.log_d(), &*res, res.blocks())?;
    ensure_shape(op, res, a.rows(), a.cols())?;
    ensure_maps(op, module, ctx.slots(), ctx.base2k(), &[a])?;
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
    let (input_k, res_k) = (a.block_layout().k(), res.block_layout().k());
    let used = KeyLayouts::new(atk);
    if extent == 1 {
        let whole = if rows { ctx.rotate_rows(k1) } else { ctx.rotate_cols(k1) };
        ensure_lt_keys(op, module, whole, &used, input_k, res_k)?;
        assert_scratch(op, used.max_bytes(|key| rotate_tmp_bytes(module, &*res, a, key)), scratch);
        for (dst, src) in res.blocks_mut().iter_mut().zip(a.blocks()) {
            module.ckks_eval_linear_transformation_self_into(dst, src, whole, atk, scratch)?;
        }
        stamp_sparsity(res.blocks_mut(), a);
        return Ok(());
    }
    let (first, second) = if rows {
        (ctx.rows_upper(k1), ctx.rows_lower(k1))
    } else {
        (ctx.cols_left(k1), ctx.cols_right(k1))
    };
    ensure_lt_keys(op, module, first, &used, input_k, res_k)?;
    ensure_lt_keys(op, module, second, &used, input_k, res_k)?;
    assert_scratch(op, used.max_bytes(|key| rotate_tmp_bytes(module, &*res, a, key)), scratch);
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
