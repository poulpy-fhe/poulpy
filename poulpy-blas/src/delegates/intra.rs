use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    api::BlasMatIntraOps,
    layouts::{BlasContext, BlasGrid, BlasGridInfos, BlasGridMut},
    oep::BlasMatIntraImpl,
};

impl<BE: Backend + BlasMatIntraImpl> BlasMatIntraOps<BE> for Module<BE> {
    fn blas_transpose_tmp_bytes<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContext<BE>, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_transpose_tmp_bytes_impl(self, res, a, ctx, atk)
    }

    fn blas_transpose<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        BE::blas_transpose_impl(self, res, a, ctx, atk, scratch)
    }

    fn blas_rotate_rows_tmp_bytes<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContext<BE>, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_rotate_rows_tmp_bytes_impl(self, res, a, ctx, atk)
    }

    fn blas_rotate_rows<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        k: usize,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        BE::blas_rotate_rows_impl(self, res, a, k, ctx, atk, scratch)
    }

    fn blas_rotate_cols_tmp_bytes<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContext<BE>, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_rotate_cols_tmp_bytes_impl(self, res, a, ctx, atk)
    }

    fn blas_rotate_cols<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        k: usize,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        BE::blas_rotate_cols_impl(self, res, a, k, ctx, atk, scratch)
    }

    fn blas_sum_rows_tmp_bytes<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContext<BE>, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_sum_rows_tmp_bytes_impl(self, res, a, ctx, atk)
    }

    fn blas_sum_rows<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        BE::blas_sum_rows_impl(self, res, a, ctx, atk, scratch)
    }

    fn blas_sum_cols_tmp_bytes<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContext<BE>, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_sum_cols_tmp_bytes_impl(self, res, a, ctx, atk)
    }

    fn blas_sum_cols<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        BE::blas_sum_cols_impl(self, res, a, ctx, atk, scratch)
    }
}
