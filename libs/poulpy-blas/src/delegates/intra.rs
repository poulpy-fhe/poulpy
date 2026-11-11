use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena};

use crate::{
    api::BlasMatIntraOps,
    layouts::{BlasContextPrepared, BlasGrid, BlasGridInfos, BlasGridMut},
    oep::BlasMatIntraImpl,
};

impl<BE: Backend + BlasMatIntraImpl<Rg>, Rg: Ring> BlasMatIntraOps<BE, Rg> for Module<BE> {
    fn blas_transpose_tmp_bytes<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContextPrepared<BE, Rg>, keys: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_transpose_tmp_bytes(self, res, a, ctx, keys)
    }

    fn blas_transpose<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        ctx: &BlasContextPrepared<BE, Rg>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE, Rg>,
        A: BlasGrid<BE, Rg>,
        K: GetAutomorphismKey<BE, Rg> + GetTensorKey<BE, Rg> + Sync,
    {
        BE::blas_transpose(self, res, a, ctx, keys, scratch)
    }

    fn blas_rotate_rows_tmp_bytes<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContextPrepared<BE, Rg>, keys: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_rotate_rows_tmp_bytes(self, res, a, ctx, keys)
    }

    fn blas_rotate_rows<R, A, K>(
        &self,
        k: usize,
        res: &mut R,
        a: &A,
        ctx: &BlasContextPrepared<BE, Rg>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE, Rg>,
        A: BlasGrid<BE, Rg>,
        K: GetAutomorphismKey<BE, Rg> + GetTensorKey<BE, Rg> + Sync,
    {
        BE::blas_rotate_rows(self, k, res, a, ctx, keys, scratch)
    }

    fn blas_rotate_cols_tmp_bytes<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContextPrepared<BE, Rg>, keys: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_rotate_cols_tmp_bytes(self, res, a, ctx, keys)
    }

    fn blas_rotate_cols<R, A, K>(
        &self,
        k: usize,
        res: &mut R,
        a: &A,
        ctx: &BlasContextPrepared<BE, Rg>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE, Rg>,
        A: BlasGrid<BE, Rg>,
        K: GetAutomorphismKey<BE, Rg> + GetTensorKey<BE, Rg> + Sync,
    {
        BE::blas_rotate_cols(self, k, res, a, ctx, keys, scratch)
    }

    fn blas_sum_rows_tmp_bytes<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContextPrepared<BE, Rg>, keys: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_sum_rows_tmp_bytes(self, res, a, ctx, keys)
    }

    fn blas_sum_rows<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        ctx: &BlasContextPrepared<BE, Rg>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE, Rg>,
        A: BlasGrid<BE, Rg>,
        K: GetAutomorphismKey<BE, Rg> + GetTensorKey<BE, Rg> + Sync,
    {
        BE::blas_sum_rows(self, res, a, ctx, keys, scratch)
    }

    fn blas_sum_cols_tmp_bytes<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContextPrepared<BE, Rg>, keys: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_sum_cols_tmp_bytes(self, res, a, ctx, keys)
    }

    fn blas_sum_cols<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        ctx: &BlasContextPrepared<BE, Rg>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE, Rg>,
        A: BlasGrid<BE, Rg>,
        K: GetAutomorphismKey<BE, Rg> + GetTensorKey<BE, Rg> + Sync,
    {
        BE::blas_sum_cols(self, res, a, ctx, keys, scratch)
    }
}
