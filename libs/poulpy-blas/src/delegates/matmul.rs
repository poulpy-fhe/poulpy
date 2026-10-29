use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::{Module, PrepareHint, ScratchArena};

use crate::{
    api::BlasMatMulOps,
    layouts::{
        BlasContext, BlasGrid, BlasGridInfos, BlasGridMut, BlasMatMulLeft, BlasMatMulOperandInfos, BlasMatMulRight,
        BlasPtOperand, Left,
    },
    oep::BlasMatMulImpl,
};

impl<BE: BlasMatMulImpl> BlasMatMulOps<BE> for Module<BE> {
    fn blas_matmul_tmp_bytes<R, A, B, T, K>(&self, res: &R, a: &A, b: &B, ctx: &BlasContext<BE>, tsk: &T, keys: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        B: BlasGridInfos,
        T: GGLWEInfos,
        K: GGLWEInfos,
    {
        BE::blas_matmul_tmp_bytes(self, res, a, b, ctx, tsk, keys)
    }

    fn blas_matmul<R, A, B, T, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &B,
        ctx: &BlasContext<BE>,
        tsk: &T,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        B: BlasGrid<BE>,
        T: GetTensorKey<BE> + Sync,
        K: GetAutomorphismKey<BE> + Sync,
    {
        BE::blas_matmul(self, res, a, b, ctx, tsk, keys, scratch)
    }

    fn blas_matmul_prepare_left_tmp_bytes<A, K>(&self, a: &A, ctx: &BlasContext<BE>, keys: &K, hint: PrepareHint) -> usize
    where
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_matmul_prepare_left_tmp_bytes(self, a, ctx, keys, hint)
    }

    fn blas_matmul_prepare_left<A, K>(
        &self,
        res: &mut BlasMatMulLeft<BE>,
        a: &A,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        BE::blas_matmul_prepare_left(self, res, a, ctx, keys, scratch)
    }

    fn blas_matmul_prepare_right_tmp_bytes<B, K>(&self, b: &B, ctx: &BlasContext<BE>, keys: &K, hint: PrepareHint) -> usize
    where
        B: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_matmul_prepare_right_tmp_bytes(self, b, ctx, keys, hint)
    }

    fn blas_matmul_prepare_right<B, K>(
        &self,
        res: &mut BlasMatMulRight<BE>,
        b: &B,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        B: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        BE::blas_matmul_prepare_right(self, res, b, ctx, keys, scratch)
    }

    fn blas_matmul_prepared_tmp_bytes<R, A, B, T, K>(
        &self,
        res: &R,
        a: &A,
        b: &B,
        ctx: &BlasContext<BE>,
        tsk: &T,
        keys: &K,
    ) -> usize
    where
        R: BlasGridInfos,
        A: BlasMatMulOperandInfos,
        B: BlasMatMulOperandInfos,
        T: GGLWEInfos,
        K: GGLWEInfos,
    {
        BE::blas_matmul_prepared_tmp_bytes(self, res, a, b, ctx, tsk, keys)
    }

    fn blas_matmul_prepared<R, T, K>(
        &self,
        res: &mut R,
        a: &BlasMatMulLeft<BE>,
        b: &BlasMatMulRight<BE>,
        ctx: &BlasContext<BE>,
        tsk: &T,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        T: GetTensorKey<BE> + Sync,
        K: GetAutomorphismKey<BE> + Sync,
    {
        BE::blas_matmul_prepared(self, res, a, b, ctx, tsk, keys, scratch)
    }

    fn blas_matmul_pt_right_tmp_bytes<R, A, K>(
        &self,
        res: &R,
        a: &A,
        b: &BlasPtOperand<BE>,
        ctx: &BlasContext<BE>,
        keys: &K,
    ) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_matmul_pt_right_tmp_bytes(self, res, a, b, ctx, keys)
    }

    fn blas_matmul_pt_right<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &BlasPtOperand<BE>,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        BE::blas_matmul_pt_right(self, res, a, b, ctx, keys, scratch)
    }

    fn blas_matmul_pt_left_tmp_bytes<R, A, K>(
        &self,
        res: &R,
        b: &BlasPtOperand<BE, Left>,
        a: &A,
        ctx: &BlasContext<BE>,
        keys: &K,
    ) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_matmul_pt_left_tmp_bytes(self, res, b, a, ctx, keys)
    }

    fn blas_matmul_pt_left<R, A, K>(
        &self,
        res: &mut R,
        b: &BlasPtOperand<BE, Left>,
        a: &A,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        BE::blas_matmul_pt_left(self, res, b, a, ctx, keys, scratch)
    }
}
