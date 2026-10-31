use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::{Module, PrepareHint, ScratchArena};

use crate::{
    api::{BlasMatMulLeftOperandPreparedFactory, BlasMatMulOps, BlasMatMulRightOperandPreparedFactory},
    layouts::{
        BlasContextPrepared, BlasGrid, BlasGridInfos, BlasGridMut, BlasMatMulLeftOperandPrepared, BlasMatMulOperandInfos,
        BlasMatMulRightOperandPrepared, BlasPtOperandPrepared, Left,
    },
    oep::BlasMatMulImpl,
};

impl<BE: BlasMatMulImpl> BlasMatMulOps<BE> for Module<BE> {
    fn blas_matmul_tmp_bytes<R, A, B, T, K>(
        &self,
        res: &R,
        a: &A,
        b: &B,
        ctx: &BlasContextPrepared<BE>,
        tsk: &T,
        keys: &K,
    ) -> usize
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
        ctx: &BlasContextPrepared<BE>,
        tsk: &T,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        B: BlasGrid<BE>,
        T: GetTensorKey<BE> + Sync,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync,
    {
        BE::blas_matmul(self, res, a, b, ctx, tsk, keys, scratch)
    }

    fn blas_matmul_prepared_tmp_bytes<R, A, B, T, K>(
        &self,
        res: &R,
        a: &A,
        b: &B,
        ctx: &BlasContextPrepared<BE>,
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
        a: &BlasMatMulLeftOperandPrepared<BE>,
        b: &BlasMatMulRightOperandPrepared<BE>,
        ctx: &BlasContextPrepared<BE>,
        tsk: &T,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        T: GetTensorKey<BE> + Sync,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync,
    {
        BE::blas_matmul_prepared(self, res, a, b, ctx, tsk, keys, scratch)
    }

    fn blas_matmul_pt_right_tmp_bytes<R, A, K>(
        &self,
        res: &R,
        a: &A,
        b: &BlasPtOperandPrepared<BE>,
        ctx: &BlasContextPrepared<BE>,
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
        b: &BlasPtOperandPrepared<BE>,
        ctx: &BlasContextPrepared<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync,
    {
        BE::blas_matmul_pt_right(self, res, a, b, ctx, keys, scratch)
    }

    fn blas_matmul_pt_left_tmp_bytes<R, A, K>(
        &self,
        res: &R,
        b: &BlasPtOperandPrepared<BE, Left>,
        a: &A,
        ctx: &BlasContextPrepared<BE>,
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
        b: &BlasPtOperandPrepared<BE, Left>,
        a: &A,
        ctx: &BlasContextPrepared<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync,
    {
        BE::blas_matmul_pt_left(self, res, b, a, ctx, keys, scratch)
    }
}

impl<BE: BlasMatMulImpl> BlasMatMulLeftOperandPreparedFactory<BE> for Module<BE> {
    fn blas_matmul_left_operand_prepare_tmp_bytes<A, K>(
        &self,
        a: &A,
        ctx: &BlasContextPrepared<BE>,
        keys: &K,
        hint: PrepareHint,
    ) -> usize
    where
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_matmul_left_operand_prepare_tmp_bytes(self, a, ctx, keys, hint)
    }

    fn blas_matmul_left_operand_prepare<A, K>(
        &self,
        res: &mut BlasMatMulLeftOperandPrepared<BE>,
        a: &A,
        ctx: &BlasContextPrepared<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync,
    {
        BE::blas_matmul_left_operand_prepare(self, res, a, ctx, keys, scratch)
    }
}

impl<BE: BlasMatMulImpl> BlasMatMulRightOperandPreparedFactory<BE> for Module<BE> {
    fn blas_matmul_right_operand_prepare_tmp_bytes<B, K>(
        &self,
        b: &B,
        ctx: &BlasContextPrepared<BE>,
        keys: &K,
        hint: PrepareHint,
    ) -> usize
    where
        B: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_matmul_right_operand_prepare_tmp_bytes(self, b, ctx, keys, hint)
    }

    fn blas_matmul_right_operand_prepare<B, K>(
        &self,
        res: &mut BlasMatMulRightOperandPrepared<BE>,
        b: &B,
        ctx: &BlasContextPrepared<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        B: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync,
    {
        BE::blas_matmul_right_operand_prepare(self, res, b, ctx, keys, scratch)
    }
}
