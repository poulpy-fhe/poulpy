use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::{Module, ScratchArena};

use crate::{
    api::BlasMatMulOps,
    layouts::{BlasContext, BlasGrid, BlasGridInfos, BlasGridMut, BlasMatMulLeft, BlasMatMulRight, BlasPtOperand},
    oep::BlasMatMulImpl,
    reference::{
        Left, validate_blas_matmul, validate_blas_matmul_prepare_left, validate_blas_matmul_prepare_right,
        validate_blas_matmul_prepared, validate_blas_matmul_pt, validate_blas_matmul_pt_left,
    },
};

impl<BE: BlasMatMulImpl> BlasMatMulOps<BE> for Module<BE> {
    fn blas_matmul_tmp_bytes<R, A, B, T, K>(&self, res: &R, a: &A, b: &B, ctx: &BlasContext<BE>, tsk: &T, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        B: BlasGridInfos,
        T: GGLWEInfos,
        K: GGLWEInfos,
    {
        BE::blas_matmul_tmp_bytes_impl(self, res, a, b, ctx, tsk, atk)
    }

    fn blas_matmul<R, A, B, T, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &B,
        ctx: &BlasContext<BE>,
        tsk: &T,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        B: BlasGrid<BE>,
        T: GetTensorKey<BE> + Sync,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let batch = validate_blas_matmul(self, &*res, a, b, ctx, tsk, atk)?;
        BE::blas_matmul_impl(self, res, a, b, ctx, tsk, atk, scratch)?;
        res.set_mat_slots(batch);
        Ok(())
    }

    fn blas_matmul_prepare_left_tmp_bytes<A, K>(&self, a: &A, ctx: &BlasContext<BE>, atk: &K) -> usize
    where
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_matmul_prepare_left_tmp_bytes_impl(self, a, ctx, atk)
    }

    fn blas_matmul_prepare_left<A, K>(
        &self,
        a: &A,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<BlasMatMulLeft<BE>>
    where
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        validate_blas_matmul_prepare_left(self, a, ctx, atk)?;
        BE::blas_matmul_prepare_left_impl(self, a, ctx, atk, scratch)
    }

    fn blas_matmul_prepare_right_tmp_bytes<B, K>(&self, b: &B, ctx: &BlasContext<BE>, atk: &K) -> usize
    where
        B: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_matmul_prepare_right_tmp_bytes_impl(self, b, ctx, atk)
    }

    fn blas_matmul_prepare_right<B, K>(
        &self,
        b: &B,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<BlasMatMulRight<BE>>
    where
        B: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        validate_blas_matmul_prepare_right(self, b, ctx, atk)?;
        BE::blas_matmul_prepare_right_impl(self, b, ctx, atk, scratch)
    }

    fn blas_matmul_prepared_tmp_bytes<R, A, B, T>(&self, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        B: BlasGridInfos,
        T: GGLWEInfos,
    {
        BE::blas_matmul_prepared_tmp_bytes_impl(self, res, a, b, tsk)
    }

    fn blas_matmul_prepared<R, T>(
        &self,
        res: &mut R,
        a: &BlasMatMulLeft<BE>,
        b: &BlasMatMulRight<BE>,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        T: GetTensorKey<BE> + Sync,
    {
        let batch = validate_blas_matmul_prepared(self, &*res, a, b, tsk)?;
        BE::blas_matmul_prepared_impl(self, res, a, b, tsk, scratch)?;
        res.set_mat_slots(batch);
        Ok(())
    }

    fn blas_matmul_pt_tmp_bytes<R, A, K>(&self, res: &R, a: &A, b: &BlasPtOperand<BE>, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_matmul_pt_tmp_bytes_impl(self, res, a, b, atk)
    }

    fn blas_matmul_pt<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &BlasPtOperand<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let batch = validate_blas_matmul_pt(self, &*res, a, b, atk)?;
        BE::blas_matmul_pt_impl(self, res, a, b, atk, scratch)?;
        res.set_mat_slots(batch);
        Ok(())
    }

    fn blas_matmul_pt_left_tmp_bytes<R, A, K>(&self, res: &R, b: &BlasPtOperand<BE, Left>, a: &A, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_matmul_pt_left_tmp_bytes_impl(self, res, b, a, atk)
    }

    fn blas_matmul_pt_left<R, A, K>(
        &self,
        res: &mut R,
        b: &BlasPtOperand<BE, Left>,
        a: &A,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let batch = validate_blas_matmul_pt_left(self, &*res, b, a, atk)?;
        BE::blas_matmul_pt_left_impl(self, res, b, a, atk, scratch)?;
        res.set_mat_slots(batch);
        Ok(())
    }
}
