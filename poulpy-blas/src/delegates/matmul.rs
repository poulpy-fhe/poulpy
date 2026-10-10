use poulpy_ckks::{CKKSResult as Result, layouts::CKKSCiphertext};
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::{
    api::BlasMatMulOps,
    layouts::{BlasCiphertext, BlasContext, BlasGridInfos, BlasMatMulLeft, BlasMatMulRight, BlasPtOperand},
    oep::BlasMatMulImpl,
};

impl<BE: Backend + BlasMatMulImpl> BlasMatMulOps<BE> for Module<BE> {
    fn blas_matmul_tmp_bytes<R, A, B, T, K>(&self, res: &R, a: &A, b: &B, tsk: &T, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        B: BlasGridInfos,
        T: GGLWEInfos,
        K: GGLWEInfos,
    {
        BE::blas_matmul_tmp_bytes_impl(self, res, a, b, tsk, atk)
    }

    fn blas_matmul<DR, DA, DB, K>(
        &self,
        res: &mut BlasCiphertext<DR, BE::ZnxWord, BE::Ring>,
        a: &BlasCiphertext<DA, BE::ZnxWord, BE::Ring>,
        b: &BlasCiphertext<DB, BE::ZnxWord, BE::Ring>,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        DR: Data,
        DA: Data,
        DB: Data,
        CKKSCiphertext<DR, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
        CKKSCiphertext<DA, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
        CKKSCiphertext<DB, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
        K: GetTensorKey<BE> + GetAutomorphismKey<BE>,
    {
        BE::blas_matmul_impl(self, res, a, b, ctx, keys, scratch)?;
        res.mat_slots = a.mat_slots.max(b.mat_slots);
        Ok(())
    }

    fn blas_matmul_prepare_left_tmp_bytes<R, A, K>(&self, res: &R, a: &A, key: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_matmul_prepare_left_tmp_bytes_impl(self, res, a, key)
    }

    fn blas_matmul_prepare_left<DR, DA, K>(
        &self,
        res: &mut BlasMatMulLeft<DR, BE::ZnxWord, BE::Ring>,
        a: &BlasCiphertext<DA, BE::ZnxWord, BE::Ring>,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        DR: Data,
        DA: Data,
        CKKSCiphertext<DR, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
        CKKSCiphertext<DA, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
        K: GetAutomorphismKey<BE>,
    {
        BE::blas_matmul_prepare_left_impl(self, res, a, ctx, keys, scratch)?;
        res.as_grid_mut().mat_slots = a.mat_slots;
        Ok(())
    }

    fn blas_matmul_prepare_right_tmp_bytes<R, B, K>(&self, res: &R, b: &B, key: &K) -> usize
    where
        R: BlasGridInfos,
        B: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_matmul_prepare_right_tmp_bytes_impl(self, res, b, key)
    }

    fn blas_matmul_prepare_right<DR, DB, K>(
        &self,
        res: &mut BlasMatMulRight<DR, BE::ZnxWord, BE::Ring>,
        b: &BlasCiphertext<DB, BE::ZnxWord, BE::Ring>,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        DR: Data,
        DB: Data,
        CKKSCiphertext<DR, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
        CKKSCiphertext<DB, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
        K: GetAutomorphismKey<BE>,
    {
        BE::blas_matmul_prepare_right_impl(self, res, b, ctx, keys, scratch)?;
        res.as_grid_mut().mat_slots = b.mat_slots;
        Ok(())
    }

    fn blas_matmul_prepared_tmp_bytes<R, A, B, T, K>(&self, res: &R, a: &A, b: &B, tsk: &T, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        B: BlasGridInfos,
        T: GGLWEInfos,
        K: GGLWEInfos,
    {
        BE::blas_matmul_prepared_tmp_bytes_impl(self, res, a, b, tsk, atk)
    }

    fn blas_matmul_prepared<DR, DA, DB, K>(
        &self,
        res: &mut BlasCiphertext<DR, BE::ZnxWord, BE::Ring>,
        a: &BlasMatMulLeft<DA, BE::ZnxWord, BE::Ring>,
        b: &BlasMatMulRight<DB, BE::ZnxWord, BE::Ring>,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        DR: Data,
        DA: Data,
        DB: Data,
        CKKSCiphertext<DR, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
        CKKSCiphertext<DA, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
        CKKSCiphertext<DB, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
        K: GetTensorKey<BE> + GetAutomorphismKey<BE>,
    {
        BE::blas_matmul_prepared_impl(self, res, a, b, ctx, keys, scratch)?;
        res.mat_slots = a.as_grid().mat_slots.max(b.as_grid().mat_slots);
        Ok(())
    }

    fn blas_matmul_pt_tmp_bytes<R, A, K>(&self, res: &R, a: &A, key: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_matmul_pt_tmp_bytes_impl(self, res, a, key)
    }

    fn blas_matmul_pt<DR, DA, K>(
        &self,
        res: &mut BlasCiphertext<DR, BE::ZnxWord, BE::Ring>,
        a: &BlasCiphertext<DA, BE::ZnxWord, BE::Ring>,
        b: &BlasPtOperand<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        DR: Data,
        DA: Data,
        CKKSCiphertext<DR, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
        CKKSCiphertext<DA, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
        K: GetAutomorphismKey<BE>,
    {
        BE::blas_matmul_pt_impl(self, res, a, b, keys, scratch)?;
        res.mat_slots = a.mat_slots;
        Ok(())
    }
}
