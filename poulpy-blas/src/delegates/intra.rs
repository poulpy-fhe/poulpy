use poulpy_ckks::{CKKSResult as Result, layouts::CKKSCiphertext};
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::{
    api::BlasMatIntraOps,
    layouts::{BlasCiphertext, BlasContext, BlasGridInfos},
    oep::BlasMatIntraImpl,
};

impl<BE: Backend + BlasMatIntraImpl> BlasMatIntraOps<BE> for Module<BE> {
    fn blas_transpose_tmp_bytes<R, A, K>(&self, res: &R, a: &A, key: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_transpose_tmp_bytes_impl(self, res, a, key)
    }

    fn blas_transpose<DR, DA, K>(
        &self,
        res: &mut BlasCiphertext<DR, BE::ZnxWord, BE::Ring>,
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
        BE::blas_transpose_impl(self, res, a, ctx, keys, scratch)
    }

    fn blas_rotate_rows_tmp_bytes<R, A, K>(&self, res: &R, a: &A, key: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_rotate_rows_tmp_bytes_impl(self, res, a, key)
    }

    fn blas_rotate_rows<DR, DA, K>(
        &self,
        res: &mut BlasCiphertext<DR, BE::ZnxWord, BE::Ring>,
        a: &BlasCiphertext<DA, BE::ZnxWord, BE::Ring>,
        k: usize,
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
        BE::blas_rotate_rows_impl(self, res, a, k, ctx, keys, scratch)
    }

    fn blas_rotate_cols_tmp_bytes<R, A, K>(&self, res: &R, a: &A, key: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_rotate_cols_tmp_bytes_impl(self, res, a, key)
    }

    fn blas_rotate_cols<DR, DA, K>(
        &self,
        res: &mut BlasCiphertext<DR, BE::ZnxWord, BE::Ring>,
        a: &BlasCiphertext<DA, BE::ZnxWord, BE::Ring>,
        k: usize,
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
        BE::blas_rotate_cols_impl(self, res, a, k, ctx, keys, scratch)
    }
}
