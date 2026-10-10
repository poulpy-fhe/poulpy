use poulpy_ckks::{CKKSResult as Result, layouts::CKKSCiphertext};
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::{
    api::BlasMatInterOps,
    layouts::{BlasCiphertext, BlasGridInfos},
    oep::BlasMatInterImpl,
};

impl<BE: Backend + BlasMatInterImpl> BlasMatInterOps<BE> for Module<BE> {
    fn blas_rotate_tmp_bytes<R, A, K>(&self, res: &R, a: &A, key: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_rotate_tmp_bytes_impl(self, res, a, key)
    }

    fn blas_rotate<DR, DA, K>(
        &self,
        res: &mut BlasCiphertext<DR, BE::ZnxWord, BE::Ring>,
        a: &BlasCiphertext<DA, BE::ZnxWord, BE::Ring>,
        r: i64,
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
        BE::blas_rotate_impl(self, res, a, r, keys, scratch)
    }

    fn blas_sum_tmp_bytes<R, A, K>(&self, res: &R, a: &A, key: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_sum_tmp_bytes_impl(self, res, a, key)
    }

    fn blas_sum<DR, DA, K>(
        &self,
        res: &mut BlasCiphertext<DR, BE::ZnxWord, BE::Ring>,
        a: &BlasCiphertext<DA, BE::ZnxWord, BE::Ring>,
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
        BE::blas_sum_impl(self, res, a, keys, scratch)
    }
}
