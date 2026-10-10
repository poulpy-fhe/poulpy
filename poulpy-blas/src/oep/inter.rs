use poulpy_ckks::{CKKSResult as Result, layouts::CKKSCiphertext};
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::layouts::{BlasCiphertext, BlasGridInfos};

/// Extension point of [`crate::api::BlasMatInterOps`].
///
/// # Safety
///
/// Every method reproduces its `_reference` counterpart in
/// [`crate::reference::BlasMatInterReference`] bit for bit, within the scratch the matching
/// `_tmp_bytes_impl` reports. With less, a call that uses a key panics before it writes `res`, as the reference does.
pub unsafe trait BlasMatInterImpl: Backend {
    fn blas_rotate_tmp_bytes_impl<R, A, K>(module: &Module<Self>, res: &R, a: &A, key: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    fn blas_rotate_impl<DR, DA, K>(
        module: &Module<Self>,
        res: &mut BlasCiphertext<DR, Self::ZnxWord, Self::Ring>,
        a: &BlasCiphertext<DA, Self::ZnxWord, Self::Ring>,
        r: i64,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        DR: Data,
        DA: Data,
        CKKSCiphertext<DR, Self::ZnxWord, Self::Ring>: GLWEToBackendMut<Self>,
        CKKSCiphertext<DA, Self::ZnxWord, Self::Ring>: GLWEToBackendRef<Self>,
        K: GetAutomorphismKey<Self>;

    fn blas_sum_tmp_bytes_impl<R, A, K>(module: &Module<Self>, res: &R, a: &A, key: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    fn blas_sum_impl<DR, DA, K>(
        module: &Module<Self>,
        res: &mut BlasCiphertext<DR, Self::ZnxWord, Self::Ring>,
        a: &BlasCiphertext<DA, Self::ZnxWord, Self::Ring>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        DR: Data,
        DA: Data,
        CKKSCiphertext<DR, Self::ZnxWord, Self::Ring>: GLWEToBackendMut<Self>,
        CKKSCiphertext<DA, Self::ZnxWord, Self::Ring>: GLWEToBackendRef<Self>,
        K: GetAutomorphismKey<Self>;
}

/// Selects [`crate::reference::BlasMatInterReference`] for `$be`.
#[macro_export]
macro_rules! impl_blas_inter_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::BlasMatInterImpl for $be {
            fn blas_rotate_tmp_bytes_impl<R, A, K>(module: &::poulpy_hal::layouts::Module<Self>, res: &R, a: &A, key: &K) -> usize
            where
                R: $crate::layouts::BlasGridInfos,
                A: $crate::layouts::BlasGridInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatInterReference::blas_rotate_tmp_bytes_reference(module, res, a, key)
            }

            fn blas_rotate_impl<DR, DA, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::layouts::BlasCiphertext<DR, Self::ZnxWord, Self::Ring>,
                a: &$crate::layouts::BlasCiphertext<DA, Self::ZnxWord, Self::Ring>,
                r: i64,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                DR: ::poulpy_hal::layouts::Data,
                DA: ::poulpy_hal::layouts::Data,
                ::poulpy_ckks::layouts::CKKSCiphertext<DR, Self::ZnxWord, Self::Ring>:
                    ::poulpy_core::layouts::GLWEToBackendMut<Self>,
                ::poulpy_ckks::layouts::CKKSCiphertext<DA, Self::ZnxWord, Self::Ring>:
                    ::poulpy_core::layouts::GLWEToBackendRef<Self>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self>,
            {
                $crate::reference::BlasMatInterReference::blas_rotate_reference(module, res, a, r, keys, scratch)
            }

            fn blas_sum_tmp_bytes_impl<R, A, K>(module: &::poulpy_hal::layouts::Module<Self>, res: &R, a: &A, key: &K) -> usize
            where
                R: $crate::layouts::BlasGridInfos,
                A: $crate::layouts::BlasGridInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatInterReference::blas_sum_tmp_bytes_reference(module, res, a, key)
            }

            fn blas_sum_impl<DR, DA, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::layouts::BlasCiphertext<DR, Self::ZnxWord, Self::Ring>,
                a: &$crate::layouts::BlasCiphertext<DA, Self::ZnxWord, Self::Ring>,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                DR: ::poulpy_hal::layouts::Data,
                DA: ::poulpy_hal::layouts::Data,
                ::poulpy_ckks::layouts::CKKSCiphertext<DR, Self::ZnxWord, Self::Ring>:
                    ::poulpy_core::layouts::GLWEToBackendMut<Self>,
                ::poulpy_ckks::layouts::CKKSCiphertext<DA, Self::ZnxWord, Self::Ring>:
                    ::poulpy_core::layouts::GLWEToBackendRef<Self>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self>,
            {
                $crate::reference::BlasMatInterReference::blas_sum_reference(module, res, a, keys, scratch)
            }
        }
    };
}
