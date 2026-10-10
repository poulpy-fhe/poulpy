use poulpy_ckks::{CKKSResult as Result, layouts::CKKSCiphertext};
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::layouts::{BlasCiphertext, BlasContext, BlasGridInfos};

/// Extension point of [`crate::api::BlasMatIntraOps`].
///
/// # Safety
///
/// Every method reproduces its `_reference` counterpart in
/// [`crate::reference::BlasMatIntraReference`] bit for bit, within the scratch the matching
/// `_tmp_bytes_impl` reports. With less, a call that uses a key panics before it writes `res`, as the reference does.
pub unsafe trait BlasMatIntraImpl: Backend {
    fn blas_transpose_tmp_bytes_impl<R, A, K>(module: &Module<Self>, res: &R, a: &A, key: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    fn blas_transpose_impl<DR, DA, K>(
        module: &Module<Self>,
        res: &mut BlasCiphertext<DR, Self::ZnxWord, Self::Ring>,
        a: &BlasCiphertext<DA, Self::ZnxWord, Self::Ring>,
        ctx: &BlasContext<Self>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        DR: Data,
        DA: Data,
        CKKSCiphertext<DR, Self::ZnxWord, Self::Ring>: GLWEToBackendMut<Self>,
        CKKSCiphertext<DA, Self::ZnxWord, Self::Ring>: GLWEToBackendRef<Self>,
        K: GetAutomorphismKey<Self>;

    fn blas_rotate_rows_tmp_bytes_impl<R, A, K>(module: &Module<Self>, res: &R, a: &A, key: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    fn blas_rotate_rows_impl<DR, DA, K>(
        module: &Module<Self>,
        res: &mut BlasCiphertext<DR, Self::ZnxWord, Self::Ring>,
        a: &BlasCiphertext<DA, Self::ZnxWord, Self::Ring>,
        k: usize,
        ctx: &BlasContext<Self>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        DR: Data,
        DA: Data,
        CKKSCiphertext<DR, Self::ZnxWord, Self::Ring>: GLWEToBackendMut<Self>,
        CKKSCiphertext<DA, Self::ZnxWord, Self::Ring>: GLWEToBackendRef<Self>,
        K: GetAutomorphismKey<Self>;

    fn blas_rotate_cols_tmp_bytes_impl<R, A, K>(module: &Module<Self>, res: &R, a: &A, key: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    fn blas_rotate_cols_impl<DR, DA, K>(
        module: &Module<Self>,
        res: &mut BlasCiphertext<DR, Self::ZnxWord, Self::Ring>,
        a: &BlasCiphertext<DA, Self::ZnxWord, Self::Ring>,
        k: usize,
        ctx: &BlasContext<Self>,
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

/// Selects [`crate::reference::BlasMatIntraReference`] for `$be`.
#[macro_export]
macro_rules! impl_blas_intra_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::BlasMatIntraImpl for $be {
            fn blas_transpose_tmp_bytes_impl<R, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                a: &A,
                key: &K,
            ) -> usize
            where
                R: $crate::layouts::BlasGridInfos,
                A: $crate::layouts::BlasGridInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatIntraReference::blas_transpose_tmp_bytes_reference(module, res, a, key)
            }

            fn blas_transpose_impl<DR, DA, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::layouts::BlasCiphertext<DR, Self::ZnxWord, Self::Ring>,
                a: &$crate::layouts::BlasCiphertext<DA, Self::ZnxWord, Self::Ring>,
                ctx: &$crate::layouts::BlasContext<Self>,
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
                $crate::reference::BlasMatIntraReference::blas_transpose_reference(module, res, a, ctx, keys, scratch)
            }

            fn blas_rotate_rows_tmp_bytes_impl<R, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                a: &A,
                key: &K,
            ) -> usize
            where
                R: $crate::layouts::BlasGridInfos,
                A: $crate::layouts::BlasGridInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatIntraReference::blas_rotate_rows_tmp_bytes_reference(module, res, a, key)
            }

            fn blas_rotate_rows_impl<DR, DA, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::layouts::BlasCiphertext<DR, Self::ZnxWord, Self::Ring>,
                a: &$crate::layouts::BlasCiphertext<DA, Self::ZnxWord, Self::Ring>,
                k: usize,
                ctx: &$crate::layouts::BlasContext<Self>,
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
                $crate::reference::BlasMatIntraReference::blas_rotate_rows_reference(module, res, a, k, ctx, keys, scratch)
            }

            fn blas_rotate_cols_tmp_bytes_impl<R, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                a: &A,
                key: &K,
            ) -> usize
            where
                R: $crate::layouts::BlasGridInfos,
                A: $crate::layouts::BlasGridInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatIntraReference::blas_rotate_cols_tmp_bytes_reference(module, res, a, key)
            }

            fn blas_rotate_cols_impl<DR, DA, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::layouts::BlasCiphertext<DR, Self::ZnxWord, Self::Ring>,
                a: &$crate::layouts::BlasCiphertext<DA, Self::ZnxWord, Self::Ring>,
                k: usize,
                ctx: &$crate::layouts::BlasContext<Self>,
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
                $crate::reference::BlasMatIntraReference::blas_rotate_cols_reference(module, res, a, k, ctx, keys, scratch)
            }
        }
    };
}
