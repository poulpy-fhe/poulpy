use poulpy_ckks::{CKKSResult as Result, layouts::CKKSCiphertext};
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::layouts::{BlasCiphertext, BlasContext, BlasGridInfos, BlasMatMulLeft, BlasMatMulRight, BlasPtOperand};

/// Extension point of [`crate::api::BlasMatMulOps`].
///
/// # Safety
///
/// Every method reproduces its `_reference` counterpart in
/// [`crate::reference::BlasMatMulReference`] bit for bit, within the scratch the matching
/// `_tmp_bytes_impl` reports. With less, a call that uses a key panics before it writes `res`, as the reference does.
///
/// No block of `res` ever sums more than `2^(63 - base2k)` terms lazily, the most `i64` digits at its radix hold
/// (poulpy-ckks's accumulation bound): the references normalize `res` before an inner index whose terms would pass it,
/// and reject products where one inner index on a normalized block already does (`d + 1` terms, `2` for
/// `blas_matmul_pt`).
pub unsafe trait BlasMatMulImpl: Backend {
    fn blas_matmul_tmp_bytes_impl<R, A, B, T, K>(module: &Module<Self>, res: &R, a: &A, b: &B, tsk: &T, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        B: BlasGridInfos,
        T: GGLWEInfos,
        K: GGLWEInfos;

    fn blas_matmul_impl<DR, DA, DB, K>(
        module: &Module<Self>,
        res: &mut BlasCiphertext<DR, Self::ZnxWord, Self::Ring>,
        a: &BlasCiphertext<DA, Self::ZnxWord, Self::Ring>,
        b: &BlasCiphertext<DB, Self::ZnxWord, Self::Ring>,
        ctx: &BlasContext<Self>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        DR: Data,
        DA: Data,
        DB: Data,
        CKKSCiphertext<DR, Self::ZnxWord, Self::Ring>: GLWEToBackendMut<Self>,
        CKKSCiphertext<DA, Self::ZnxWord, Self::Ring>: GLWEToBackendRef<Self>,
        CKKSCiphertext<DB, Self::ZnxWord, Self::Ring>: GLWEToBackendRef<Self>,
        K: GetTensorKey<Self> + GetAutomorphismKey<Self>;

    fn blas_matmul_prepare_left_tmp_bytes_impl<R, A, K>(module: &Module<Self>, res: &R, a: &A, key: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    fn blas_matmul_prepare_left_impl<DR, DA, K>(
        module: &Module<Self>,
        res: &mut BlasMatMulLeft<DR, Self::ZnxWord, Self::Ring>,
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

    fn blas_matmul_prepare_right_tmp_bytes_impl<R, B, K>(module: &Module<Self>, res: &R, b: &B, key: &K) -> usize
    where
        R: BlasGridInfos,
        B: BlasGridInfos,
        K: GGLWEInfos;

    fn blas_matmul_prepare_right_impl<DR, DB, K>(
        module: &Module<Self>,
        res: &mut BlasMatMulRight<DR, Self::ZnxWord, Self::Ring>,
        b: &BlasCiphertext<DB, Self::ZnxWord, Self::Ring>,
        ctx: &BlasContext<Self>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        DR: Data,
        DB: Data,
        CKKSCiphertext<DR, Self::ZnxWord, Self::Ring>: GLWEToBackendMut<Self>,
        CKKSCiphertext<DB, Self::ZnxWord, Self::Ring>: GLWEToBackendRef<Self>,
        K: GetAutomorphismKey<Self>;

    fn blas_matmul_prepared_tmp_bytes_impl<R, A, B, T, K>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        b: &B,
        tsk: &T,
        atk: &K,
    ) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        B: BlasGridInfos,
        T: GGLWEInfos,
        K: GGLWEInfos;

    fn blas_matmul_prepared_impl<DR, DA, DB, K>(
        module: &Module<Self>,
        res: &mut BlasCiphertext<DR, Self::ZnxWord, Self::Ring>,
        a: &BlasMatMulLeft<DA, Self::ZnxWord, Self::Ring>,
        b: &BlasMatMulRight<DB, Self::ZnxWord, Self::Ring>,
        ctx: &BlasContext<Self>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        DR: Data,
        DA: Data,
        DB: Data,
        CKKSCiphertext<DR, Self::ZnxWord, Self::Ring>: GLWEToBackendMut<Self>,
        CKKSCiphertext<DA, Self::ZnxWord, Self::Ring>: GLWEToBackendRef<Self>,
        CKKSCiphertext<DB, Self::ZnxWord, Self::Ring>: GLWEToBackendRef<Self>,
        K: GetTensorKey<Self> + GetAutomorphismKey<Self>;

    fn blas_matmul_pt_tmp_bytes_impl<R, A, K>(module: &Module<Self>, res: &R, a: &A, key: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    fn blas_matmul_pt_impl<DR, DA, K>(
        module: &Module<Self>,
        res: &mut BlasCiphertext<DR, Self::ZnxWord, Self::Ring>,
        a: &BlasCiphertext<DA, Self::ZnxWord, Self::Ring>,
        b: &BlasPtOperand<Self>,
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

/// Selects [`crate::reference::BlasMatMulReference`] for `$be`.
#[macro_export]
macro_rules! impl_blas_matmul_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::BlasMatMulImpl for $be {
            fn blas_matmul_tmp_bytes_impl<R, A, B, T, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                a: &A,
                b: &B,
                tsk: &T,
                atk: &K,
            ) -> usize
            where
                R: $crate::layouts::BlasGridInfos,
                A: $crate::layouts::BlasGridInfos,
                B: $crate::layouts::BlasGridInfos,
                T: ::poulpy_core::layouts::GGLWEInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatMulReference::blas_matmul_tmp_bytes_reference(module, res, a, b, tsk, atk)
            }

            fn blas_matmul_impl<DR, DA, DB, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::layouts::BlasCiphertext<DR, Self::ZnxWord, Self::Ring>,
                a: &$crate::layouts::BlasCiphertext<DA, Self::ZnxWord, Self::Ring>,
                b: &$crate::layouts::BlasCiphertext<DB, Self::ZnxWord, Self::Ring>,
                ctx: &$crate::layouts::BlasContext<Self>,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                DR: ::poulpy_hal::layouts::Data,
                DA: ::poulpy_hal::layouts::Data,
                DB: ::poulpy_hal::layouts::Data,
                ::poulpy_ckks::layouts::CKKSCiphertext<DR, Self::ZnxWord, Self::Ring>:
                    ::poulpy_core::layouts::GLWEToBackendMut<Self>,
                ::poulpy_ckks::layouts::CKKSCiphertext<DA, Self::ZnxWord, Self::Ring>:
                    ::poulpy_core::layouts::GLWEToBackendRef<Self>,
                ::poulpy_ckks::layouts::CKKSCiphertext<DB, Self::ZnxWord, Self::Ring>:
                    ::poulpy_core::layouts::GLWEToBackendRef<Self>,
                K: ::poulpy_core::layouts::GetTensorKey<Self> + ::poulpy_core::layouts::GetAutomorphismKey<Self>,
            {
                $crate::reference::BlasMatMulReference::blas_matmul_reference(module, res, a, b, ctx, keys, scratch)
            }

            fn blas_matmul_prepare_left_tmp_bytes_impl<R, A, K>(
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
                $crate::reference::BlasMatMulReference::blas_matmul_prepare_left_tmp_bytes_reference(module, res, a, key)
            }

            fn blas_matmul_prepare_left_impl<DR, DA, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::layouts::BlasMatMulLeft<DR, Self::ZnxWord, Self::Ring>,
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
                $crate::reference::BlasMatMulReference::blas_matmul_prepare_left_reference(module, res, a, ctx, keys, scratch)
            }

            fn blas_matmul_prepare_right_tmp_bytes_impl<R, B, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                b: &B,
                key: &K,
            ) -> usize
            where
                R: $crate::layouts::BlasGridInfos,
                B: $crate::layouts::BlasGridInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatMulReference::blas_matmul_prepare_right_tmp_bytes_reference(module, res, b, key)
            }

            fn blas_matmul_prepare_right_impl<DR, DB, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::layouts::BlasMatMulRight<DR, Self::ZnxWord, Self::Ring>,
                b: &$crate::layouts::BlasCiphertext<DB, Self::ZnxWord, Self::Ring>,
                ctx: &$crate::layouts::BlasContext<Self>,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                DR: ::poulpy_hal::layouts::Data,
                DB: ::poulpy_hal::layouts::Data,
                ::poulpy_ckks::layouts::CKKSCiphertext<DR, Self::ZnxWord, Self::Ring>:
                    ::poulpy_core::layouts::GLWEToBackendMut<Self>,
                ::poulpy_ckks::layouts::CKKSCiphertext<DB, Self::ZnxWord, Self::Ring>:
                    ::poulpy_core::layouts::GLWEToBackendRef<Self>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self>,
            {
                $crate::reference::BlasMatMulReference::blas_matmul_prepare_right_reference(module, res, b, ctx, keys, scratch)
            }

            fn blas_matmul_prepared_tmp_bytes_impl<R, A, B, T, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                a: &A,
                b: &B,
                tsk: &T,
                atk: &K,
            ) -> usize
            where
                R: $crate::layouts::BlasGridInfos,
                A: $crate::layouts::BlasGridInfos,
                B: $crate::layouts::BlasGridInfos,
                T: ::poulpy_core::layouts::GGLWEInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatMulReference::blas_matmul_prepared_tmp_bytes_reference(module, res, a, b, tsk, atk)
            }

            fn blas_matmul_prepared_impl<DR, DA, DB, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::layouts::BlasCiphertext<DR, Self::ZnxWord, Self::Ring>,
                a: &$crate::layouts::BlasMatMulLeft<DA, Self::ZnxWord, Self::Ring>,
                b: &$crate::layouts::BlasMatMulRight<DB, Self::ZnxWord, Self::Ring>,
                ctx: &$crate::layouts::BlasContext<Self>,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                DR: ::poulpy_hal::layouts::Data,
                DA: ::poulpy_hal::layouts::Data,
                DB: ::poulpy_hal::layouts::Data,
                ::poulpy_ckks::layouts::CKKSCiphertext<DR, Self::ZnxWord, Self::Ring>:
                    ::poulpy_core::layouts::GLWEToBackendMut<Self>,
                ::poulpy_ckks::layouts::CKKSCiphertext<DA, Self::ZnxWord, Self::Ring>:
                    ::poulpy_core::layouts::GLWEToBackendRef<Self>,
                ::poulpy_ckks::layouts::CKKSCiphertext<DB, Self::ZnxWord, Self::Ring>:
                    ::poulpy_core::layouts::GLWEToBackendRef<Self>,
                K: ::poulpy_core::layouts::GetTensorKey<Self> + ::poulpy_core::layouts::GetAutomorphismKey<Self>,
            {
                $crate::reference::BlasMatMulReference::blas_matmul_prepared_reference(module, res, a, b, ctx, keys, scratch)
            }

            fn blas_matmul_pt_tmp_bytes_impl<R, A, K>(
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
                $crate::reference::BlasMatMulReference::blas_matmul_pt_tmp_bytes_reference(module, res, a, key)
            }

            fn blas_matmul_pt_impl<DR, DA, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::layouts::BlasCiphertext<DR, Self::ZnxWord, Self::Ring>,
                a: &$crate::layouts::BlasCiphertext<DA, Self::ZnxWord, Self::Ring>,
                b: &$crate::layouts::BlasPtOperand<Self>,
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
                $crate::reference::BlasMatMulReference::blas_matmul_pt_reference(module, res, a, b, keys, scratch)
            }
        }
    };
}
