use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::layouts::{BlasGrid, BlasGridInfos, BlasGridMut};

/// Extension point of [`crate::api::BlasMatInterOps`].
///
/// # Safety
///
/// Every method reproduces its `_reference` counterpart in
/// [`crate::reference::BlasMatInterReference`] bit for bit, within the scratch the matching
/// `_tmp_bytes_impl` reports. With less, a call that uses a key panics before it writes `res`, as the reference does.
pub unsafe trait BlasMatInterImpl: Backend {
    fn blas_rotate_tmp_bytes_impl<R, A, K>(module: &Module<Self>, res: &R, a: &A, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    fn blas_rotate_impl<R, A, K>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        r: i64,
        atk: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasGridMut<Self>,
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + Sync;

    fn blas_sum_tmp_bytes_impl<R, A, K>(module: &Module<Self>, res: &R, a: &A, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    fn blas_sum_impl<R, A, K>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        atk: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasGridMut<Self>,
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + Sync;
}

/// Selects [`crate::reference::BlasMatInterReference`] for `$be`.
#[macro_export]
macro_rules! impl_blas_inter_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::BlasMatInterImpl for $be {
            fn blas_rotate_tmp_bytes_impl<R, A, K>(module: &::poulpy_hal::layouts::Module<Self>, res: &R, a: &A, atk: &K) -> usize
            where
                R: $crate::layouts::BlasGridInfos,
                A: $crate::layouts::BlasGridInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatInterReference::blas_rotate_tmp_bytes_reference(module, res, a, atk)
            }

            fn blas_rotate_impl<R, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                a: &A,
                r: i64,
                atk: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                R: $crate::layouts::BlasGridMut<Self>,
                A: $crate::layouts::BlasGrid<Self>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + Sync,
            {
                $crate::reference::BlasMatInterReference::blas_rotate_reference(module, res, a, r, atk, scratch)
            }

            fn blas_sum_tmp_bytes_impl<R, A, K>(module: &::poulpy_hal::layouts::Module<Self>, res: &R, a: &A, atk: &K) -> usize
            where
                R: $crate::layouts::BlasGridInfos,
                A: $crate::layouts::BlasGridInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatInterReference::blas_sum_tmp_bytes_reference(module, res, a, atk)
            }

            fn blas_sum_impl<R, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                a: &A,
                atk: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                R: $crate::layouts::BlasGridMut<Self>,
                A: $crate::layouts::BlasGrid<Self>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + Sync,
            {
                $crate::reference::BlasMatInterReference::blas_sum_reference(module, res, a, atk, scratch)
            }
        }
    };
}
