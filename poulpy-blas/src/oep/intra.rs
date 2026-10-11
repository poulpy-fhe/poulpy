use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::layouts::{BlasContext, BlasGrid, BlasGridInfos, BlasGridMut};

/// Extension point of [`crate::api::BlasMatIntraOps`].
///
/// # Safety
///
/// Every method reproduces its `_reference` counterpart in
/// [`crate::reference::BlasMatIntraReference`] bit for bit, within the scratch the matching
/// `_tmp_bytes_impl` reports. With less, a call that uses a key panics before it writes `res`, as the reference does.
pub unsafe trait BlasMatIntraImpl: Backend {
    fn blas_transpose_tmp_bytes_impl<R, A, K>(module: &Module<Self>, res: &R, a: &A, ctx: &BlasContext<Self>, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    fn blas_transpose_impl<R, A, K>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        ctx: &BlasContext<Self>,
        atk: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasGridMut<Self>,
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + Sync;

    fn blas_rotate_rows_tmp_bytes_impl<R, A, K>(module: &Module<Self>, res: &R, a: &A, ctx: &BlasContext<Self>, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    fn blas_rotate_rows_impl<R, A, K>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        k: usize,
        ctx: &BlasContext<Self>,
        atk: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasGridMut<Self>,
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + Sync;

    fn blas_rotate_cols_tmp_bytes_impl<R, A, K>(module: &Module<Self>, res: &R, a: &A, ctx: &BlasContext<Self>, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    fn blas_rotate_cols_impl<R, A, K>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        k: usize,
        ctx: &BlasContext<Self>,
        atk: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasGridMut<Self>,
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + Sync;
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
                ctx: &$crate::layouts::BlasContext<Self>,
                atk: &K,
            ) -> usize
            where
                R: $crate::layouts::BlasGridInfos,
                A: $crate::layouts::BlasGridInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatIntraReference::blas_transpose_tmp_bytes_reference(module, res, a, ctx, atk)
            }

            fn blas_transpose_impl<R, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                a: &A,
                ctx: &$crate::layouts::BlasContext<Self>,
                atk: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                R: $crate::layouts::BlasGridMut<Self>,
                A: $crate::layouts::BlasGrid<Self>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + Sync,
            {
                $crate::reference::BlasMatIntraReference::blas_transpose_reference(module, res, a, ctx, atk, scratch)
            }

            fn blas_rotate_rows_tmp_bytes_impl<R, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                a: &A,
                ctx: &$crate::layouts::BlasContext<Self>,
                atk: &K,
            ) -> usize
            where
                R: $crate::layouts::BlasGridInfos,
                A: $crate::layouts::BlasGridInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatIntraReference::blas_rotate_rows_tmp_bytes_reference(module, res, a, ctx, atk)
            }

            fn blas_rotate_rows_impl<R, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                a: &A,
                k: usize,
                ctx: &$crate::layouts::BlasContext<Self>,
                atk: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                R: $crate::layouts::BlasGridMut<Self>,
                A: $crate::layouts::BlasGrid<Self>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + Sync,
            {
                $crate::reference::BlasMatIntraReference::blas_rotate_rows_reference(module, res, a, k, ctx, atk, scratch)
            }

            fn blas_rotate_cols_tmp_bytes_impl<R, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                a: &A,
                ctx: &$crate::layouts::BlasContext<Self>,
                atk: &K,
            ) -> usize
            where
                R: $crate::layouts::BlasGridInfos,
                A: $crate::layouts::BlasGridInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatIntraReference::blas_rotate_cols_tmp_bytes_reference(module, res, a, ctx, atk)
            }

            fn blas_rotate_cols_impl<R, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                a: &A,
                k: usize,
                ctx: &$crate::layouts::BlasContext<Self>,
                atk: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                R: $crate::layouts::BlasGridMut<Self>,
                A: $crate::layouts::BlasGrid<Self>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + Sync,
            {
                $crate::reference::BlasMatIntraReference::blas_rotate_cols_reference(module, res, a, k, ctx, atk, scratch)
            }
        }
    };
}
