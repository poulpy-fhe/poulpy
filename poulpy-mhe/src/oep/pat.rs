use poulpy_core::layouts::{GGLWEInfos, GGLWEToBackendMut, GLWEInfos, GLWEToBackendMut};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::layouts::{GGLWEPatCompressedOwned, GGLWEPatOwned, GLWEPatCompressedOwned};

/// # Safety
/// Reproduce the reference sum, including the seed and layout checks, and clear
/// the canonical flag of `res`.
pub unsafe trait PatAggregateImpl: Backend {
    fn glwe_pat_compressed_aggregate_assign(
        module: &Module<Self>,
        res: &mut GLWEPatCompressedOwned<Self>,
        a: &GLWEPatCompressedOwned<Self>,
    );

    fn gglwe_pat_compressed_aggregate_assign(
        module: &Module<Self>,
        res: &mut GGLWEPatCompressedOwned<Self>,
        a: &GGLWEPatCompressedOwned<Self>,
    );

    fn gglwe_pat_aggregate_assign(module: &Module<Self>, res: &mut GGLWEPatOwned<Self>, a: &GGLWEPatOwned<Self>);
}

/// # Safety
/// Produce the reference canonical digits within the queried scratch budget.
pub unsafe trait PatNormalizeImpl: Backend {
    fn pat_normalize_tmp_bytes(module: &Module<Self>) -> usize;

    fn glwe_pat_compressed_normalize_assign(
        module: &Module<Self>,
        res: &mut GLWEPatCompressedOwned<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    );

    fn gglwe_pat_compressed_normalize_assign(
        module: &Module<Self>,
        res: &mut GGLWEPatCompressedOwned<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    );

    fn gglwe_pat_normalize_assign(module: &Module<Self>, res: &mut GGLWEPatOwned<Self>, scratch: &mut ScratchArena<'_, Self>);
}

/// # Safety
/// Reproduce the reference ciphertext, masks included, within the queried
/// scratch budget.
pub unsafe trait PatFinalizeImpl: Backend {
    fn pat_finalize_tmp_bytes(module: &Module<Self>) -> usize;

    fn glwe_pat_compressed_finalize<R>(
        module: &Module<Self>,
        res: &mut R,
        pat: &GLWEPatCompressedOwned<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos;

    fn gglwe_pat_compressed_finalize<R>(
        module: &Module<Self>,
        res: &mut R,
        pat: &GGLWEPatCompressedOwned<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self> + GGLWEInfos;

    fn gglwe_pat_finalize<R>(module: &Module<Self>, res: &mut R, pat: &GGLWEPatOwned<Self>, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos;
}

/// Selects the reference aggregation, normalization and finalization of every PAT shape.
#[macro_export]
macro_rules! impl_mhe_pat_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::PatAggregateImpl for $be {
            fn glwe_pat_compressed_aggregate_assign(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GLWEPatCompressedOwned<$be>,
                a: &$crate::layouts::GLWEPatCompressedOwned<$be>,
            ) {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::PatAggregateReference<$be>>::glwe_pat_compressed_aggregate_assign_reference(module, res, a)
            }

            fn gglwe_pat_compressed_aggregate_assign(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GGLWEPatCompressedOwned<$be>,
                a: &$crate::layouts::GGLWEPatCompressedOwned<$be>,
            ) {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::PatAggregateReference<$be>>::gglwe_pat_compressed_aggregate_assign_reference(module, res, a)
            }

            fn gglwe_pat_aggregate_assign(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GGLWEPatOwned<$be>,
                a: &$crate::layouts::GGLWEPatOwned<$be>,
            ) {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::PatAggregateReference<$be>>::gglwe_pat_aggregate_assign_reference(module, res, a)
            }
        }

        unsafe impl $crate::oep::PatNormalizeImpl for $be {
            fn pat_normalize_tmp_bytes(module: &::poulpy_hal::layouts::Module<$be>) -> usize {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::PatNormalizeReference<$be>>::pat_normalize_tmp_bytes_reference(module)
            }

            fn glwe_pat_compressed_normalize_assign(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GLWEPatCompressedOwned<$be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::PatNormalizeReference<$be>>::glwe_pat_compressed_normalize_assign_reference(module, res, scratch)
            }

            fn gglwe_pat_compressed_normalize_assign(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GGLWEPatCompressedOwned<$be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::PatNormalizeReference<$be>>::gglwe_pat_compressed_normalize_assign_reference(module, res, scratch)
            }

            fn gglwe_pat_normalize_assign(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GGLWEPatOwned<$be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::PatNormalizeReference<$be>>::gglwe_pat_normalize_assign_reference(module, res, scratch)
            }
        }

        unsafe impl $crate::oep::PatFinalizeImpl for $be {
            fn pat_finalize_tmp_bytes(module: &::poulpy_hal::layouts::Module<$be>) -> usize {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::PatFinalizeReference<$be>>::pat_finalize_tmp_bytes_reference(module)
            }

            fn glwe_pat_compressed_finalize<R>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                pat: &$crate::layouts::GLWEPatCompressedOwned<$be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::PatFinalizeReference<$be>>::glwe_pat_compressed_finalize_reference(module, res, pat, scratch)
            }

            fn gglwe_pat_compressed_finalize<R>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                pat: &$crate::layouts::GGLWEPatCompressedOwned<$be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GGLWEToBackendMut<$be> + ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::PatFinalizeReference<$be>>::gglwe_pat_compressed_finalize_reference(module, res, pat, scratch)
            }

            fn gglwe_pat_finalize<R>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                pat: &$crate::layouts::GGLWEPatOwned<$be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GGLWEToBackendMut<$be> + ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::PatFinalizeReference<$be>>::gglwe_pat_finalize_reference(module, res, pat, scratch)
            }
        }
    };
}
