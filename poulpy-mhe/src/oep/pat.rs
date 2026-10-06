use poulpy_core::layouts::{
    GGLWECompressedSeed, GGLWECompressedToBackendMut, GGLWECompressedToBackendRef, GGLWEInfos, GGLWEToBackendMut,
    GGLWEToBackendRef, GLWECompressedSeed, GLWECompressedToBackendMut, GLWECompressedToBackendRef, GLWEInfos, GLWEToBackendMut,
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

/// # Safety
/// Outputs follow the [noise metadata rule](crate::oep#noise-metadata). Delegates only forward calls.
/// Reproduce the reference sum, including the seed and layout checks, and the
/// reference canonical ciphertext, masks included, within the queried scratch
/// budget.
pub unsafe trait GLWEPatCompressedImpl: Backend {
    fn glwe_pat_compressed_aggregate_assign<R, A>(module: &Module<Self>, res: &mut R, a: &A)
    where
        R: GLWECompressedToBackendMut<Self> + GLWECompressedSeed + GLWEInfos,
        A: GLWECompressedToBackendRef<Self> + GLWECompressedSeed + GLWEInfos;

    fn glwe_pat_compressed_finalize_tmp_bytes(module: &Module<Self>) -> usize;

    fn glwe_pat_compressed_finalize<R, P>(module: &Module<Self>, res: &mut R, pat: &P, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        P: GLWECompressedToBackendRef<Self> + GLWECompressedSeed + GLWEInfos;
}

/// # Safety
/// Outputs follow the [noise metadata rule](crate::oep#noise-metadata). Delegates only forward calls.
/// Reproduce the reference sum, including the seed and layout checks, and the
/// reference canonical ciphertext, masks included, within the queried scratch
/// budget.
pub unsafe trait GGLWEPatCompressedImpl: Backend {
    fn gglwe_pat_compressed_aggregate_assign<R, A>(module: &Module<Self>, res: &mut R, a: &A)
    where
        R: GGLWECompressedToBackendMut<Self> + GGLWECompressedSeed + GGLWEInfos,
        A: GGLWECompressedToBackendRef<Self> + GGLWECompressedSeed + GGLWEInfos;

    fn gglwe_pat_compressed_finalize_tmp_bytes(module: &Module<Self>) -> usize;

    fn gglwe_pat_compressed_finalize<R, P>(module: &Module<Self>, res: &mut R, pat: &P, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos,
        P: GGLWECompressedToBackendRef<Self> + GGLWEInfos;
}

/// # Safety
/// Outputs follow the [noise metadata rule](crate::oep#noise-metadata). Delegates only forward calls.
/// Reproduce the reference sum, including the layout check, and the reference
/// canonical ciphertext, masks included, within the queried scratch budget.
pub unsafe trait GGLWEPatImpl: Backend {
    fn gglwe_pat_aggregate_assign<R, A>(module: &Module<Self>, res: &mut R, a: &A)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos,
        A: GGLWEToBackendRef<Self> + GGLWEInfos;

    fn gglwe_pat_finalize_tmp_bytes(module: &Module<Self>) -> usize;

    fn gglwe_pat_finalize<R, P>(module: &Module<Self>, res: &mut R, pat: &P, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos,
        P: GGLWEToBackendRef<Self> + GGLWEInfos;
}

/// Selects the reference aggregation and finalization of every PAT type.
#[macro_export]
macro_rules! impl_mhe_pat_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::GLWEPatCompressedImpl for $be {
            fn glwe_pat_compressed_aggregate_assign<R, A>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, a: &A)
            where
                R: ::poulpy_core::layouts::GLWECompressedToBackendMut<$be>
                    + ::poulpy_core::layouts::GLWECompressedSeed
                    + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWECompressedToBackendRef<$be>
                    + ::poulpy_core::layouts::GLWECompressedSeed
                    + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEPatCompressedReference<$be>>::glwe_pat_compressed_aggregate_assign_reference(module, res, a)
            }

            fn glwe_pat_compressed_finalize_tmp_bytes(module: &::poulpy_hal::layouts::Module<$be>) -> usize {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEPatCompressedReference<$be>>::glwe_pat_compressed_finalize_tmp_bytes_reference(module)
            }

            fn glwe_pat_compressed_finalize<R, P>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                pat: &P,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                P: ::poulpy_core::layouts::GLWECompressedToBackendRef<$be>
                    + ::poulpy_core::layouts::GLWECompressedSeed
                    + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEPatCompressedReference<$be>>::glwe_pat_compressed_finalize_reference(module, res, pat, scratch)
            }
        }

        unsafe impl $crate::oep::GGLWEPatCompressedImpl for $be {
            fn gglwe_pat_compressed_aggregate_assign<R, A>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, a: &A)
            where
                R: ::poulpy_core::layouts::GGLWECompressedToBackendMut<$be>
                    + ::poulpy_core::layouts::GGLWECompressedSeed
                    + ::poulpy_core::layouts::GGLWEInfos,
                A: ::poulpy_core::layouts::GGLWECompressedToBackendRef<$be>
                    + ::poulpy_core::layouts::GGLWECompressedSeed
                    + ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GGLWEPatCompressedReference<$be>>::gglwe_pat_compressed_aggregate_assign_reference(module, res, a)
            }

            fn gglwe_pat_compressed_finalize_tmp_bytes(module: &::poulpy_hal::layouts::Module<$be>) -> usize {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GGLWEPatCompressedReference<$be>>::gglwe_pat_compressed_finalize_tmp_bytes_reference(module)
            }

            fn gglwe_pat_compressed_finalize<R, P>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                pat: &P,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GGLWEToBackendMut<$be> + ::poulpy_core::layouts::GGLWEInfos,
                P: ::poulpy_core::layouts::GGLWECompressedToBackendRef<$be> + ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GGLWEPatCompressedReference<$be>>::gglwe_pat_compressed_finalize_reference(module, res, pat, scratch)
            }
        }

        unsafe impl $crate::oep::GGLWEPatImpl for $be {
            fn gglwe_pat_aggregate_assign<R, A>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, a: &A)
            where
                R: ::poulpy_core::layouts::GGLWEToBackendMut<$be> + ::poulpy_core::layouts::GGLWEInfos,
                A: ::poulpy_core::layouts::GGLWEToBackendRef<$be> + ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GGLWEPatReference<$be>>::gglwe_pat_aggregate_assign_reference(module, res, a)
            }

            fn gglwe_pat_finalize_tmp_bytes(module: &::poulpy_hal::layouts::Module<$be>) -> usize {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GGLWEPatReference<$be>>::gglwe_pat_finalize_tmp_bytes_reference(module)
            }

            fn gglwe_pat_finalize<R, P>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                pat: &P,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GGLWEToBackendMut<$be> + ::poulpy_core::layouts::GGLWEInfos,
                P: ::poulpy_core::layouts::GGLWEToBackendRef<$be> + ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GGLWEPatReference<$be>>::gglwe_pat_finalize_reference(module, res, pat, scratch)
            }
        }
    };
}
