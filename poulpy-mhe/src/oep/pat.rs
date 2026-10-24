use poulpy_core::layouts::{
    GGLWECompressedSeed, GGLWECompressedToBackendMut, GGLWECompressedToBackendRef, GGLWEInfos, GGLWEToBackendMut,
    GGLWEToBackendRef, GLWECompressedSeed, GLWECompressedToBackendMut, GLWECompressedToBackendRef, GLWEInfos, GLWEToBackendMut,
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

/// # Safety
/// Every method reproduces the reference or derived body its doc names, checks
/// included, and stays within the scratch its `_tmp_bytes` query returns. The
/// synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md`
/// apply.
pub unsafe trait GLWEPatCompressedImpl: Backend {
    /// Reproduces `GLWEPatCompressedReference::glwe_pat_compressed_aggregate_assign_reference`: adds `a` into `res` without normalizing.
    /// Panics unless the layouts and the mask seeds agree.
    fn glwe_pat_compressed_aggregate_assign<R, A>(module: &Module<Self>, res: &mut R, a: &A)
    where
        R: GLWECompressedToBackendMut<Self> + GLWECompressedSeed + GLWEInfos,
        A: GLWECompressedToBackendRef<Self> + GLWECompressedSeed + GLWEInfos;

    /// Upper bound on the scratch of `glwe_pat_compressed_finalize`, as `GLWEPatCompressedReference::glwe_pat_compressed_finalize_tmp_bytes_reference`.
    fn glwe_pat_compressed_finalize_tmp_bytes(module: &Module<Self>) -> usize;

    /// Reproduces `GLWEPatCompressedReference::glwe_pat_compressed_finalize_reference`: writes the canonical normalization of the aggregate into `res`.
    /// Panics unless the layouts agree; the masks are regenerated from the seed, already canonical, and only the body is normalized.
    fn glwe_pat_compressed_finalize<R, P>(module: &Module<Self>, res: &mut R, pat: &P, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        P: GLWECompressedToBackendRef<Self> + GLWECompressedSeed + GLWEInfos;
}

/// # Safety
/// Every method reproduces the reference or derived body its doc names, checks
/// included, and stays within the scratch its `_tmp_bytes` query returns. The
/// synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md`
/// apply.
pub unsafe trait GGLWEPatCompressedImpl: Backend {
    /// Reproduces `GGLWEPatCompressedReference::gglwe_pat_compressed_aggregate_assign_reference`: adds `a` into `res` without normalizing.
    /// Panics unless the layouts and the mask seeds agree.
    fn gglwe_pat_compressed_aggregate_assign<R, A>(module: &Module<Self>, res: &mut R, a: &A)
    where
        R: GGLWECompressedToBackendMut<Self> + GGLWECompressedSeed + GGLWEInfos,
        A: GGLWECompressedToBackendRef<Self> + GGLWECompressedSeed + GGLWEInfos;

    /// Upper bound on the scratch of `gglwe_pat_compressed_finalize`, as `GGLWEPatCompressedReference::gglwe_pat_compressed_finalize_tmp_bytes_reference`.
    fn gglwe_pat_compressed_finalize_tmp_bytes(module: &Module<Self>) -> usize;

    /// Reproduces `GGLWEPatCompressedReference::gglwe_pat_compressed_finalize_reference`: writes the canonical normalization of the aggregate into `res`.
    /// Panics unless the layouts agree; the masks are regenerated from the seed, already canonical, and only the bodies are normalized.
    fn gglwe_pat_compressed_finalize<R, P>(module: &Module<Self>, res: &mut R, pat: &P, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos,
        P: GGLWECompressedToBackendRef<Self> + GGLWEInfos;
}

/// # Safety
/// Every method reproduces the reference or derived body its doc names, checks
/// included, and stays within the scratch its `_tmp_bytes` query returns. The
/// synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md`
/// apply.
pub unsafe trait GGLWEPatImpl: Backend {
    /// Reproduces `GGLWEPatReference::gglwe_pat_aggregate_assign_reference`: adds `a` into `res` without normalizing.
    /// Panics unless the layouts agree.
    fn gglwe_pat_aggregate_assign<R, A>(module: &Module<Self>, res: &mut R, a: &A)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos,
        A: GGLWEToBackendRef<Self> + GGLWEInfos;

    /// Upper bound on the scratch of `gglwe_pat_finalize`, as `GGLWEPatReference::gglwe_pat_finalize_tmp_bytes_reference`.
    fn gglwe_pat_finalize_tmp_bytes(module: &Module<Self>) -> usize;

    /// Reproduces `GGLWEPatReference::gglwe_pat_finalize_reference`: writes the canonical normalization of the aggregate into `res`.
    /// Panics unless the layouts agree; every column is normalized.
    fn gglwe_pat_finalize<R, P>(module: &Module<Self>, res: &mut R, pat: &P, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos,
        P: GGLWEToBackendRef<Self> + GGLWEInfos;
}

/// Selects the reference aggregation and finalization of `GLWEPatCompressed`.
#[macro_export]
macro_rules! impl_glwe_pat_compressed_reference_full {
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
    };
}

/// Selects the reference aggregation and finalization of `GGLWEPatCompressed`.
#[macro_export]
macro_rules! impl_gglwe_pat_compressed_reference_full {
    ($be:ty) => {
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
    };
}

/// Selects the reference aggregation and finalization of `GGLWEPat`.
#[macro_export]
macro_rules! impl_gglwe_pat_reference_full {
    ($be:ty) => {
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
