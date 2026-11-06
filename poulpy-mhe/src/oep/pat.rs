use poulpy_core::layouts::{
    GGLWECompressedSeed, GGLWECompressedToBackendMut, GGLWECompressedToBackendRef, GGLWEInfos, GGLWEToBackendMut,
    GGLWEToBackendRef, GLWECompressedSeed, GLWECompressedToBackendMut, GLWECompressedToBackendRef, GLWEInfos, GLWEToBackendMut,
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

/// # Safety
/// Every method reproduces the reference or derived body its doc names, checks
/// included, and stays within the scratch its `_tmp_bytes` query returns. The
/// memory, layout, synchrony and aliasing rules of
/// `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait MHEGLWEPatCompressedImpl: Backend {
    /// Reproduces
    /// `MHEGLWEPatCompressedReference::mhe_glwe_pat_compressed_aggregate_assign_reference`:
    /// adds `a` into `res` without normalizing.
    /// Panics unless the layouts and the mask seeds agree.
    fn mhe_glwe_pat_compressed_aggregate_assign<R, A>(module: &Module<Self>, res: &mut R, a: &A)
    where
        R: GLWECompressedToBackendMut<Self> + GLWECompressedSeed + GLWEInfos,
        A: GLWECompressedToBackendRef<Self> + GLWECompressedSeed + GLWEInfos;

    /// Upper bound on the scratch of `mhe_glwe_pat_compressed_finalize`, as
    /// `MHEGLWEPatCompressedReference::mhe_glwe_pat_compressed_finalize_tmp_bytes_reference`.
    fn mhe_glwe_pat_compressed_finalize_tmp_bytes(module: &Module<Self>) -> usize;

    /// Reproduces
    /// `MHEGLWEPatCompressedReference::mhe_glwe_pat_compressed_finalize_reference`:
    /// writes the canonical normalization of the aggregate into `res`.
    /// Panics unless the layouts agree; the masks are regenerated from the
    /// seed, already canonical, and only the body is normalized.
    fn mhe_glwe_pat_compressed_finalize<R, P>(module: &Module<Self>, res: &mut R, pat: &P, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        P: GLWECompressedToBackendRef<Self> + GLWECompressedSeed + GLWEInfos;
}

/// # Safety
/// Every method reproduces the reference or derived body its doc names, checks
/// included, and stays within the scratch its `_tmp_bytes` query returns. The
/// memory, layout, synchrony and aliasing rules of
/// `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait MHEGGLWEPatCompressedImpl: Backend {
    /// Reproduces
    /// `MHEGGLWEPatCompressedReference::mhe_gglwe_pat_compressed_aggregate_assign_reference`:
    /// adds `a` into `res` without normalizing.
    /// Panics unless the layouts and the mask seeds agree.
    fn mhe_gglwe_pat_compressed_aggregate_assign<R, A>(module: &Module<Self>, res: &mut R, a: &A)
    where
        R: GGLWECompressedToBackendMut<Self> + GGLWECompressedSeed + GGLWEInfos,
        A: GGLWECompressedToBackendRef<Self> + GGLWECompressedSeed + GGLWEInfos;

    /// Upper bound on the scratch of `mhe_gglwe_pat_compressed_finalize`, as
    /// `MHEGGLWEPatCompressedReference::mhe_gglwe_pat_compressed_finalize_tmp_bytes_reference`.
    fn mhe_gglwe_pat_compressed_finalize_tmp_bytes(module: &Module<Self>) -> usize;

    /// Reproduces
    /// `MHEGGLWEPatCompressedReference::mhe_gglwe_pat_compressed_finalize_reference`:
    /// writes the canonical normalization of the aggregate into `res`.
    /// Panics unless the layouts agree; the masks are regenerated from the
    /// seed, already canonical, and only the bodies are normalized.
    fn mhe_gglwe_pat_compressed_finalize<R, P>(module: &Module<Self>, res: &mut R, pat: &P, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos,
        P: GGLWECompressedToBackendRef<Self> + GGLWEInfos;
}

/// # Safety
/// Every method reproduces the reference or derived body its doc names, checks
/// included, and stays within the scratch its `_tmp_bytes` query returns. The
/// memory, layout, synchrony and aliasing rules of
/// `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait MHEGGLWEPatImpl: Backend {
    /// Reproduces `MHEGGLWEPatReference::mhe_gglwe_pat_aggregate_assign_reference`:
    /// adds `a` into `res` without normalizing.
    /// Panics unless the layouts agree.
    fn mhe_gglwe_pat_aggregate_assign<R, A>(module: &Module<Self>, res: &mut R, a: &A)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos,
        A: GGLWEToBackendRef<Self> + GGLWEInfos;

    /// Upper bound on the scratch of `mhe_gglwe_pat_finalize`, as
    /// `MHEGGLWEPatReference::mhe_gglwe_pat_finalize_tmp_bytes_reference`.
    fn mhe_gglwe_pat_finalize_tmp_bytes(module: &Module<Self>) -> usize;

    /// Reproduces `MHEGGLWEPatReference::mhe_gglwe_pat_finalize_reference`: writes the
    /// canonical normalization of the aggregate into `res`.
    /// Panics unless the layouts agree; every column is normalized.
    fn mhe_gglwe_pat_finalize<R, P>(module: &Module<Self>, res: &mut R, pat: &P, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos,
        P: GGLWEToBackendRef<Self> + GGLWEInfos;
}

/// Selects the reference aggregation and finalization of `MHEGLWEPatCompressed`.
#[macro_export]
macro_rules! impl_mhe_glwe_pat_compressed_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::MHEGLWEPatCompressedImpl for $be {
            fn mhe_glwe_pat_compressed_aggregate_assign<R, A>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, a: &A)
            where
                R: ::poulpy_core::layouts::GLWECompressedToBackendMut<$be>
                    + ::poulpy_core::layouts::GLWECompressedSeed
                    + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWECompressedToBackendRef<$be>
                    + ::poulpy_core::layouts::GLWECompressedSeed
                    + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGLWEPatCompressedReference<$be>>::mhe_glwe_pat_compressed_aggregate_assign_reference(module, res, a)
            }

            fn mhe_glwe_pat_compressed_finalize_tmp_bytes(module: &::poulpy_hal::layouts::Module<$be>) -> usize {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGLWEPatCompressedReference<$be>>::mhe_glwe_pat_compressed_finalize_tmp_bytes_reference(module)
            }

            fn mhe_glwe_pat_compressed_finalize<R, P>(
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
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGLWEPatCompressedReference<$be>>::mhe_glwe_pat_compressed_finalize_reference(module, res, pat, scratch)
            }
        }
    };
}

/// Selects the reference aggregation and finalization of `MHEGGLWEPatCompressed`.
#[macro_export]
macro_rules! impl_mhe_gglwe_pat_compressed_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::MHEGGLWEPatCompressedImpl for $be {
            fn mhe_gglwe_pat_compressed_aggregate_assign<R, A>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, a: &A)
            where
                R: ::poulpy_core::layouts::GGLWECompressedToBackendMut<$be>
                    + ::poulpy_core::layouts::GGLWECompressedSeed
                    + ::poulpy_core::layouts::GGLWEInfos,
                A: ::poulpy_core::layouts::GGLWECompressedToBackendRef<$be>
                    + ::poulpy_core::layouts::GGLWECompressedSeed
                    + ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGGLWEPatCompressedReference<$be>>::mhe_gglwe_pat_compressed_aggregate_assign_reference(module, res, a)
            }

            fn mhe_gglwe_pat_compressed_finalize_tmp_bytes(module: &::poulpy_hal::layouts::Module<$be>) -> usize {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGGLWEPatCompressedReference<$be>>::mhe_gglwe_pat_compressed_finalize_tmp_bytes_reference(module)
            }

            fn mhe_gglwe_pat_compressed_finalize<R, P>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                pat: &P,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GGLWEToBackendMut<$be> + ::poulpy_core::layouts::GGLWEInfos,
                P: ::poulpy_core::layouts::GGLWECompressedToBackendRef<$be> + ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGGLWEPatCompressedReference<$be>>::mhe_gglwe_pat_compressed_finalize_reference(module, res, pat, scratch)
            }
        }
    };
}

/// Selects the reference aggregation and finalization of `MHEGGLWEPat`.
#[macro_export]
macro_rules! impl_mhe_gglwe_pat_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::MHEGGLWEPatImpl for $be {
            fn mhe_gglwe_pat_aggregate_assign<R, A>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, a: &A)
            where
                R: ::poulpy_core::layouts::GGLWEToBackendMut<$be> + ::poulpy_core::layouts::GGLWEInfos,
                A: ::poulpy_core::layouts::GGLWEToBackendRef<$be> + ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGGLWEPatReference<$be>>::mhe_gglwe_pat_aggregate_assign_reference(module, res, a)
            }

            fn mhe_gglwe_pat_finalize_tmp_bytes(module: &::poulpy_hal::layouts::Module<$be>) -> usize {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGGLWEPatReference<$be>>::mhe_gglwe_pat_finalize_tmp_bytes_reference(module)
            }

            fn mhe_gglwe_pat_finalize<R, P>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                pat: &P,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GGLWEToBackendMut<$be> + ::poulpy_core::layouts::GGLWEInfos,
                P: ::poulpy_core::layouts::GGLWEToBackendRef<$be> + ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGGLWEPatReference<$be>>::mhe_gglwe_pat_finalize_reference(module, res, pat, scratch)
            }
        }
    };
}
