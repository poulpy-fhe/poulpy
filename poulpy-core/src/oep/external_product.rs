use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::layouts::{
    GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut,
    GGSWToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GGSWPreparedBackendRef,
};

/// Backend hook for GLWE external products.
///
/// # Safety
/// Implementors must preserve the semantics, scratch requirements, and aliasing
/// guarantees expected by the public and reference external-product layers.
pub unsafe trait GLWEExternalProductImpl: Backend {
    fn glwe_external_product_tmp_bytes<R, A, G>(module: &Module<Self>, res_infos: &R, a_infos: &A, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        G: GGSWInfos;

    fn glwe_external_product<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        ggsw: &GGSWPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos;

    fn glwe_external_product_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        ggsw: &GGSWPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos;
}

/// Backend hook for batched GGLWE external products.
///
/// # Safety
/// Implementors must preserve the semantics, scratch requirements, and aliasing
/// guarantees expected by the public and reference external-product layers.
pub unsafe trait GGLWEExternalProductImpl: Backend + GLWEExternalProductImpl + crate::oep::GLWEZeroImpl {
    fn gglwe_external_product_tmp_bytes<R, A, B>(module: &Module<Self>, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        B: GGSWInfos,
    {
        crate::oep::derived::external_product::gglwe_external_product_tmp_bytes_derived::<Self, _, _, _, _>(
            module, res_infos, a_infos, b_infos,
        )
    }

    fn gglwe_external_product<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        b: &GGSWPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self> + GGLWEInfos,
        A: GGLWEToBackendRef<Self> + GGLWEInfos,
    {
        crate::oep::derived::external_product::gglwe_external_product_derived::<Self, _, _, _>(module, res, a, b, scratch)
    }

    fn gglwe_external_product_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        a: &GGSWPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self> + GGLWEInfos,
    {
        crate::oep::derived::external_product::gglwe_external_product_assign_derived::<Self, _, _>(module, res, a, scratch)
    }
}

/// Backend hook for GGSW external products.
///
/// # Safety
/// Implementors must preserve the semantics, scratch requirements, and aliasing
/// guarantees expected by the public and reference external-product layers.
pub unsafe trait GGSWExternalProductImpl: Backend + GLWEExternalProductImpl + crate::oep::GLWEZeroImpl {
    fn ggsw_external_product_tmp_bytes<R, A, B>(module: &Module<Self>, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        B: GGSWInfos,
    {
        crate::oep::derived::external_product::ggsw_external_product_tmp_bytes_derived::<Self, _, _, _, _>(
            module, res_infos, a_infos, b_infos,
        )
    }

    fn ggsw_external_product<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        b: &GGSWPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self> + GGSWAtViewMut<Self> + GGSWInfos,
        A: GGSWToBackendRef<Self> + GGSWAtViewRef<Self> + GGSWInfos,
    {
        crate::oep::derived::external_product::ggsw_external_product_derived::<Self, _, _, _>(module, res, a, b, scratch)
    }

    fn ggsw_external_product_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        a: &GGSWPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self> + GGSWAtViewMut<Self> + GGSWInfos,
    {
        crate::oep::derived::external_product::ggsw_external_product_assign_derived::<Self, _, _>(module, res, a, scratch)
    }
}

/// Selects the portable HAL algorithms for this backend operation family.
#[macro_export]
macro_rules! impl_glwe_external_product_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::GLWEExternalProductImpl for $be {
            fn glwe_external_product_tmp_bytes<R, A, G>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res_infos: &R,
                a_infos: &A,
                ggsw_infos: &G,
            ) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEInfos,
                G: $crate::layouts::GGSWInfos,
            {
                $crate::reference::external_product::glwe::glwe_external_product_tmp_bytes_reference::<$be, _, _, _, _>(
                    module, res_infos, a_infos, ggsw_infos,
                )
            }

            fn glwe_external_product<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                a: &A,
                ggsw: &$crate::layouts::prepared::GGSWPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::external_product::glwe::glwe_external_product_reference::<$be, _, _, _>(
                    module, res, a, ggsw, scratch,
                )
            }

            fn glwe_external_product_assign<R>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                ggsw: &$crate::layouts::prepared::GGSWPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::external_product::glwe::glwe_external_product_assign_reference::<$be, _, _>(
                    module, res, ggsw, scratch,
                )
            }
        }
    };
}

/// Selects the same-layer row composition for this backend.
#[macro_export]
macro_rules! impl_gglwe_external_product_derived_full {
    ($be:ty) => {
        unsafe impl $crate::oep::GGLWEExternalProductImpl for $be {}
    };
}

/// Selects the same-layer row composition for this backend.
#[macro_export]
macro_rules! impl_ggsw_external_product_derived_full {
    ($be:ty) => {
        unsafe impl $crate::oep::GGSWExternalProductImpl for $be {}
    };
}

// Reference helpers remain available through OEP for source compatibility.
pub use crate::reference::external_product::GLWEExternalProductReference;
