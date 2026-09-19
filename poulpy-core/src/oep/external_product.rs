use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::layouts::{GGSWInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GGSWPreparedBackendRef};

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

/// Override surface for the GLWE external-product sub-family.
///
/// Abstract: no HAL supertraits, no default method bodies. See
/// [`glwe_external_product_reference`] for reference algorithms a backend may forward to.
pub trait GLWEExternalProductReference<BE: Backend> {
    fn glwe_external_product_dft_fill_tmp_bytes_reference<A, G>(&self, a_infos: &A, ggsw_infos: &G) -> usize
    where
        A: GLWEInfos,
        G: GGSWInfos;

    fn glwe_external_product_tmp_bytes_reference<R, A, G>(&self, res_infos: &R, a_infos: &A, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        G: GGSWInfos;

    fn glwe_external_product_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        ggsw: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_external_product_assign_reference<R>(
        &self,
        res: &mut R,
        ggsw: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos;
}

unsafe impl<BE: Backend> GLWEExternalProductImpl for BE
where
    Module<BE>: GLWEExternalProductReference<BE>,
{
    fn glwe_external_product_tmp_bytes<R, A, G>(module: &Module<BE>, res_infos: &R, a_infos: &A, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        G: GGSWInfos,
    {
        module.glwe_external_product_tmp_bytes_reference(res_infos, a_infos, ggsw_infos)
    }

    fn glwe_external_product<R, A>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        ggsw: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        module.glwe_external_product_reference(res, a, ggsw, scratch)
    }

    fn glwe_external_product_assign<R>(
        module: &Module<BE>,
        res: &mut R,
        ggsw: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
    {
        module.glwe_external_product_assign_reference(res, ggsw, scratch)
    }
}

/// Implements [`GLWEExternalProductReference`] for `Module<$be>` by forwarding every method to
/// the corresponding [`glwe_external_product_reference`] free function.
#[macro_export]
macro_rules! impl_glwe_external_product_reference_full {
    ($be:ty) => {
        impl $crate::oep::GLWEExternalProductReference<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn glwe_external_product_dft_fill_tmp_bytes_reference<A, G>(&self, a_infos: &A, ggsw_infos: &G) -> usize
            where
                A: $crate::layouts::GLWEInfos,
                G: $crate::layouts::GGSWInfos,
            {
                $crate::reference::external_product::glwe::glwe_external_product_dft_fill_tmp_bytes_reference::<$be, _, _, _>(
                    self, a_infos, ggsw_infos,
                )
            }

            fn glwe_external_product_tmp_bytes_reference<R, A, G>(&self, res_infos: &R, a_infos: &A, ggsw_infos: &G) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEInfos,
                G: $crate::layouts::GGSWInfos,
            {
                $crate::reference::external_product::glwe::glwe_external_product_tmp_bytes_reference::<$be, _, _, _, _>(
                    self, res_infos, a_infos, ggsw_infos,
                )
            }

            fn glwe_external_product_reference<R, A>(
                &self,
                res: &mut R,
                a: &A,
                ggsw: &$crate::layouts::prepared::GGSWPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::external_product::glwe::glwe_external_product_reference::<$be, _, _, _>(
                    self, res, a, ggsw, scratch,
                )
            }

            fn glwe_external_product_assign_reference<R>(
                &self,
                res: &mut R,
                ggsw: &$crate::layouts::prepared::GGSWPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::external_product::glwe::glwe_external_product_assign_reference::<$be, _, _>(
                    self, res, ggsw, scratch,
                )
            }
        }
    };
}
