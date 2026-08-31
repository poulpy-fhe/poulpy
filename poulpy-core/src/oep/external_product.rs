use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::layouts::{
    GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut,
    GGSWToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GGSWPreparedBackendRef,
};

/// Backend hook for GLWE external products.
///
/// # Safety
/// Implementors must preserve the semantics, scratch requirements, and aliasing
/// guarantees expected by the public and default external-product layers.
pub unsafe trait GLWEExternalProductImpl<BE: Backend>: Backend {
    fn glwe_external_product_tmp_bytes<R, A, G>(module: &Module<BE>, res_infos: &R, a_infos: &A, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        G: GGSWInfos;

    fn glwe_external_product<R, A>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        ggsw: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_external_product_assign<R>(
        module: &Module<BE>,
        res: &mut R,
        ggsw: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos;
}

/// Backend hook for batched GGLWE external products.
///
/// # Safety
/// Implementors must preserve the semantics, scratch requirements, and aliasing
/// guarantees expected by the public and default external-product layers.
pub unsafe trait GGLWEExternalProductImpl<BE: Backend>: Backend {
    fn gglwe_external_product_tmp_bytes<R, A, B>(module: &Module<BE>, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        B: GGSWInfos;

    fn gglwe_external_product<R, A>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        b: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos;

    fn gglwe_external_product_assign<R>(
        module: &Module<BE>,
        res: &mut R,
        a: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos;
}

/// Backend hook for GGSW external products.
///
/// # Safety
/// Implementors must preserve the semantics, scratch requirements, and aliasing
/// guarantees expected by the public and default external-product layers.
pub unsafe trait GGSWExternalProductImpl<BE: Backend>: Backend {
    fn ggsw_external_product_tmp_bytes<R, A, B>(module: &Module<BE>, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        B: GGSWInfos;

    fn ggsw_external_product<R, A>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        b: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWAtViewRef<BE> + GGSWInfos;

    fn ggsw_external_product_assign<R>(
        module: &Module<BE>,
        res: &mut R,
        a: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos;
}

/// Override surface for the GLWE external-product sub-family.
///
/// Abstract: no HAL supertraits, no default method bodies. See
/// [`glwe_external_product_defaults`] for reference algorithms a backend may forward to.
pub trait GLWEExternalProductDefault<BE: Backend> {
    fn glwe_external_product_dft_fill_tmp_bytes_default<A, G>(&self, a_infos: &A, ggsw_infos: &G) -> usize
    where
        A: GLWEInfos,
        G: GGSWInfos;

    fn glwe_external_product_tmp_bytes_default<R, A, G>(&self, res_infos: &R, a_infos: &A, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        G: GGSWInfos;

    fn glwe_external_product_default<R, A>(
        &self,
        res: &mut R,
        a: &A,
        ggsw: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_external_product_assign_default<R>(
        &self,
        res: &mut R,
        ggsw: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos;
}

/// Override surface for the GGLWE external-product sub-family.
pub trait GGLWEExternalProductDefault<BE: Backend> {
    fn gglwe_external_product_tmp_bytes_default<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        B: GGSWInfos;

    fn gglwe_external_product_default<R, A>(
        &self,
        res: &mut R,
        a: &A,
        b: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos;

    fn gglwe_external_product_assign_default<R>(
        &self,
        res: &mut R,
        a: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos;
}

/// Override surface for the GGSW external-product sub-family.
pub trait GGSWExternalProductDefault<BE: Backend> {
    fn ggsw_external_product_tmp_bytes_default<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        B: GGSWInfos;

    fn ggsw_external_product_default<R, A>(
        &self,
        res: &mut R,
        a: &A,
        b: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWAtViewRef<BE> + GGSWInfos;

    fn ggsw_external_product_assign_default<R>(
        &self,
        res: &mut R,
        a: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos;
}

unsafe impl<BE: Backend> GLWEExternalProductImpl<BE> for BE
where
    Module<BE>: GLWEExternalProductDefault<BE>,
{
    fn glwe_external_product_tmp_bytes<R, A, G>(module: &Module<BE>, res_infos: &R, a_infos: &A, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        G: GGSWInfos,
    {
        module.glwe_external_product_tmp_bytes_default(res_infos, a_infos, ggsw_infos)
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
        module.glwe_external_product_default(res, a, ggsw, scratch)
    }

    fn glwe_external_product_assign<R>(
        module: &Module<BE>,
        res: &mut R,
        ggsw: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
    {
        module.glwe_external_product_assign_default(res, ggsw, scratch)
    }
}

unsafe impl<BE: Backend> GGLWEExternalProductImpl<BE> for BE
where
    Module<BE>: GGLWEExternalProductDefault<BE>,
{
    fn gglwe_external_product_tmp_bytes<R, A, B>(module: &Module<BE>, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        B: GGSWInfos,
    {
        module.gglwe_external_product_tmp_bytes_default(res_infos, a_infos, b_infos)
    }

    fn gglwe_external_product<R, A>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        b: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
    {
        module.gglwe_external_product_default(res, a, b, scratch)
    }

    fn gglwe_external_product_assign<R>(
        module: &Module<BE>,
        res: &mut R,
        a: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
    {
        module.gglwe_external_product_assign_default(res, a, scratch)
    }
}

unsafe impl<BE: Backend> GGSWExternalProductImpl<BE> for BE
where
    Module<BE>: GGSWExternalProductDefault<BE>,
{
    fn ggsw_external_product_tmp_bytes<R, A, B>(module: &Module<BE>, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        B: GGSWInfos,
    {
        module.ggsw_external_product_tmp_bytes_default(res_infos, a_infos, b_infos)
    }

    fn ggsw_external_product<R, A>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        b: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWAtViewRef<BE> + GGSWInfos,
    {
        module.ggsw_external_product_default(res, a, b, scratch)
    }

    fn ggsw_external_product_assign<R>(
        module: &Module<BE>,
        res: &mut R,
        a: &GGSWPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos + GGSWAtViewMut<BE>,
    {
        module.ggsw_external_product_assign_default(res, a, scratch)
    }
}

/// Implements [`GLWEExternalProductDefault`] for `Module<$be>` by forwarding every method to
/// the corresponding [`glwe_external_product_defaults`] free function.
#[macro_export]
macro_rules! impl_glwe_external_product_defaults_full {
    ($be:ty) => {
        impl $crate::oep::GLWEExternalProductDefault<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn glwe_external_product_dft_fill_tmp_bytes_default<A, G>(&self, a_infos: &A, ggsw_infos: &G) -> usize
            where
                A: $crate::layouts::GLWEInfos,
                G: $crate::layouts::GGSWInfos,
            {
                $crate::default::external_product::glwe::glwe_external_product_dft_fill_tmp_bytes_default::<$be, _, _, _>(
                    self, a_infos, ggsw_infos,
                )
            }

            fn glwe_external_product_tmp_bytes_default<R, A, G>(&self, res_infos: &R, a_infos: &A, ggsw_infos: &G) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEInfos,
                G: $crate::layouts::GGSWInfos,
            {
                $crate::default::external_product::glwe::glwe_external_product_tmp_bytes_default::<$be, _, _, _, _>(
                    self, res_infos, a_infos, ggsw_infos,
                )
            }

            fn glwe_external_product_default<R, A>(
                &self,
                res: &mut R,
                a: &A,
                ggsw: &$crate::layouts::prepared::GGSWPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::default::external_product::glwe::glwe_external_product_default::<$be, _, _, _>(
                    self, res, a, ggsw, scratch,
                )
            }

            fn glwe_external_product_assign_default<R>(
                &self,
                res: &mut R,
                ggsw: &$crate::layouts::prepared::GGSWPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::default::external_product::glwe::glwe_external_product_assign_default::<$be, _, _>(
                    self, res, ggsw, scratch,
                )
            }
        }
    };
}

/// Implements [`GGLWEExternalProductDefault`] for `Module<$be>` by forwarding every method to
/// the corresponding [`gglwe_external_product_defaults`] free function.
#[macro_export]
macro_rules! impl_gglwe_external_product_defaults_full {
    ($be:ty) => {
        impl $crate::oep::GGLWEExternalProductDefault<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn gglwe_external_product_tmp_bytes_default<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
            where
                R: $crate::layouts::GGLWEInfos,
                A: $crate::layouts::GGLWEInfos,
                B: $crate::layouts::GGSWInfos,
            {
                $crate::default::external_product::gglwe::gglwe_external_product_tmp_bytes_default::<$be, _, _, _, _>(
                    self, res_infos, a_infos, b_infos,
                )
            }

            fn gglwe_external_product_default<R, A>(
                &self,
                res: &mut R,
                a: &A,
                b: &$crate::layouts::prepared::GGSWPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::GGLWEInfos,
                A: $crate::layouts::GGLWEToBackendRef<$be> + $crate::layouts::GGLWEInfos,
            {
                $crate::default::external_product::gglwe::gglwe_external_product_default::<$be, _, _, _>(self, res, a, b, scratch)
            }

            fn gglwe_external_product_assign_default<R>(
                &self,
                res: &mut R,
                a: &$crate::layouts::prepared::GGSWPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::GGLWEInfos,
            {
                $crate::default::external_product::gglwe::gglwe_external_product_assign_default::<$be, _, _>(
                    self, res, a, scratch,
                )
            }
        }
    };
}

/// Implements [`GGSWExternalProductDefault`] for `Module<$be>` by forwarding every method to
/// the corresponding [`ggsw_external_product_defaults`] free function.
#[macro_export]
macro_rules! impl_ggsw_external_product_defaults_full {
    ($be:ty) => {
        impl $crate::oep::GGSWExternalProductDefault<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn ggsw_external_product_tmp_bytes_default<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
            where
                R: $crate::layouts::GGSWInfos,
                A: $crate::layouts::GGSWInfos,
                B: $crate::layouts::GGSWInfos,
            {
                $crate::default::external_product::ggsw::ggsw_external_product_tmp_bytes_default::<$be, _, _, _, _>(
                    self, res_infos, a_infos, b_infos,
                )
            }

            fn ggsw_external_product_default<R, A>(
                &self,
                res: &mut R,
                a: &A,
                b: &$crate::layouts::prepared::GGSWPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GGSWToBackendMut<$be> + $crate::layouts::GGSWAtViewMut<$be> + $crate::layouts::GGSWInfos,
                A: $crate::layouts::GGSWToBackendRef<$be> + $crate::layouts::GGSWAtViewRef<$be> + $crate::layouts::GGSWInfos,
            {
                $crate::default::external_product::ggsw::ggsw_external_product_default::<$be, _, _, _>(self, res, a, b, scratch)
            }

            fn ggsw_external_product_assign_default<R>(
                &self,
                res: &mut R,
                a: &$crate::layouts::prepared::GGSWPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GGSWToBackendMut<$be> + $crate::layouts::GGSWAtViewMut<$be> + $crate::layouts::GGSWInfos,
            {
                $crate::default::external_product::ggsw::ggsw_external_product_assign_default::<$be, _, _>(self, res, a, scratch)
            }
        }
    };
}
