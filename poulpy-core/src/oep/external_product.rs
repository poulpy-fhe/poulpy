use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut,
        GGSWToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GGSWPreparedToBackendRef,
    },
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

    fn glwe_external_product<'s, R, A, G>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        ggsw: &G,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        G: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        BE: 's;

    fn glwe_external_product_assign<'s, R, G>(
        module: &Module<BE>,
        res: &mut R,
        ggsw: &G,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        G: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        BE: 's;
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

    fn gglwe_external_product<'s, R, A, B>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        b: &B,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
        B: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        BE: 's;

    fn gglwe_external_product_assign<'s, R, A>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        BE: 's;
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

    fn ggsw_external_product<'s, R, A, B>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        b: &B,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWAtViewRef<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWAtViewRef<BE> + GGSWInfos,
        B: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        BE: 's;

    fn ggsw_external_product_assign<'s, R, A>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos + GGSWAtViewMut<BE>,
        A: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        BE: 's;
}

/// Override surface for the GLWE external-product sub-family.
///
/// Abstract: no HAL supertraits, no default method bodies. See
/// [`glwe_external_product_defaults`] for reference algorithms a backend may forward to.
#[doc(hidden)]
#[allow(private_bounds)]
pub trait GLWEExternalProductDefault<BE: Backend>
where
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn glwe_external_product_dft_fill_tmp_bytes<A, G>(&self, a_infos: &A, ggsw_infos: &G) -> usize
    where
        A: GLWEInfos,
        G: GGSWInfos;

    fn glwe_external_product_tmp_bytes<R, A, G>(&self, res_infos: &R, a_infos: &A, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        G: GGSWInfos;

    fn glwe_external_product<'s, R, A, G>(
        &self,
        res: &mut R,
        a: &A,
        ggsw: &G,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        G: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        BE: 's;

    fn glwe_external_product_assign<'s, R, G>(&self, res: &mut R, ggsw: &G, key_size: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        G: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        BE: 's;
}

/// Override surface for the GGLWE external-product sub-family.
#[doc(hidden)]
#[allow(private_bounds)]
pub trait GGLWEExternalProductDefault<BE: Backend>
where
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn gglwe_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        B: GGSWInfos;

    fn gglwe_external_product<'s, R, A, B>(&self, res: &mut R, a: &A, b: &B, key_size: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
        B: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        BE: 's;

    fn gglwe_external_product_assign<'s, R, A>(&self, res: &mut R, a: &A, key_size: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        BE: 's;
}

/// Override surface for the GGSW external-product sub-family.
#[doc(hidden)]
#[allow(private_bounds)]
pub trait GGSWExternalProductDefault<BE: Backend>
where
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn ggsw_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        B: GGSWInfos;

    fn ggsw_external_product<'s, R, A, B>(&self, res: &mut R, a: &A, b: &B, key_size: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWAtViewRef<BE> + GGSWInfos,
        B: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        BE: 's;

    fn ggsw_external_product_assign<'s, R, A>(&self, res: &mut R, a: &A, key_size: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        A: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        BE: 's;
}

#[allow(private_bounds)]
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
        module.glwe_external_product_tmp_bytes(res_infos, a_infos, ggsw_infos)
    }

    fn glwe_external_product<'s, R, A, G>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        ggsw: &G,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        G: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        BE: 's,
    {
        module.glwe_external_product(res, a, ggsw, key_size, scratch)
    }

    fn glwe_external_product_assign<'s, R, G>(
        module: &Module<BE>,
        res: &mut R,
        ggsw: &G,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        G: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        BE: 's,
    {
        module.glwe_external_product_assign(res, ggsw, key_size, scratch)
    }
}

#[allow(private_bounds)]
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
        module.gglwe_external_product_tmp_bytes(res_infos, a_infos, b_infos)
    }

    fn gglwe_external_product<'s, R, A, B>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        b: &B,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
        B: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        BE: 's,
    {
        module.gglwe_external_product(res, a, b, key_size, scratch)
    }

    fn gglwe_external_product_assign<'s, R, A>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        BE: 's,
    {
        module.gglwe_external_product_assign(res, a, key_size, scratch)
    }
}

#[allow(private_bounds)]
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
        module.ggsw_external_product_tmp_bytes(res_infos, a_infos, b_infos)
    }

    fn ggsw_external_product<'s, R, A, B>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        b: &B,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWAtViewRef<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWAtViewRef<BE> + GGSWInfos,
        B: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        BE: 's,
    {
        module.ggsw_external_product(res, a, b, key_size, scratch)
    }

    fn ggsw_external_product_assign<'s, R, A>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos + GGSWAtViewMut<BE>,
        A: GGSWPreparedToBackendRef<BE> + GGSWInfos,
        BE: 's,
    {
        module.ggsw_external_product_assign(res, a, key_size, scratch)
    }
}

/// Implements [`GLWEExternalProductDefault`] for `Module<$be>` by forwarding every method to
/// the corresponding [`glwe_external_product_defaults`] free function.
#[macro_export]
macro_rules! impl_glwe_external_product_defaults_full {
    ($be:ty) => {
        impl $crate::oep::GLWEExternalProductDefault<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn glwe_external_product_dft_fill_tmp_bytes<A, G>(&self, a_infos: &A, ggsw_infos: &G) -> usize
            where
                A: $crate::layouts::GLWEInfos,
                G: $crate::layouts::GGSWInfos,
            {
                $crate::default::external_product::glwe::glwe_external_product_dft_fill_tmp_bytes_default::<$be, _, _, _>(
                    self, a_infos, ggsw_infos,
                )
            }

            fn glwe_external_product_tmp_bytes<R, A, G>(&self, res_infos: &R, a_infos: &A, ggsw_infos: &G) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEInfos,
                G: $crate::layouts::GGSWInfos,
            {
                $crate::default::external_product::glwe::glwe_external_product_tmp_bytes_default::<$be, _, _, _, _>(
                    self, res_infos, a_infos, ggsw_infos,
                )
            }

            fn glwe_external_product<'s, R, A, G>(
                &self,
                res: &mut R,
                a: &A,
                ggsw: &G,
                key_size: usize,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'s, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
                G: $crate::layouts::prepared::GGSWPreparedToBackendRef<$be> + $crate::layouts::GGSWInfos,
                $be: 's,
            {
                $crate::default::external_product::glwe::glwe_external_product_default::<$be, _, _, _, _>(
                    self, res, a, ggsw, key_size, scratch,
                )
            }

            fn glwe_external_product_assign<'s, R, G>(
                &self,
                res: &mut R,
                ggsw: &G,
                key_size: usize,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'s, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                G: $crate::layouts::prepared::GGSWPreparedToBackendRef<$be> + $crate::layouts::GGSWInfos,
                $be: 's,
            {
                $crate::default::external_product::glwe::glwe_external_product_assign_default::<$be, _, _, _>(
                    self, res, ggsw, key_size, scratch,
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
            fn gglwe_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
            where
                R: $crate::layouts::GGLWEInfos,
                A: $crate::layouts::GGLWEInfos,
                B: $crate::layouts::GGSWInfos,
            {
                $crate::default::external_product::gglwe::gglwe_external_product_tmp_bytes_default::<$be, _, _, _, _>(
                    self, res_infos, a_infos, b_infos,
                )
            }

            fn gglwe_external_product<'s, R, A, B>(
                &self,
                res: &mut R,
                a: &A,
                b: &B,
                key_size: usize,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'s, $be>,
            ) where
                R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::GGLWEInfos,
                A: $crate::layouts::GGLWEToBackendRef<$be> + $crate::layouts::GGLWEInfos,
                B: $crate::layouts::prepared::GGSWPreparedToBackendRef<$be> + $crate::layouts::GGSWInfos,
                $be: 's,
            {
                $crate::default::external_product::gglwe::gglwe_external_product_default::<$be, _, _, _, _>(
                    self, res, a, b, key_size, scratch,
                )
            }

            fn gglwe_external_product_assign<'s, R, A>(
                &self,
                res: &mut R,
                a: &A,
                key_size: usize,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'s, $be>,
            ) where
                R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::GGLWEInfos,
                A: $crate::layouts::prepared::GGSWPreparedToBackendRef<$be> + $crate::layouts::GGSWInfos,
                $be: 's,
            {
                $crate::default::external_product::gglwe::gglwe_external_product_assign_default::<$be, _, _, _>(
                    self, res, a, key_size, scratch,
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
            fn ggsw_external_product_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
            where
                R: $crate::layouts::GGSWInfos,
                A: $crate::layouts::GGSWInfos,
                B: $crate::layouts::GGSWInfos,
            {
                $crate::default::external_product::ggsw::ggsw_external_product_tmp_bytes_default::<$be, _, _, _, _>(
                    self, res_infos, a_infos, b_infos,
                )
            }

            fn ggsw_external_product<'s, R, A, B>(
                &self,
                res: &mut R,
                a: &A,
                b: &B,
                key_size: usize,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'s, $be>,
            ) where
                R: $crate::layouts::GGSWToBackendMut<$be> + $crate::layouts::GGSWAtViewMut<$be> + $crate::layouts::GGSWInfos,
                A: $crate::layouts::GGSWToBackendRef<$be> + $crate::layouts::GGSWAtViewRef<$be> + $crate::layouts::GGSWInfos,
                B: $crate::layouts::prepared::GGSWPreparedToBackendRef<$be> + $crate::layouts::GGSWInfos,
                $be: 's,
            {
                $crate::default::external_product::ggsw::ggsw_external_product_default::<$be, _, _, _, _>(
                    self, res, a, b, key_size, scratch,
                )
            }

            fn ggsw_external_product_assign<'s, R, A>(
                &self,
                res: &mut R,
                a: &A,
                key_size: usize,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'s, $be>,
            ) where
                R: $crate::layouts::GGSWToBackendMut<$be> + $crate::layouts::GGSWAtViewMut<$be> + $crate::layouts::GGSWInfos,
                A: $crate::layouts::prepared::GGSWPreparedToBackendRef<$be> + $crate::layouts::GGSWInfos,
                $be: 's,
            {
                $crate::default::external_product::ggsw::ggsw_external_product_assign_default::<$be, _, _, _>(
                    self, res, a, key_size, scratch,
                )
            }
        }
    };
}
