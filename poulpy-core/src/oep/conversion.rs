use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEToBackendRef, GGSWInfos, GGSWToBackendMut, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        LWEToBackendMut, LWEToBackendRef,
        prepared::{GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedToBackendRef},
    },
};

/// Backend-provided ciphertext conversion operations.
///
/// # Safety
/// Implementations must only read and write the regions described by the provided layouts, respect
/// scratch-space requirements, and produce results equivalent to the documented conversion
/// semantics for the backend.
#[allow(private_bounds)]
pub unsafe trait ConversionImpl<BE: Backend>: Backend {
    fn lwe_sample_extract<R, A>(module: &Module<BE>, res: &mut R, a: &A)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_from_lwe_tmp_bytes<R, A, K>(module: &Module<BE>, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn glwe_from_lwe<'s, R, A, K>(
        module: &Module<BE>,
        res: &mut R,
        lwe: &A,
        ksk: &K,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;

    fn lwe_from_glwe_tmp_bytes<R, A, K>(module: &Module<BE>, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn lwe_from_glwe<'s, R, A, K>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        a_idx: usize,
        key: &K,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;

    fn ggsw_from_gglwe_tmp_bytes<R, A>(module: &Module<BE>, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_from_gglwe<'s, R, A, T>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        tsk: &T,
        tsk_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;

    fn ggsw_expand_rows_tmp_bytes<R, A>(module: &Module<BE>, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_expand_row<'s, R, T>(module: &Module<BE>, res: &mut R, tsk: &T, tsk_size: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos;
}

/// Override surface for the conversion family.
///
/// Abstract: no HAL supertraits, no default method bodies. See [`conversion_defaults`]
/// for reference algorithms a backend may forward to.
#[doc(hidden)]
#[allow(private_bounds)]
pub trait ConversionDefault<BE: Backend>
where
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn lwe_sample_extract<R, A>(&self, res: &mut R, a: &A)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_from_lwe_tmp_bytes<R, A, K>(&self, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn glwe_from_lwe<'s, R, A, K>(&self, res: &mut R, lwe: &A, ksk: &K, key_size: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;

    fn lwe_from_glwe_tmp_bytes<R, A, K>(&self, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn lwe_from_glwe<'s, R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        a_idx: usize,
        key: &K,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;

    fn ggsw_from_gglwe_tmp_bytes<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_from_gglwe<'s, R, A, T>(&self, res: &mut R, a: &A, tsk: &T, tsk_size: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;

    fn ggsw_expand_rows_tmp_bytes<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_expand_row<'s, R, T>(&self, res: &mut R, tsk: &T, tsk_size: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos;
}

#[allow(private_bounds)]
unsafe impl<BE: Backend> ConversionImpl<BE> for BE
where
    Module<BE>: ConversionDefault<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn lwe_sample_extract<R, A>(module: &Module<BE>, res: &mut R, a: &A)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        module.lwe_sample_extract(res, a)
    }

    fn glwe_from_lwe_tmp_bytes<R, A, K>(module: &Module<BE>, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
    {
        module.glwe_from_lwe_tmp_bytes(glwe_infos, lwe_infos, key_infos)
    }

    fn glwe_from_lwe<'s, R, A, K>(
        module: &Module<BE>,
        res: &mut R,
        lwe: &A,
        ksk: &K,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    {
        module.glwe_from_lwe(res, lwe, ksk, key_size, scratch)
    }

    fn lwe_from_glwe_tmp_bytes<R, A, K>(module: &Module<BE>, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        module.lwe_from_glwe_tmp_bytes(lwe_infos, glwe_infos, key_infos)
    }

    fn lwe_from_glwe<'s, R, A, K>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        a_idx: usize,
        key: &K,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    {
        module.lwe_from_glwe(res, a, a_idx, key, key_size, scratch)
    }

    fn ggsw_from_gglwe_tmp_bytes<R, A>(module: &Module<BE>, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos,
    {
        module.ggsw_from_gglwe_tmp_bytes(res_infos, tsk_infos)
    }

    fn ggsw_from_gglwe<'s, R, A, T>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        tsk: &T,
        tsk_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    {
        module.ggsw_from_gglwe(res, a, tsk, tsk_size, scratch)
    }

    fn ggsw_expand_rows_tmp_bytes<R, A>(module: &Module<BE>, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos,
    {
        module.ggsw_expand_rows_tmp_bytes(res_infos, tsk_infos)
    }

    fn ggsw_expand_row<'s, R, T>(module: &Module<BE>, res: &mut R, tsk: &T, tsk_size: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        module.ggsw_expand_row(res, tsk, tsk_size, scratch)
    }
}

/// Implements [`ConversionDefault`] for `Module<$be>` by forwarding every method to
/// the corresponding [`conversion_defaults`] free function.
#[macro_export]
macro_rules! impl_conversion_defaults_full {
    ($be:ty) => {
        impl $crate::oep::ConversionDefault<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn lwe_sample_extract<R, A>(&self, res: &mut R, a: &A)
            where
                R: $crate::layouts::LWEToBackendMut<$be> + $crate::layouts::LWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::default::conversion::lwe_sample_extract_default::<$be, _, _, _>(self, res, a)
            }

            fn glwe_from_lwe_tmp_bytes<R, A, K>(&self, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::LWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::default::conversion::glwe_from_lwe_tmp_bytes_default::<$be, _, _, _, _>(
                    self, glwe_infos, lwe_infos, key_infos,
                )
            }

            fn glwe_from_lwe<'s, R, A, K>(
                &self,
                res: &mut R,
                lwe: &A,
                ksk: &K,
                key_size: usize,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'s, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::LWEToBackendRef<$be> + $crate::layouts::LWEInfos,
                K: $crate::layouts::prepared::GGLWEPreparedToBackendRef<$be> + $crate::layouts::GGLWEInfos,
                $be: 's,
            {
                $crate::default::conversion::glwe_from_lwe_default::<$be, _, _, _, _>(self, res, lwe, ksk, key_size, scratch)
            }

            fn lwe_from_glwe_tmp_bytes<R, A, K>(&self, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
            where
                R: $crate::layouts::LWEInfos,
                A: $crate::layouts::GLWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::default::conversion::lwe_from_glwe_tmp_bytes_default::<$be, _, _, _, _>(
                    self, lwe_infos, glwe_infos, key_infos,
                )
            }

            fn lwe_from_glwe<'s, R, A, K>(
                &self,
                res: &mut R,
                a: &A,
                a_idx: usize,
                key: &K,
                key_size: usize,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'s, $be>,
            ) where
                R: $crate::layouts::LWEToBackendMut<$be> + $crate::layouts::LWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
                K: $crate::layouts::prepared::GGLWEPreparedToBackendRef<$be> + $crate::layouts::GGLWEInfos,
                $be: 's,
            {
                $crate::default::conversion::lwe_from_glwe_default::<$be, _, _, _, _>(self, res, a, a_idx, key, key_size, scratch)
            }

            fn ggsw_from_gglwe_tmp_bytes<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
            where
                R: $crate::layouts::GGSWInfos,
                A: $crate::layouts::GGLWEInfos,
            {
                $crate::default::conversion::ggsw_from_gglwe_tmp_bytes_default::<$be, _, _, _>(self, res_infos, tsk_infos)
            }

            fn ggsw_from_gglwe<'s, R, A, T>(
                &self,
                res: &mut R,
                a: &A,
                tsk: &T,
                tsk_size: usize,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'s, $be>,
            ) where
                R: $crate::layouts::GGSWToBackendMut<$be> + $crate::layouts::GGSWInfos,
                A: $crate::layouts::GGLWEToBackendRef<$be> + $crate::layouts::GGLWEInfos,
                T: $crate::layouts::prepared::GGLWEToGGSWKeyPreparedToBackendRef<$be> + $crate::layouts::GGLWEInfos,
                $be: 's,
            {
                $crate::default::conversion::ggsw_from_gglwe_default::<$be, _, _, _, _>(self, res, a, tsk, tsk_size, scratch)
            }

            fn ggsw_expand_rows_tmp_bytes<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
            where
                R: $crate::layouts::GGSWInfos,
                A: $crate::layouts::GGLWEInfos,
            {
                $crate::default::conversion::ggsw_expand_rows_tmp_bytes_default::<$be, _, _, _>(self, res_infos, tsk_infos)
            }

            fn ggsw_expand_row<'s, R, T>(
                &self,
                res: &mut R,
                tsk: &T,
                tsk_size: usize,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'s, $be>,
            ) where
                R: $crate::layouts::GGSWToBackendMut<$be> + $crate::layouts::GGSWInfos,
                T: $crate::layouts::prepared::GGLWEToGGSWKeyPreparedToBackendRef<$be> + $crate::layouts::GGLWEInfos,
            {
                $crate::default::conversion::ggsw_expand_row_default::<$be, _, _, _>(self, res, tsk, tsk_size, scratch)
            }
        }
    };
}
