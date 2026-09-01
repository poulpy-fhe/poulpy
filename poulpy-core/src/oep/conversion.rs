use poulpy_hal::layouts::{Backend, Module, Normalized, ScratchArena};

use crate::layouts::{
    GGLWEInfos, GGLWEToBackendRef, GGSWInfos, GGSWToBackendMut, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
    LWEMatrixInfos, LWEMatrixToBackendMut, LWEToBackendMut, LWEToBackendRef,
    prepared::{GGLWEPreparedBackendRef, GGLWEToGGSWKeyPreparedBackendRef},
};

/// Backend-provided ciphertext conversion operations.
///
/// # Safety
/// Implementations must only read and write the regions described by the provided layouts, respect
/// scratch-space requirements, and produce results equivalent to the documented conversion
/// semantics for the backend.
pub unsafe trait ConversionImpl<BE: Backend>: Backend {
    fn lwe_sample_extract<R, A>(module: &Module<BE>, res: &mut R, a: &A)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos;

    fn glwe_from_lwe_tmp_bytes<R, A, K>(module: &Module<BE>, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn glwe_from_lwe<R, A>(
        module: &Module<BE>,
        res: &mut R,
        lwe: &A,
        ksk: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos;

    fn lwe_from_glwe_tmp_bytes<R, A, K>(module: &Module<BE>, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn lwe_from_glwe<R, A>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        a_idx: usize,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos;

    fn ggsw_from_gglwe_tmp_bytes<R, A>(module: &Module<BE>, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_from_gglwe<R, A>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos;

    fn glwe_expand_lwe_tmp_bytes<R, A>(module: &Module<BE>, lwe_infos: &R, a_infos: &A) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos;

    fn glwe_expand_lwe<R, A>(module: &Module<BE>, res: &mut [R], a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos;

    fn glwe_expand_lwe_matrix_tmp_bytes<R, A>(module: &Module<BE>, res_infos: &R, a_infos: &A) -> usize
    where
        R: LWEMatrixInfos,
        A: GLWEInfos;

    fn glwe_expand_lwe_matrix<R, A>(module: &Module<BE>, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEMatrixToBackendMut<BE> + LWEMatrixInfos,
        A: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos;

    fn ggsw_expand_rows_tmp_bytes<R, A>(module: &Module<BE>, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_expand_row<R>(
        module: &Module<BE>,
        res: &mut R,
        tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos;
}

/// Override surface for the conversion family.
///
/// Abstract: no HAL supertraits, no default method bodies. See [`conversion_defaults`]
/// for reference algorithms a backend may forward to.
pub trait ConversionDefault<BE: Backend> {
    fn lwe_sample_extract_default<R, A>(&self, res: &mut R, a: &A)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos;

    fn glwe_from_lwe_tmp_bytes_default<R, A, K>(&self, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn glwe_from_lwe_default<R, A>(
        &self,
        res: &mut R,
        lwe: &A,
        ksk: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos;

    fn lwe_from_glwe_tmp_bytes_default<R, A, K>(&self, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn lwe_from_glwe_default<R, A>(
        &self,
        res: &mut R,
        a: &A,
        a_idx: usize,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos;

    fn ggsw_from_gglwe_tmp_bytes_default<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_from_gglwe_default<R, A>(
        &self,
        res: &mut R,
        a: &A,
        tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos;

    fn glwe_expand_lwe_tmp_bytes_default<R, A>(&self, lwe_infos: &R, a_infos: &A) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos;

    fn glwe_expand_lwe_default<R, A>(&self, res: &mut [R], a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos;

    fn glwe_expand_lwe_matrix_tmp_bytes_default<R, A>(&self, res_infos: &R, a_infos: &A) -> usize
    where
        R: LWEMatrixInfos,
        A: GLWEInfos;

    fn glwe_expand_lwe_matrix_default<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEMatrixToBackendMut<BE> + LWEMatrixInfos,
        A: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos;

    fn ggsw_expand_rows_tmp_bytes_default<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_expand_row_default<R>(
        &self,
        res: &mut R,
        tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos;
}

unsafe impl<BE: Backend> ConversionImpl<BE> for BE
where
    Module<BE>: ConversionDefault<BE>,
{
    fn lwe_sample_extract<R, A>(module: &Module<BE>, res: &mut R, a: &A)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos,
    {
        module.lwe_sample_extract_default(res, a)
    }

    fn glwe_from_lwe_tmp_bytes<R, A, K>(module: &Module<BE>, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
    {
        module.glwe_from_lwe_tmp_bytes_default(glwe_infos, lwe_infos, key_infos)
    }

    fn glwe_from_lwe<R, A>(
        module: &Module<BE>,
        res: &mut R,
        lwe: &A,
        ksk: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos,
    {
        module.glwe_from_lwe_default(res, lwe, ksk, scratch)
    }

    fn lwe_from_glwe_tmp_bytes<R, A, K>(module: &Module<BE>, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        module.lwe_from_glwe_tmp_bytes_default(lwe_infos, glwe_infos, key_infos)
    }

    fn lwe_from_glwe<R, A>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        a_idx: usize,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos,
    {
        module.lwe_from_glwe_default(res, a, a_idx, key, scratch)
    }

    fn ggsw_from_gglwe_tmp_bytes<R, A>(module: &Module<BE>, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos,
    {
        module.ggsw_from_gglwe_tmp_bytes_default(res_infos, tsk_infos)
    }

    fn ggsw_from_gglwe<R, A>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
    {
        module.ggsw_from_gglwe_default(res, a, tsk, scratch)
    }

    fn glwe_expand_lwe_tmp_bytes<R, A>(module: &Module<BE>, lwe_infos: &R, a_infos: &A) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos,
    {
        module.glwe_expand_lwe_tmp_bytes_default(lwe_infos, a_infos)
    }

    fn glwe_expand_lwe<R, A>(module: &Module<BE>, res: &mut [R], a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos,
    {
        module.glwe_expand_lwe_default(res, a, scratch)
    }

    fn glwe_expand_lwe_matrix_tmp_bytes<R, A>(module: &Module<BE>, res_infos: &R, a_infos: &A) -> usize
    where
        R: LWEMatrixInfos,
        A: GLWEInfos,
    {
        module.glwe_expand_lwe_matrix_tmp_bytes_default(res_infos, a_infos)
    }

    fn glwe_expand_lwe_matrix<R, A>(module: &Module<BE>, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEMatrixToBackendMut<BE> + LWEMatrixInfos,
        A: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos,
    {
        module.glwe_expand_lwe_matrix_default(res, a, scratch)
    }

    fn ggsw_expand_rows_tmp_bytes<R, A>(module: &Module<BE>, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos,
    {
        module.ggsw_expand_rows_tmp_bytes_default(res_infos, tsk_infos)
    }

    fn ggsw_expand_row<R>(
        module: &Module<BE>,
        res: &mut R,
        tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos,
    {
        module.ggsw_expand_row_default(res, tsk, scratch)
    }
}

/// Implements [`ConversionDefault`] for `Module<$be>` by forwarding every method to
/// the corresponding [`conversion_defaults`] free function.
#[macro_export]
macro_rules! impl_conversion_defaults_full {
    ($be:ty) => {
        impl $crate::oep::ConversionDefault<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn lwe_sample_extract_default<R, A>(&self, res: &mut R, a: &A)
            where
                R: $crate::layouts::LWEToBackendMut<$be> + $crate::layouts::LWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be, State = ::poulpy_hal::layouts::Normalized> + $crate::layouts::GLWEInfos,
            {
                $crate::default::conversion::lwe_sample_extract_default::<$be, _, _, _>(self, res, a)
            }

            fn glwe_from_lwe_tmp_bytes_default<R, A, K>(&self, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::LWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::default::conversion::glwe_from_lwe_tmp_bytes_default::<$be, _, _, _, _>(
                    self, glwe_infos, lwe_infos, key_infos,
                )
            }

            fn glwe_from_lwe_default<R, A>(
                &self,
                res: &mut R,
                lwe: &A,
                ksk: &$crate::layouts::prepared::GGLWEPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::LWEToBackendRef<$be> + $crate::layouts::LWEInfos,
            {
                $crate::default::conversion::glwe_from_lwe_default::<$be, _, _, _>(self, res, lwe, ksk, scratch)
            }

            fn lwe_from_glwe_tmp_bytes_default<R, A, K>(&self, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
            where
                R: $crate::layouts::LWEInfos,
                A: $crate::layouts::GLWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::default::conversion::lwe_from_glwe_tmp_bytes_default::<$be, _, _, _, _>(
                    self, lwe_infos, glwe_infos, key_infos,
                )
            }

            fn lwe_from_glwe_default<R, A>(
                &self,
                res: &mut R,
                a: &A,
                a_idx: usize,
                key: &$crate::layouts::prepared::GGLWEPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::LWEToBackendMut<$be> + $crate::layouts::LWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be, State = ::poulpy_hal::layouts::Normalized> + $crate::layouts::GLWEInfos,
            {
                $crate::default::conversion::lwe_from_glwe_default::<$be, _, _, _>(self, res, a, a_idx, key, scratch)
            }

            fn ggsw_from_gglwe_tmp_bytes_default<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
            where
                R: $crate::layouts::GGSWInfos,
                A: $crate::layouts::GGLWEInfos,
            {
                $crate::default::conversion::ggsw_from_gglwe_tmp_bytes_default::<$be, _, _, _>(self, res_infos, tsk_infos)
            }

            fn ggsw_from_gglwe_default<R, A>(
                &self,
                res: &mut R,
                a: &A,
                tsk: &$crate::layouts::prepared::GGLWEToGGSWKeyPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GGSWToBackendMut<$be> + $crate::layouts::GGSWInfos,
                A: $crate::layouts::GGLWEToBackendRef<$be> + $crate::layouts::GGLWEInfos,
            {
                $crate::default::conversion::ggsw_from_gglwe_default::<$be, _, _, _>(self, res, a, tsk, scratch)
            }

            fn glwe_expand_lwe_tmp_bytes_default<R, A>(&self, lwe_infos: &R, a_infos: &A) -> usize
            where
                R: $crate::layouts::LWEInfos,
                A: $crate::layouts::GLWEInfos,
            {
                $crate::default::conversion::glwe_expand_lwe_tmp_bytes_default::<$be, _, _, _>(self, lwe_infos, a_infos)
            }

            fn glwe_expand_lwe_default<R, A>(
                &self,
                res: &mut [R],
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::LWEToBackendMut<$be> + $crate::layouts::LWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be, State = ::poulpy_hal::layouts::Normalized> + $crate::layouts::GLWEInfos,
            {
                $crate::default::conversion::glwe_expand_lwe_default::<$be, _, _, _>(self, res, a, scratch)
            }

            fn glwe_expand_lwe_matrix_tmp_bytes_default<R, A>(&self, res_infos: &R, a_infos: &A) -> usize
            where
                R: $crate::layouts::LWEMatrixInfos,
                A: $crate::layouts::GLWEInfos,
            {
                $crate::default::conversion::glwe_expand_lwe_matrix_tmp_bytes_default::<$be, _, _, _>(self, res_infos, a_infos)
            }

            fn glwe_expand_lwe_matrix_default<R, A>(
                &self,
                res: &mut R,
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::LWEMatrixToBackendMut<$be> + $crate::layouts::LWEMatrixInfos,
                A: $crate::layouts::GLWEToBackendRef<$be, State = ::poulpy_hal::layouts::Normalized> + $crate::layouts::GLWEInfos,
            {
                $crate::default::conversion::glwe_expand_lwe_matrix_default::<$be, _, _, _>(self, res, a, scratch)
            }

            fn ggsw_expand_rows_tmp_bytes_default<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
            where
                R: $crate::layouts::GGSWInfos,
                A: $crate::layouts::GGLWEInfos,
            {
                $crate::default::conversion::ggsw_expand_rows_tmp_bytes_default::<$be, _, _, _>(self, res_infos, tsk_infos)
            }

            fn ggsw_expand_row_default<R>(
                &self,
                res: &mut R,
                tsk: &$crate::layouts::prepared::GGLWEToGGSWKeyPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GGSWToBackendMut<$be> + $crate::layouts::GGSWInfos,
            {
                $crate::default::conversion::ggsw_expand_row_default::<$be, _, _>(self, res, tsk, scratch)
            }
        }
    };
}
