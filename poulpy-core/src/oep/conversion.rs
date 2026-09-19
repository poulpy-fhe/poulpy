use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::layouts::{
    GGLWEInfos, GGSWInfos, GGSWToBackendMut, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, LWEMatrixInfos,
    LWEMatrixToBackendMut, LWEToBackendMut, LWEToBackendRef,
    prepared::{GGLWEPreparedBackendRef, GGLWEToGGSWKeyPreparedBackendRef},
};

/// Backend-provided ciphertext conversion operations.
///
/// # Safety
/// Implementations must only read and write the regions described by the provided layouts, respect
/// scratch-space requirements, and produce results equivalent to the documented conversion
/// semantics for the backend.
pub unsafe trait ConversionImpl: Backend {
    fn lwe_sample_extract<R, A>(module: &Module<Self>, res: &mut R, a: &A)
    where
        R: LWEToBackendMut<Self> + LWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos;

    fn glwe_from_lwe_tmp_bytes<R, A, K>(module: &Module<Self>, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn glwe_from_lwe<R, A>(
        module: &Module<Self>,
        res: &mut R,
        lwe: &A,
        ksk: &GGLWEPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: LWEToBackendRef<Self> + LWEInfos;

    fn lwe_from_glwe_tmp_bytes<R, A, K>(module: &Module<Self>, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn lwe_from_glwe<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        a_idx: usize,
        key: &GGLWEPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: LWEToBackendMut<Self> + LWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos;

    fn glwe_expand_lwe_tmp_bytes<R, A>(module: &Module<Self>, lwe_infos: &R, a_infos: &A) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos;

    fn glwe_expand_lwe<R, A>(module: &Module<Self>, res: &mut [R], a: &A, scratch: &mut ScratchArena<'_, Self>)
    where
        R: LWEToBackendMut<Self> + LWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos;

    fn glwe_expand_lwe_matrix_tmp_bytes<R, A>(module: &Module<Self>, res_infos: &R, a_infos: &A) -> usize
    where
        R: LWEMatrixInfos,
        A: GLWEInfos;

    fn glwe_expand_lwe_matrix<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, Self>)
    where
        R: LWEMatrixToBackendMut<Self> + LWEMatrixInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos;

    fn ggsw_expand_rows_tmp_bytes<R, A>(module: &Module<Self>, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_expand_row<R>(
        module: &Module<Self>,
        res: &mut R,
        tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self> + GGSWInfos;
}

/// Override surface for the conversion family.
///
/// Abstract: no HAL supertraits, no default method bodies. See [`conversion_reference`]
/// for reference algorithms a backend may forward to.
pub trait ConversionReference<BE: Backend> {
    fn lwe_sample_extract_reference<R, A>(&self, res: &mut R, a: &A)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_from_lwe_tmp_bytes_reference<R, A, K>(&self, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn glwe_from_lwe_reference<R, A>(
        &self,
        res: &mut R,
        lwe: &A,
        ksk: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos;

    fn lwe_from_glwe_tmp_bytes_reference<R, A, K>(&self, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn lwe_from_glwe_reference<R, A>(
        &self,
        res: &mut R,
        a: &A,
        a_idx: usize,
        key: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_expand_lwe_tmp_bytes_reference<R, A>(&self, lwe_infos: &R, a_infos: &A) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos;

    fn glwe_expand_lwe_reference<R, A>(&self, res: &mut [R], a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_expand_lwe_matrix_tmp_bytes_reference<R, A>(&self, res_infos: &R, a_infos: &A) -> usize
    where
        R: LWEMatrixInfos,
        A: GLWEInfos;

    fn glwe_expand_lwe_matrix_reference<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEMatrixToBackendMut<BE> + LWEMatrixInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn ggsw_expand_rows_tmp_bytes_reference<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_expand_row_reference<R>(
        &self,
        res: &mut R,
        tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos;
}

unsafe impl<BE: Backend> ConversionImpl for BE
where
    Module<BE>: ConversionReference<BE>,
{
    fn lwe_sample_extract<R, A>(module: &Module<BE>, res: &mut R, a: &A)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        module.lwe_sample_extract_reference(res, a)
    }

    fn glwe_from_lwe_tmp_bytes<R, A, K>(module: &Module<BE>, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
    {
        module.glwe_from_lwe_tmp_bytes_reference(glwe_infos, lwe_infos, key_infos)
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
        module.glwe_from_lwe_reference(res, lwe, ksk, scratch)
    }

    fn lwe_from_glwe_tmp_bytes<R, A, K>(module: &Module<BE>, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        module.lwe_from_glwe_tmp_bytes_reference(lwe_infos, glwe_infos, key_infos)
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
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        module.lwe_from_glwe_reference(res, a, a_idx, key, scratch)
    }

    fn glwe_expand_lwe_tmp_bytes<R, A>(module: &Module<BE>, lwe_infos: &R, a_infos: &A) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos,
    {
        module.glwe_expand_lwe_tmp_bytes_reference(lwe_infos, a_infos)
    }

    fn glwe_expand_lwe<R, A>(module: &Module<BE>, res: &mut [R], a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        module.glwe_expand_lwe_reference(res, a, scratch)
    }

    fn glwe_expand_lwe_matrix_tmp_bytes<R, A>(module: &Module<BE>, res_infos: &R, a_infos: &A) -> usize
    where
        R: LWEMatrixInfos,
        A: GLWEInfos,
    {
        module.glwe_expand_lwe_matrix_tmp_bytes_reference(res_infos, a_infos)
    }

    fn glwe_expand_lwe_matrix<R, A>(module: &Module<BE>, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEMatrixToBackendMut<BE> + LWEMatrixInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        module.glwe_expand_lwe_matrix_reference(res, a, scratch)
    }

    fn ggsw_expand_rows_tmp_bytes<R, A>(module: &Module<BE>, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos,
    {
        module.ggsw_expand_rows_tmp_bytes_reference(res_infos, tsk_infos)
    }

    fn ggsw_expand_row<R>(
        module: &Module<BE>,
        res: &mut R,
        tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos,
    {
        module.ggsw_expand_row_reference(res, tsk, scratch)
    }
}

/// Implements [`ConversionReference`] for `Module<$be>` by forwarding every method to
/// the corresponding [`conversion_reference`] free function.
#[macro_export]
macro_rules! impl_conversion_reference_full {
    ($be:ty) => {
        impl $crate::oep::ConversionReference<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn lwe_sample_extract_reference<R, A>(&self, res: &mut R, a: &A)
            where
                R: $crate::layouts::LWEToBackendMut<$be> + $crate::layouts::LWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::conversion::lwe_sample_extract_reference::<$be, _, _, _>(self, res, a)
            }

            fn glwe_from_lwe_tmp_bytes_reference<R, A, K>(&self, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::LWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::reference::conversion::glwe_from_lwe_tmp_bytes_reference::<$be, _, _, _, _>(
                    self, glwe_infos, lwe_infos, key_infos,
                )
            }

            fn glwe_from_lwe_reference<R, A>(
                &self,
                res: &mut R,
                lwe: &A,
                ksk: &$crate::layouts::prepared::GGLWEPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::LWEToBackendRef<$be> + $crate::layouts::LWEInfos,
            {
                $crate::reference::conversion::glwe_from_lwe_reference::<$be, _, _, _>(self, res, lwe, ksk, scratch)
            }

            fn lwe_from_glwe_tmp_bytes_reference<R, A, K>(&self, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
            where
                R: $crate::layouts::LWEInfos,
                A: $crate::layouts::GLWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::reference::conversion::lwe_from_glwe_tmp_bytes_reference::<$be, _, _, _, _>(
                    self, lwe_infos, glwe_infos, key_infos,
                )
            }

            fn lwe_from_glwe_reference<R, A>(
                &self,
                res: &mut R,
                a: &A,
                a_idx: usize,
                key: &$crate::layouts::prepared::GGLWEPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::LWEToBackendMut<$be> + $crate::layouts::LWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::conversion::lwe_from_glwe_reference::<$be, _, _, _>(self, res, a, a_idx, key, scratch)
            }

            fn glwe_expand_lwe_tmp_bytes_reference<R, A>(&self, lwe_infos: &R, a_infos: &A) -> usize
            where
                R: $crate::layouts::LWEInfos,
                A: $crate::layouts::GLWEInfos,
            {
                $crate::reference::conversion::glwe_expand_lwe_tmp_bytes_reference::<$be, _, _, _>(self, lwe_infos, a_infos)
            }

            fn glwe_expand_lwe_reference<R, A>(
                &self,
                res: &mut [R],
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::LWEToBackendMut<$be> + $crate::layouts::LWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::conversion::glwe_expand_lwe_reference::<$be, _, _, _>(self, res, a, scratch)
            }

            fn glwe_expand_lwe_matrix_tmp_bytes_reference<R, A>(&self, res_infos: &R, a_infos: &A) -> usize
            where
                R: $crate::layouts::LWEMatrixInfos,
                A: $crate::layouts::GLWEInfos,
            {
                $crate::reference::conversion::glwe_expand_lwe_matrix_tmp_bytes_reference::<$be, _, _, _>(
                    self, res_infos, a_infos,
                )
            }

            fn glwe_expand_lwe_matrix_reference<R, A>(
                &self,
                res: &mut R,
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::LWEMatrixToBackendMut<$be> + $crate::layouts::LWEMatrixInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::conversion::glwe_expand_lwe_matrix_reference::<$be, _, _, _>(self, res, a, scratch)
            }

            fn ggsw_expand_rows_tmp_bytes_reference<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
            where
                R: $crate::layouts::GGSWInfos,
                A: $crate::layouts::GGLWEInfos,
            {
                $crate::reference::conversion::ggsw_expand_rows_tmp_bytes_reference::<$be, _, _, _>(self, res_infos, tsk_infos)
            }

            fn ggsw_expand_row_reference<R>(
                &self,
                res: &mut R,
                tsk: &$crate::layouts::prepared::GGLWEToGGSWKeyPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GGSWToBackendMut<$be> + $crate::layouts::GGSWInfos,
            {
                $crate::reference::conversion::ggsw_expand_row_reference::<$be, _, _>(self, res, tsk, scratch)
            }
        }
    };
}
