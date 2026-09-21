use poulpy_hal::layouts::{Backend, Module, ScratchArena};

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
pub unsafe trait ConversionImpl:
    Backend + crate::oep::GLWECopyImpl + crate::oep::GLWEKeyswitchImpl + crate::oep::GLWERotateImpl
{
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
        K: GGLWEInfos,
    {
        crate::oep::derived::conversion::lwe_from_glwe_tmp_bytes_derived::<Self, _, _, _, _>(
            module, lwe_infos, glwe_infos, key_infos,
        )
    }

    fn lwe_from_glwe<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        a_idx: usize,
        key: &GGLWEPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: LWEToBackendMut<Self> + LWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos,
    {
        crate::oep::derived::conversion::lwe_from_glwe_derived::<Self, _, _, _>(module, res, a, a_idx, key, scratch)
    }

    fn ggsw_from_gglwe_tmp_bytes<R, A, T>(module: &Module<Self>, res_infos: &R, a_infos: &A, tsk_infos: &T) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos,
        T: GGLWEInfos,
    {
        crate::oep::derived::conversion::ggsw_from_gglwe_tmp_bytes_derived::<Self, _, _, _, _>(
            module, res_infos, a_infos, tsk_infos,
        )
    }

    fn ggsw_from_gglwe<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self> + GGSWInfos,
        A: GGLWEToBackendRef<Self> + GGLWEInfos,
    {
        crate::oep::derived::conversion::ggsw_from_gglwe_derived::<Self, _, _, _>(module, res, a, tsk, scratch)
    }

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

/// Selects the portable HAL algorithms for this backend operation family.
#[macro_export]
macro_rules! impl_conversion_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::ConversionImpl for $be {
            fn lwe_sample_extract<R, A>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, a: &A)
            where
                R: $crate::layouts::LWEToBackendMut<$be> + $crate::layouts::LWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::conversion::lwe_sample_extract_reference::<$be, _, _, _>(module, res, a)
            }

            fn glwe_from_lwe_tmp_bytes<R, A, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                glwe_infos: &R,
                lwe_infos: &A,
                key_infos: &K,
            ) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::LWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::reference::conversion::glwe_from_lwe_tmp_bytes_reference::<$be, _, _, _, _>(
                    module, glwe_infos, lwe_infos, key_infos,
                )
            }

            fn glwe_from_lwe<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                lwe: &A,
                ksk: &$crate::layouts::prepared::GGLWEPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::LWEToBackendRef<$be> + $crate::layouts::LWEInfos,
            {
                $crate::reference::conversion::glwe_from_lwe_reference::<$be, _, _, _>(module, res, lwe, ksk, scratch)
            }

            fn glwe_expand_lwe_tmp_bytes<R, A>(module: &::poulpy_hal::layouts::Module<$be>, lwe_infos: &R, a_infos: &A) -> usize
            where
                R: $crate::layouts::LWEInfos,
                A: $crate::layouts::GLWEInfos,
            {
                $crate::reference::conversion::glwe_expand_lwe_tmp_bytes_reference::<$be, _, _, _>(module, lwe_infos, a_infos)
            }

            fn glwe_expand_lwe<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut [R],
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::LWEToBackendMut<$be> + $crate::layouts::LWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::conversion::glwe_expand_lwe_reference::<$be, _, _, _>(module, res, a, scratch)
            }

            fn glwe_expand_lwe_matrix_tmp_bytes<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res_infos: &R,
                a_infos: &A,
            ) -> usize
            where
                R: $crate::layouts::LWEMatrixInfos,
                A: $crate::layouts::GLWEInfos,
            {
                $crate::reference::conversion::glwe_expand_lwe_matrix_tmp_bytes_reference::<$be, _, _, _>(
                    module, res_infos, a_infos,
                )
            }

            fn glwe_expand_lwe_matrix<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::LWEMatrixToBackendMut<$be> + $crate::layouts::LWEMatrixInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::conversion::glwe_expand_lwe_matrix_reference::<$be, _, _, _>(module, res, a, scratch)
            }

            fn ggsw_expand_rows_tmp_bytes<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res_infos: &R,
                tsk_infos: &A,
            ) -> usize
            where
                R: $crate::layouts::GGSWInfos,
                A: $crate::layouts::GGLWEInfos,
            {
                $crate::reference::conversion::ggsw_expand_rows_tmp_bytes_reference::<$be, _, _, _>(module, res_infos, tsk_infos)
            }

            fn ggsw_expand_row<R>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                tsk: &$crate::layouts::prepared::GGLWEToGGSWKeyPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GGSWToBackendMut<$be> + $crate::layouts::GGSWInfos,
            {
                $crate::reference::conversion::ggsw_expand_row_reference::<$be, _, _>(module, res, tsk, scratch)
            }
        }
    };
}

// Reference helpers remain available through OEP for source compatibility.
pub use crate::reference::conversion::ConversionReference;
