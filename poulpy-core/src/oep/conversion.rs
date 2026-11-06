use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::layouts::{
    GGLWEInfos, GGLWEToBackendRef, GGSWInfos, GGSWToBackendMut, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
    LWEMatrixInfos, LWEMatrixToBackendMut, LWEToBackendMut, LWEToBackendRef,
    prepared::{GGLWEPreparedBackendRef, GGLWEToGGSWKeyPreparedBackendRef},
};

/// Backend-provided ciphertext conversion operations, on the standard ring only.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait ConversionImpl: Backend + crate::oep::GLWEKeyswitchImpl + crate::oep::GLWERotateImpl {
    /// Reproduces [`lwe_sample_extract_reference`](crate::reference::conversion::lwe_sample_extract_reference) bit
    /// for bit; free in kernels.
    fn lwe_sample_extract<R, A>(module: &Module<Self>, res: &mut R, a: &A)
    where
        R: LWEToBackendMut<Self> + LWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos;

    /// Scratch bound of [`Self::glwe_from_lwe`]; may report less than the reference, never less than the body takes.
    fn glwe_from_lwe_tmp_bytes<R, A, K>(module: &Module<Self>, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    /// Reproduces [`glwe_from_lwe_reference`](crate::reference::conversion::glwe_from_lwe_reference) bit for bit
    /// under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within [`Self::glwe_from_lwe_tmp_bytes`]; free in
    /// kernels and scratch use.
    fn glwe_from_lwe<R, A>(
        module: &Module<Self>,
        res: &mut R,
        lwe: &A,
        ksk: &GGLWEPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: LWEToBackendRef<Self> + LWEInfos;

    /// Default body: `lwe_from_glwe_tmp_bytes_derived`, a rank-one GLWE temporary plus the larger of
    /// [`GLWEKeyswitchImpl::glwe_keyswitch_tmp_bytes`](crate::oep::GLWEKeyswitchImpl::glwe_keyswitch_tmp_bytes) and
    /// [`GLWERotateImpl::glwe_rotate_tmp_bytes`](crate::oep::GLWERotateImpl::glwe_rotate_tmp_bytes).
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

    /// Default body: `lwe_from_glwe_derived`,
    /// [`GLWEKeyswitchImpl::glwe_keyswitch`](crate::oep::GLWEKeyswitchImpl::glwe_keyswitch) of `a` into a rank-one
    /// GLWE, [`GLWERotateImpl::glwe_rotate_assign`](crate::oep::GLWERotateImpl::glwe_rotate_assign) by `-a_idx` when
    /// `a_idx` is not zero, then [`Self::lwe_sample_extract`], within [`Self::lwe_from_glwe_tmp_bytes`], under any
    /// [`PrepareHint`](poulpy_hal::layouts::PrepareHint) of its prepared operands.
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

    /// Scratch bound of [`Self::glwe_expand_lwe`]; may report less than the reference, never less than the body
    /// takes.
    fn glwe_expand_lwe_tmp_bytes<R, A>(module: &Module<Self>, lwe_infos: &R, a_infos: &A) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos;

    /// Reproduces [`glwe_expand_lwe_reference`](crate::reference::conversion::glwe_expand_lwe_reference) bit for bit,
    /// within [`Self::glwe_expand_lwe_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_expand_lwe<R, A>(module: &Module<Self>, res: &mut [R], a: &A, scratch: &mut ScratchArena<'_, Self>)
    where
        R: LWEToBackendMut<Self> + LWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos;

    /// Scratch bound of [`Self::glwe_expand_lwe_matrix`]; may report less than the reference, never less than the
    /// body takes.
    fn glwe_expand_lwe_matrix_tmp_bytes<R, A>(module: &Module<Self>, res_infos: &R, a_infos: &A) -> usize
    where
        R: LWEMatrixInfos,
        A: GLWEInfos;

    /// Reproduces
    /// [`glwe_expand_lwe_matrix_reference`](crate::reference::conversion::glwe_expand_lwe_matrix_reference) bit for
    /// bit, within [`Self::glwe_expand_lwe_matrix_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_expand_lwe_matrix<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, Self>)
    where
        R: LWEMatrixToBackendMut<Self> + LWEMatrixInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos;
}

/// Backend-provided GGSW conversions (ring independent).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GGSWConversionImpl: Backend + crate::oep::GLWECopyImpl {
    /// Scratch bound of [`Self::ggsw_expand_rows_assign`]; may report less than the reference, never less than the body
    /// takes.
    fn ggsw_expand_rows_assign_tmp_bytes<R, A>(module: &Module<Self>, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    /// Reproduces [`ggsw_expand_rows_assign_reference`](crate::reference::conversion::ggsw_expand_rows_assign_reference) bit for bit
    /// under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within [`Self::ggsw_expand_rows_assign_tmp_bytes`]; free
    /// in kernels and scratch use.
    fn ggsw_expand_rows_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self> + GGSWInfos;

    /// Default body: `ggsw_from_gglwe_tmp_bytes_derived`, the larger of [`Self::ggsw_expand_rows_assign_tmp_bytes`] and
    /// [`GLWECopyImpl::glwe_copy_tmp_bytes`](crate::oep::GLWECopyImpl::glwe_copy_tmp_bytes).
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

    /// Default body: `ggsw_from_gglwe_derived`, [`GLWECopyImpl::glwe_copy`](crate::oep::GLWECopyImpl::glwe_copy) of
    /// the first column of each row of `a`, then [`Self::ggsw_expand_rows_assign`], within
    /// [`Self::ggsw_from_gglwe_tmp_bytes`], under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint) of its
    /// prepared operands.
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
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::conversion::ConversionReference<$be>>::lwe_sample_extract_reference(module, res, a)
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
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::conversion::ConversionReference<$be>>::glwe_from_lwe_tmp_bytes_reference(
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
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::conversion::ConversionReference<$be>>::glwe_from_lwe_reference(module, res, lwe, ksk, scratch)
            }

            fn glwe_expand_lwe_tmp_bytes<R, A>(module: &::poulpy_hal::layouts::Module<$be>, lwe_infos: &R, a_infos: &A) -> usize
            where
                R: $crate::layouts::LWEInfos,
                A: $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::conversion::ConversionReference<$be>>::glwe_expand_lwe_tmp_bytes_reference(module, lwe_infos, a_infos)
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
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::conversion::ConversionReference<$be>>::glwe_expand_lwe_reference(module, res, a, scratch)
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
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::conversion::ConversionReference<$be>>::glwe_expand_lwe_matrix_tmp_bytes_reference(
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
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::conversion::ConversionReference<$be>>::glwe_expand_lwe_matrix_reference(module, res, a, scratch)
            }
        }
    };
}

/// Selects the portable GGSW conversion algorithms.
#[macro_export]
macro_rules! impl_ggsw_conversion_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::GGSWConversionImpl for $be {
            fn ggsw_expand_rows_assign_tmp_bytes<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res_infos: &R,
                tsk_infos: &A,
            ) -> usize
            where
                R: $crate::layouts::GGSWInfos,
                A: $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::conversion::GGSWConversionReference<$be>>::ggsw_expand_rows_assign_tmp_bytes_reference(
                    module, res_infos, tsk_infos,
                )
            }

            fn ggsw_expand_rows_assign<R>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                tsk: &$crate::layouts::prepared::GGLWEToGGSWKeyPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GGSWToBackendMut<$be> + $crate::layouts::GGSWInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::conversion::GGSWConversionReference<$be>>::ggsw_expand_rows_assign_reference(module, res, tsk, scratch)
            }
        }
    };
}
