#![allow(clippy::too_many_arguments)]

use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::layouts::{
    GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, GLWEInfos, GLWEToBackendMut,
    GLWEToBackendRef, GetGaloisElement, SetGaloisElement,
    prepared::{GGLWEToGGSWKeyPreparedBackendRef, GLWEAutomorphismKeyPreparedBackendRef},
};

/// Backend hook for automorphism-family operations.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait AutomorphismImpl: Backend + crate::oep::GGSWConversionImpl {
    /// Scratch bound of [`Self::glwe_automorphism`], [`Self::glwe_automorphism_assign`],
    /// [`Self::glwe_automorphism_add`], [`Self::glwe_automorphism_add_assign`], [`Self::glwe_automorphism_sub`],
    /// [`Self::glwe_automorphism_sub_negate`], [`Self::glwe_automorphism_sub_assign`] and
    /// [`Self::glwe_automorphism_sub_negate_assign`]; may report less than the reference, never less than the body
    /// takes.
    fn glwe_automorphism_tmp_bytes<R, A, K>(module: &Module<Self>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    /// Reproduces [`glwe_automorphism_reference`](crate::reference::automorphism::glwe::glwe_automorphism_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_automorphism_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_automorphism<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos;

    /// Reproduces
    /// [`glwe_automorphism_assign_reference`](crate::reference::automorphism::glwe::glwe_automorphism_assign_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_automorphism_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_automorphism_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos;

    /// Reproduces
    /// [`glwe_automorphism_add_reference`](crate::reference::automorphism::glwe::glwe_automorphism_add_reference) bit
    /// for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_automorphism_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_automorphism_add<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos;

    /// Reproduces
    /// [`glwe_automorphism_add_assign_reference`](crate::reference::automorphism::glwe::glwe_automorphism_add_assign_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_automorphism_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_automorphism_add_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos;

    /// Reproduces
    /// [`glwe_automorphism_sub_reference`](crate::reference::automorphism::glwe::glwe_automorphism_sub_reference) bit
    /// for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_automorphism_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_automorphism_sub<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos;

    /// Reproduces
    /// [`glwe_automorphism_sub_negate_reference`](crate::reference::automorphism::glwe::glwe_automorphism_sub_negate_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_automorphism_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_automorphism_sub_negate<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos;

    /// Reproduces
    /// [`glwe_automorphism_sub_assign_reference`](crate::reference::automorphism::glwe::glwe_automorphism_sub_assign_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_automorphism_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_automorphism_sub_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos;

    /// Reproduces
    /// [`glwe_automorphism_sub_negate_assign_reference`](crate::reference::automorphism::glwe::glwe_automorphism_sub_negate_assign_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_automorphism_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_automorphism_sub_negate_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos;

    /// Scratch bound of [`Self::glwe_automorphism_many`] for keys of the layout `key_infos`; may report less than the
    /// reference, never less than the body takes.
    fn glwe_automorphism_many_tmp_bytes<R, A, K>(module: &Module<Self>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    /// Reproduces
    /// [`glwe_automorphism_many_reference`](crate::reference::automorphism::glwe::glwe_automorphism_many_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within the largest
    /// [`Self::glwe_automorphism_many_tmp_bytes`] over its keys; free in kernels and scratch use.
    fn glwe_automorphism_many<R, A>(
        module: &Module<Self>,
        res: &mut [&mut R],
        a: &A,
        keys: &[&GLWEAutomorphismKeyPreparedBackendRef<'_, Self>],
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos;

    /// Default body: `ggsw_automorphism_tmp_bytes_derived`, the larger of [`Self::glwe_automorphism_tmp_bytes`] and
    /// [`GGSWConversionImpl::ggsw_expand_rows_tmp_bytes`](crate::oep::GGSWConversionImpl::ggsw_expand_rows_tmp_bytes).
    fn ggsw_automorphism_tmp_bytes<R, A, K, T>(
        module: &Module<Self>,
        res_infos: &R,
        a_infos: &A,
        key_infos: &K,
        tsk_infos: &T,
    ) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        crate::oep::derived::automorphism::ggsw_automorphism_tmp_bytes_derived::<Self, _, _, _, _, _>(
            module, res_infos, a_infos, key_infos, tsk_infos,
        )
    }

    /// Default body: `ggsw_automorphism_derived`, [`Self::glwe_automorphism`] on the first column of each row, then
    /// [`GGSWConversionImpl::ggsw_expand_row`](crate::oep::GGSWConversionImpl::ggsw_expand_row), within
    /// [`Self::ggsw_automorphism_tmp_bytes`], under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint) of its
    /// prepared operands.
    fn ggsw_automorphism<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self> + GGSWInfos,
        A: GGSWToBackendRef<Self> + GGSWInfos,
    {
        crate::oep::derived::automorphism::ggsw_automorphism_derived::<Self, _, _, _>(module, res, a, key, tsk, scratch)
    }

    /// Default body: `ggsw_automorphism_assign_derived`, [`Self::glwe_automorphism_assign`] on the first column of
    /// each row, then [`GGSWConversionImpl::ggsw_expand_row`](crate::oep::GGSWConversionImpl::ggsw_expand_row),
    /// within [`Self::ggsw_automorphism_tmp_bytes`], under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint) of
    /// its prepared operands.
    fn ggsw_automorphism_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self> + GGSWInfos,
    {
        crate::oep::derived::automorphism::ggsw_automorphism_assign_derived::<Self, _, _>(module, res, key, tsk, scratch)
    }

    /// Scratch bound of [`Self::glwe_automorphism_key_automorphism`] and
    /// [`Self::glwe_automorphism_key_automorphism_assign`]; may report less than the reference, never less than the
    /// body takes.
    fn glwe_automorphism_key_automorphism_tmp_bytes<R, A, K>(
        module: &Module<Self>,
        res_infos: &R,
        a_infos: &A,
        key_infos: &K,
    ) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos;

    /// Reproduces
    /// [`glwe_automorphism_key_automorphism_reference`](crate::reference::automorphism::gglwe::glwe_automorphism_key_automorphism_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_automorphism_key_automorphism_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_automorphism_key_automorphism<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self> + SetGaloisElement + GGLWEInfos,
        A: GGLWEToBackendRef<Self> + GetGaloisElement + GGLWEInfos;

    /// Reproduces
    /// [`glwe_automorphism_key_automorphism_assign_reference`](crate::reference::automorphism::gglwe::glwe_automorphism_key_automorphism_assign_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_automorphism_key_automorphism_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_automorphism_key_automorphism_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self> + SetGaloisElement + GetGaloisElement + GGLWEInfos;
}

/// Selects the HAL automorphism algorithms and the derived GGSW defaults.
#[macro_export]
macro_rules! impl_automorphism_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::AutomorphismImpl for $be {
            fn glwe_automorphism_tmp_bytes<R, A, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res_infos: &R,
                a_infos: &A,
                key_infos: &K,
            ) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::reference::automorphism::glwe::glwe_automorphism_tmp_bytes_reference::<$be, _, _, _, _>(
                    module, res_infos, a_infos, key_infos,
                )
            }

            fn glwe_automorphism<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                a: &A,
                key: &$crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::automorphism::glwe::glwe_automorphism_reference::<$be, _, _, _>(module, res, a, key, scratch)
            }

            fn glwe_automorphism_assign<R>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                key: &$crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::automorphism::glwe::glwe_automorphism_assign_reference::<$be, _, _>(module, res, key, scratch)
            }

            fn glwe_automorphism_add<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                a: &A,
                key: &$crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::automorphism::glwe::glwe_automorphism_add_reference::<$be, _, _, _>(
                    module, res, a, key, scratch,
                )
            }

            fn glwe_automorphism_add_assign<R>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                key: &$crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::automorphism::glwe::glwe_automorphism_add_assign_reference::<$be, _, _>(
                    module, res, key, scratch,
                )
            }

            fn glwe_automorphism_sub<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                a: &A,
                key: &$crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::automorphism::glwe::glwe_automorphism_sub_reference::<$be, _, _, _>(
                    module, res, a, key, scratch,
                )
            }

            fn glwe_automorphism_sub_negate<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                a: &A,
                key: &$crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::automorphism::glwe::glwe_automorphism_sub_negate_reference::<$be, _, _, _>(
                    module, res, a, key, scratch,
                )
            }

            fn glwe_automorphism_sub_assign<R>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                key: &$crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::automorphism::glwe::glwe_automorphism_sub_assign_reference::<$be, _, _>(
                    module, res, key, scratch,
                )
            }

            fn glwe_automorphism_sub_negate_assign<R>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                key: &$crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::automorphism::glwe::glwe_automorphism_sub_negate_assign_reference::<$be, _, _>(
                    module, res, key, scratch,
                )
            }

            fn glwe_automorphism_many_tmp_bytes<R, A, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res_infos: &R,
                a_infos: &A,
                key_infos: &K,
            ) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::reference::automorphism::glwe::glwe_automorphism_many_tmp_bytes_reference::<$be, _, _, _>(
                    module, res_infos, a_infos, key_infos,
                )
            }

            fn glwe_automorphism_many<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut [&mut R],
                a: &A,
                keys: &[&$crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, $be>],
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                $crate::reference::automorphism::glwe::glwe_automorphism_many_reference::<$be, _, _>(
                    module, res, a, keys, scratch,
                )
            }

            fn glwe_automorphism_key_automorphism_tmp_bytes<R, A, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res_infos: &R,
                a_infos: &A,
                key_infos: &K,
            ) -> usize
            where
                R: $crate::layouts::GGLWEInfos,
                A: $crate::layouts::GGLWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::reference::automorphism::gglwe::glwe_automorphism_key_automorphism_tmp_bytes_reference::<$be, _, _, _, _>(
                    module, res_infos, a_infos, key_infos,
                )
            }

            fn glwe_automorphism_key_automorphism<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                a: &A,
                key: &$crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::SetGaloisElement + $crate::layouts::GGLWEInfos,
                A: $crate::layouts::GGLWEToBackendRef<$be> + $crate::layouts::GetGaloisElement + $crate::layouts::GGLWEInfos,
            {
                $crate::reference::automorphism::gglwe::glwe_automorphism_key_automorphism_reference::<$be, _, _, _>(
                    module, res, a, key, scratch,
                )
            }

            fn glwe_automorphism_key_automorphism_assign<R>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                key: &$crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GGLWEToBackendMut<$be>
                    + $crate::layouts::SetGaloisElement
                    + $crate::layouts::GetGaloisElement
                    + $crate::layouts::GGLWEInfos,
            {
                $crate::reference::automorphism::gglwe::glwe_automorphism_key_automorphism_assign_reference::<$be, _, _>(
                    module, res, key, scratch,
                )
            }
        }
    };
}
