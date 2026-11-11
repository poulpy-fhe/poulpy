#![allow(clippy::too_many_arguments)]

use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena, Standard};

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
pub unsafe trait AutomorphismImpl<Rg: Ring = Standard>: Backend + crate::oep::GGSWConversionImpl<Rg> {
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
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self, Rg> + GLWEInfos,
        A: GLWEToBackendRef<Self, Rg> + GLWEInfos;

    /// Reproduces
    /// [`glwe_automorphism_assign_reference`](crate::reference::automorphism::glwe::glwe_automorphism_assign_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_automorphism_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_automorphism_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self, Rg> + GLWEInfos;

    /// Reproduces
    /// [`glwe_automorphism_add_reference`](crate::reference::automorphism::glwe::glwe_automorphism_add_reference) bit
    /// for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_automorphism_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_automorphism_add<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self, Rg> + GLWEInfos,
        A: GLWEToBackendRef<Self, Rg> + GLWEInfos;

    /// Reproduces
    /// [`glwe_automorphism_add_assign_reference`](crate::reference::automorphism::glwe::glwe_automorphism_add_assign_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_automorphism_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_automorphism_add_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self, Rg> + GLWEInfos;

    /// Reproduces
    /// [`glwe_automorphism_sub_reference`](crate::reference::automorphism::glwe::glwe_automorphism_sub_reference) bit
    /// for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_automorphism_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_automorphism_sub<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self, Rg> + GLWEInfos,
        A: GLWEToBackendRef<Self, Rg> + GLWEInfos;

    /// Reproduces
    /// [`glwe_automorphism_sub_negate_reference`](crate::reference::automorphism::glwe::glwe_automorphism_sub_negate_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_automorphism_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_automorphism_sub_negate<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self, Rg> + GLWEInfos,
        A: GLWEToBackendRef<Self, Rg> + GLWEInfos;

    /// Reproduces
    /// [`glwe_automorphism_sub_assign_reference`](crate::reference::automorphism::glwe::glwe_automorphism_sub_assign_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_automorphism_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_automorphism_sub_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self, Rg> + GLWEInfos;

    /// Reproduces
    /// [`glwe_automorphism_sub_negate_assign_reference`](crate::reference::automorphism::glwe::glwe_automorphism_sub_negate_assign_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_automorphism_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_automorphism_sub_negate_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self, Rg> + GLWEInfos;

    /// Scratch bound of [`Self::glwe_automorphism_many`] for keys of the layout `key_infos`; may report less than the
    /// reference, never less than the body takes.
    fn glwe_automorphism_many_tmp_bytes<R, A, K>(module: &Module<Self>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    /// Reproduces
    /// [`glwe_automorphism_many_reference`](crate::reference::automorphism::GLWEAutomorphismReference::glwe_automorphism_many_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within the largest
    /// [`Self::glwe_automorphism_many_tmp_bytes`] over its keys; free in kernels and scratch use.
    fn glwe_automorphism_many<R, A>(
        module: &Module<Self>,
        res: &mut [&mut R],
        a: &A,
        keys: &[&GLWEAutomorphismKeyPreparedBackendRef<'_, Self, Rg>],
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self, Rg> + GLWEInfos,
        A: GLWEToBackendRef<Self, Rg> + GLWEInfos;

    /// Default body: `ggsw_automorphism_tmp_bytes_derived`, the larger of [`Self::glwe_automorphism_tmp_bytes`] and
    /// [`GGSWConversionImpl::ggsw_expand_rows_assign_tmp_bytes`](crate::oep::GGSWConversionImpl::ggsw_expand_rows_assign_tmp_bytes).
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
        crate::oep::derived::automorphism::ggsw_automorphism_tmp_bytes_derived::<Self, _, _, _, _, _, Rg>(
            module, res_infos, a_infos, key_infos, tsk_infos,
        )
    }

    /// Default body: `ggsw_automorphism_derived`, [`Self::glwe_automorphism`] on the first column of each row, then
    /// [`GGSWConversionImpl::ggsw_expand_rows_assign`](crate::oep::GGSWConversionImpl::ggsw_expand_rows_assign), within
    /// [`Self::ggsw_automorphism_tmp_bytes`], under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint) of its
    /// prepared operands.
    fn ggsw_automorphism<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self, Rg>,
        tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self, Rg> + GGSWInfos,
        A: GGSWToBackendRef<Self, Rg> + GGSWInfos,
    {
        crate::oep::derived::automorphism::ggsw_automorphism_derived::<Self, _, _, _, Rg>(module, res, a, key, tsk, scratch)
    }

    /// Default body: `ggsw_automorphism_assign_derived`, [`Self::glwe_automorphism_assign`] on the first column of
    /// each row, then [`GGSWConversionImpl::ggsw_expand_rows_assign`](crate::oep::GGSWConversionImpl::ggsw_expand_rows_assign),
    /// within [`Self::ggsw_automorphism_tmp_bytes`], under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint) of
    /// its prepared operands.
    fn ggsw_automorphism_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self, Rg>,
        tsk: &GGLWEToGGSWKeyPreparedBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self, Rg> + GGSWInfos,
    {
        crate::oep::derived::automorphism::ggsw_automorphism_assign_derived::<Self, _, _, Rg>(module, res, key, tsk, scratch)
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
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self, Rg> + SetGaloisElement + GGLWEInfos,
        A: GGLWEToBackendRef<Self, Rg> + GetGaloisElement + GGLWEInfos;

    /// Reproduces
    /// [`glwe_automorphism_key_automorphism_assign_reference`](crate::reference::automorphism::gglwe::glwe_automorphism_key_automorphism_assign_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_automorphism_key_automorphism_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_automorphism_key_automorphism_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self, Rg>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self, Rg> + SetGaloisElement + GetGaloisElement + GGLWEInfos;
}

/// Selects the HAL automorphism algorithms and the derived GGSW defaults.
///
/// `($be)` implements both rings and `($be, $ring)` one ring, as `docs/guides/layering-contract.md` describes.
#[macro_export]
macro_rules! impl_automorphism_reference_full {
    ($be:ty) => {
        $crate::impl_automorphism_reference_full!($be, ::poulpy_hal::layouts::Standard);
        $crate::impl_automorphism_reference_full!($be, ::poulpy_hal::layouts::ConjugateInvariant);
    };
    ($be:ty, $ring:ty) => {
        unsafe impl $crate::oep::AutomorphismImpl<$ring> for $be {
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
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::automorphism::GLWEAutomorphismReference<$be, $ring>>::glwe_automorphism_tmp_bytes_reference(
                    module, res_infos, a_infos, key_infos,
                )
            }

            fn glwe_automorphism<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                a: &A,
                key: &$crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, $be, $ring>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be, $ring> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be, $ring> + $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::automorphism::GLWEAutomorphismReference<$be, $ring>>::glwe_automorphism_reference(module, res, a, key, scratch)
            }

            fn glwe_automorphism_assign<R>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                key: &$crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, $be, $ring>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be, $ring> + $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::automorphism::GLWEAutomorphismReference<$be, $ring>>::glwe_automorphism_assign_reference(module, res, key, scratch)
            }

            fn glwe_automorphism_add<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                a: &A,
                key: &$crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, $be, $ring>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be, $ring> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be, $ring> + $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::automorphism::GLWEAutomorphismReference<$be, $ring>>::glwe_automorphism_add_reference(
                    module, res, a, key, scratch,
                )
            }

            fn glwe_automorphism_add_assign<R>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                key: &$crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, $be, $ring>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be, $ring> + $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::automorphism::GLWEAutomorphismReference<$be, $ring>>::glwe_automorphism_add_assign_reference(
                    module, res, key, scratch,
                )
            }

            fn glwe_automorphism_sub<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                a: &A,
                key: &$crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, $be, $ring>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be, $ring> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be, $ring> + $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::automorphism::GLWEAutomorphismReference<$be, $ring>>::glwe_automorphism_sub_reference(
                    module, res, a, key, scratch,
                )
            }

            fn glwe_automorphism_sub_negate<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                a: &A,
                key: &$crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, $be, $ring>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be, $ring> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be, $ring> + $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::automorphism::GLWEAutomorphismReference<$be, $ring>>::glwe_automorphism_sub_negate_reference(
                    module, res, a, key, scratch,
                )
            }

            fn glwe_automorphism_sub_assign<R>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                key: &$crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, $be, $ring>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be, $ring> + $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::automorphism::GLWEAutomorphismReference<$be, $ring>>::glwe_automorphism_sub_assign_reference(
                    module, res, key, scratch,
                )
            }

            fn glwe_automorphism_sub_negate_assign<R>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                key: &$crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, $be, $ring>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be, $ring> + $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::automorphism::GLWEAutomorphismReference<$be, $ring>>::glwe_automorphism_sub_negate_assign_reference(
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
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::automorphism::GLWEAutomorphismReference<$be, $ring>>::glwe_automorphism_many_tmp_bytes_reference(
                    module, res_infos, a_infos, key_infos,
                )
            }

            fn glwe_automorphism_many<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut [&mut R],
                a: &A,
                keys: &[&$crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, $be, $ring>],
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be, $ring> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be, $ring> + $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::automorphism::GLWEAutomorphismReference<$be, $ring>>::glwe_automorphism_many_reference(
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
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::automorphism::GLWEAutomorphismKeyAutomorphismReference<$be, $ring>>::glwe_automorphism_key_automorphism_tmp_bytes_reference(
                    module, res_infos, a_infos, key_infos,
                )
            }

            fn glwe_automorphism_key_automorphism<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                a: &A,
                key: &$crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, $be, $ring>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GGLWEToBackendMut<$be, $ring>
                    + $crate::layouts::SetGaloisElement
                    + $crate::layouts::GGLWEInfos,
                A: $crate::layouts::GGLWEToBackendRef<$be, $ring>
                    + $crate::layouts::GetGaloisElement
                    + $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::automorphism::GLWEAutomorphismKeyAutomorphismReference<$be, $ring>>::glwe_automorphism_key_automorphism_reference(
                    module, res, a, key, scratch,
                )
            }

            fn glwe_automorphism_key_automorphism_assign<R>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                key: &$crate::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, $be, $ring>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GGLWEToBackendMut<$be, $ring>
                    + $crate::layouts::SetGaloisElement
                    + $crate::layouts::GetGaloisElement
                    + $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::automorphism::GLWEAutomorphismKeyAutomorphismReference<$be, $ring>>::glwe_automorphism_key_automorphism_assign_reference(
                    module, res, key, scratch,
                )
            }
        }
    };
}
