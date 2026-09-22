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
/// Implementors must preserve the semantics, scratch requirements, aliasing
/// guarantees, and backend bit-parity contract expected by end-to-end pipelines.
pub unsafe trait AutomorphismImpl: Backend + crate::oep::ConversionImpl {
    fn glwe_automorphism_tmp_bytes<R, A, K>(module: &Module<Self>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_automorphism<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos;

    fn glwe_automorphism_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos;

    fn glwe_automorphism_add<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos;

    fn glwe_automorphism_add_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos;

    fn glwe_automorphism_sub<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos;

    fn glwe_automorphism_sub_negate<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos;

    fn glwe_automorphism_sub_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos;

    fn glwe_automorphism_sub_negate_assign<R>(
        module: &Module<Self>,
        res: &mut R,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos;

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

    fn glwe_automorphism_key_automorphism<R, A>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self> + SetGaloisElement + GGLWEInfos,
        A: GGLWEToBackendRef<Self> + GetGaloisElement + GGLWEInfos;

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

// Reference helpers remain available through OEP for source compatibility.
pub use crate::reference::automorphism::{GGLWEAutomorphismReference, GLWEAutomorphismReference};
