#![allow(clippy::too_many_arguments)]

use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::layouts::{
    GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, GLWEInfos, GLWEToBackendMut,
    GLWEToBackendRef, LWEInfos, LWEToBackendMut, LWEToBackendRef,
    prepared::{GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedToBackendRef},
};

/// Backend-provided GLWE key-switching operations.
///
/// # Safety
/// Implementations must satisfy the documented key-switch semantics, honor layout metadata and
/// prepared-key interpretation, and keep all reads and writes within the described backend buffers.
#[allow(private_bounds)]
pub unsafe trait GLWEKeyswitchImpl<BE: Backend>: Backend {
    fn glwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_keyswitch<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;

    fn glwe_keyswitch_assign<R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;
}

/// Backend-provided GGLWE key-switching operations.
///
/// # Safety
/// Implementations must preserve ciphertext invariants, use scratch space according to the
/// advertised temporary-size contract, and uphold aliasing guarantees for backend-owned buffers.
#[allow(private_bounds)]
pub unsafe trait GGLWEKeyswitchImpl<BE: Backend>: Backend {
    fn gglwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos;

    fn gglwe_keyswitch<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;

    fn gglwe_keyswitch_assign<R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;
}

/// Backend-provided GGSW key-switching operations.
///
/// # Safety
/// Implementations must correctly interpret prepared key material for the backend, respect all
/// layout-derived bounds, and avoid invalid aliasing or mutation through scratch-backed views.
#[allow(private_bounds)]
pub unsafe trait GGSWKeyswitchImpl<BE: Backend>: Backend {
    fn ggsw_keyswitch_tmp_bytes<R, A, K, T>(
        module: &Module<BE>,
        res_infos: &R,
        a_infos: &A,
        key_infos: &K,
        tsk_infos: &T,
    ) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos;

    fn ggsw_keyswitch<R, A, K, T>(module: &Module<BE>, res: &mut R, a: &A, key: &K, tsk: &T, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos;

    fn ggsw_keyswitch_assign<R, K, T>(module: &Module<BE>, res: &mut R, key: &K, tsk: &T, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos;
}

/// Backend-provided LWE key-switching operations.
///
/// # Safety
/// Implementations must only access the ciphertext and key regions described by the layouts and
/// must produce results matching the logical key-switch operation for the backend.
#[allow(private_bounds)]
pub unsafe trait LWEKeyswitchImpl<BE: Backend>: Backend {
    fn lwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn lwe_keyswitch<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, ksk: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;
}

/// Override surface for the GLWE key-switching sub-family.
///
/// Abstract: no HAL supertraits, no default method bodies. See [`glwe_keyswitch_defaults`]
/// for reference algorithms a backend may forward to.
#[doc(hidden)]
#[allow(private_bounds)]
pub trait GLWEKeyswitchDefault<BE: Backend> {
    fn glwe_keyswitch_tmp_bytes_default<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_keyswitch_default<R, A, K>(&self, res: &mut R, a: &A, key: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;

    fn glwe_keyswitch_assign_default<R, K>(&self, res: &mut R, key: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;
}

/// Override surface for the GGLWE key-switching sub-family.
#[doc(hidden)]
#[allow(private_bounds)]
pub trait GGLWEKeyswitchDefault<BE: Backend> {
    fn gglwe_keyswitch_tmp_bytes_default<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos;

    fn gglwe_keyswitch_default<R, A, B>(&self, res: &mut R, a: &A, b: &B, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
        B: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;

    fn gglwe_keyswitch_assign_default<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;
}

/// Override surface for the GGSW key-switching sub-family.
#[doc(hidden)]
#[allow(private_bounds)]
pub trait GGSWKeyswitchDefault<BE: Backend> {
    fn ggsw_keyswitch_tmp_bytes_default<R, A, K, T>(&self, res_infos: &R, a_infos: &A, key_infos: &K, tsk_infos: &T) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos;

    fn ggsw_keyswitch_default<R, A, K, T>(&self, res: &mut R, a: &A, key: &K, tsk: &T, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos;

    fn ggsw_keyswitch_assign_default<R, K, T>(&self, res: &mut R, key: &K, tsk: &T, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos;
}

/// Override surface for the LWE key-switching sub-family.
#[doc(hidden)]
#[allow(private_bounds)]
pub trait LWEKeyswitchDefault<BE: Backend> {
    fn lwe_keyswitch_tmp_bytes_default<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn lwe_keyswitch_default<R, A, K>(&self, res: &mut R, a: &A, ksk: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos;
}

#[allow(private_bounds)]
unsafe impl<BE: Backend> GLWEKeyswitchImpl<BE> for BE
where
    Module<BE>: GLWEKeyswitchDefault<BE>,
{
    fn glwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        module.glwe_keyswitch_tmp_bytes_default(res_infos, a_infos, key_infos)
    }

    fn glwe_keyswitch<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        module.glwe_keyswitch_default(res, a, key, scratch)
    }

    fn glwe_keyswitch_assign<R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        module.glwe_keyswitch_assign_default(res, key, scratch)
    }
}

#[allow(private_bounds)]
unsafe impl<BE: Backend> GGLWEKeyswitchImpl<BE> for BE
where
    Module<BE>: GGLWEKeyswitchDefault<BE>,
{
    fn gglwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos,
    {
        module.gglwe_keyswitch_tmp_bytes_default(res_infos, a_infos, key_infos)
    }

    fn gglwe_keyswitch<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToBackendRef<BE> + GGLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        module.gglwe_keyswitch_default(res, a, key, scratch)
    }

    fn gglwe_keyswitch_assign<R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        module.gglwe_keyswitch_assign_default(res, key, scratch)
    }
}

#[allow(private_bounds)]
unsafe impl<BE: Backend> GGSWKeyswitchImpl<BE> for BE
where
    Module<BE>: GGSWKeyswitchDefault<BE>,
{
    fn ggsw_keyswitch_tmp_bytes<R, A, K, T>(
        module: &Module<BE>,
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
        module.ggsw_keyswitch_tmp_bytes_default(res_infos, a_infos, key_infos, tsk_infos)
    }

    fn ggsw_keyswitch<R, A, K, T>(module: &Module<BE>, res: &mut R, a: &A, key: &K, tsk: &T, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        module.ggsw_keyswitch_default(res, a, key, tsk, scratch)
    }

    fn ggsw_keyswitch_assign<R, K, T>(module: &Module<BE>, res: &mut R, key: &K, tsk: &T, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        module.ggsw_keyswitch_assign_default(res, key, tsk, scratch)
    }
}

#[allow(private_bounds)]
unsafe impl<BE: Backend> LWEKeyswitchImpl<BE> for BE
where
    Module<BE>: LWEKeyswitchDefault<BE>,
{
    fn lwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
    {
        module.lwe_keyswitch_tmp_bytes_default(res_infos, a_infos, key_infos)
    }

    fn lwe_keyswitch<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, ksk: &K, scratch: &mut ScratchArena<'_, BE>)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        module.lwe_keyswitch_default(res, a, ksk, scratch)
    }
}

// === Convenience macros for full-default opt-in ===

/// Implements [`GLWEKeyswitchDefault`] for `Module<$be>` by forwarding every method to
/// the corresponding [`glwe_keyswitch_defaults`] free function.
#[macro_export]
macro_rules! impl_glwe_keyswitch_defaults_full {
    ($be:ty) => {
        impl $crate::oep::GLWEKeyswitchDefault<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn glwe_keyswitch_tmp_bytes_default<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::default::keyswitching::glwe::glwe_keyswitch_tmp_bytes_default::<$be, _, _, _, _>(
                    self, res_infos, a_infos, key_infos,
                )
            }

            fn glwe_keyswitch_default<R, A, K>(
                &self,
                res: &mut R,
                a: &A,
                key: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
                K: $crate::layouts::prepared::GGLWEPreparedToBackendRef<$be> + $crate::layouts::GGLWEInfos,
            {
                $crate::default::keyswitching::glwe::glwe_keyswitch_default::<$be, _, _, _, _>(self, res, a, key, scratch)
            }

            fn glwe_keyswitch_assign_default<R, K>(
                &self,
                res: &mut R,
                key: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                K: $crate::layouts::prepared::GGLWEPreparedToBackendRef<$be> + $crate::layouts::GGLWEInfos,
            {
                $crate::default::keyswitching::glwe::glwe_keyswitch_assign_default::<$be, _, _, _>(self, res, key, scratch)
            }
        }
    };
}

/// Implements [`GGLWEKeyswitchDefault`] for `Module<$be>` by forwarding every method to
/// the corresponding [`gglwe_keyswitch_defaults`] free function.
#[macro_export]
macro_rules! impl_gglwe_keyswitch_defaults_full {
    ($be:ty) => {
        impl $crate::oep::GGLWEKeyswitchDefault<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn gglwe_keyswitch_tmp_bytes_default<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
            where
                R: $crate::layouts::GGLWEInfos,
                A: $crate::layouts::GGLWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::default::keyswitching::gglwe::gglwe_keyswitch_tmp_bytes_default::<$be, _, _, _, _>(
                    self, res_infos, a_infos, key_infos,
                )
            }

            fn gglwe_keyswitch_default<R, A, B>(
                &self,
                res: &mut R,
                a: &A,
                b: &B,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::GGLWEInfos,
                A: $crate::layouts::GGLWEToBackendRef<$be> + $crate::layouts::GGLWEInfos,
                B: $crate::layouts::prepared::GGLWEPreparedToBackendRef<$be> + $crate::layouts::GGLWEInfos,
            {
                $crate::default::keyswitching::gglwe::gglwe_keyswitch_default::<$be, _, _, _, _>(self, res, a, b, scratch)
            }

            fn gglwe_keyswitch_assign_default<R, A>(
                &self,
                res: &mut R,
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::GGLWEInfos,
                A: $crate::layouts::prepared::GGLWEPreparedToBackendRef<$be> + $crate::layouts::GGLWEInfos,
            {
                $crate::default::keyswitching::gglwe::gglwe_keyswitch_assign_default::<$be, _, _, _>(self, res, a, scratch)
            }
        }
    };
}

/// Implements [`GGSWKeyswitchDefault`] for `Module<$be>` by forwarding every method to
/// the corresponding [`ggsw_keyswitch_defaults`] free function.
#[macro_export]
macro_rules! impl_ggsw_keyswitch_defaults_full {
    ($be:ty) => {
        impl $crate::oep::GGSWKeyswitchDefault<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn ggsw_keyswitch_tmp_bytes_default<R, A, K, T>(
                &self,
                res_infos: &R,
                a_infos: &A,
                key_infos: &K,
                tsk_infos: &T,
            ) -> usize
            where
                R: $crate::layouts::GGSWInfos,
                A: $crate::layouts::GGSWInfos,
                K: $crate::layouts::GGLWEInfos,
                T: $crate::layouts::GGLWEInfos,
            {
                $crate::default::keyswitching::ggsw::ggsw_keyswitch_tmp_bytes_default::<$be, _, _, _, _, _>(
                    self, res_infos, a_infos, key_infos, tsk_infos,
                )
            }

            fn ggsw_keyswitch_default<R, A, K, T>(
                &self,
                res: &mut R,
                a: &A,
                key: &K,
                tsk: &T,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GGSWToBackendMut<$be> + $crate::layouts::GGSWInfos,
                A: $crate::layouts::GGSWToBackendRef<$be> + $crate::layouts::GGSWInfos,
                K: $crate::layouts::prepared::GGLWEPreparedToBackendRef<$be> + $crate::layouts::GGLWEInfos,
                T: $crate::layouts::prepared::GGLWEToGGSWKeyPreparedToBackendRef<$be> + $crate::layouts::GGLWEInfos,
            {
                $crate::default::keyswitching::ggsw::ggsw_keyswitch_default::<$be, _, _, _, _, _>(self, res, a, key, tsk, scratch)
            }

            fn ggsw_keyswitch_assign_default<R, K, T>(
                &self,
                res: &mut R,
                key: &K,
                tsk: &T,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::GGSWToBackendMut<$be> + $crate::layouts::GGSWInfos,
                K: $crate::layouts::prepared::GGLWEPreparedToBackendRef<$be> + $crate::layouts::GGLWEInfos,
                T: $crate::layouts::prepared::GGLWEToGGSWKeyPreparedToBackendRef<$be> + $crate::layouts::GGLWEInfos,
            {
                $crate::default::keyswitching::ggsw::ggsw_keyswitch_assign_default::<$be, _, _, _, _>(
                    self, res, key, tsk, scratch,
                )
            }
        }
    };
}

/// Implements [`LWEKeyswitchDefault`] for `Module<$be>` by forwarding every method to
/// the corresponding [`lwe_keyswitch_defaults`] free function.
#[macro_export]
macro_rules! impl_lwe_keyswitch_defaults_full {
    ($be:ty) => {
        impl $crate::oep::LWEKeyswitchDefault<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn lwe_keyswitch_tmp_bytes_default<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
            where
                R: $crate::layouts::LWEInfos,
                A: $crate::layouts::LWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::default::keyswitching::lwe::lwe_keyswitch_tmp_bytes_default::<$be, _, _, _, _>(
                    self, res_infos, a_infos, key_infos,
                )
            }

            fn lwe_keyswitch_default<R, A, K>(
                &self,
                res: &mut R,
                a: &A,
                ksk: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: $crate::layouts::LWEToBackendMut<$be> + $crate::layouts::LWEInfos,
                A: $crate::layouts::LWEToBackendRef<$be> + $crate::layouts::LWEInfos,
                K: $crate::layouts::prepared::GGLWEPreparedToBackendRef<$be> + $crate::layouts::GGLWEInfos,
            {
                $crate::default::keyswitching::lwe::lwe_keyswitch_default::<$be, _, _, _, _>(self, res, a, ksk, scratch)
            }
        }
    };
}
