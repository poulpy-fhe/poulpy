use poulpy_core::{
    EncryptionInfos, GetDistribution,
    layouts::{GGLWEInfos, GGLWEToBackendMut, GLWEInfos, GLWESecretToBackendRef, GLWESwitchingKeyDegreesMut, SetGaloisElement},
};
use poulpy_hal::{
    layouts::{Module, ScratchArena},
    source::Source,
};

use super::derived::evaluation_key as derived;
use crate::{
    layouts::{GLWEAutomorphismKeyPatCompressedOwned, GLWESwitchingKeyPatCompressedOwned},
    oep::{PatAggregateImpl, PatFinalizeImpl, PatNormalizeImpl},
};

/// # Safety
/// Reproduce the reference share, mask seeds and degrees included, within the
/// queried scratch budget.
pub unsafe trait GLWESwitchingKeyShareImpl: PatAggregateImpl + PatNormalizeImpl + PatFinalizeImpl {
    fn glwe_switching_key_share_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_switching_key_share<S1, S2, E>(
        module: &Module<Self>,
        res: &mut GLWESwitchingKeyPatCompressedOwned<Self>,
        sk_in: &S1,
        sk_out: &S2,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        S1: GLWESecretToBackendRef<Self> + GLWEInfos,
        S2: GLWESecretToBackendRef<Self> + GetDistribution + GLWEInfos,
        E: EncryptionInfos;

    fn glwe_switching_key_share_aggregate_assign(
        module: &Module<Self>,
        res: &mut GLWESwitchingKeyPatCompressedOwned<Self>,
        a: &GLWESwitchingKeyPatCompressedOwned<Self>,
    ) {
        derived::glwe_switching_key_share_aggregate_assign_derived(module, res, a)
    }

    fn glwe_switching_key_share_normalize_tmp_bytes(module: &Module<Self>) -> usize {
        derived::glwe_switching_key_share_normalize_tmp_bytes_derived(module)
    }

    fn glwe_switching_key_share_normalize_assign(
        module: &Module<Self>,
        res: &mut GLWESwitchingKeyPatCompressedOwned<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        derived::glwe_switching_key_share_normalize_assign_derived(module, res, scratch)
    }

    fn glwe_switching_key_finalize_tmp_bytes(module: &Module<Self>) -> usize {
        derived::glwe_switching_key_finalize_tmp_bytes_derived(module)
    }

    fn glwe_switching_key_finalize<R>(
        module: &Module<Self>,
        res: &mut R,
        pat: &GLWESwitchingKeyPatCompressedOwned<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
    {
        derived::glwe_switching_key_finalize_derived(module, res, pat, scratch)
    }
}

/// # Safety
/// Reproduce the reference share, mask seeds and Galois element included,
/// within the queried scratch budget.
pub unsafe trait GLWEAutomorphismKeyShareImpl: PatAggregateImpl + PatNormalizeImpl + PatFinalizeImpl {
    fn glwe_automorphism_key_share_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_automorphism_key_share<S, E>(
        module: &Module<Self>,
        res: &mut GLWEAutomorphismKeyPatCompressedOwned<Self>,
        p: i64,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        S: GLWESecretToBackendRef<Self> + GLWEInfos,
        E: EncryptionInfos;

    fn glwe_automorphism_key_share_aggregate_assign(
        module: &Module<Self>,
        res: &mut GLWEAutomorphismKeyPatCompressedOwned<Self>,
        a: &GLWEAutomorphismKeyPatCompressedOwned<Self>,
    ) {
        derived::glwe_automorphism_key_share_aggregate_assign_derived(module, res, a)
    }

    fn glwe_automorphism_key_share_normalize_tmp_bytes(module: &Module<Self>) -> usize {
        derived::glwe_automorphism_key_share_normalize_tmp_bytes_derived(module)
    }

    fn glwe_automorphism_key_share_normalize_assign(
        module: &Module<Self>,
        res: &mut GLWEAutomorphismKeyPatCompressedOwned<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        derived::glwe_automorphism_key_share_normalize_assign_derived(module, res, scratch)
    }

    fn glwe_automorphism_key_finalize_tmp_bytes(module: &Module<Self>) -> usize {
        derived::glwe_automorphism_key_finalize_tmp_bytes_derived(module)
    }

    fn glwe_automorphism_key_finalize<R>(
        module: &Module<Self>,
        res: &mut R,
        pat: &GLWEAutomorphismKeyPatCompressedOwned<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self> + GGLWEInfos + SetGaloisElement,
    {
        derived::glwe_automorphism_key_finalize_derived(module, res, pat, scratch)
    }
}

/// Selects the reference evaluation key shares; aggregation, normalization and
/// finalization keep their derived defaults.
#[macro_export]
macro_rules! impl_mhe_evaluation_key_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::GLWESwitchingKeyShareImpl for $be {
            fn glwe_switching_key_share_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWESwitchingKeyShareReference<$be>>::glwe_switching_key_share_tmp_bytes_reference(module, infos)
            }

            fn glwe_switching_key_share<S1, S2, E>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GLWESwitchingKeyPatCompressedOwned<$be>,
                sk_in: &S1,
                sk_out: &S2,
                seed: [u8; 32],
                enc_infos: &E,
                source_xe: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                S1: ::poulpy_core::layouts::GLWESecretToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                S2: ::poulpy_core::layouts::GLWESecretToBackendRef<$be>
                    + ::poulpy_core::GetDistribution
                    + ::poulpy_core::layouts::GLWEInfos,
                E: ::poulpy_core::EncryptionInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWESwitchingKeyShareReference<$be>>::glwe_switching_key_share_reference(module, res, sk_in, sk_out, seed, enc_infos, source_xe, scratch)
            }
        }

        unsafe impl $crate::oep::GLWEAutomorphismKeyShareImpl for $be {
            fn glwe_automorphism_key_share_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEAutomorphismKeyShareReference<$be>>::glwe_automorphism_key_share_tmp_bytes_reference(module, infos)
            }

            fn glwe_automorphism_key_share<S, E>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GLWEAutomorphismKeyPatCompressedOwned<$be>,
                p: i64,
                sk: &S,
                seed: [u8; 32],
                enc_infos: &E,
                source_xe: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                S: ::poulpy_core::layouts::GLWESecretToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                E: ::poulpy_core::EncryptionInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEAutomorphismKeyShareReference<$be>>::glwe_automorphism_key_share_reference(module, res, p, sk, seed, enc_infos, source_xe, scratch)
            }
        }
    };
}
