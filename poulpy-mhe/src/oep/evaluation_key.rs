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
    layouts::{GLWEAutomorphismKeyShareOwned, GLWESwitchingKeyShareOwned},
    oep::GGLWEPatCompressedImpl,
};

/// # Safety
/// Reproduce the reference share, mask seeds and degrees included, within the
/// queried scratch budget.
pub unsafe trait GLWESwitchingKeyMHEProtocolImpl: GGLWEPatCompressedImpl {
    fn glwe_switching_key_gen_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_switching_key_gen<S1, S2, E>(
        module: &Module<Self>,
        res: &mut GLWESwitchingKeyShareOwned<Self>,
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

    fn glwe_switching_key_aggregate(
        module: &Module<Self>,
        res: &mut GLWESwitchingKeyShareOwned<Self>,
        a: &GLWESwitchingKeyShareOwned<Self>,
    ) {
        derived::glwe_switching_key_aggregate_derived(module, res, a)
    }

    fn glwe_switching_key_finalize_tmp_bytes(module: &Module<Self>) -> usize {
        Self::gglwe_pat_compressed_finalize_tmp_bytes(module)
    }

    fn glwe_switching_key_finalize<R>(
        module: &Module<Self>,
        res: &mut R,
        share: &GLWESwitchingKeyShareOwned<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
    {
        derived::glwe_switching_key_finalize_derived(module, res, share, scratch)
    }
}

/// # Safety
/// Reproduce the reference share, mask seeds and Galois element included,
/// within the queried scratch budget.
pub unsafe trait GLWEAutomorphismKeyMHEProtocolImpl: GGLWEPatCompressedImpl {
    fn glwe_automorphism_key_gen_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_automorphism_key_gen<S, E>(
        module: &Module<Self>,
        res: &mut GLWEAutomorphismKeyShareOwned<Self>,
        p: i64,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        S: GLWESecretToBackendRef<Self> + GLWEInfos,
        E: EncryptionInfos;

    fn glwe_automorphism_key_aggregate(
        module: &Module<Self>,
        res: &mut GLWEAutomorphismKeyShareOwned<Self>,
        a: &GLWEAutomorphismKeyShareOwned<Self>,
    ) {
        derived::glwe_automorphism_key_aggregate_derived(module, res, a)
    }

    fn glwe_automorphism_key_finalize_tmp_bytes(module: &Module<Self>) -> usize {
        Self::gglwe_pat_compressed_finalize_tmp_bytes(module)
    }

    fn glwe_automorphism_key_finalize<R>(
        module: &Module<Self>,
        res: &mut R,
        share: &GLWEAutomorphismKeyShareOwned<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self> + GGLWEInfos + SetGaloisElement,
    {
        derived::glwe_automorphism_key_finalize_derived(module, res, share, scratch)
    }
}

/// Selects the reference share generation; aggregation and finalization keep
/// their derived defaults.
#[macro_export]
macro_rules! impl_mhe_evaluation_key_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::GLWESwitchingKeyMHEProtocolImpl for $be {
            fn glwe_switching_key_gen_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWESwitchingKeyMHEProtocolReference<$be>>::glwe_switching_key_gen_tmp_bytes_reference(module, infos)
            }

            fn glwe_switching_key_gen<S1, S2, E>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GLWESwitchingKeyShareOwned<$be>,
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
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWESwitchingKeyMHEProtocolReference<$be>>::glwe_switching_key_gen_reference(module, res, sk_in, sk_out, seed, enc_infos, source_xe, scratch)
            }
        }

        unsafe impl $crate::oep::GLWEAutomorphismKeyMHEProtocolImpl for $be {
            fn glwe_automorphism_key_gen_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEAutomorphismKeyMHEProtocolReference<$be>>::glwe_automorphism_key_gen_tmp_bytes_reference(module, infos)
            }

            fn glwe_automorphism_key_gen<S, E>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GLWEAutomorphismKeyShareOwned<$be>,
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
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEAutomorphismKeyMHEProtocolReference<$be>>::glwe_automorphism_key_gen_reference(module, res, p, sk, seed, enc_infos, source_xe, scratch)
            }
        }
    };
}
