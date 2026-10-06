use poulpy_core::{
    GetDistribution,
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
/// Outputs follow the [noise metadata rule](crate::oep#noise-metadata). Delegates only forward calls.
/// Reproduce the reference share, mask seeds and degrees included, within the
/// queried scratch budget.
pub unsafe trait GLWESwitchingKeyMHEProtocolImpl: GGLWEPatCompressedImpl {
    fn mhe_glwe_switching_key_share_gen_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_switching_key_share_gen<S1, S2>(
        module: &Module<Self>,
        res: &mut GLWESwitchingKeyShareOwned<Self>,
        sk_in: &S1,
        sk_out: &S2,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        S1: GLWESecretToBackendRef<Self> + GLWEInfos,
        S2: GLWESecretToBackendRef<Self> + GetDistribution + GLWEInfos;

    fn mhe_glwe_switching_key_share_aggregate(
        module: &Module<Self>,
        res: &mut GLWESwitchingKeyShareOwned<Self>,
        a: &GLWESwitchingKeyShareOwned<Self>,
    ) {
        derived::mhe_glwe_switching_key_share_aggregate_derived(module, res, a)
    }

    fn mhe_glwe_switching_key_share_finalize_tmp_bytes(module: &Module<Self>) -> usize {
        Self::gglwe_pat_compressed_finalize_tmp_bytes(module)
    }

    fn mhe_glwe_switching_key_share_finalize<R>(
        module: &Module<Self>,
        res: &mut R,
        share: &GLWESwitchingKeyShareOwned<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
    {
        derived::mhe_glwe_switching_key_share_finalize_derived(module, res, share, scratch)
    }
}

/// # Safety
/// Outputs follow the [noise metadata rule](crate::oep#noise-metadata). Delegates only forward calls.
/// Reproduce the reference share, mask seeds and Galois element included,
/// within the queried scratch budget.
pub unsafe trait GLWEAutomorphismKeyMHEProtocolImpl: GGLWEPatCompressedImpl {
    fn mhe_glwe_automorphism_key_share_gen_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_automorphism_key_share_gen<S>(
        module: &Module<Self>,
        res: &mut GLWEAutomorphismKeyShareOwned<Self>,
        p: i64,
        sk: &S,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        S: GLWESecretToBackendRef<Self> + GLWEInfos;

    fn mhe_glwe_automorphism_key_share_aggregate(
        module: &Module<Self>,
        res: &mut GLWEAutomorphismKeyShareOwned<Self>,
        a: &GLWEAutomorphismKeyShareOwned<Self>,
    ) {
        derived::mhe_glwe_automorphism_key_share_aggregate_derived(module, res, a)
    }

    fn mhe_glwe_automorphism_key_share_finalize_tmp_bytes(module: &Module<Self>) -> usize {
        Self::gglwe_pat_compressed_finalize_tmp_bytes(module)
    }

    fn mhe_glwe_automorphism_key_share_finalize<R>(
        module: &Module<Self>,
        res: &mut R,
        share: &GLWEAutomorphismKeyShareOwned<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self> + GGLWEInfos + SetGaloisElement,
    {
        derived::mhe_glwe_automorphism_key_share_finalize_derived(module, res, share, scratch)
    }
}

/// Selects the reference share generation; aggregation and finalization keep
/// their derived defaults.
#[macro_export]
macro_rules! impl_mhe_evaluation_key_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::GLWESwitchingKeyMHEProtocolImpl for $be {
            fn mhe_glwe_switching_key_share_gen_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWESwitchingKeyMHEProtocolReference<$be>>::mhe_glwe_switching_key_share_gen_tmp_bytes_reference(module, infos)
            }

            fn mhe_glwe_switching_key_share_gen<S1, S2>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GLWESwitchingKeyShareOwned<$be>,
                sk_in: &S1,
                sk_out: &S2,
                seed: [u8; 32],
                source_xe: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                S1: ::poulpy_core::layouts::GLWESecretToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                S2: ::poulpy_core::layouts::GLWESecretToBackendRef<$be>
                    + ::poulpy_core::GetDistribution
                    + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWESwitchingKeyMHEProtocolReference<$be>>::mhe_glwe_switching_key_share_gen_reference(module, res, sk_in, sk_out, seed, source_xe, scratch)
            }
        }

        unsafe impl $crate::oep::GLWEAutomorphismKeyMHEProtocolImpl for $be {
            fn mhe_glwe_automorphism_key_share_gen_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEAutomorphismKeyMHEProtocolReference<$be>>::mhe_glwe_automorphism_key_share_gen_tmp_bytes_reference(module, infos)
            }

            fn mhe_glwe_automorphism_key_share_gen<S>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GLWEAutomorphismKeyShareOwned<$be>,
                p: i64,
                sk: &S,
                seed: [u8; 32],
                source_xe: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                S: ::poulpy_core::layouts::GLWESecretToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEAutomorphismKeyMHEProtocolReference<$be>>::mhe_glwe_automorphism_key_share_gen_reference(module, res, p, sk, seed, source_xe, scratch)
            }
        }
    };
}
