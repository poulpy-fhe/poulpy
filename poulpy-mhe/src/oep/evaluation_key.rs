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
/// Every method reproduces the reference or derived body its doc names, checks
/// included, and stays within the scratch its `_tmp_bytes` query returns.
/// Share generation wipes that scratch before returning.
/// The memory, layout, synchrony and aliasing rules of
/// `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GLWESwitchingKeyMHEProtocolImpl: GGLWEPatCompressedImpl {
    /// Upper bound on the scratch of `mhe_glwe_switching_key_share_gen`, as
    /// `GLWESwitchingKeyMHEProtocolReference::mhe_glwe_switching_key_share_gen_tmp_bytes_reference`.
    fn mhe_glwe_switching_key_share_gen_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    /// Reproduces
    /// `GLWESwitchingKeyMHEProtocolReference::mhe_glwe_switching_key_share_gen_reference`:
    /// the compressed GGLWE encryption of `sk_in` under `sk_out`, masks drawn
    /// from `seed`, errors from `source_xe`, recording the secret degrees in
    /// `res`.
    /// Panics unless `res` has the module's degree, each secret's degree is a
    /// power of two up to the key's, and `sk_in` and `sk_out` have the key's
    /// input and output ranks.
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

    /// Default body `mhe_glwe_switching_key_share_aggregate_derived`: the
    /// compressed PAT aggregation, panicking unless the secret degrees agree.
    fn mhe_glwe_switching_key_share_aggregate(
        module: &Module<Self>,
        res: &mut GLWESwitchingKeyShareOwned<Self>,
        a: &GLWESwitchingKeyShareOwned<Self>,
    ) {
        derived::mhe_glwe_switching_key_share_aggregate_derived(module, res, a)
    }

    /// Default body: the scratch of the compressed GGLWE PAT finalization.
    fn mhe_glwe_switching_key_share_finalize_tmp_bytes(module: &Module<Self>) -> usize {
        Self::gglwe_pat_compressed_finalize_tmp_bytes(module)
    }

    /// Default body `mhe_glwe_switching_key_share_finalize_derived`: the
    /// compressed GGLWE PAT finalization, then copies the share's secret
    /// degrees into `res`.
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
/// Every method reproduces the reference or derived body its doc names, checks
/// included, and stays within the scratch its `_tmp_bytes` query returns.
/// Share generation wipes that scratch before returning.
/// The memory, layout, synchrony and aliasing rules of
/// `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GLWEAutomorphismKeyMHEProtocolImpl: GGLWEPatCompressedImpl {
    /// Upper bound on the scratch of `mhe_glwe_automorphism_key_share_gen`, as
    /// `GLWEAutomorphismKeyMHEProtocolReference::mhe_glwe_automorphism_key_share_gen_tmp_bytes_reference`.
    fn mhe_glwe_automorphism_key_share_gen_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    /// Reproduces
    /// `GLWEAutomorphismKeyMHEProtocolReference::mhe_glwe_automorphism_key_share_gen_reference`:
    /// the compressed GGLWE encryption of the automorphism `p` of `sk` under
    /// `sk`, masks drawn from `seed`, errors from `source_xe`, recording `p` in
    /// `res`.
    /// Panics unless `res` has the module's degree and equal input and output
    /// ranks, `sk` has the key's degree and rank, and `p` is odd.
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

    /// Default body `mhe_glwe_automorphism_key_share_aggregate_derived`: the
    /// compressed PAT aggregation, panicking unless the Galois elements agree.
    fn mhe_glwe_automorphism_key_share_aggregate(
        module: &Module<Self>,
        res: &mut GLWEAutomorphismKeyShareOwned<Self>,
        a: &GLWEAutomorphismKeyShareOwned<Self>,
    ) {
        derived::mhe_glwe_automorphism_key_share_aggregate_derived(module, res, a)
    }

    /// Default body: the scratch of the compressed GGLWE PAT finalization.
    fn mhe_glwe_automorphism_key_share_finalize_tmp_bytes(module: &Module<Self>) -> usize {
        Self::gglwe_pat_compressed_finalize_tmp_bytes(module)
    }

    /// Default body `mhe_glwe_automorphism_key_share_finalize_derived`: the
    /// compressed GGLWE PAT finalization, then copies the share's Galois
    /// element into `res`.
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

/// Selects the reference switching key share generation; aggregation and
/// finalization keep their derived defaults.
#[macro_export]
macro_rules! impl_glwe_switching_key_mhe_protocol_reference_full {
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
    };
}

/// Selects the reference automorphism key share generation; aggregation and
/// finalization keep their derived defaults.
#[macro_export]
macro_rules! impl_glwe_automorphism_key_mhe_protocol_reference_full {
    ($be:ty) => {
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
