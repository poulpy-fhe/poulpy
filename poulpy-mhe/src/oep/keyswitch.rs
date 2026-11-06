use crate::layouts::{MHEGLWEPrivateKeyswitchShareOwned, MHEGLWEPublicKeyswitchShareOwned};
use poulpy_core::{
    Noise,
    layouts::{
        GLWEInfos, GLWEMaskToBackendRef, GLWEPublicKeyPreparedToBackendRef, GLWESecretPreparedToBackendRef, GLWEToBackendMut,
        GLWEToBackendRef,
    },
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

/// # Safety
/// Every method reproduces the reference or derived body its doc names, checks
/// included, and stays within the scratch its `_tmp_bytes` query returns.
/// Share generation wipes that scratch before returning.
/// The memory, layout, synchrony and aliasing rules of
/// `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait MHEGLWEPrivateKeyswitchProtocolImpl: Backend {
    /// Upper bound on the scratch of `mhe_glwe_private_keyswitch_share_gen`, as
    /// `MHEGLWEPrivateKeyswitchProtocolReference::mhe_glwe_private_keyswitch_share_gen_tmp_bytes_reference`;
    /// panics unless `infos` has the module's degree.
    fn mhe_glwe_private_keyswitch_share_gen_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Reproduces
    /// `MHEGLWEPrivateKeyswitchProtocolReference::mhe_glwe_private_keyswitch_share_gen_reference`:
    /// `<mask, sk_in> - <mask, sk_out>` plus the `flood` noise drawn from
    /// `source_smudge`; the mask is read only.
    /// Panics unless `mask` is canonical and has the module's degree, `res` has
    /// rank 0 and the mask's layout, the secrets have the mask's degree and
    /// rank, and `flood` is valid at the share's precision, all before any
    /// draw.
    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_private_keyswitch_share_gen<C, S1, S2>(
        module: &Module<Self>,
        res: &mut MHEGLWEPrivateKeyswitchShareOwned<Self>,
        mask: &C,
        sk_in: &S1,
        sk_out: &S2,
        flood: Noise,
        source_smudge: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        C: GLWEMaskToBackendRef<Self> + GLWEInfos,
        S1: GLWESecretPreparedToBackendRef<Self> + GLWEInfos,
        S2: GLWESecretPreparedToBackendRef<Self> + GLWEInfos;

    /// Reproduces
    /// `MHEGLWEPrivateKeyswitchProtocolReference::mhe_glwe_private_keyswitch_share_aggregate_reference`:
    /// adds `a` into `res` without normalizing and combines the noise with
    /// `aggregate_metadata`.
    /// Panics unless the layouts agree.
    fn mhe_glwe_private_keyswitch_share_aggregate(
        module: &Module<Self>,
        res: &mut MHEGLWEPrivateKeyswitchShareOwned<Self>,
        a: &MHEGLWEPrivateKeyswitchShareOwned<Self>,
    );

    /// Upper bound on the scratch of
    /// `mhe_glwe_private_keyswitch_share_finalize`, as the reference.
    fn mhe_glwe_private_keyswitch_share_finalize_tmp_bytes(module: &Module<Self>) -> usize;

    /// Reproduces
    /// `MHEGLWEPrivateKeyswitchProtocolReference::mhe_glwe_private_keyswitch_share_finalize_reference`:
    /// `res = ct + share`, the share added to the body and normalized.
    /// Panics unless the ciphertext has the module's degree, `res` has its
    /// layout and the share is its rank-0 body layout.
    fn mhe_glwe_private_keyswitch_share_finalize<R, C>(
        module: &Module<Self>,
        res: &mut R,
        ct: &C,
        share: &MHEGLWEPrivateKeyswitchShareOwned<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        C: GLWEToBackendRef<Self> + GLWEInfos;
}

/// # Safety
/// Every method reproduces the reference or derived body its doc names, checks
/// included, and stays within the scratch its `_tmp_bytes` query returns.
/// Share generation wipes that scratch before returning.
/// The memory, layout, synchrony and aliasing rules of
/// `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait MHEGLWEPublicKeyswitchProtocolImpl: Backend {
    /// Upper bound on the scratch of `mhe_glwe_public_keyswitch_share_gen`, as
    /// `MHEGLWEPublicKeyswitchProtocolReference::mhe_glwe_public_keyswitch_share_gen_tmp_bytes_reference`;
    /// panics unless all three layouts have the module's degree.
    fn mhe_glwe_public_keyswitch_share_gen_tmp_bytes<A, B, P>(
        module: &Module<Self>,
        ct_infos: &A,
        res_infos: &B,
        pk_infos: &P,
    ) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos,
        P: GLWEInfos;

    /// Reproduces
    /// `MHEGLWEPublicKeyswitchProtocolReference::mhe_glwe_public_keyswitch_share_gen_reference`:
    /// `<mask, sk_in>` plus an encryption of zero under `pk_out` (`source_xu`
    /// for the ephemeral, `source_xe` for the errors) plus the `flood` noise
    /// from `source_smudge`; the mask is read only.
    /// Panics unless `mask` is canonical and has the module's degree, `res` has
    /// the mask's layout, `sk_in` has the mask's degree and rank, `pk_out` has
    /// the share's degree, radix and rank, is at least as precise and has a
    /// consistent secret provenance, and `flood` is valid, all before any draw.
    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_public_keyswitch_share_gen<C, S, K>(
        module: &Module<Self>,
        res: &mut MHEGLWEPublicKeyswitchShareOwned<Self>,
        mask: &C,
        sk_in: &S,
        pk_out: &K,
        flood: Noise,
        source_xu: &mut Source,
        source_xe: &mut Source,
        source_smudge: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        C: GLWEMaskToBackendRef<Self> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<Self> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<Self> + GLWEInfos;

    /// Reproduces
    /// `MHEGLWEPublicKeyswitchProtocolReference::mhe_glwe_public_keyswitch_share_aggregate_reference`:
    /// adds `a` into `res` without normalizing and combines the noise with
    /// `aggregate_common_key_metadata`.
    /// Panics unless the layouts agree.
    fn mhe_glwe_public_keyswitch_share_aggregate(
        module: &Module<Self>,
        res: &mut MHEGLWEPublicKeyswitchShareOwned<Self>,
        a: &MHEGLWEPublicKeyswitchShareOwned<Self>,
    );

    /// Upper bound on the scratch of
    /// `mhe_glwe_public_keyswitch_share_finalize`, as the reference.
    fn mhe_glwe_public_keyswitch_share_finalize_tmp_bytes(module: &Module<Self>) -> usize;

    /// Reproduces
    /// `MHEGLWEPublicKeyswitchProtocolReference::mhe_glwe_public_keyswitch_share_finalize_reference`:
    /// `res = ct + share` with the share's mask and body, normalized.
    /// Panics unless the ciphertext has the module's degree and the share and
    /// `res` have its layout.
    fn mhe_glwe_public_keyswitch_share_finalize<R, C>(
        module: &Module<Self>,
        res: &mut R,
        ct: &C,
        share: &MHEGLWEPublicKeyswitchShareOwned<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        C: GLWEToBackendRef<Self> + GLWEInfos;
}

/// Selects the reference private key switching share generation, aggregation
/// and finalization.
#[macro_export]
macro_rules! impl_mhe_glwe_private_keyswitch_protocol_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::MHEGLWEPrivateKeyswitchProtocolImpl for $be {
            fn mhe_glwe_private_keyswitch_share_gen_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGLWEPrivateKeyswitchProtocolReference<$be>>::mhe_glwe_private_keyswitch_share_gen_tmp_bytes_reference(module, infos)
            }

            fn mhe_glwe_private_keyswitch_share_gen<C, S1, S2>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::MHEGLWEPrivateKeyswitchShareOwned<$be>,
                mask: &C,
                sk_in: &S1,
                sk_out: &S2,
                flood: ::poulpy_core::Noise,
                source_smudge: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                C: ::poulpy_core::layouts::GLWEMaskToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                S1: ::poulpy_core::layouts::GLWESecretPreparedToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                S2: ::poulpy_core::layouts::GLWESecretPreparedToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGLWEPrivateKeyswitchProtocolReference<$be>>::mhe_glwe_private_keyswitch_share_gen_reference(module, res, mask, sk_in, sk_out, flood, source_smudge, scratch)
            }

            fn mhe_glwe_private_keyswitch_share_aggregate(module: &::poulpy_hal::layouts::Module<$be>, res: &mut $crate::layouts::MHEGLWEPrivateKeyswitchShareOwned<$be>, a: &$crate::layouts::MHEGLWEPrivateKeyswitchShareOwned<$be>) {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGLWEPrivateKeyswitchProtocolReference<$be>>::mhe_glwe_private_keyswitch_share_aggregate_reference(module, res, a)
            }

            fn mhe_glwe_private_keyswitch_share_finalize_tmp_bytes(module: &::poulpy_hal::layouts::Module<$be>) -> usize {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGLWEPrivateKeyswitchProtocolReference<$be>>::mhe_glwe_private_keyswitch_share_finalize_tmp_bytes_reference(module)
            }

            fn mhe_glwe_private_keyswitch_share_finalize<R, C>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                ct: &C,
                share: &$crate::layouts::MHEGLWEPrivateKeyswitchShareOwned<$be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                C: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGLWEPrivateKeyswitchProtocolReference<$be>>::mhe_glwe_private_keyswitch_share_finalize_reference(module, res, ct, share, scratch)
            }
        }
    };
}

/// Selects the reference public key switching share generation, aggregation
/// and finalization.
#[macro_export]
macro_rules! impl_mhe_glwe_public_keyswitch_protocol_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::MHEGLWEPublicKeyswitchProtocolImpl for $be {
            fn mhe_glwe_public_keyswitch_share_gen_tmp_bytes<A, B, P>(
                module: &::poulpy_hal::layouts::Module<$be>,
                ct_infos: &A,
                res_infos: &B,
                pk_infos: &P,
            ) -> usize
            where
                A: ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GLWEInfos,
                P: ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGLWEPublicKeyswitchProtocolReference<$be>>::mhe_glwe_public_keyswitch_share_gen_tmp_bytes_reference(module, ct_infos, res_infos, pk_infos)
            }

            fn mhe_glwe_public_keyswitch_share_gen<C, S, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::MHEGLWEPublicKeyswitchShareOwned<$be>,
                mask: &C,
                sk_in: &S,
                pk_out: &K,
                flood: ::poulpy_core::Noise,
                source_xu: &mut ::poulpy_hal::source::Source,
                source_xe: &mut ::poulpy_hal::source::Source,
                source_smudge: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                C: ::poulpy_core::layouts::GLWEMaskToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                S: ::poulpy_core::layouts::GLWESecretPreparedToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                K: ::poulpy_core::layouts::GLWEPublicKeyPreparedToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGLWEPublicKeyswitchProtocolReference<$be>>::mhe_glwe_public_keyswitch_share_gen_reference(module, res, mask, sk_in, pk_out, flood, source_xu, source_xe, source_smudge, scratch)
            }

            fn mhe_glwe_public_keyswitch_share_aggregate(module: &::poulpy_hal::layouts::Module<$be>, res: &mut $crate::layouts::MHEGLWEPublicKeyswitchShareOwned<$be>, a: &$crate::layouts::MHEGLWEPublicKeyswitchShareOwned<$be>) {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGLWEPublicKeyswitchProtocolReference<$be>>::mhe_glwe_public_keyswitch_share_aggregate_reference(module, res, a)
            }

            fn mhe_glwe_public_keyswitch_share_finalize_tmp_bytes(module: &::poulpy_hal::layouts::Module<$be>) -> usize {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGLWEPublicKeyswitchProtocolReference<$be>>::mhe_glwe_public_keyswitch_share_finalize_tmp_bytes_reference(module)
            }

            fn mhe_glwe_public_keyswitch_share_finalize<R, C>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                ct: &C,
                share: &$crate::layouts::MHEGLWEPublicKeyswitchShareOwned<$be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                C: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGLWEPublicKeyswitchProtocolReference<$be>>::mhe_glwe_public_keyswitch_share_finalize_reference(module, res, ct, share, scratch)
            }
        }
    };
}
