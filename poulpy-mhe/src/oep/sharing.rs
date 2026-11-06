use poulpy_core::{
    Noise,
    layouts::{GLWEInfos, GLWEMaskToBackendRef, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{
    layouts::{MHEGLWEEncToShareShareOwned, MHEGLWEShareToEncShareOwned},
    oep::MHEGLWEPatCompressedImpl,
};

/// # Safety
/// Every method reproduces the reference or derived body its doc names, checks
/// included, and stays within the scratch its `_tmp_bytes` query returns.
/// Share generation and finalization, which writes the party's private
/// share, wipe that scratch before returning.
/// The memory, layout, synchrony and aliasing rules of
/// `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait MHEGLWEEncToShareProtocolImpl: Backend {
    /// Upper bound on the scratch of `mhe_glwe_enc_to_share_share_gen`, as
    /// `MHEGLWEEncToShareProtocolReference::mhe_glwe_enc_to_share_share_gen_tmp_bytes_reference`.
    fn mhe_glwe_enc_to_share_share_gen_tmp_bytes<A>(module: &Module<Self>, ct_infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Reproduces
    /// `MHEGLWEEncToShareProtocolReference::mhe_glwe_enc_to_share_share_gen_reference`:
    /// the secret share is uniform from `source_xm`, the public share is
    /// `<mask, sk> - secret` plus the `flood` noise from `source_smudge`; the
    /// mask is read only.
    /// Panics unless `mask` is canonical and has the module's degree, `sk` has
    /// the mask's degree and rank, both shares have rank 0 and the mask's
    /// layout, and `flood` is valid at the share's precision, all before any
    /// draw.
    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_enc_to_share_share_gen<P, C, S>(
        module: &Module<Self>,
        public: &mut MHEGLWEEncToShareShareOwned<Self>,
        secret: &mut P,
        mask: &C,
        sk: &S,
        flood: Noise,
        source_xm: &mut Source,
        source_smudge: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        P: GLWEToBackendMut<Self> + GLWEInfos,
        C: GLWEMaskToBackendRef<Self> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<Self> + GLWEInfos;

    /// Reproduces
    /// `MHEGLWEEncToShareProtocolReference::mhe_glwe_enc_to_share_share_aggregate_reference`:
    /// adds `a` into `res` without normalizing and combines the noise with
    /// `aggregate_metadata`.
    /// Panics unless the layouts agree.
    fn mhe_glwe_enc_to_share_share_aggregate(
        module: &Module<Self>,
        res: &mut MHEGLWEEncToShareShareOwned<Self>,
        a: &MHEGLWEEncToShareShareOwned<Self>,
    );

    /// Upper bound on the scratch of `mhe_glwe_enc_to_share_share_finalize`, as
    /// the reference.
    fn mhe_glwe_enc_to_share_share_finalize_tmp_bytes(module: &Module<Self>) -> usize;

    /// Reproduces
    /// `MHEGLWEEncToShareProtocolReference::mhe_glwe_enc_to_share_share_finalize_reference`:
    /// `secret += ct body + public`, normalized.
    /// Panics unless the ciphertext and the public share have the same layout
    /// and both additive shares have rank 0.
    fn mhe_glwe_enc_to_share_share_finalize<P, C>(
        module: &Module<Self>,
        secret: &mut P,
        ct: &C,
        public: &MHEGLWEEncToShareShareOwned<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        P: GLWEToBackendMut<Self> + GLWEInfos,
        C: GLWEToBackendRef<Self> + GLWEInfos;
}

/// # Safety
/// Every method reproduces the reference or derived body its doc names, checks
/// included, and stays within the scratch its `_tmp_bytes` query returns.
/// Share generation wipes that scratch before returning.
/// The memory, layout, synchrony and aliasing rules of
/// `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait MHEGLWEShareToEncProtocolImpl: MHEGLWEPatCompressedImpl {
    /// Upper bound on the scratch of `mhe_glwe_share_to_enc_share_gen`, as
    /// `MHEGLWEShareToEncProtocolReference::mhe_glwe_share_to_enc_share_gen_tmp_bytes_reference`.
    fn mhe_glwe_share_to_enc_share_gen_tmp_bytes<A, B>(module: &Module<Self>, res_infos: &A, secret_infos: &B) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos;

    /// Reproduces
    /// `MHEGLWEShareToEncProtocolReference::mhe_glwe_share_to_enc_share_gen_reference`:
    /// a compressed encryption of the additive `secret` under `sk`, mask drawn
    /// from `seed`, errors from `source_xe`.
    /// Panics unless `secret` has rank 0, is at most as precise as the output
    /// and shares its layout with the plaintext.
    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_share_to_enc_share_gen<P, S>(
        module: &Module<Self>,
        res: &mut MHEGLWEShareToEncShareOwned<Self>,
        secret: &P,
        sk: &S,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        P: GLWEToBackendRef<Self> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<Self> + GLWEInfos;

    /// Default body: the `MHEGLWEPatCompressed` aggregation of the inner shares.
    fn mhe_glwe_share_to_enc_share_aggregate(
        module: &Module<Self>,
        res: &mut MHEGLWEShareToEncShareOwned<Self>,
        a: &MHEGLWEShareToEncShareOwned<Self>,
    ) {
        Self::mhe_glwe_pat_compressed_aggregate_assign(module, &mut res.inner, &a.inner)
    }

    /// Default body: the scratch of the `MHEGLWEPatCompressed` finalization.
    fn mhe_glwe_share_to_enc_share_finalize_tmp_bytes(module: &Module<Self>) -> usize {
        Self::mhe_glwe_pat_compressed_finalize_tmp_bytes(module)
    }

    /// Default body: the `MHEGLWEPatCompressed` finalization of the inner share.
    fn mhe_glwe_share_to_enc_share_finalize<R>(
        module: &Module<Self>,
        res: &mut R,
        share: &MHEGLWEShareToEncShareOwned<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
    {
        Self::mhe_glwe_pat_compressed_finalize(module, res, &share.inner, scratch)
    }
}

/// Selects the reference encryption-to-shares share generation, aggregation and
/// finalization.
#[macro_export]
macro_rules! impl_mhe_glwe_enc_to_share_protocol_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::MHEGLWEEncToShareProtocolImpl for $be {
            fn mhe_glwe_enc_to_share_share_gen_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, ct_infos: &A) -> usize
            where
                A: ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGLWEEncToShareProtocolReference<$be>>::mhe_glwe_enc_to_share_share_gen_tmp_bytes_reference(module, ct_infos)
            }

            fn mhe_glwe_enc_to_share_share_gen<P, C, S>(
                module: &::poulpy_hal::layouts::Module<$be>,
                public: &mut $crate::layouts::MHEGLWEEncToShareShareOwned<$be>,
                secret: &mut P,
                mask: &C,
                sk: &S,
                flood: ::poulpy_core::Noise,
                source_xm: &mut ::poulpy_hal::source::Source,
                source_smudge: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                P: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                C: ::poulpy_core::layouts::GLWEMaskToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                S: ::poulpy_core::layouts::GLWESecretPreparedToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGLWEEncToShareProtocolReference<$be>>::mhe_glwe_enc_to_share_share_gen_reference(module, public, secret, mask, sk, flood, source_xm, source_smudge, scratch)
            }

            fn mhe_glwe_enc_to_share_share_aggregate(module: &::poulpy_hal::layouts::Module<$be>, res: &mut $crate::layouts::MHEGLWEEncToShareShareOwned<$be>, a: &$crate::layouts::MHEGLWEEncToShareShareOwned<$be>) {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGLWEEncToShareProtocolReference<$be>>::mhe_glwe_enc_to_share_share_aggregate_reference(module, res, a)
            }

            fn mhe_glwe_enc_to_share_share_finalize_tmp_bytes(module: &::poulpy_hal::layouts::Module<$be>) -> usize {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGLWEEncToShareProtocolReference<$be>>::mhe_glwe_enc_to_share_share_finalize_tmp_bytes_reference(module)
            }

            fn mhe_glwe_enc_to_share_share_finalize<P, C>(
                module: &::poulpy_hal::layouts::Module<$be>,
                secret: &mut P,
                ct: &C,
                public: &$crate::layouts::MHEGLWEEncToShareShareOwned<$be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                P: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                C: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGLWEEncToShareProtocolReference<$be>>::mhe_glwe_enc_to_share_share_finalize_reference(module, secret, ct, public, scratch)
            }
        }
    };
}

/// Selects the reference shares-to-encryption share generation; aggregation and
/// finalization keep their derived defaults.
#[macro_export]
macro_rules! impl_mhe_glwe_share_to_enc_protocol_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::MHEGLWEShareToEncProtocolImpl for $be {
            fn mhe_glwe_share_to_enc_share_gen_tmp_bytes<A, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res_infos: &A,
                secret_infos: &B,
            ) -> usize
            where
                A: ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGLWEShareToEncProtocolReference<$be>>::mhe_glwe_share_to_enc_share_gen_tmp_bytes_reference(module, res_infos, secret_infos)
            }

            fn mhe_glwe_share_to_enc_share_gen<P, S>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::MHEGLWEShareToEncShareOwned<$be>,
                secret: &P,
                sk: &S,
                seed: [u8; 32],
                source_xe: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                P: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                S: ::poulpy_core::layouts::GLWESecretPreparedToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGLWEShareToEncProtocolReference<$be>>::mhe_glwe_share_to_enc_share_gen_reference(module, res, secret, sk, seed, source_xe, scratch)
            }
        }
    };
}
