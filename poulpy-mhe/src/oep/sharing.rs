use poulpy_core::{
    Noise,
    layouts::{GLWEInfos, GLWEMaskToBackendRef, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{
    layouts::{GLWEEncToShareShareOwned, GLWEShareToEncShareOwned},
    oep::GLWEPatCompressedImpl,
};

/// # Safety
/// Ciphertext outputs must reproduce the reference noise metadata and provenance
/// checks, including clearing invalidated estimates. Delegates only forward calls.
/// Reproduce the reference share and finalization within the queried scratch
/// budgets.
pub unsafe trait GLWEEncToShareMHEProtocolImpl: Backend {
    fn mhe_glwe_enc_to_share_share_gen_tmp_bytes<A>(module: &Module<Self>, ct_infos: &A) -> usize
    where
        A: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_enc_to_share_share_gen<P, C, S>(
        module: &Module<Self>,
        public: &mut GLWEEncToShareShareOwned<Self>,
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

    fn mhe_glwe_enc_to_share_share_aggregate(
        module: &Module<Self>,
        res: &mut GLWEEncToShareShareOwned<Self>,
        a: &GLWEEncToShareShareOwned<Self>,
    );

    fn mhe_glwe_enc_to_share_share_finalize_tmp_bytes(module: &Module<Self>) -> usize;

    fn mhe_glwe_enc_to_share_share_finalize<P, C>(
        module: &Module<Self>,
        secret: &mut P,
        ct: &C,
        public: &GLWEEncToShareShareOwned<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        P: GLWEToBackendMut<Self> + GLWEInfos,
        C: GLWEToBackendRef<Self> + GLWEInfos;
}

/// # Safety
/// Ciphertext outputs must reproduce the reference noise metadata and provenance
/// checks, including clearing invalidated estimates. Delegates only forward calls.
/// Reproduce the reference share within the queried scratch budget.
pub unsafe trait GLWEShareToEncMHEProtocolImpl: GLWEPatCompressedImpl {
    fn mhe_glwe_share_to_enc_share_gen_tmp_bytes<A, B>(module: &Module<Self>, res_infos: &A, secret_infos: &B) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_share_to_enc_share_gen<P, S>(
        module: &Module<Self>,
        res: &mut GLWEShareToEncShareOwned<Self>,
        secret: &P,
        sk: &S,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        P: GLWEToBackendRef<Self> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<Self> + GLWEInfos;

    fn mhe_glwe_share_to_enc_share_aggregate(
        module: &Module<Self>,
        res: &mut GLWEShareToEncShareOwned<Self>,
        a: &GLWEShareToEncShareOwned<Self>,
    ) {
        Self::glwe_pat_compressed_aggregate_assign(module, &mut res.inner, &a.inner)
    }

    fn mhe_glwe_share_to_enc_share_finalize_tmp_bytes(module: &Module<Self>) -> usize {
        Self::glwe_pat_compressed_finalize_tmp_bytes(module)
    }

    fn mhe_glwe_share_to_enc_share_finalize<R>(
        module: &Module<Self>,
        res: &mut R,
        share: &GLWEShareToEncShareOwned<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
    {
        Self::glwe_pat_compressed_finalize(module, res, &share.inner, scratch)
    }
}

/// Selects the reference encryption-to-shares and shares-to-encryption
/// operations.
#[macro_export]
macro_rules! impl_mhe_sharing_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::GLWEEncToShareMHEProtocolImpl for $be {
            fn mhe_glwe_enc_to_share_share_gen_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, ct_infos: &A) -> usize
            where
                A: ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEEncToShareMHEProtocolReference<$be>>::mhe_glwe_enc_to_share_share_gen_tmp_bytes_reference(module, ct_infos)
            }

            fn mhe_glwe_enc_to_share_share_gen<P, C, S>(
                module: &::poulpy_hal::layouts::Module<$be>,
                public: &mut $crate::layouts::GLWEEncToShareShareOwned<$be>,
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
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEEncToShareMHEProtocolReference<$be>>::mhe_glwe_enc_to_share_share_gen_reference(module, public, secret, mask, sk, flood, source_xm, source_smudge, scratch)
            }

            fn mhe_glwe_enc_to_share_share_aggregate(module: &::poulpy_hal::layouts::Module<$be>, res: &mut $crate::layouts::GLWEEncToShareShareOwned<$be>, a: &$crate::layouts::GLWEEncToShareShareOwned<$be>) {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEEncToShareMHEProtocolReference<$be>>::mhe_glwe_enc_to_share_share_aggregate_reference(module, res, a)
            }

            fn mhe_glwe_enc_to_share_share_finalize_tmp_bytes(module: &::poulpy_hal::layouts::Module<$be>) -> usize {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEEncToShareMHEProtocolReference<$be>>::mhe_glwe_enc_to_share_share_finalize_tmp_bytes_reference(module)
            }

            fn mhe_glwe_enc_to_share_share_finalize<P, C>(
                module: &::poulpy_hal::layouts::Module<$be>,
                secret: &mut P,
                ct: &C,
                public: &$crate::layouts::GLWEEncToShareShareOwned<$be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                P: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                C: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEEncToShareMHEProtocolReference<$be>>::mhe_glwe_enc_to_share_share_finalize_reference(module, secret, ct, public, scratch)
            }
        }

        unsafe impl $crate::oep::GLWEShareToEncMHEProtocolImpl for $be {
            fn mhe_glwe_share_to_enc_share_gen_tmp_bytes<A, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res_infos: &A,
                secret_infos: &B,
            ) -> usize
            where
                A: ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEShareToEncMHEProtocolReference<$be>>::mhe_glwe_share_to_enc_share_gen_tmp_bytes_reference(module, res_infos, secret_infos)
            }

            fn mhe_glwe_share_to_enc_share_gen<P, S>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GLWEShareToEncShareOwned<$be>,
                secret: &P,
                sk: &S,
                seed: [u8; 32],
                source_xe: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                P: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                S: ::poulpy_core::layouts::GLWESecretPreparedToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEShareToEncMHEProtocolReference<$be>>::mhe_glwe_share_to_enc_share_gen_reference(module, res, secret, sk, seed, source_xe, scratch)
            }
        }
    };
}
