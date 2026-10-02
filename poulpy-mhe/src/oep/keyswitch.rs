use crate::layouts::{GLWEKeyswitchShareOwned, GLWEPublicKeyswitchShareOwned};
use poulpy_core::{
    EncryptionInfos, SmudgingNoise,
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
/// Reproduce the reference share and finalization within the queried scratch
/// budgets.
pub unsafe trait GLWEKeyswitchMHEProtocolImpl: Backend {
    fn mhe_glwe_keyswitch_share_gen_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_keyswitch_share_gen<C, S1, S2>(
        module: &Module<Self>,
        res: &mut GLWEKeyswitchShareOwned<Self>,
        mask: &C,
        sk_in: &S1,
        sk_out: &S2,
        flood: SmudgingNoise,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        C: GLWEMaskToBackendRef<Self> + GLWEInfos,
        S1: GLWESecretPreparedToBackendRef<Self> + GLWEInfos,
        S2: GLWESecretPreparedToBackendRef<Self> + GLWEInfos;

    fn mhe_glwe_keyswitch_share_aggregate(
        module: &Module<Self>,
        res: &mut GLWEKeyswitchShareOwned<Self>,
        a: &GLWEKeyswitchShareOwned<Self>,
    );

    fn mhe_glwe_keyswitch_share_finalize_tmp_bytes(module: &Module<Self>) -> usize;

    fn mhe_glwe_keyswitch_share_finalize<R, C>(
        module: &Module<Self>,
        res: &mut R,
        ct: &C,
        share: &GLWEKeyswitchShareOwned<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        C: GLWEToBackendRef<Self> + GLWEInfos;
}

/// # Safety
/// Reproduce the reference share and finalization within the queried scratch
/// budgets.
pub unsafe trait GLWEPublicKeyswitchMHEProtocolImpl: Backend {
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

    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_public_keyswitch_share_gen<C, S, K, E>(
        module: &Module<Self>,
        res: &mut GLWEPublicKeyswitchShareOwned<Self>,
        mask: &C,
        sk_in: &S,
        pk_out: &K,
        flood: SmudgingNoise,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        C: GLWEMaskToBackendRef<Self> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<Self> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<Self> + GLWEInfos,
        E: EncryptionInfos;

    fn mhe_glwe_public_keyswitch_share_aggregate(
        module: &Module<Self>,
        res: &mut GLWEPublicKeyswitchShareOwned<Self>,
        a: &GLWEPublicKeyswitchShareOwned<Self>,
    );

    fn mhe_glwe_public_keyswitch_share_finalize_tmp_bytes(module: &Module<Self>) -> usize;

    fn mhe_glwe_public_keyswitch_share_finalize<R, C>(
        module: &Module<Self>,
        res: &mut R,
        ct: &C,
        share: &GLWEPublicKeyswitchShareOwned<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        C: GLWEToBackendRef<Self> + GLWEInfos;
}

/// Selects the reference key switching shares and finalizations.
#[macro_export]
macro_rules! impl_mhe_keyswitch_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::GLWEKeyswitchMHEProtocolImpl for $be {
            fn mhe_glwe_keyswitch_share_gen_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEKeyswitchMHEProtocolReference<$be>>::mhe_glwe_keyswitch_share_gen_tmp_bytes_reference(module, infos)
            }

            fn mhe_glwe_keyswitch_share_gen<C, S1, S2>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GLWEKeyswitchShareOwned<$be>,
                mask: &C,
                sk_in: &S1,
                sk_out: &S2,
                flood: ::poulpy_core::SmudgingNoise,
                source_xe: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                C: ::poulpy_core::layouts::GLWEMaskToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                S1: ::poulpy_core::layouts::GLWESecretPreparedToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                S2: ::poulpy_core::layouts::GLWESecretPreparedToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEKeyswitchMHEProtocolReference<$be>>::mhe_glwe_keyswitch_share_gen_reference(module, res, mask, sk_in, sk_out, flood, source_xe, scratch)
            }

            fn mhe_glwe_keyswitch_share_aggregate(module: &::poulpy_hal::layouts::Module<$be>, res: &mut $crate::layouts::GLWEKeyswitchShareOwned<$be>, a: &$crate::layouts::GLWEKeyswitchShareOwned<$be>) {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEKeyswitchMHEProtocolReference<$be>>::mhe_glwe_keyswitch_share_aggregate_reference(module, res, a)
            }

            fn mhe_glwe_keyswitch_share_finalize_tmp_bytes(module: &::poulpy_hal::layouts::Module<$be>) -> usize {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEKeyswitchMHEProtocolReference<$be>>::mhe_glwe_keyswitch_share_finalize_tmp_bytes_reference(module)
            }

            fn mhe_glwe_keyswitch_share_finalize<R, C>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                ct: &C,
                share: &$crate::layouts::GLWEKeyswitchShareOwned<$be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                C: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEKeyswitchMHEProtocolReference<$be>>::mhe_glwe_keyswitch_share_finalize_reference(module, res, ct, share, scratch)
            }
        }

        unsafe impl $crate::oep::GLWEPublicKeyswitchMHEProtocolImpl for $be {
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
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEPublicKeyswitchMHEProtocolReference<$be>>::mhe_glwe_public_keyswitch_share_gen_tmp_bytes_reference(module, ct_infos, res_infos, pk_infos)
            }

            fn mhe_glwe_public_keyswitch_share_gen<C, S, K, E>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GLWEPublicKeyswitchShareOwned<$be>,
                mask: &C,
                sk_in: &S,
                pk_out: &K,
                flood: ::poulpy_core::SmudgingNoise,
                enc_infos: &E,
                source_xu: &mut ::poulpy_hal::source::Source,
                source_xe: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                C: ::poulpy_core::layouts::GLWEMaskToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                S: ::poulpy_core::layouts::GLWESecretPreparedToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                K: ::poulpy_core::layouts::GLWEPublicKeyPreparedToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                E: ::poulpy_core::EncryptionInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEPublicKeyswitchMHEProtocolReference<$be>>::mhe_glwe_public_keyswitch_share_gen_reference(module, res, mask, sk_in, pk_out, flood, enc_infos, source_xu, source_xe, scratch)
            }

            fn mhe_glwe_public_keyswitch_share_aggregate(module: &::poulpy_hal::layouts::Module<$be>, res: &mut $crate::layouts::GLWEPublicKeyswitchShareOwned<$be>, a: &$crate::layouts::GLWEPublicKeyswitchShareOwned<$be>) {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEPublicKeyswitchMHEProtocolReference<$be>>::mhe_glwe_public_keyswitch_share_aggregate_reference(module, res, a)
            }

            fn mhe_glwe_public_keyswitch_share_finalize_tmp_bytes(module: &::poulpy_hal::layouts::Module<$be>) -> usize {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEPublicKeyswitchMHEProtocolReference<$be>>::mhe_glwe_public_keyswitch_share_finalize_tmp_bytes_reference(module)
            }

            fn mhe_glwe_public_keyswitch_share_finalize<R, C>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                ct: &C,
                share: &$crate::layouts::GLWEPublicKeyswitchShareOwned<$be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                C: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEPublicKeyswitchMHEProtocolReference<$be>>::mhe_glwe_public_keyswitch_share_finalize_reference(module, res, ct, share, scratch)
            }
        }
    };
}
