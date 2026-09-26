use poulpy_core::{
    EncryptionInfos, GetDistribution,
    layouts::{GLWEInfos, GLWEPreparedToBackendRef, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

/// # Safety
/// Reproduce the reference share and finalization within the queried scratch
/// budgets.
pub unsafe trait GLWEKeyswitchShareImpl: Backend {
    fn glwe_keyswitch_share_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_keyswitch_share<R, C, S1, S2, E>(
        module: &Module<Self>,
        res: &mut R,
        ct: &C,
        sk_in: &S1,
        sk_out: &S2,
        flood: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        C: GLWEToBackendRef<Self> + GLWEInfos,
        S1: GLWESecretPreparedToBackendRef<Self> + GLWEInfos,
        S2: GLWESecretPreparedToBackendRef<Self> + GLWEInfos,
        E: EncryptionInfos;

    fn glwe_keyswitch_finalize_tmp_bytes(module: &Module<Self>) -> usize;

    fn glwe_keyswitch_finalize<R, C, H>(
        module: &Module<Self>,
        res: &mut R,
        ct: &C,
        share: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        C: GLWEToBackendRef<Self> + GLWEInfos,
        H: GLWEToBackendRef<Self> + GLWEInfos;
}

/// # Safety
/// Reproduce the reference share and finalization within the queried scratch
/// budgets.
pub unsafe trait GLWEPublicKeyswitchShareImpl: Backend {
    fn glwe_public_keyswitch_share_tmp_bytes<A, B, P>(module: &Module<Self>, ct_infos: &A, res_infos: &B, pk_infos: &P) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos,
        P: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_public_keyswitch_share<R, C, S, K, E1, E2>(
        module: &Module<Self>,
        res: &mut R,
        ct: &C,
        sk_in: &S,
        pk_out: &K,
        flood: &E1,
        enc_infos: &E2,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        C: GLWEToBackendRef<Self> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<Self> + GLWEInfos,
        K: GLWEPreparedToBackendRef<Self> + GetDistribution + GLWEInfos,
        E1: EncryptionInfos,
        E2: EncryptionInfos;

    fn glwe_public_keyswitch_finalize_tmp_bytes(module: &Module<Self>) -> usize;

    fn glwe_public_keyswitch_finalize<R, C, H>(
        module: &Module<Self>,
        res: &mut R,
        ct: &C,
        share: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        C: GLWEToBackendRef<Self> + GLWEInfos,
        H: GLWEToBackendRef<Self> + GLWEInfos;
}

/// Selects the reference key switching shares and finalizations.
#[macro_export]
macro_rules! impl_mhe_keyswitch_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::GLWEKeyswitchShareImpl for $be {
            fn glwe_keyswitch_share_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEKeyswitchShareReference<$be>>::glwe_keyswitch_share_tmp_bytes_reference(module, infos)
            }

            fn glwe_keyswitch_share<R, C, S1, S2, E>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                ct: &C,
                sk_in: &S1,
                sk_out: &S2,
                flood: &E,
                source_xe: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                C: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                S1: ::poulpy_core::layouts::GLWESecretPreparedToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                S2: ::poulpy_core::layouts::GLWESecretPreparedToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                E: ::poulpy_core::EncryptionInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEKeyswitchShareReference<$be>>::glwe_keyswitch_share_reference(module, res, ct, sk_in, sk_out, flood, source_xe, scratch)
            }

            fn glwe_keyswitch_finalize_tmp_bytes(module: &::poulpy_hal::layouts::Module<$be>) -> usize {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEKeyswitchShareReference<$be>>::glwe_keyswitch_finalize_tmp_bytes_reference(module)
            }

            fn glwe_keyswitch_finalize<R, C, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                ct: &C,
                share: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                C: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                H: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEKeyswitchShareReference<$be>>::glwe_keyswitch_finalize_reference(module, res, ct, share, scratch)
            }
        }

        unsafe impl $crate::oep::GLWEPublicKeyswitchShareImpl for $be {
            fn glwe_public_keyswitch_share_tmp_bytes<A, B, P>(
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
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEPublicKeyswitchShareReference<$be>>::glwe_public_keyswitch_share_tmp_bytes_reference(module, ct_infos, res_infos, pk_infos)
            }

            fn glwe_public_keyswitch_share<R, C, S, K, E1, E2>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                ct: &C,
                sk_in: &S,
                pk_out: &K,
                flood: &E1,
                enc_infos: &E2,
                source_xu: &mut ::poulpy_hal::source::Source,
                source_xe: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                C: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                S: ::poulpy_core::layouts::GLWESecretPreparedToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                K: ::poulpy_core::layouts::GLWEPreparedToBackendRef<$be>
                    + ::poulpy_core::GetDistribution
                    + ::poulpy_core::layouts::GLWEInfos,
                E1: ::poulpy_core::EncryptionInfos,
                E2: ::poulpy_core::EncryptionInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEPublicKeyswitchShareReference<$be>>::glwe_public_keyswitch_share_reference(module, res, ct, sk_in, pk_out, flood, enc_infos, source_xu, source_xe, scratch)
            }

            fn glwe_public_keyswitch_finalize_tmp_bytes(module: &::poulpy_hal::layouts::Module<$be>) -> usize {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEPublicKeyswitchShareReference<$be>>::glwe_public_keyswitch_finalize_tmp_bytes_reference(module)
            }

            fn glwe_public_keyswitch_finalize<R, C, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                ct: &C,
                share: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                C: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                H: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEPublicKeyswitchShareReference<$be>>::glwe_public_keyswitch_finalize_reference(module, res, ct, share, scratch)
            }
        }
    };
}
