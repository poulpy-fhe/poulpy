use poulpy_core::{
    EncryptionInfos,
    layouts::{GLWEInfos, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef},
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
    };
}
