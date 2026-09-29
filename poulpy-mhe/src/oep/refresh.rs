use poulpy_core::{
    EncryptionInfos, SmudgingInfos,
    layouts::{GLWEInfos, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::layouts::GLWERefreshShareOwned;

/// # Safety
/// Reproduce the reference share and finalization within the queried scratch
/// budgets.
pub unsafe trait GLWERefreshMHEProtocolImpl: Backend {
    fn mhe_glwe_refresh_share_gen_tmp_bytes<A, B>(module: &Module<Self>, ct_infos: &A, res_infos: &B) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_refresh_share_gen<C, S, E1, E2>(
        module: &Module<Self>,
        res: &mut GLWERefreshShareOwned<Self>,
        ct: &C,
        sk: &S,
        log_bound: usize,
        seed: [u8; 32],
        flood: &E1,
        enc_infos: &E2,
        source_xm: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        C: GLWEToBackendRef<Self> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<Self> + GLWEInfos,
        E1: SmudgingInfos,
        E2: EncryptionInfos;

    fn mhe_glwe_refresh_share_aggregate(
        module: &Module<Self>,
        res: &mut GLWERefreshShareOwned<Self>,
        a: &GLWERefreshShareOwned<Self>,
    );

    fn mhe_glwe_refresh_share_finalize_tmp_bytes<A>(module: &Module<Self>, res_infos: &A) -> usize
    where
        A: GLWEInfos;

    fn mhe_glwe_refresh_share_finalize<R, C>(
        module: &Module<Self>,
        res: &mut R,
        ct: &C,
        share: &GLWERefreshShareOwned<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        C: GLWEToBackendRef<Self> + GLWEInfos;
}

/// Selects the reference refresh share and finalization.
#[macro_export]
macro_rules! impl_mhe_refresh_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::GLWERefreshMHEProtocolImpl for $be {
            fn mhe_glwe_refresh_share_gen_tmp_bytes<A, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                ct_infos: &A,
                res_infos: &B,
            ) -> usize
            where
                A: ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWERefreshMHEProtocolReference<$be>>::mhe_glwe_refresh_share_gen_tmp_bytes_reference(module, ct_infos, res_infos)
            }

            fn mhe_glwe_refresh_share_gen<C, S, E1, E2>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GLWERefreshShareOwned<$be>,
                ct: &C,
                sk: &S,
                log_bound: usize,
                seed: [u8; 32],
                flood: &E1,
                enc_infos: &E2,
                source_xm: &mut ::poulpy_hal::source::Source,
                source_xe: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                C: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                S: ::poulpy_core::layouts::GLWESecretPreparedToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                E1: ::poulpy_core::SmudgingInfos,
                E2: ::poulpy_core::EncryptionInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWERefreshMHEProtocolReference<$be>>::mhe_glwe_refresh_share_gen_reference(module, res, ct, sk, log_bound, seed, flood, enc_infos, source_xm, source_xe, scratch)
            }

            fn mhe_glwe_refresh_share_aggregate(module: &::poulpy_hal::layouts::Module<$be>, res: &mut $crate::layouts::GLWERefreshShareOwned<$be>, a: &$crate::layouts::GLWERefreshShareOwned<$be>) {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWERefreshMHEProtocolReference<$be>>::mhe_glwe_refresh_share_aggregate_reference(module, res, a)
            }

            fn mhe_glwe_refresh_share_finalize_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, res_infos: &A) -> usize
            where
                A: ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWERefreshMHEProtocolReference<$be>>::mhe_glwe_refresh_share_finalize_tmp_bytes_reference(module, res_infos)
            }

            fn mhe_glwe_refresh_share_finalize<R, C>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                ct: &C,
                share: &$crate::layouts::GLWERefreshShareOwned<$be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                C: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWERefreshMHEProtocolReference<$be>>::mhe_glwe_refresh_share_finalize_reference(module, res, ct, share, scratch)
            }
        }
    };
}
