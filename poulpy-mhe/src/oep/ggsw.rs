use poulpy_core::layouts::{
    GGLWEInfos, GGSWInfos, GGSWToBackendMut, GLWEInfos,
    prepared::{GGLWEPreparedToBackendRef, GLWESecretPreparedToBackendRef},
};
use poulpy_hal::{
    layouts::{Module, ScalarZnxToBackendRef, ScratchArena},
    source::Source,
};

use super::derived::ggsw as derived;
use crate::{layouts::GGSWShareOwned, oep::GGLWEPatCompressedImpl};

/// # Safety
/// Outputs follow the [noise metadata rule](crate::oep#noise-metadata). Delegates only forward calls.
/// Reproduce the reference share, mask seeds included, and the reference
/// finalized GGSW, within the queried scratch budgets.
pub unsafe trait GGSWMHEProtocolImpl: GGLWEPatCompressedImpl {
    fn mhe_ggsw_share_gen_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGSWInfos;

    #[allow(clippy::too_many_arguments)]
    fn mhe_ggsw_share_gen<P, S, U>(
        module: &Module<Self>,
        res: &mut GGSWShareOwned<Self>,
        pt: &P,
        sk: &S,
        u: &U,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        P: ScalarZnxToBackendRef<Self>,
        S: GLWESecretPreparedToBackendRef<Self> + GLWEInfos,
        U: GLWESecretPreparedToBackendRef<Self> + GLWEInfos;

    fn mhe_ggsw_share_aggregate(module: &Module<Self>, res: &mut GGSWShareOwned<Self>, a: &GGSWShareOwned<Self>) {
        derived::mhe_ggsw_share_aggregate_derived(module, res, a)
    }

    fn mhe_ggsw_share_finalize_tmp_bytes<R, K>(module: &Module<Self>, res_infos: &R, key_infos: &K) -> usize
    where
        R: GGSWInfos,
        K: GGLWEInfos;

    fn mhe_ggsw_share_finalize<R, K>(
        module: &Module<Self>,
        res: &mut R,
        share: &GGSWShareOwned<Self>,
        key: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<Self> + GGLWEInfos;
}

/// Selects the reference share generation and finalization; aggregation keeps
/// its derived default.
#[macro_export]
macro_rules! impl_mhe_ggsw_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::GGSWMHEProtocolImpl for $be {
            fn mhe_ggsw_share_gen_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: ::poulpy_core::layouts::GGSWInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GGSWMHEProtocolReference<$be>>::mhe_ggsw_share_gen_tmp_bytes_reference(module, infos)
            }

            fn mhe_ggsw_share_gen<P, S, U>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GGSWShareOwned<$be>,
                pt: &P,
                sk: &S,
                u: &U,
                seed: [u8; 32],
                source_xe: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                P: ::poulpy_hal::layouts::ScalarZnxToBackendRef<$be>,
                S: ::poulpy_core::layouts::prepared::GLWESecretPreparedToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                U: ::poulpy_core::layouts::prepared::GLWESecretPreparedToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GGSWMHEProtocolReference<$be>>::mhe_ggsw_share_gen_reference(module, res, pt, sk, u, seed, source_xe, scratch)
            }

            fn mhe_ggsw_share_finalize_tmp_bytes<R, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res_infos: &R,
                key_infos: &K,
            ) -> usize
            where
                R: ::poulpy_core::layouts::GGSWInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GGSWMHEProtocolReference<$be>>::mhe_ggsw_share_finalize_tmp_bytes_reference(module, res_infos, key_infos)
            }

            fn mhe_ggsw_share_finalize<R, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                share: &$crate::layouts::GGSWShareOwned<$be>,
                key: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GGSWToBackendMut<$be> + ::poulpy_core::layouts::GGSWInfos,
                K: ::poulpy_core::layouts::prepared::GGLWEPreparedToBackendRef<$be> + ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GGSWMHEProtocolReference<$be>>::mhe_ggsw_share_finalize_reference(module, res, share, key, scratch)
            }
        }
    };
}
