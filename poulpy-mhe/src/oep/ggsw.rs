use poulpy_core::layouts::{
    GGLWEInfos, GGSWInfos, GGSWToBackendMut, GLWEInfos,
    prepared::{GGLWEPreparedToBackendRef, GLWESecretPreparedToBackendRef},
};
use poulpy_hal::{
    layouts::{Module, ScalarZnxToBackendRef, ScratchArena},
    source::Source,
};

use super::derived::ggsw as derived;
use crate::{layouts::MHEGGSWShareOwned, oep::MHEGGLWEPatCompressedImpl};

/// # Safety
/// Every method reproduces the reference or derived body its doc names, checks
/// included, and stays within the scratch its `_tmp_bytes` query returns.
/// Share generation wipes that scratch before returning.
/// The memory, layout, synchrony and aliasing rules of
/// `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait MHEGGSWProtocolImpl: MHEGGLWEPatCompressedImpl {
    /// Upper bound on the scratch of `mhe_ggsw_share_gen`, as
    /// `MHEGGSWProtocolReference::mhe_ggsw_share_gen_tmp_bytes_reference`;
    /// panics unless `infos` has the module's degree.
    fn mhe_ggsw_share_gen_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGSWInfos;

    /// Reproduces `MHEGGSWProtocolReference::mhe_ggsw_share_gen_reference`: the
    /// child seeds of the part masks derive from `seed` in the reference order,
    /// `source_xe` supplies the errors, and the message-half masks reuse the
    /// key-half seeds.
    /// Panics unless `res`, `pt` and both secrets have the module's degree, the
    /// secrets have the share's rank and `pt` has one column.
    #[allow(clippy::too_many_arguments)]
    fn mhe_ggsw_share_gen<P, S, U>(
        module: &Module<Self>,
        res: &mut MHEGGSWShareOwned<Self>,
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

    /// Default body `mhe_ggsw_share_aggregate_derived`: aggregates every part
    /// with the compressed PAT aggregation and combines the noise with
    /// `aggregate_metadata`.
    /// Panics unless the GGSW layouts and every part's mask seeds agree.
    fn mhe_ggsw_share_aggregate(module: &Module<Self>, res: &mut MHEGGSWShareOwned<Self>, a: &MHEGGSWShareOwned<Self>) {
        derived::mhe_ggsw_share_aggregate_derived(module, res, a)
    }

    /// Upper bound on the scratch of `mhe_ggsw_share_finalize`, as
    /// `MHEGGSWProtocolReference::mhe_ggsw_share_finalize_tmp_bytes_reference`;
    /// panics unless `res_infos` and `key_infos` have the module's degree.
    fn mhe_ggsw_share_finalize_tmp_bytes<R, K>(module: &Module<Self>, res_infos: &R, key_infos: &K) -> usize
    where
        R: GGSWInfos,
        K: GGLWEInfos;

    /// Reproduces
    /// `MHEGGSWProtocolReference::mhe_ggsw_share_finalize_reference`:
    /// regenerates the masks from the share's seeds, normalizes the bodies,
    /// then expands the rows with the tensor key.
    /// Panics unless the degrees are the module's, `res` has the share's
    /// layout, the key ranks equal the share's, `share.k()` is within the key's
    /// `max_input_k` (`assert_key_covers`) and the key and share are tagged
    /// with one secret.
    fn mhe_ggsw_share_finalize<R, K>(
        module: &Module<Self>,
        res: &mut R,
        share: &MHEGGSWShareOwned<Self>,
        key: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<Self> + GGLWEInfos;
}

/// Selects the reference share generation and finalization; aggregation keeps
/// its derived default.
#[macro_export]
macro_rules! impl_mhe_ggsw_protocol_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::MHEGGSWProtocolImpl for $be {
            fn mhe_ggsw_share_gen_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: ::poulpy_core::layouts::GGSWInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGGSWProtocolReference<$be>>::mhe_ggsw_share_gen_tmp_bytes_reference(module, infos)
            }

            fn mhe_ggsw_share_gen<P, S, U>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::MHEGGSWShareOwned<$be>,
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
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGGSWProtocolReference<$be>>::mhe_ggsw_share_gen_reference(module, res, pt, sk, u, seed, source_xe, scratch)
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
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGGSWProtocolReference<$be>>::mhe_ggsw_share_finalize_tmp_bytes_reference(module, res_infos, key_infos)
            }

            fn mhe_ggsw_share_finalize<R, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                share: &$crate::layouts::MHEGGSWShareOwned<$be>,
                key: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GGSWToBackendMut<$be> + ::poulpy_core::layouts::GGSWInfos,
                K: ::poulpy_core::layouts::prepared::GGLWEPreparedToBackendRef<$be> + ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::MHEGGSWProtocolReference<$be>>::mhe_ggsw_share_finalize_reference(module, res, share, key, scratch)
            }
        }
    };
}
