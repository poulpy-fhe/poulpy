use poulpy_core::{
    GetDistribution, GetDistributionMut,
    layouts::{GLWEInfos, GLWEPublicKeyAtViewMut, GLWEPublicKeyToBackendMut, GLWESecretPreparedToBackendRef},
    oep::EncryptionImpl,
};
use poulpy_hal::{
    layouts::{Module, ScratchArena},
    oep::HalMatZnxImpl,
    source::Source,
};

use super::derived::public_key as derived;
use crate::layouts::GLWEPublicKeyShareOwned;

/// # Safety
/// Every method reproduces the reference or derived body its doc names, checks
/// included, and stays within the scratch its `_tmp_bytes` query returns.
/// Share generation wipes that scratch before returning.
/// The memory, layout, synchrony and aliasing rules of
/// `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GLWEPublicKeyMHEProtocolImpl: HalMatZnxImpl + EncryptionImpl {
    /// Upper bound on the scratch of `mhe_glwe_public_key_share_gen`, as `GLWEPublicKeyMHEProtocolReference::mhe_glwe_public_key_share_gen_tmp_bytes_reference`.
    fn mhe_glwe_public_key_share_gen_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Reproduces `GLWEPublicKeyMHEProtocolReference::mhe_glwe_public_key_share_gen_reference`: one compressed encryption of zero per entry, entry masks drawn from `seed`, errors from `source_xe`.
    /// Panics unless `res` and `sk` have the module's degree and the share's rank and the secret's distribution is samplable.
    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_public_key_share_gen<S>(
        module: &Module<Self>,
        res: &mut GLWEPublicKeyShareOwned<Self>,
        sk: &S,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        S: GLWESecretPreparedToBackendRef<Self> + GetDistribution;

    /// Default body `mhe_glwe_public_key_share_aggregate_derived`: adds the entries and combines the noise with `aggregate_metadata`.
    /// Panics unless the layouts, secret distributions and entry seeds agree.
    fn mhe_glwe_public_key_share_aggregate(
        module: &Module<Self>,
        res: &mut GLWEPublicKeyShareOwned<Self>,
        a: &GLWEPublicKeyShareOwned<Self>,
    ) {
        derived::mhe_glwe_public_key_share_aggregate_derived(module, res, a)
    }

    /// Default body: the scratch of the `MatZnx` normalization.
    fn mhe_glwe_public_key_share_finalize_tmp_bytes(module: &Module<Self>) -> usize {
        Self::mat_znx_normalize_tmp_bytes(module)
    }

    /// Default body `mhe_glwe_public_key_share_finalize_derived`: regenerates every entry mask from its seed, normalizes the bodies and copies the distribution into `res`.
    /// Panics unless the distribution is samplable, the entry seeds are pairwise distinct and the layouts agree.
    fn mhe_glwe_public_key_share_finalize<R>(
        module: &Module<Self>,
        res: &mut R,
        share: &GLWEPublicKeyShareOwned<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEPublicKeyAtViewMut<Self> + GLWEPublicKeyToBackendMut<Self> + GetDistributionMut + GLWEInfos,
    {
        derived::mhe_glwe_public_key_share_finalize_derived(module, res, share, scratch)
    }
}

/// Selects the reference share generation; aggregation and finalization keep
/// their derived defaults.
#[macro_export]
macro_rules! impl_glwe_public_key_mhe_protocol_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::GLWEPublicKeyMHEProtocolImpl for $be {
            fn mhe_glwe_public_key_share_gen_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEPublicKeyMHEProtocolReference<$be>>::mhe_glwe_public_key_share_gen_tmp_bytes_reference(module, infos)
            }

            fn mhe_glwe_public_key_share_gen<S>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GLWEPublicKeyShareOwned<$be>,
                sk: &S,
                seed: [u8; 32],
                source_xe: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                S: ::poulpy_core::layouts::GLWESecretPreparedToBackendRef<$be> + ::poulpy_core::GetDistribution,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEPublicKeyMHEProtocolReference<$be>>::mhe_glwe_public_key_share_gen_reference(module, res, sk, seed, source_xe, scratch)
            }
        }
    };
}
