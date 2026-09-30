use poulpy_core::{
    Distribution, EncryptionInfos, GetDistributionMut,
    layouts::{GLWEInfos, GLWEPublicKeyAtViewMut, GLWESecretPreparedToBackendRef},
};
use poulpy_hal::{
    layouts::{Module, ScratchArena},
    source::Source,
};

use super::derived::public_key as derived;
use crate::{layouts::GLWEPublicKeyShareOwned, oep::GLWEPatCompressedImpl};

/// # Safety
/// Reproduce the reference share, entry seeds included, within the queried
/// scratch budget.
pub unsafe trait GLWEPublicKeyProtocolImpl: GLWEPatCompressedImpl {
    fn glwe_public_key_gen_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_public_key_gen<S, E>(
        module: &Module<Self>,
        res: &mut GLWEPublicKeyShareOwned<Self>,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        S: GLWESecretPreparedToBackendRef<Self>,
        E: EncryptionInfos;

    fn glwe_public_key_aggregate(
        module: &Module<Self>,
        res: &mut GLWEPublicKeyShareOwned<Self>,
        a: &GLWEPublicKeyShareOwned<Self>,
    ) {
        derived::glwe_public_key_aggregate_derived(module, res, a)
    }

    fn glwe_public_key_finalize_tmp_bytes(module: &Module<Self>) -> usize {
        Self::glwe_pat_compressed_finalize_tmp_bytes(module)
    }

    fn glwe_public_key_finalize<R>(
        module: &Module<Self>,
        res: &mut R,
        share: &GLWEPublicKeyShareOwned<Self>,
        dist: Distribution,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEPublicKeyAtViewMut<Self> + GetDistributionMut + GLWEInfos,
    {
        derived::glwe_public_key_finalize_derived(module, res, share, dist, scratch)
    }
}

/// Selects the reference share generation; aggregation and finalization keep
/// their derived defaults.
#[macro_export]
macro_rules! impl_mhe_public_key_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::GLWEPublicKeyProtocolImpl for $be {
            fn glwe_public_key_gen_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEPublicKeyProtocolReference<$be>>::glwe_public_key_gen_tmp_bytes_reference(module, infos)
            }

            fn glwe_public_key_gen<S, E>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GLWEPublicKeyShareOwned<$be>,
                sk: &S,
                seed: [u8; 32],
                enc_infos: &E,
                source_xe: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                S: ::poulpy_core::layouts::GLWESecretPreparedToBackendRef<$be>,
                E: ::poulpy_core::EncryptionInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEPublicKeyProtocolReference<$be>>::glwe_public_key_gen_reference(module, res, sk, seed, enc_infos, source_xe, scratch)
            }
        }
    };
}
