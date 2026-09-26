use poulpy_core::{
    Distribution, EncryptionInfos, GetDistributionMut,
    layouts::{GLWEInfos, GLWESecretPreparedToBackendRef, GLWEToBackendMut},
};
use poulpy_hal::{
    layouts::{Module, ScratchArena},
    source::Source,
};

use crate::{layouts::GLWEPatCompressedOwned, oep::PatFinalizeImpl};

/// # Safety
/// Reproduce the reference share, mask seed included, within the queried
/// scratch budget.
pub unsafe trait GLWEPublicKeyShareImpl: PatFinalizeImpl {
    fn glwe_public_key_share_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_public_key_share<S, E>(
        module: &Module<Self>,
        res: &mut GLWEPatCompressedOwned<Self>,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        S: GLWESecretPreparedToBackendRef<Self>,
        E: EncryptionInfos;

    fn glwe_public_key_finalize_tmp_bytes(module: &Module<Self>) -> usize {
        super::derived::public_key::glwe_public_key_finalize_tmp_bytes_derived(module)
    }

    fn glwe_public_key_finalize<R>(
        module: &Module<Self>,
        res: &mut R,
        pat: &GLWEPatCompressedOwned<Self>,
        dist: Distribution,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GetDistributionMut + GLWEInfos,
    {
        super::derived::public_key::glwe_public_key_finalize_derived(module, res, pat, dist, scratch)
    }
}

/// Selects the reference collective public key share; finalization keeps its
/// derived default.
#[macro_export]
macro_rules! impl_mhe_public_key_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::GLWEPublicKeyShareImpl for $be {
            fn glwe_public_key_share_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEPublicKeyShareReference<$be>>::glwe_public_key_share_tmp_bytes_reference(module, infos)
            }

            fn glwe_public_key_share<S, E>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut $crate::layouts::GLWEPatCompressedOwned<$be>,
                sk: &S,
                seed: [u8; 32],
                enc_infos: &E,
                source_xe: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                S: ::poulpy_core::layouts::GLWESecretPreparedToBackendRef<$be>,
                E: ::poulpy_core::EncryptionInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEPublicKeyShareReference<$be>>::glwe_public_key_share_reference(module, res, sk, seed, enc_infos, source_xe, scratch)
            }
        }
    };
}
