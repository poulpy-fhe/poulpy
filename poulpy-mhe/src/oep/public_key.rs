use poulpy_core::{
    Distribution, EncryptionInfos, GetDistributionMut,
    layouts::{
        GLWECompressedSeed, GLWECompressedSeedMut, GLWECompressedToBackendMut, GLWECompressedToBackendRef, GLWEInfos,
        GLWEPublicKeyAtViewMut, GLWESecretPreparedToBackendRef,
    },
};
use poulpy_hal::{
    layouts::{Module, ScratchArena},
    source::Source,
};

use crate::oep::GLWEPatCompressedImpl;

/// # Safety
/// Reproduce the reference share, mask seed included, within the queried
/// scratch budget.
pub unsafe trait GLWEPublicKeyShareImpl: GLWEPatCompressedImpl {
    fn glwe_public_key_share_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_public_key_share<R, S, E>(
        module: &Module<Self>,
        res: &mut R,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWECompressedToBackendMut<Self> + GLWECompressedSeedMut + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<Self>,
        E: EncryptionInfos;

    fn glwe_public_key_finalize_tmp_bytes(module: &Module<Self>) -> usize {
        super::derived::public_key::glwe_public_key_finalize_tmp_bytes_derived(module)
    }

    fn glwe_public_key_finalize<R, P>(
        module: &Module<Self>,
        res: &mut R,
        pats: &[P],
        dist: Distribution,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEPublicKeyAtViewMut<Self> + GetDistributionMut + GLWEInfos,
        P: GLWECompressedToBackendRef<Self> + GLWECompressedSeed + GLWEInfos,
    {
        super::derived::public_key::glwe_public_key_finalize_derived(module, res, pats, dist, scratch)
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

            fn glwe_public_key_share<R, S, E>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                sk: &S,
                seed: [u8; 32],
                enc_infos: &E,
                source_xe: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWECompressedToBackendMut<$be>
                    + ::poulpy_core::layouts::GLWECompressedSeedMut
                    + ::poulpy_core::layouts::GLWEInfos,
                S: ::poulpy_core::layouts::GLWESecretPreparedToBackendRef<$be>,
                E: ::poulpy_core::EncryptionInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::GLWEPublicKeyShareReference<$be>>::glwe_public_key_share_reference(module, res, sk, seed, enc_infos, source_xe, scratch)
            }
        }
    };
}
