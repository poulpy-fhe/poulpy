use poulpy_core::{
    Distribution, EncryptionInfos, GetDistributionMut,
    layouts::{
        GLWECompressedSeed, GLWECompressedSeedMut, GLWECompressedToBackendMut, GLWECompressedToBackendRef, GLWEInfos,
        GLWEPublicKeyAtViewMut, GLWESecretPreparedToBackendRef,
    },
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{api::GLWEPublicKeyShare, oep::GLWEPublicKeyShareImpl};

impl<BE: Backend + GLWEPublicKeyShareImpl> GLWEPublicKeyShare<BE> for Module<BE> {
    fn glwe_public_key_share_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::glwe_public_key_share_tmp_bytes(self, infos)
    }

    fn glwe_public_key_share<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWECompressedToBackendMut<BE> + GLWECompressedSeedMut + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        E: EncryptionInfos,
    {
        BE::glwe_public_key_share(self, res, sk, seed, enc_infos, source_xe, scratch)
    }

    fn glwe_public_key_finalize_tmp_bytes(&self) -> usize {
        BE::glwe_public_key_finalize_tmp_bytes(self)
    }

    fn glwe_public_key_finalize<R, P>(&self, res: &mut R, pats: &[P], dist: Distribution, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEPublicKeyAtViewMut<BE> + GetDistributionMut + GLWEInfos,
        P: GLWECompressedToBackendRef<BE> + GLWECompressedSeed + GLWEInfos,
    {
        BE::glwe_public_key_finalize(self, res, pats, dist, scratch)
    }
}
