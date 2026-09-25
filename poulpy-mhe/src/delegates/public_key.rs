use poulpy_core::{
    Distribution, EncryptionInfos, GetDistributionMut,
    layouts::{GLWEInfos, GLWESecretPreparedToBackendRef, GLWEToBackendMut},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{api::GLWEPublicKeyShare, layouts::GLWEPatCompressedOwned, oep::GLWEPublicKeyShareImpl};

impl<BE: Backend + GLWEPublicKeyShareImpl> GLWEPublicKeyShare<BE> for Module<BE> {
    fn glwe_public_key_share_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::glwe_public_key_share_tmp_bytes(self, infos)
    }

    fn glwe_public_key_share<S, E>(
        &self,
        res: &mut GLWEPatCompressedOwned<BE>,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretPreparedToBackendRef<BE>,
        E: EncryptionInfos,
    {
        BE::glwe_public_key_share(self, res, sk, seed, enc_infos, source_xe, scratch)
    }

    fn glwe_public_key_finalize_tmp_bytes(&self) -> usize {
        BE::glwe_public_key_finalize_tmp_bytes(self)
    }

    fn glwe_public_key_finalize<R>(
        &self,
        res: &mut R,
        pat: &GLWEPatCompressedOwned<BE>,
        dist: Distribution,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GetDistributionMut + GLWEInfos,
    {
        BE::glwe_public_key_finalize(self, res, pat, dist, scratch)
    }
}
