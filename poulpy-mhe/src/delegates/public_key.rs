use poulpy_core::{
    EncryptionInfos, GetDistributionMut,
    layouts::{GLWEInfos, GLWEPublicKeyAtViewMut, GLWESecretPreparedToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{api::GLWEPublicKeyMHEProtocol, layouts::GLWEPublicKeyShareOwned, oep::GLWEPublicKeyMHEProtocolImpl};

impl<BE: Backend + GLWEPublicKeyMHEProtocolImpl> GLWEPublicKeyMHEProtocol<BE> for Module<BE> {
    fn glwe_public_key_gen_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::glwe_public_key_gen_tmp_bytes(self, infos)
    }

    fn glwe_public_key_gen<S, E>(
        &self,
        res: &mut GLWEPublicKeyShareOwned<BE>,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretPreparedToBackendRef<BE>,
        E: EncryptionInfos,
    {
        BE::glwe_public_key_gen(self, res, sk, seed, enc_infos, source_xe, scratch)
    }

    fn glwe_public_key_aggregate(&self, res: &mut GLWEPublicKeyShareOwned<BE>, a: &GLWEPublicKeyShareOwned<BE>) {
        BE::glwe_public_key_aggregate(self, res, a)
    }

    fn glwe_public_key_finalize_tmp_bytes(&self) -> usize {
        BE::glwe_public_key_finalize_tmp_bytes(self)
    }

    fn glwe_public_key_finalize<R>(&self, res: &mut R, share: &GLWEPublicKeyShareOwned<BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEPublicKeyAtViewMut<BE> + GetDistributionMut + GLWEInfos,
    {
        BE::glwe_public_key_finalize(self, res, share, scratch)
    }
}
