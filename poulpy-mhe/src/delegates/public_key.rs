use poulpy_core::{
    GetDistribution, GetDistributionMut,
    layouts::{GLWEInfos, GLWEPublicKeyAtViewMut, GLWEPublicKeyToBackendMut, GLWESecretPreparedToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{api::MHEGLWEPublicKeyProtocol, layouts::MHEGLWEPublicKeyShareOwned, oep::MHEGLWEPublicKeyProtocolImpl};

impl<BE: Backend + MHEGLWEPublicKeyProtocolImpl> MHEGLWEPublicKeyProtocol<BE> for Module<BE> {
    fn mhe_glwe_public_key_share_gen_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::mhe_glwe_public_key_share_gen_tmp_bytes(self, infos)
    }

    fn mhe_glwe_public_key_share_gen<S>(
        &self,
        res: &mut MHEGLWEPublicKeyShareOwned<BE>,
        sk: &S,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretPreparedToBackendRef<BE> + GetDistribution,
    {
        BE::mhe_glwe_public_key_share_gen(self, res, sk, seed, source_xe, scratch)
    }

    fn mhe_glwe_public_key_share_aggregate(&self, res: &mut MHEGLWEPublicKeyShareOwned<BE>, a: &MHEGLWEPublicKeyShareOwned<BE>) {
        BE::mhe_glwe_public_key_share_aggregate(self, res, a)
    }

    fn mhe_glwe_public_key_share_finalize_tmp_bytes(&self) -> usize {
        BE::mhe_glwe_public_key_share_finalize_tmp_bytes(self)
    }

    fn mhe_glwe_public_key_share_finalize<R>(
        &self,
        res: &mut R,
        share: &MHEGLWEPublicKeyShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEPublicKeyAtViewMut<BE> + GLWEPublicKeyToBackendMut<BE> + GetDistributionMut + GLWEInfos,
    {
        BE::mhe_glwe_public_key_share_finalize(self, res, share, scratch)
    }
}
