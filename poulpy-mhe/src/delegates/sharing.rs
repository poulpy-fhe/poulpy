use poulpy_core::{
    EncryptionInfos, SmudgingNoise,
    layouts::{GLWEInfos, GLWEMaskToBackendRef, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{
    api::{GLWEEncToShareMHEProtocol, GLWEShareToEncMHEProtocol},
    layouts::{GLWEEncToShareShareOwned, GLWEShareToEncShareOwned},
    oep::{GLWEEncToShareMHEProtocolImpl, GLWEShareToEncMHEProtocolImpl},
};

impl<BE: Backend + GLWEEncToShareMHEProtocolImpl> GLWEEncToShareMHEProtocol<BE> for Module<BE> {
    fn mhe_glwe_enc_to_share_share_gen_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::mhe_glwe_enc_to_share_share_gen_tmp_bytes(self, ct_infos)
    }

    fn mhe_glwe_enc_to_share_share_gen<P, C, S>(
        &self,
        public: &mut GLWEEncToShareShareOwned<BE>,
        secret: &mut P,
        mask: &C,
        sk: &S,
        flood: SmudgingNoise,
        source_xm: &mut Source,
        source_smudge: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEMaskToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
    {
        BE::mhe_glwe_enc_to_share_share_gen(self, public, secret, mask, sk, flood, source_xm, source_smudge, scratch)
    }

    fn mhe_glwe_enc_to_share_share_aggregate(&self, res: &mut GLWEEncToShareShareOwned<BE>, a: &GLWEEncToShareShareOwned<BE>) {
        BE::mhe_glwe_enc_to_share_share_aggregate(self, res, a)
    }

    fn mhe_glwe_enc_to_share_share_finalize_tmp_bytes(&self) -> usize {
        BE::mhe_glwe_enc_to_share_share_finalize_tmp_bytes(self)
    }

    fn mhe_glwe_enc_to_share_share_finalize<P, C>(
        &self,
        secret: &mut P,
        ct: &C,
        public: &GLWEEncToShareShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::mhe_glwe_enc_to_share_share_finalize(self, secret, ct, public, scratch)
    }
}

impl<BE: Backend + GLWEShareToEncMHEProtocolImpl> GLWEShareToEncMHEProtocol<BE> for Module<BE> {
    fn mhe_glwe_share_to_enc_share_gen_tmp_bytes<A, B>(&self, res_infos: &A, secret_infos: &B) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos,
    {
        BE::mhe_glwe_share_to_enc_share_gen_tmp_bytes(self, res_infos, secret_infos)
    }

    fn mhe_glwe_share_to_enc_share_gen<P, S, E>(
        &self,
        res: &mut GLWEShareToEncShareOwned<BE>,
        secret: &P,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos,
    {
        BE::mhe_glwe_share_to_enc_share_gen(self, res, secret, sk, seed, enc_infos, source_xe, scratch)
    }

    fn mhe_glwe_share_to_enc_share_aggregate(&self, res: &mut GLWEShareToEncShareOwned<BE>, a: &GLWEShareToEncShareOwned<BE>) {
        BE::mhe_glwe_share_to_enc_share_aggregate(self, res, a)
    }

    fn mhe_glwe_share_to_enc_share_finalize_tmp_bytes(&self) -> usize {
        BE::mhe_glwe_share_to_enc_share_finalize_tmp_bytes(self)
    }

    fn mhe_glwe_share_to_enc_share_finalize<R>(
        &self,
        res: &mut R,
        share: &GLWEShareToEncShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
    {
        BE::mhe_glwe_share_to_enc_share_finalize(self, res, share, scratch)
    }
}
