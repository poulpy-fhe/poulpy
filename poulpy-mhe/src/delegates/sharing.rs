use poulpy_core::{
    Noise,
    layouts::{GLWEInfos, GLWEMaskToBackendRef, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{
    api::{MHEGLWEEncToShareProtocol, MHEGLWEShareToEncProtocol},
    layouts::{MHEGLWEEncToShareShareOwned, MHEGLWEShareToEncShareOwned},
    oep::{MHEGLWEEncToShareProtocolImpl, MHEGLWEShareToEncProtocolImpl},
};

impl<BE: Backend + MHEGLWEEncToShareProtocolImpl> MHEGLWEEncToShareProtocol<BE> for Module<BE> {
    fn mhe_glwe_enc_to_share_share_gen_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::mhe_glwe_enc_to_share_share_gen_tmp_bytes(self, ct_infos)
    }

    fn mhe_glwe_enc_to_share_share_gen<P, C, S>(
        &self,
        public: &mut MHEGLWEEncToShareShareOwned<BE>,
        secret: &mut P,
        mask: &C,
        sk: &S,
        flood: Noise,
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

    fn mhe_glwe_enc_to_share_share_aggregate(
        &self,
        res: &mut MHEGLWEEncToShareShareOwned<BE>,
        a: &MHEGLWEEncToShareShareOwned<BE>,
    ) {
        BE::mhe_glwe_enc_to_share_share_aggregate(self, res, a)
    }

    fn mhe_glwe_enc_to_share_share_finalize_tmp_bytes(&self) -> usize {
        BE::mhe_glwe_enc_to_share_share_finalize_tmp_bytes(self)
    }

    fn mhe_glwe_enc_to_share_share_finalize<P, C>(
        &self,
        secret: &mut P,
        ct: &C,
        public: &MHEGLWEEncToShareShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::mhe_glwe_enc_to_share_share_finalize(self, secret, ct, public, scratch)
    }
}

impl<BE: Backend + MHEGLWEShareToEncProtocolImpl> MHEGLWEShareToEncProtocol<BE> for Module<BE> {
    fn mhe_glwe_share_to_enc_share_gen_tmp_bytes<A, B>(&self, res_infos: &A, secret_infos: &B) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos,
    {
        BE::mhe_glwe_share_to_enc_share_gen_tmp_bytes(self, res_infos, secret_infos)
    }

    fn mhe_glwe_share_to_enc_share_gen<P, S>(
        &self,
        res: &mut MHEGLWEShareToEncShareOwned<BE>,
        secret: &P,
        sk: &S,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
    {
        BE::mhe_glwe_share_to_enc_share_gen(self, res, secret, sk, seed, source_xe, scratch)
    }

    fn mhe_glwe_share_to_enc_share_aggregate(
        &self,
        res: &mut MHEGLWEShareToEncShareOwned<BE>,
        a: &MHEGLWEShareToEncShareOwned<BE>,
    ) {
        BE::mhe_glwe_share_to_enc_share_aggregate(self, res, a)
    }

    fn mhe_glwe_share_to_enc_share_finalize_tmp_bytes(&self) -> usize {
        BE::mhe_glwe_share_to_enc_share_finalize_tmp_bytes(self)
    }

    fn mhe_glwe_share_to_enc_share_finalize<R>(
        &self,
        res: &mut R,
        share: &MHEGLWEShareToEncShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
    {
        BE::mhe_glwe_share_to_enc_share_finalize(self, res, share, scratch)
    }
}
