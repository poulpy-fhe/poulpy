use crate::layouts::{MHEGLWEPrivateKeyswitchShareOwned, MHEGLWEPublicKeyswitchShareOwned};
use poulpy_core::{
    Noise,
    layouts::{
        GLWEInfos, GLWEMaskToBackendRef, GLWEPublicKeyPreparedToBackendRef, GLWESecretPreparedToBackendRef, GLWEToBackendMut,
        GLWEToBackendRef,
    },
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{
    api::{MHEGLWEPrivateKeyswitchProtocol, MHEGLWEPublicKeyswitchProtocol},
    oep::{MHEGLWEPrivateKeyswitchProtocolImpl, MHEGLWEPublicKeyswitchProtocolImpl},
};

impl<BE: Backend + MHEGLWEPrivateKeyswitchProtocolImpl> MHEGLWEPrivateKeyswitchProtocol<BE> for Module<BE> {
    fn mhe_glwe_private_keyswitch_share_gen_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::mhe_glwe_private_keyswitch_share_gen_tmp_bytes(self, infos)
    }

    fn mhe_glwe_private_keyswitch_share_gen<C, S1, S2>(
        &self,
        res: &mut MHEGLWEPrivateKeyswitchShareOwned<BE>,
        mask: &C,
        sk_in: &S1,
        sk_out: &S2,
        flood: Noise,
        source_smudge: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEMaskToBackendRef<BE> + GLWEInfos,
        S1: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
    {
        BE::mhe_glwe_private_keyswitch_share_gen(self, res, mask, sk_in, sk_out, flood, source_smudge, scratch)
    }

    fn mhe_glwe_private_keyswitch_share_aggregate(
        &self,
        res: &mut MHEGLWEPrivateKeyswitchShareOwned<BE>,
        a: &MHEGLWEPrivateKeyswitchShareOwned<BE>,
    ) {
        BE::mhe_glwe_private_keyswitch_share_aggregate(self, res, a)
    }

    fn mhe_glwe_private_keyswitch_share_finalize_tmp_bytes(&self) -> usize {
        BE::mhe_glwe_private_keyswitch_share_finalize_tmp_bytes(self)
    }

    fn mhe_glwe_private_keyswitch_share_finalize<R, C>(
        &self,
        res: &mut R,
        ct: &C,
        share: &MHEGLWEPrivateKeyswitchShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::mhe_glwe_private_keyswitch_share_finalize(self, res, ct, share, scratch)
    }
}

impl<BE: Backend + MHEGLWEPublicKeyswitchProtocolImpl> MHEGLWEPublicKeyswitchProtocol<BE> for Module<BE> {
    fn mhe_glwe_public_keyswitch_share_gen_tmp_bytes<A, B, P>(&self, ct_infos: &A, res_infos: &B, pk_infos: &P) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos,
        P: GLWEInfos,
    {
        BE::mhe_glwe_public_keyswitch_share_gen_tmp_bytes(self, ct_infos, res_infos, pk_infos)
    }

    fn mhe_glwe_public_keyswitch_share_gen<C, S, K>(
        &self,
        res: &mut MHEGLWEPublicKeyswitchShareOwned<BE>,
        mask: &C,
        sk_in: &S,
        pk_out: &K,
        flood: Noise,
        source_xu: &mut Source,
        source_xe: &mut Source,
        source_smudge: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEMaskToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
    {
        BE::mhe_glwe_public_keyswitch_share_gen(
            self,
            res,
            mask,
            sk_in,
            pk_out,
            flood,
            source_xu,
            source_xe,
            source_smudge,
            scratch,
        )
    }

    fn mhe_glwe_public_keyswitch_share_aggregate(
        &self,
        res: &mut MHEGLWEPublicKeyswitchShareOwned<BE>,
        a: &MHEGLWEPublicKeyswitchShareOwned<BE>,
    ) {
        BE::mhe_glwe_public_keyswitch_share_aggregate(self, res, a)
    }

    fn mhe_glwe_public_keyswitch_share_finalize_tmp_bytes(&self) -> usize {
        BE::mhe_glwe_public_keyswitch_share_finalize_tmp_bytes(self)
    }

    fn mhe_glwe_public_keyswitch_share_finalize<R, C>(
        &self,
        res: &mut R,
        ct: &C,
        share: &MHEGLWEPublicKeyswitchShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::mhe_glwe_public_keyswitch_share_finalize(self, res, ct, share, scratch)
    }
}
