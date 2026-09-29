use crate::layouts::{GLWEKeyswitchShareOwned, GLWEPublicKeyswitchShareOwned};
use poulpy_core::{
    EncryptionInfos, SmudgingInfos,
    layouts::{GLWEInfos, GLWEPublicKeyPreparedToBackendRef, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{
    api::{GLWEKeyswitchMHEProtocol, GLWEPublicKeyswitchMHEProtocol},
    oep::{GLWEKeyswitchMHEProtocolImpl, GLWEPublicKeyswitchMHEProtocolImpl},
};

impl<BE: Backend + GLWEKeyswitchMHEProtocolImpl> GLWEKeyswitchMHEProtocol<BE> for Module<BE> {
    fn mhe_glwe_keyswitch_share_gen_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::mhe_glwe_keyswitch_share_gen_tmp_bytes(self, infos)
    }

    fn mhe_glwe_keyswitch_share_gen<C, S1, S2, E>(
        &self,
        res: &mut GLWEKeyswitchShareOwned<BE>,
        ct: &C,
        sk_in: &S1,
        sk_out: &S2,
        flood: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEToBackendRef<BE> + GLWEInfos,
        S1: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: SmudgingInfos,
    {
        BE::mhe_glwe_keyswitch_share_gen(self, res, ct, sk_in, sk_out, flood, source_xe, scratch)
    }

    fn mhe_glwe_keyswitch_share_aggregate(&self, res: &mut GLWEKeyswitchShareOwned<BE>, a: &GLWEKeyswitchShareOwned<BE>) {
        BE::mhe_glwe_keyswitch_share_aggregate(self, res, a)
    }

    fn mhe_glwe_keyswitch_share_finalize_tmp_bytes(&self) -> usize {
        BE::mhe_glwe_keyswitch_share_finalize_tmp_bytes(self)
    }

    fn mhe_glwe_keyswitch_share_finalize<R, C>(
        &self,
        res: &mut R,
        ct: &C,
        share: &GLWEKeyswitchShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::mhe_glwe_keyswitch_share_finalize(self, res, ct, share, scratch)
    }
}

impl<BE: Backend + GLWEPublicKeyswitchMHEProtocolImpl> GLWEPublicKeyswitchMHEProtocol<BE> for Module<BE> {
    fn mhe_glwe_public_keyswitch_share_gen_tmp_bytes<A, B, P>(&self, ct_infos: &A, res_infos: &B, pk_infos: &P) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos,
        P: GLWEInfos,
    {
        BE::mhe_glwe_public_keyswitch_share_gen_tmp_bytes(self, ct_infos, res_infos, pk_infos)
    }

    fn mhe_glwe_public_keyswitch_share_gen<C, S, K, E1, E2>(
        &self,
        res: &mut GLWEPublicKeyswitchShareOwned<BE>,
        ct: &C,
        sk_in: &S,
        pk_out: &K,
        flood: &E1,
        enc_infos: &E2,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
        E1: SmudgingInfos,
        E2: EncryptionInfos,
    {
        BE::mhe_glwe_public_keyswitch_share_gen(self, res, ct, sk_in, pk_out, flood, enc_infos, source_xu, source_xe, scratch)
    }

    fn mhe_glwe_public_keyswitch_share_aggregate(
        &self,
        res: &mut GLWEPublicKeyswitchShareOwned<BE>,
        a: &GLWEPublicKeyswitchShareOwned<BE>,
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
        share: &GLWEPublicKeyswitchShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::mhe_glwe_public_keyswitch_share_finalize(self, res, ct, share, scratch)
    }
}
