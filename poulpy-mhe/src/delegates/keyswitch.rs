use poulpy_core::{
    EncryptionInfos, GetDistribution,
    layouts::{GLWEInfos, GLWEPreparedToBackendRef, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{
    api::{GLWEKeyswitchShare, GLWEPublicKeyswitchShare},
    oep::{GLWEKeyswitchShareImpl, GLWEPublicKeyswitchShareImpl},
};

impl<BE: Backend + GLWEKeyswitchShareImpl> GLWEKeyswitchShare<BE> for Module<BE> {
    fn glwe_keyswitch_share_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::glwe_keyswitch_share_tmp_bytes(self, infos)
    }

    fn glwe_keyswitch_share<R, C, S1, S2, E>(
        &self,
        res: &mut R,
        ct: &C,
        sk_in: &S1,
        sk_out: &S2,
        flood: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
        S1: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos,
    {
        BE::glwe_keyswitch_share(self, res, ct, sk_in, sk_out, flood, source_xe, scratch)
    }

    fn glwe_keyswitch_finalize_tmp_bytes(&self) -> usize {
        BE::glwe_keyswitch_finalize_tmp_bytes(self)
    }

    fn glwe_keyswitch_finalize<R, C, H>(&self, res: &mut R, ct: &C, share: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
        H: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_keyswitch_finalize(self, res, ct, share, scratch)
    }
}

impl<BE: Backend + GLWEPublicKeyswitchShareImpl> GLWEPublicKeyswitchShare<BE> for Module<BE> {
    fn glwe_public_keyswitch_share_tmp_bytes<A, B, P>(&self, ct_infos: &A, res_infos: &B, pk_infos: &P) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos,
        P: GLWEInfos,
    {
        BE::glwe_public_keyswitch_share_tmp_bytes(self, ct_infos, res_infos, pk_infos)
    }

    fn glwe_public_keyswitch_share<R, C, S, K, E1, E2>(
        &self,
        res: &mut R,
        ct: &C,
        sk_in: &S,
        pk_out: &K,
        flood: &E1,
        enc_infos: &E2,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        K: GLWEPreparedToBackendRef<BE> + GetDistribution + GLWEInfos,
        E1: EncryptionInfos,
        E2: EncryptionInfos,
    {
        BE::glwe_public_keyswitch_share(self, res, ct, sk_in, pk_out, flood, enc_infos, source_xu, source_xe, scratch)
    }

    fn glwe_public_keyswitch_finalize_tmp_bytes(&self) -> usize {
        BE::glwe_public_keyswitch_finalize_tmp_bytes(self)
    }

    fn glwe_public_keyswitch_finalize<R, C, H>(&self, res: &mut R, ct: &C, share: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
        H: GLWEToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_public_keyswitch_finalize(self, res, ct, share, scratch)
    }
}
