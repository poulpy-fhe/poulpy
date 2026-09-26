use poulpy_core::{
    EncryptionInfos,
    layouts::{GLWEInfos, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{api::GLWEKeyswitchShare, oep::GLWEKeyswitchShareImpl};

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
