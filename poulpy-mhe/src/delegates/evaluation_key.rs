use poulpy_core::{
    GetDistribution,
    layouts::{GGLWEInfos, GGLWEToBackendMut, GLWEInfos, GLWESecretToBackendRef, GLWESwitchingKeyDegreesMut, SetGaloisElement},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{
    api::{MHEGLWEAutomorphismKeyProtocol, MHEGLWESwitchingKeyProtocol},
    layouts::{MHEGLWEAutomorphismKeyShareOwned, MHEGLWESwitchingKeyShareOwned},
    oep::{MHEGLWEAutomorphismKeyProtocolImpl, MHEGLWESwitchingKeyProtocolImpl},
};

impl<BE: Backend + MHEGLWESwitchingKeyProtocolImpl> MHEGLWESwitchingKeyProtocol<BE> for Module<BE> {
    fn mhe_glwe_switching_key_share_gen_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::mhe_glwe_switching_key_share_gen_tmp_bytes(self, infos)
    }

    fn mhe_glwe_switching_key_share_gen<S1, S2>(
        &self,
        res: &mut MHEGLWESwitchingKeyShareOwned<BE>,
        sk_in: &S1,
        sk_out: &S2,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S1: GLWESecretToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        BE::mhe_glwe_switching_key_share_gen(self, res, sk_in, sk_out, seed, source_xe, scratch)
    }

    fn mhe_glwe_switching_key_share_aggregate(
        &self,
        res: &mut MHEGLWESwitchingKeyShareOwned<BE>,
        a: &MHEGLWESwitchingKeyShareOwned<BE>,
    ) {
        BE::mhe_glwe_switching_key_share_aggregate(self, res, a)
    }

    fn mhe_glwe_switching_key_share_finalize_tmp_bytes(&self) -> usize {
        BE::mhe_glwe_switching_key_share_finalize_tmp_bytes(self)
    }

    fn mhe_glwe_switching_key_share_finalize<R>(
        &self,
        res: &mut R,
        share: &MHEGLWESwitchingKeyShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
    {
        BE::mhe_glwe_switching_key_share_finalize(self, res, share, scratch)
    }
}

impl<BE: Backend + MHEGLWEAutomorphismKeyProtocolImpl> MHEGLWEAutomorphismKeyProtocol<BE> for Module<BE> {
    fn mhe_glwe_automorphism_key_share_gen_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::mhe_glwe_automorphism_key_share_gen_tmp_bytes(self, infos)
    }

    fn mhe_glwe_automorphism_key_share_gen<S>(
        &self,
        res: &mut MHEGLWEAutomorphismKeyShareOwned<BE>,
        p: i64,
        sk: &S,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretToBackendRef<BE> + GLWEInfos,
    {
        BE::mhe_glwe_automorphism_key_share_gen(self, res, p, sk, seed, source_xe, scratch)
    }

    fn mhe_glwe_automorphism_key_share_aggregate(
        &self,
        res: &mut MHEGLWEAutomorphismKeyShareOwned<BE>,
        a: &MHEGLWEAutomorphismKeyShareOwned<BE>,
    ) {
        BE::mhe_glwe_automorphism_key_share_aggregate(self, res, a)
    }

    fn mhe_glwe_automorphism_key_share_finalize_tmp_bytes(&self) -> usize {
        BE::mhe_glwe_automorphism_key_share_finalize_tmp_bytes(self)
    }

    fn mhe_glwe_automorphism_key_share_finalize<R>(
        &self,
        res: &mut R,
        share: &MHEGLWEAutomorphismKeyShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + SetGaloisElement,
    {
        BE::mhe_glwe_automorphism_key_share_finalize(self, res, share, scratch)
    }
}
