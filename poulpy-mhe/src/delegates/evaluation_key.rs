use poulpy_core::{
    EncryptionInfos, GetDistribution,
    layouts::{GGLWEInfos, GGLWEToBackendMut, GLWEInfos, GLWESecretToBackendRef, GLWESwitchingKeyDegreesMut, SetGaloisElement},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{
    api::{GLWEAutomorphismKeyMHEProtocol, GLWESwitchingKeyMHEProtocol},
    layouts::{GLWEAutomorphismKeyShareOwned, GLWESwitchingKeyShareOwned},
    oep::{GLWEAutomorphismKeyMHEProtocolImpl, GLWESwitchingKeyMHEProtocolImpl},
};

impl<BE: Backend + GLWESwitchingKeyMHEProtocolImpl> GLWESwitchingKeyMHEProtocol<BE> for Module<BE> {
    fn glwe_switching_key_gen_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_switching_key_gen_tmp_bytes(self, infos)
    }

    fn glwe_switching_key_gen<S1, S2, E>(
        &self,
        res: &mut GLWESwitchingKeyShareOwned<BE>,
        sk_in: &S1,
        sk_out: &S2,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S1: GLWESecretToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
        E: EncryptionInfos,
    {
        BE::glwe_switching_key_gen(self, res, sk_in, sk_out, seed, enc_infos, source_xe, scratch)
    }

    fn glwe_switching_key_aggregate(&self, res: &mut GLWESwitchingKeyShareOwned<BE>, a: &GLWESwitchingKeyShareOwned<BE>) {
        BE::glwe_switching_key_aggregate(self, res, a)
    }

    fn glwe_switching_key_finalize_tmp_bytes(&self) -> usize {
        BE::glwe_switching_key_finalize_tmp_bytes(self)
    }

    fn glwe_switching_key_finalize<R>(
        &self,
        res: &mut R,
        share: &GLWESwitchingKeyShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
    {
        BE::glwe_switching_key_finalize(self, res, share, scratch)
    }
}

impl<BE: Backend + GLWEAutomorphismKeyMHEProtocolImpl> GLWEAutomorphismKeyMHEProtocol<BE> for Module<BE> {
    fn glwe_automorphism_key_gen_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_automorphism_key_gen_tmp_bytes(self, infos)
    }

    fn glwe_automorphism_key_gen<S, E>(
        &self,
        res: &mut GLWEAutomorphismKeyShareOwned<BE>,
        p: i64,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos,
    {
        BE::glwe_automorphism_key_gen(self, res, p, sk, seed, enc_infos, source_xe, scratch)
    }

    fn glwe_automorphism_key_aggregate(
        &self,
        res: &mut GLWEAutomorphismKeyShareOwned<BE>,
        a: &GLWEAutomorphismKeyShareOwned<BE>,
    ) {
        BE::glwe_automorphism_key_aggregate(self, res, a)
    }

    fn glwe_automorphism_key_finalize_tmp_bytes(&self) -> usize {
        BE::glwe_automorphism_key_finalize_tmp_bytes(self)
    }

    fn glwe_automorphism_key_finalize<R>(
        &self,
        res: &mut R,
        share: &GLWEAutomorphismKeyShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + SetGaloisElement,
    {
        BE::glwe_automorphism_key_finalize(self, res, share, scratch)
    }
}
