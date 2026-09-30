use poulpy_core::{
    EncryptionInfos, GetDistribution,
    layouts::{GGLWEInfos, GLWEInfos, GLWESecretToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{
    api::{GLWEAutomorphismKeyShare, GLWESwitchingKeyShare},
    layouts::{GLWEAutomorphismKeyPatCompressedOwned, GLWESwitchingKeyPatCompressedOwned},
    oep::{GLWEAutomorphismKeyShareImpl, GLWESwitchingKeyShareImpl},
};

impl<BE: Backend + GLWESwitchingKeyShareImpl> GLWESwitchingKeyShare<BE> for Module<BE> {
    fn glwe_switching_key_share_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_switching_key_share_tmp_bytes(self, infos)
    }

    fn glwe_switching_key_share<S1, S2, E>(
        &self,
        res: &mut GLWESwitchingKeyPatCompressedOwned<BE>,
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
        BE::glwe_switching_key_share(self, res, sk_in, sk_out, seed, enc_infos, source_xe, scratch)
    }
}

impl<BE: Backend + GLWEAutomorphismKeyShareImpl> GLWEAutomorphismKeyShare<BE> for Module<BE> {
    fn glwe_automorphism_key_share_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_automorphism_key_share_tmp_bytes(self, infos)
    }

    fn glwe_automorphism_key_share<S, E>(
        &self,
        res: &mut GLWEAutomorphismKeyPatCompressedOwned<BE>,
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
        BE::glwe_automorphism_key_share(self, res, p, sk, seed, enc_infos, source_xe, scratch)
    }
}
