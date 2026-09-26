use poulpy_core::{
    EncryptionInfos, GetDistribution,
    layouts::{GGLWEInfos, GGLWEToBackendMut, GLWEInfos, GLWESecretToBackendRef, GLWESwitchingKeyDegreesMut, SetGaloisElement},
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

    fn glwe_switching_key_share_aggregate_assign(
        &self,
        res: &mut GLWESwitchingKeyPatCompressedOwned<BE>,
        a: &GLWESwitchingKeyPatCompressedOwned<BE>,
    ) {
        BE::glwe_switching_key_share_aggregate_assign(self, res, a)
    }

    fn glwe_switching_key_share_normalize_tmp_bytes(&self) -> usize {
        BE::glwe_switching_key_share_normalize_tmp_bytes(self)
    }

    fn glwe_switching_key_share_normalize_assign(
        &self,
        res: &mut GLWESwitchingKeyPatCompressedOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        BE::glwe_switching_key_share_normalize_assign(self, res, scratch)
    }

    fn glwe_switching_key_finalize_tmp_bytes(&self) -> usize {
        BE::glwe_switching_key_finalize_tmp_bytes(self)
    }

    fn glwe_switching_key_finalize<R>(
        &self,
        res: &mut R,
        pat: &GLWESwitchingKeyPatCompressedOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
    {
        BE::glwe_switching_key_finalize(self, res, pat, scratch)
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

    fn glwe_automorphism_key_share_aggregate_assign(
        &self,
        res: &mut GLWEAutomorphismKeyPatCompressedOwned<BE>,
        a: &GLWEAutomorphismKeyPatCompressedOwned<BE>,
    ) {
        BE::glwe_automorphism_key_share_aggregate_assign(self, res, a)
    }

    fn glwe_automorphism_key_share_normalize_tmp_bytes(&self) -> usize {
        BE::glwe_automorphism_key_share_normalize_tmp_bytes(self)
    }

    fn glwe_automorphism_key_share_normalize_assign(
        &self,
        res: &mut GLWEAutomorphismKeyPatCompressedOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        BE::glwe_automorphism_key_share_normalize_assign(self, res, scratch)
    }

    fn glwe_automorphism_key_finalize_tmp_bytes(&self) -> usize {
        BE::glwe_automorphism_key_finalize_tmp_bytes(self)
    }

    fn glwe_automorphism_key_finalize<R>(
        &self,
        res: &mut R,
        pat: &GLWEAutomorphismKeyPatCompressedOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + SetGaloisElement,
    {
        BE::glwe_automorphism_key_finalize(self, res, pat, scratch)
    }
}
