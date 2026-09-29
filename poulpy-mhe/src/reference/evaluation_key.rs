use poulpy_core::{
    EncryptionInfos, GLWEAutomorphismKeyCompressedEncryptSk, GLWESwitchingKeyCompressedEncryptSk, GetDistribution,
    layouts::{GGLWEInfos, GLWEInfos, GLWESecretToBackendRef, LWEInfos},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::layouts::{GLWEAutomorphismKeyPatCompressedOwned, GLWESwitchingKeyPatCompressedOwned};

pub trait GLWESwitchingKeyShareReference<BE: Backend> {
    fn glwe_switching_key_share_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_switching_key_share_reference<S1, S2, E>(
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
        E: EncryptionInfos;
}

impl<BE: Backend> GLWESwitchingKeyShareReference<BE> for Module<BE>
where
    Self: GLWESwitchingKeyCompressedEncryptSk<BE>,
{
    fn glwe_switching_key_share_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert!(
            infos.n().as_usize() == self.n(),
            "invalid layout: degree differs from the module's"
        );
        self.glwe_switching_key_compressed_encrypt_sk_tmp_bytes(infos)
    }

    fn glwe_switching_key_share_reference<S1, S2, E>(
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
        assert!(
            res.n().as_usize() == self.n(),
            "invalid share: degree differs from the module's"
        );
        assert!(
            sk_in.n().as_usize().is_power_of_two() && sk_in.n() <= res.n(),
            "invalid share: unsupported input secret degree"
        );
        assert!(
            sk_out.n().as_usize().is_power_of_two() && sk_out.n() <= res.n(),
            "invalid share: unsupported output secret degree"
        );
        assert!(
            sk_in.rank() == res.rank_in(),
            "invalid share: input secret rank differs from the key's"
        );
        assert!(
            sk_out.rank() == res.rank_out(),
            "invalid share: output secret rank differs from the key's"
        );
        self.glwe_switching_key_compressed_encrypt_sk(res, sk_in, sk_out, seed, enc_infos, source_xe, scratch);
    }
}

pub trait GLWEAutomorphismKeyShareReference<BE: Backend> {
    fn glwe_automorphism_key_share_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_automorphism_key_share_reference<S, E>(
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
        E: EncryptionInfos;
}

impl<BE: Backend> GLWEAutomorphismKeyShareReference<BE> for Module<BE>
where
    Self: GLWEAutomorphismKeyCompressedEncryptSk<BE>,
{
    fn glwe_automorphism_key_share_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert!(
            infos.n().as_usize() == self.n(),
            "invalid layout: degree differs from the module's"
        );
        self.glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes(infos)
    }

    fn glwe_automorphism_key_share_reference<S, E>(
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
        assert!(
            res.n().as_usize() == self.n(),
            "invalid share: degree differs from the module's"
        );
        assert!(sk.n() == res.n(), "invalid share: secret degree differs from the key's");
        assert!(
            res.rank_in() == res.rank_out(),
            "invalid share: automorphism key input and output ranks differ"
        );
        assert!(
            sk.rank() == res.rank_out(),
            "invalid share: secret rank differs from the key's"
        );
        assert!(p & 1 != 0, "invalid share: Galois element must be odd");
        self.glwe_automorphism_key_compressed_encrypt_sk(res, p, sk, seed, enc_infos, source_xe, scratch);
    }
}
