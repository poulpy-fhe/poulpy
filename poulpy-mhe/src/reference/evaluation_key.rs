use poulpy_core::{
    EncryptionInfos, GLWEAutomorphismKeyCompressedEncryptSk, GLWESwitchingKeyCompressedEncryptSk, GetDistribution,
    layouts::{GGLWEInfos, GLWEInfos, GLWESecretToBackendRef, LWEInfos},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::layouts::{GLWEAutomorphismKeyShareOwned, GLWESwitchingKeyShareOwned};

pub trait GLWESwitchingKeyMHEProtocolReference<BE: Backend> {
    fn mhe_glwe_switching_key_share_gen_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_switching_key_share_gen_reference<S1, S2, E>(
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
        E: EncryptionInfos;
}

impl<BE: Backend> GLWESwitchingKeyMHEProtocolReference<BE> for Module<BE>
where
    Self: GLWESwitchingKeyCompressedEncryptSk<BE>,
{
    fn mhe_glwe_switching_key_share_gen_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert!(
            infos.n().as_usize() == self.n(),
            "invalid layout: degree differs from the module's"
        );
        self.glwe_switching_key_compressed_encrypt_sk_tmp_bytes(infos)
    }

    fn mhe_glwe_switching_key_share_gen_reference<S1, S2, E>(
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

pub trait GLWEAutomorphismKeyMHEProtocolReference<BE: Backend> {
    fn mhe_glwe_automorphism_key_share_gen_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_automorphism_key_share_gen_reference<S, E>(
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
        E: EncryptionInfos;
}

impl<BE: Backend> GLWEAutomorphismKeyMHEProtocolReference<BE> for Module<BE>
where
    Self: GLWEAutomorphismKeyCompressedEncryptSk<BE>,
{
    fn mhe_glwe_automorphism_key_share_gen_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert!(
            infos.n().as_usize() == self.n(),
            "invalid layout: degree differs from the module's"
        );
        self.glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes(infos)
    }

    fn mhe_glwe_automorphism_key_share_gen_reference<S, E>(
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
