use poulpy_core::{
    Distribution, EncryptionInfos, GLWECompressedEncryptSk, GetDistribution,
    layouts::{GLWEInfos, GLWESecretPreparedToBackendRef, LWEInfos},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::layouts::GLWEPublicKeyShareOwned;

pub trait GLWEPublicKeyMHEProtocolReference<BE: Backend> {
    fn mhe_glwe_public_key_gen_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_public_key_gen_reference<S, E>(
        &self,
        res: &mut GLWEPublicKeyShareOwned<BE>,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretPreparedToBackendRef<BE>,
        E: EncryptionInfos;
}

impl<BE: Backend> GLWEPublicKeyMHEProtocolReference<BE> for Module<BE>
where
    Self: GLWECompressedEncryptSk<BE>,
{
    fn mhe_glwe_public_key_gen_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        assert!(
            infos.n().as_usize() == self.n(),
            "invalid layout: degree differs from the module's"
        );
        self.glwe_compressed_encrypt_sk_tmp_bytes(infos)
    }

    fn mhe_glwe_public_key_gen_reference<S, E>(
        &self,
        res: &mut GLWEPublicKeyShareOwned<BE>,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretPreparedToBackendRef<BE>,
        E: EncryptionInfos,
    {
        assert!(
            res.n().as_usize() == self.n(),
            "invalid share: degree differs from the module's"
        );
        let sk_ref = sk.to_backend_ref();
        assert!(sk_ref.n() == res.n(), "invalid share: secret degree differs from the share's");
        assert!(
            sk_ref.rank() == res.rank(),
            "invalid share: secret rank differs from the share's"
        );
        assert!(
            !matches!(sk_ref.dist(), Distribution::NONE | Distribution::ENCAPSULATED(_)),
            "invalid secret: a public key share needs a samplable distribution"
        );
        res.dist = *sk_ref.dist();
        // Every party derives the same distinct entry seeds from `seed`.
        let mut seeds = Source::new(seed);
        for entry in res.entries.iter_mut() {
            self.glwe_compressed_encrypt_zero_sk(entry, sk, seeds.new_seed(), enc_infos, source_xe, scratch);
        }
    }
}
