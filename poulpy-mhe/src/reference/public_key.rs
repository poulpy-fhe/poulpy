use poulpy_core::{
    Distribution, EncryptionInfos, GLWECompressedEncryptSk, GetDistribution,
    layouts::{GLWECompressedSeedMut, GLWECompressedToBackendMut, GLWEInfos, GLWESecretPreparedToBackendRef, LWEInfos},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

pub trait GLWEPublicKeyShareReference<BE: Backend> {
    fn glwe_public_key_share_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_public_key_share_reference<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWECompressedToBackendMut<BE> + GLWECompressedSeedMut + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        E: EncryptionInfos;
}

impl<BE: Backend> GLWEPublicKeyShareReference<BE> for Module<BE>
where
    Self: GLWECompressedEncryptSk<BE>,
{
    fn glwe_public_key_share_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        assert!(
            infos.n().as_usize() == self.n(),
            "invalid layout: degree differs from the module's"
        );
        self.glwe_compressed_encrypt_sk_tmp_bytes(infos)
    }

    fn glwe_public_key_share_reference<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWECompressedToBackendMut<BE> + GLWECompressedSeedMut + GLWEInfos,
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
        self.glwe_compressed_encrypt_zero_sk(res, sk, seed, enc_infos, source_xe, scratch);
    }
}
