use poulpy_core::{
    Distribution, EncryptionInfos, GLWEBytesOf, GLWECompressedEncryptSk, GetDistribution, ScratchArenaTakeCore,
    layouts::{GLWEInfos, GLWESecretPreparedToBackendRef},
};
use poulpy_hal::{
    api::VecZnxZero,
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::layouts::GLWEPatCompressedOwned;

pub trait GLWEPublicKeyShareReference<BE: Backend> {
    fn glwe_public_key_share_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_public_key_share_reference<S, E>(
        &self,
        res: &mut GLWEPatCompressedOwned<BE>,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretPreparedToBackendRef<BE>,
        E: EncryptionInfos;
}

impl<BE: Backend> GLWEPublicKeyShareReference<BE> for Module<BE>
where
    Self: GLWECompressedEncryptSk<BE> + GLWEBytesOf<BE> + VecZnxZero<BE>,
{
    fn glwe_public_key_share_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::scratch_aligned(self.glwe_plaintext_bytes_of_from_infos(infos)) + self.glwe_compressed_encrypt_sk_tmp_bytes(infos)
    }

    fn glwe_public_key_share_reference<S, E>(
        &self,
        res: &mut GLWEPatCompressedOwned<BE>,
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
            !matches!(sk.to_backend_ref().dist(), Distribution::NONE | Distribution::ENCAPSULATED(_)),
            "invalid secret: a public key share needs a samplable distribution"
        );
        let infos = res.glwe_layout();
        let (mut pt, mut scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(&infos);
        self.vec_znx_zero(pt.data_mut(), 0);
        self.glwe_compressed_encrypt_sk(res, &pt, sk, seed, enc_infos, source_xe, &mut scratch_1);
        res.canonical = true;
    }
}
