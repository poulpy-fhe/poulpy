use poulpy_core::{
    Distribution, GLWEPublicKeyCompressedGenerate, GetDistribution,
    layouts::{GLWEInfos, GLWESecretPreparedToBackendRef, LWEInfos},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::layouts::GLWEPublicKeyShareOwned;

pub trait GLWEPublicKeyMHEProtocolReference<BE: Backend> {
    fn mhe_glwe_public_key_share_gen_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_public_key_share_gen_reference<S>(
        &self,
        res: &mut GLWEPublicKeyShareOwned<BE>,
        sk: &S,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretPreparedToBackendRef<BE> + GetDistribution;
}

impl<BE: Backend> GLWEPublicKeyMHEProtocolReference<BE> for Module<BE>
where
    Self: GLWEPublicKeyCompressedGenerate<BE>,
{
    fn mhe_glwe_public_key_share_gen_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        assert!(
            infos.n().as_usize() == self.n(),
            "invalid layout: degree differs from the module's"
        );
        self.glwe_public_key_compressed_generate_tmp_bytes(infos)
    }

    fn mhe_glwe_public_key_share_gen_reference<S>(
        &self,
        res: &mut GLWEPublicKeyShareOwned<BE>,
        sk: &S,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S: GLWESecretPreparedToBackendRef<BE> + GetDistribution,
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
        self.glwe_public_key_compressed_generate(&mut res.key, sk, seed, source_xe, scratch);
    }
}
