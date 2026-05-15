#![allow(clippy::too_many_arguments)]

use poulpy_hal::{
    api::{ModuleN, VecZnxAutomorphismBackend},
    layouts::{
        Backend, GaloisElement, Module, ScratchArena, scalar_znx_as_vec_znx_backend_mut_from_mut,
        scalar_znx_as_vec_znx_backend_ref_from_ref,
    },
    source::Source,
};

use crate::{
    EncryptionInfos, GGLWECompressedEncryptSk, ScratchArenaTakeCore,
    layouts::{
        GGLWECompressedSeedMut, GGLWECompressedToBackendMut, GGLWEInfos, GLWEInfos, GLWESecret, GLWESecretPreparedFactory,
        GLWESecretToBackendRef, LWEInfos, SetGaloisElement,
    },
};

#[doc(hidden)]
pub trait GLWEAutomorphismKeyCompressedEncryptSkDefault<BE: Backend> {
    fn glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_automorphism_key_compressed_encrypt_sk_default<R, S, E>(
        &self,
        res: &mut R,
        p: i64,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeedMut + SetGaloisElement + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToBackendRef<BE> + GLWEInfos;
}

impl<BE: Backend> GLWEAutomorphismKeyCompressedEncryptSkDefault<BE> for Module<BE>
where
    Self: ModuleN + GaloisElement + VecZnxAutomorphismBackend<BE> + GGLWECompressedEncryptSk<BE> + GLWESecretPreparedFactory<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());
        let lvl_0: usize = self.glwe_secret_prepared_bytes_of_from_infos(infos);
        let lvl_1: usize = GLWESecret::bytes_of_from_infos(infos);
        let lvl_2_encrypt: usize = self.gglwe_compressed_encrypt_sk_tmp_bytes(infos);

        lvl_0 + lvl_1 + lvl_2_encrypt
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_automorphism_key_compressed_encrypt_sk_default<R, S, E>(
        &self,
        res: &mut R,
        p: i64,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeedMut + SetGaloisElement + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToBackendRef<BE> + GLWEInfos,
    {
        let sk = sk.to_backend_ref();
        assert_eq!(res.n(), sk.n());
        assert_eq!(res.rank_out(), res.rank_in());
        assert_eq!(sk.rank(), res.rank_out());
        assert!(
            scratch.available() >= self.glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes_default(res),
            "scratch.available(): {} < GLWEAutomorphismKeyCompressedEncryptSk::glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes_default(res)
        );

        let scratch = scratch.borrow();
        let (mut sk_out_prepared, scratch_1) = scratch.take_glwe_secret_prepared_scratch(self, sk.rank());
        let (mut sk_out, scratch_2) = scratch_1.take_glwe_secret_scratch(self.n().into(), sk.rank());
        sk_out.dist = sk.dist;
        {
            let sk_backend = scalar_znx_as_vec_znx_backend_ref_from_ref::<BE>(&sk.data);
            let mut sk_out_backend = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(&mut sk_out.data);
            for i in 0..sk.rank().into() {
                self.vec_znx_automorphism_backend(self.galois_element_inv(p), &mut sk_out_backend, i, &sk_backend, i);
            }
        }
        self.glwe_secret_prepare(&mut sk_out_prepared, &sk_out);

        let (mut enc_scratch, _scratch_3) = scratch_2.split_at(self.gglwe_compressed_encrypt_sk_tmp_bytes(res));
        let sk_data_ref = &sk.data;
        self.gglwe_compressed_encrypt_sk(
            res,
            &sk_data_ref,
            &sk_out_prepared,
            seed_xa,
            enc_infos,
            source_xe,
            &mut enc_scratch,
        );

        res.set_p(p);
    }
}
