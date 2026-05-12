use poulpy_hal::{
    api::{ScratchOwnedAlloc, SvpPPolBytesOf, VecZnxAutomorphismBackend},
    layouts::{
        Backend, GaloisElement, Module, ScratchArena, ScratchOwned, scalar_znx_as_vec_znx_backend_mut_from_mut,
        scalar_znx_as_vec_znx_backend_ref_from_ref,
    },
    source::Source,
};

use crate::{
    EncryptionInfos, GGLWEEncryptSk, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GLWEInfos, GLWESecret, GLWESecretPreparedFactory, GLWESecretToBackendRef, LWEInfos,
        SetGaloisElement,
    },
};

#[doc(hidden)]
pub trait GLWEAutomorphismKeyEncryptSkDefault<BE: Backend> {
    fn glwe_automorphism_key_encrypt_sk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_automorphism_key_encrypt_sk_default<R, S, E>(
        &self,
        res: &mut R,
        p: i64,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + SetGaloisElement + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToBackendRef<BE> + GLWEInfos;
}

impl<BE: Backend> GLWEAutomorphismKeyEncryptSkDefault<BE> for Module<BE>
where
    Self: GGLWEEncryptSk<BE> + VecZnxAutomorphismBackend<BE> + GaloisElement + SvpPPolBytesOf + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn glwe_automorphism_key_encrypt_sk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEAutomorphismKey"
        );
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = self.glwe_secret_prepared_bytes_of_from_infos(infos);
        let lvl_1_sk: usize = GLWESecret::bytes_of_from_infos(infos);
        let lvl_1: usize = lvl_1_sk;

        lvl_0 + lvl_1
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_automorphism_key_encrypt_sk_default<R, S, E>(
        &self,
        res: &mut R,
        p: i64,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,

        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + SetGaloisElement + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToBackendRef<BE> + GLWEInfos,
    {
        let sk = sk.to_backend_ref();

        assert_eq!(res.n(), sk.n());
        assert_eq!(res.rank_out(), res.rank_in());
        assert_eq!(sk.rank(), res.rank_out());
        assert!(
            scratch.available() >= self.glwe_automorphism_key_encrypt_sk_tmp_bytes_default(res),
            "scratch.available(): {} < GLWEAutomorphismKeyEncryptSk::glwe_automorphism_key_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.glwe_automorphism_key_encrypt_sk_tmp_bytes_default(res)
        );

        let scratch = scratch.borrow();
        let (mut sk_out_prepared, scratch_1) = scratch.take_glwe_secret_prepared_scratch(self, sk.rank());
        let (mut sk_out, _scratch_2) = scratch_1.take_glwe_secret_scratch(self.n().into(), sk.rank());
        sk_out.dist = sk.dist;
        {
            let sk_backend = scalar_znx_as_vec_znx_backend_ref_from_ref::<BE>(&sk.data);
            let mut sk_out_backend = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(&mut sk_out.data);
            for i in 0..sk.rank().into() {
                self.vec_znx_automorphism_backend(self.galois_element_inv(p), &mut sk_out_backend, i, &sk_backend, i);
            }
        }
        self.glwe_secret_prepare(&mut sk_out_prepared, &sk_out);

        let mut enc_scratch: ScratchOwned<BE> = ScratchOwned::alloc(self.gglwe_encrypt_sk_tmp_bytes(res));
        let sk_data_ref = &sk.data;
        self.gglwe_encrypt_sk(
            res,
            &sk_data_ref,
            &sk_out_prepared,
            enc_infos,
            source_xe,
            source_xa,
            &mut enc_scratch.arena(),
        );

        res.set_p(p);
    }
}

#[doc(hidden)]
pub trait GLWEAutomorphismKeyEncryptPkDefault<BE: Backend> {
    fn glwe_automorphism_key_encrypt_pk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;
}

impl<BE: Backend> GLWEAutomorphismKeyEncryptPkDefault<BE> for Module<BE>
where
    Self:,
{
    fn glwe_automorphism_key_encrypt_pk_tmp_bytes_default<A>(&self, _infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        unimplemented!()
    }
}
