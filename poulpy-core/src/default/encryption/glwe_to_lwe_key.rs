use poulpy_hal::{
    api::{ModuleN, ScratchOwnedAlloc, VecZnxAutomorphismBackend, VecZnxCopyRangeBackend, VecZnxZeroBackend},
    layouts::{
        Backend, Module, ScratchArena, ScratchOwned, scalar_znx_as_vec_znx_backend_mut_from_mut,
        scalar_znx_as_vec_znx_backend_ref_from_mut, scalar_znx_as_vec_znx_backend_ref_from_ref,
    },
    source::Source,
};

use crate::{
    EncryptionInfos, GGLWEEncryptSk, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GLWESecret, GLWESecretToBackendRef, LWEInfos, LWESecretToBackendRef, Rank,
        prepared::GLWESecretPreparedFactory,
    },
};

#[doc(hidden)]
pub trait GLWEToLWESwitchingKeyEncryptSkDefault<BE: Backend> {
    fn glwe_to_lwe_key_encrypt_sk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_to_lwe_key_encrypt_sk_default<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_lwe: &S1,
        sk_glwe: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S1: LWESecretToBackendRef<BE>,
        S2: GLWESecretToBackendRef<BE>,
        E: EncryptionInfos,
        R: GGLWEToBackendMut<BE> + GGLWEInfos;
}

impl<BE: Backend> GLWEToLWESwitchingKeyEncryptSkDefault<BE> for Module<BE>
where
    Self: ModuleN
        + GGLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxAutomorphismBackend<BE>
        + VecZnxCopyRangeBackend<BE>
        + VecZnxZeroBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn glwe_to_lwe_key_encrypt_sk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = self.glwe_secret_prepared_bytes_of(infos.rank_in());
        let lvl_1_sk_lwe_as_glwe_src: usize = GLWESecret::bytes_of(self.n().into(), Rank(1));
        let lvl_2_sk_lwe_as_glwe: usize = GLWESecret::bytes_of(self.n().into(), Rank(1));

        lvl_0 + lvl_1_sk_lwe_as_glwe_src + lvl_2_sk_lwe_as_glwe
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_to_lwe_key_encrypt_sk_default<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_lwe: &S1,
        sk_glwe: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S1: LWESecretToBackendRef<BE>,
        S2: GLWESecretToBackendRef<BE>,
        E: EncryptionInfos,
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
    {
        let sk_lwe = sk_lwe.to_backend_ref();
        let sk_glwe = sk_glwe.to_backend_ref();

        assert!(sk_lwe.n().0 <= self.n() as u32);
        assert!(
            scratch.available() >= self.glwe_to_lwe_key_encrypt_sk_tmp_bytes_default(res),
            "scratch.available(): {} < GLWEToLWESwitchingKeyEncryptSk::glwe_to_lwe_key_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.glwe_to_lwe_key_encrypt_sk_tmp_bytes_default(res)
        );

        let scratch = scratch.borrow();
        let (mut sk_lwe_as_glwe_prep, scratch_1) = scratch.take_glwe_secret_prepared_scratch(self, Rank(1));
        let (mut sk_lwe_as_glwe_src, scratch_2) = scratch_1.take_glwe_secret_scratch(self.n().into(), Rank(1));
        let (mut sk_lwe_as_glwe, _scratch_3) = scratch_2.take_glwe_secret_scratch(self.n().into(), Rank(1));

        sk_lwe_as_glwe_src.dist = sk_lwe.dist;
        sk_lwe_as_glwe.dist = sk_lwe.dist;
        {
            let mut sk_lwe_as_glwe_src_backend = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(&mut sk_lwe_as_glwe_src.data);
            let sk_lwe_backend = scalar_znx_as_vec_znx_backend_ref_from_ref::<BE>(&sk_lwe.data);
            self.vec_znx_zero_backend(&mut sk_lwe_as_glwe_src_backend, 0);
            self.vec_znx_copy_range_backend(
                &mut sk_lwe_as_glwe_src_backend,
                0,
                0,
                0,
                &sk_lwe_backend,
                0,
                0,
                0,
                sk_lwe.n().into(),
            );
        }
        {
            let sk_lwe_as_glwe_src_backend = scalar_znx_as_vec_znx_backend_ref_from_mut::<BE>(&sk_lwe_as_glwe_src.data);
            let mut sk_lwe_as_glwe_backend = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(&mut sk_lwe_as_glwe.data);
            self.vec_znx_automorphism_backend(-1, &mut sk_lwe_as_glwe_backend, 0, &sk_lwe_as_glwe_src_backend, 0);
        }
        self.glwe_secret_prepare(&mut sk_lwe_as_glwe_prep, &sk_lwe_as_glwe);

        let mut enc_scratch: ScratchOwned<BE> = ScratchOwned::alloc(self.gglwe_encrypt_sk_tmp_bytes(res));
        let sk_glwe_data_ref = &sk_glwe.data;
        self.gglwe_encrypt_sk(
            res,
            &sk_glwe_data_ref,
            &sk_lwe_as_glwe_prep,
            enc_infos,
            source_xe,
            source_xa,
            &mut enc_scratch.arena(),
        );
    }
}
