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
        GGLWEInfos, GGLWEToBackendMut, GLWESecret, GLWESecretPreparedFactory, GLWESecretPreparedToBackendRef, LWEInfos,
        LWESecretToBackendRef, Rank,
    },
};

#[doc(hidden)]
pub trait LWEToGLWESwitchingKeyEncryptSkDefault<BE: Backend> {
    fn lwe_to_glwe_key_encrypt_sk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn lwe_to_glwe_key_encrypt_sk_default<R, S1, S2, E>(
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
        S2: GLWESecretPreparedToBackendRef<BE>,
        E: EncryptionInfos,
        R: GGLWEToBackendMut<BE> + GGLWEInfos;
}

impl<BE: Backend> LWEToGLWESwitchingKeyEncryptSkDefault<BE> for Module<BE>
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
    fn lwe_to_glwe_key_encrypt_sk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(
            infos.rank_in(),
            Rank(1),
            "rank_in != 1 is not supported for LWEToGLWEKeyPrepared"
        );
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = GLWESecret::bytes_of(self.n().into(), Rank(1));
        let lvl_1: usize = GLWESecret::bytes_of(self.n().into(), Rank(1));

        lvl_0 + lvl_1
    }

    #[allow(clippy::too_many_arguments)]
    fn lwe_to_glwe_key_encrypt_sk_default<R, S1, S2, E>(
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
        S2: GLWESecretPreparedToBackendRef<BE>,
        E: EncryptionInfos,
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
    {
        let sk_lwe = sk_lwe.to_backend_ref();

        assert!(sk_lwe.n().0 <= self.n() as u32);
        assert!(
            scratch.available() >= self.lwe_to_glwe_key_encrypt_sk_tmp_bytes_default(res),
            "scratch.available(): {} < LWEToGLWESwitchingKeyEncryptSk::lwe_to_glwe_key_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.lwe_to_glwe_key_encrypt_sk_tmp_bytes_default(res)
        );

        let scratch = scratch.borrow();
        let (mut sk_lwe_as_glwe_src, scratch_1) = scratch.take_glwe_secret_scratch(self.n().into(), Rank(1));
        let (mut sk_lwe_as_glwe, _scratch_2) = scratch_1.take_glwe_secret_scratch(self.n().into(), Rank(1));

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

        let mut enc_scratch: ScratchOwned<BE> = ScratchOwned::alloc(self.gglwe_encrypt_sk_tmp_bytes(res));
        let sk_lwe_as_glwe_data = &mut sk_lwe_as_glwe.data;
        self.gglwe_encrypt_sk(
            res,
            &sk_lwe_as_glwe_data,
            sk_glwe,
            enc_infos,
            source_xe,
            source_xa,
            &mut enc_scratch.arena(),
        );
    }
}
