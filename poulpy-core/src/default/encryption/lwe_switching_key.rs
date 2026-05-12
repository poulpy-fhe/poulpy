use poulpy_hal::{
    api::{ModuleN, VecZnxAutomorphismBackend, VecZnxCopyRangeBackend, VecZnxZeroBackend},
    layouts::{
        Backend, Module, ScratchArena, scalar_znx_as_vec_znx_backend_mut_from_mut, scalar_znx_as_vec_znx_backend_ref_from_mut,
        scalar_znx_as_vec_znx_backend_ref_from_ref,
    },
    source::Source,
};

use crate::{
    EncryptionInfos, ScratchArenaTakeCore,
    encryption::glwe_switching_key::GLWESwitchingKeyEncryptSk,
    layouts::{GGLWEInfos, GGLWEToBackendMut, GLWESecret, GLWESwitchingKeyDegreesMut, LWEInfos, LWESecretToBackendRef, Rank},
};

#[doc(hidden)]
pub trait LWESwitchingKeyEncryptDefault<BE: Backend> {
    fn lwe_switching_key_encrypt_sk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn lwe_switching_key_encrypt_sk_default<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_lwe_in: &S1,
        sk_lwe_out: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: LWESecretToBackendRef<BE>,
        S2: LWESecretToBackendRef<BE>;
}

impl<BE: Backend> LWESwitchingKeyEncryptDefault<BE> for Module<BE>
where
    Self: ModuleN
        + GLWESwitchingKeyEncryptSk<BE>
        + VecZnxAutomorphismBackend<BE>
        + VecZnxCopyRangeBackend<BE>
        + VecZnxZeroBackend<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn lwe_switching_key_encrypt_sk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(infos.dsize().0, 1, "dsize > 1 is not supported for LWESwitchingKey");
        assert_eq!(infos.rank_in().0, 1, "rank_in > 1 is not supported for LWESwitchingKey");
        assert_eq!(infos.rank_out().0, 1, "rank_out > 1 is not supported for LWESwitchingKey");
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = GLWESecret::bytes_of(self.n().into(), Rank(1));
        let lvl_1: usize = GLWESecret::bytes_of(self.n().into(), Rank(1));
        let lvl_2: usize = GLWESecret::bytes_of(self.n().into(), Rank(1));
        let lvl_3: usize = self.glwe_switching_key_encrypt_sk_tmp_bytes(infos);

        lvl_0 + lvl_1 + lvl_2 + lvl_3
    }

    #[allow(clippy::too_many_arguments)]
    fn lwe_switching_key_encrypt_sk_default<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_lwe_in: &S1,
        sk_lwe_out: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: LWESecretToBackendRef<BE>,
        S2: LWESecretToBackendRef<BE>,
    {
        let sk_lwe_in = sk_lwe_in.to_backend_ref();
        let sk_lwe_out = sk_lwe_out.to_backend_ref();

        assert!(sk_lwe_in.n().0 <= res.n().0);
        assert!(sk_lwe_out.n().0 <= res.n().0);
        assert!(res.n() <= self.n() as u32);
        assert!(
            scratch.available() >= self.lwe_switching_key_encrypt_sk_tmp_bytes_default(res),
            "scratch.available(): {} < LWESwitchingKeyEncrypt::lwe_switching_key_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.lwe_switching_key_encrypt_sk_tmp_bytes_default(res)
        );

        let scratch = scratch.borrow();
        let (mut sk_glwe_src, scratch_1) = scratch.take_glwe_secret_scratch(self.n().into(), Rank(1));
        let (mut sk_glwe_out, scratch_2) = scratch_1.take_glwe_secret_scratch(self.n().into(), Rank(1));
        let (mut sk_glwe_in, mut enc_scratch) = scratch_2.take_glwe_secret_scratch(self.n().into(), Rank(1));

        sk_glwe_out.dist = sk_lwe_out.dist;
        sk_glwe_src.dist = sk_lwe_out.dist;
        {
            let mut sk_glwe_src_backend = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(&mut sk_glwe_src.data);
            let sk_lwe_out_backend = scalar_znx_as_vec_znx_backend_ref_from_ref::<BE>(&sk_lwe_out.data);
            self.vec_znx_zero_backend(&mut sk_glwe_src_backend, 0);
            self.vec_znx_copy_range_backend(
                &mut sk_glwe_src_backend,
                0,
                0,
                0,
                &sk_lwe_out_backend,
                0,
                0,
                0,
                sk_lwe_out.n().into(),
            );
        }
        {
            let sk_glwe_src_backend = scalar_znx_as_vec_znx_backend_ref_from_mut::<BE>(&sk_glwe_src.data);
            let mut sk_glwe_out_backend = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(&mut sk_glwe_out.data);
            self.vec_znx_automorphism_backend(-1, &mut sk_glwe_out_backend, 0, &sk_glwe_src_backend, 0);
        }

        sk_glwe_src.dist = sk_lwe_in.dist;
        sk_glwe_in.dist = sk_lwe_in.dist;
        {
            let mut sk_glwe_src_backend = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(&mut sk_glwe_src.data);
            let sk_lwe_in_backend = scalar_znx_as_vec_znx_backend_ref_from_ref::<BE>(&sk_lwe_in.data);
            self.vec_znx_zero_backend(&mut sk_glwe_src_backend, 0);
            self.vec_znx_copy_range_backend(
                &mut sk_glwe_src_backend,
                0,
                0,
                0,
                &sk_lwe_in_backend,
                0,
                0,
                0,
                sk_lwe_in.n().into(),
            );
        }
        {
            let sk_glwe_src_backend = scalar_znx_as_vec_znx_backend_ref_from_mut::<BE>(&sk_glwe_src.data);
            let mut sk_glwe_in_backend = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(&mut sk_glwe_in.data);
            self.vec_znx_automorphism_backend(-1, &mut sk_glwe_in_backend, 0, &sk_glwe_src_backend, 0);
        }

        self.glwe_switching_key_encrypt_sk(
            res,
            &sk_glwe_in,
            &sk_glwe_out,
            enc_infos,
            source_xe,
            source_xa,
            &mut enc_scratch,
        );
    }
}
