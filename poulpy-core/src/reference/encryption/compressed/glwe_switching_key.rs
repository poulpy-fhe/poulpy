use poulpy_hal::{
    api::{ModuleN, ScratchArenaTakeBasic, VecZnxSwitchRing},
    layouts::{
        Backend, Module, ScratchArena, scalar_znx_as_vec_znx_backend_mut_from_mut, scalar_znx_as_vec_znx_backend_ref_from_ref,
    },
    source::Source,
};

use crate::layouts::operand_degree;
use crate::{
    GGLWECompressedEncryptSk, GetDistribution, ScratchArenaTakeCore,
    layouts::{
        GGLWECompressedSeedMut, GGLWECompressedToBackendMut, GGLWEInfos, GLWEInfos, GLWESecretToBackendRef,
        GLWESwitchingKeyDegreesMut, LWEInfos, prepared::GLWESecretPreparedFactory,
    },
};

/// Portable implementation using HAL operations.
///
/// Backend implementations may call this helper without changing their override selection.
pub trait GLWESwitchingKeyCompressedEncryptSkReference<BE: Backend> {
    fn glwe_switching_key_compressed_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_switching_key_compressed_encrypt_sk_reference<R, S1, S2>(
        &self,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeedMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        S1: GLWESecretToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos;
}

impl<BE: Backend> GLWESwitchingKeyCompressedEncryptSkReference<BE> for Module<BE>
where
    Self: ModuleN + GGLWECompressedEncryptSk<BE> + GLWESecretPreparedFactory<BE> + VecZnxSwitchRing<BE>,
{
    fn glwe_switching_key_compressed_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let n: usize = operand_degree(self.n(), &[infos.n()]);

        let lvl_0: usize = BE::bytes_of_scalar_znx(n, infos.rank_in().into());
        let lvl_1: usize = BE::bytes_of_scalar_znx(n, infos.rank_out().into());
        let lvl_2: usize = self.glwe_secret_prepared_bytes_of(infos.rank_out());
        let lvl_3_encrypt: usize = self.gglwe_compressed_encrypt_sk_tmp_bytes(infos);
        lvl_0 + lvl_1 + lvl_2 + lvl_3_encrypt
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_switching_key_compressed_encrypt_sk_reference<R, S1, S2>(
        &self,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeedMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        S1: GLWESecretToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        let sk_in = sk_in.to_backend_ref();
        let sk_out_ref = sk_out.to_backend_ref();

        assert!(sk_in.n().0 <= res.n().0);
        assert!(sk_out_ref.n().0 <= res.n().0);
        assert!(
            scratch.available() >= self.glwe_switching_key_compressed_encrypt_sk_tmp_bytes_reference(res),
            "scratch.available(): {} < GLWESwitchingKeyCompressedEncryptSk::glwe_switching_key_compressed_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.glwe_switching_key_compressed_encrypt_sk_tmp_bytes_reference(res)
        );
        let tmp_bytes: usize = self.glwe_switching_key_compressed_encrypt_sk_tmp_bytes_reference(res);
        {
            let (mut sk_in_lifted, scratch_1) = scratch
                .borrow()
                .take_scalar_znx_scratch(res.n().as_usize(), sk_in.rank().into());
            let sk_in_backend_vec = scalar_znx_as_vec_znx_backend_ref_from_ref::<BE>(sk_in.data());
            for i in 0..sk_in.rank().into() {
                let mut sk_in_lifted_backend_vec = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(&mut sk_in_lifted);
                self.vec_znx_switch_ring(&mut sk_in_lifted_backend_vec, i, &sk_in_backend_vec, i);
            }

            let (mut sk_out_lifted, scratch_2) = scratch_1.take_glwe_secret_scratch(res.n().as_usize().into(), sk_out_ref.rank());
            sk_out_lifted.dist = *sk_out.dist();
            let sk_out_backend_vec = scalar_znx_as_vec_znx_backend_ref_from_ref::<BE>(sk_out_ref.data());
            for i in 0..sk_out_ref.rank().into() {
                let mut sk_out_lifted_backend_vec = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(sk_out_lifted.data_mut());
                self.vec_znx_switch_ring(&mut sk_out_lifted_backend_vec, i, &sk_out_backend_vec, i);
            }

            let (mut sk_out_prepared, scratch_3) = scratch_2.take_glwe_secret_prepared_scratch(res.n(), sk_out_ref.rank());
            self.glwe_secret_prepare(&mut sk_out_prepared, &sk_out_lifted);

            let (mut enc_scratch, _scratch_4) = scratch_3.split_at(self.gglwe_compressed_encrypt_sk_tmp_bytes(res));
            self.gglwe_compressed_encrypt_sk(res, &sk_in_lifted, &sk_out_prepared, seed_xa, source_xe, &mut enc_scratch);

            *res.input_degree() = sk_in.n();
            *res.output_degree() = sk_out_ref.n();
        }
        scratch.wipe(tmp_bytes);
    }
}
