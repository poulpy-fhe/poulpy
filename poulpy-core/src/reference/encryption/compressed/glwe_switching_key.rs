use poulpy_hal::{
    api::{ModuleN, ScratchArenaTakeBasic, VecZnxSwitchRing},
    layouts::{
        Backend, Module, ScratchArena, scalar_znx_as_vec_znx_backend_mut_from_mut, scalar_znx_as_vec_znx_backend_ref_from_ref,
    },
    source::Source,
};

use crate::{
    EncryptionInfos, GGLWECompressedEncryptSk, GetDistribution, ScratchArenaTakeCore,
    layouts::{
        GGLWECompressedSeedMut, GGLWECompressedToBackendMut, GGLWEInfos, GLWEInfos, GLWESecretToBackendRef,
        GLWESwitchingKeyDegreesMut, LWEInfos, prepared::GLWESecretPreparedFactory,
    },
};

/// Backend override contract; opt into its portable body with the matching forwarding macro.
pub trait GLWESwitchingKeyCompressedEncryptSkReference<BE: Backend> {
    fn glwe_switching_key_compressed_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_switching_key_compressed_encrypt_sk_reference<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeedMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: GLWESecretToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos;
}

/// Independently callable portable composition for [`GLWESwitchingKeyCompressedEncryptSkReference`].
///
/// HAL bounds belong to this helper, not to the backend override contract.
pub trait GLWESwitchingKeyCompressedEncryptSkComposition<BE: Backend> {
    fn glwe_switching_key_compressed_encrypt_sk_tmp_bytes_composition<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_switching_key_compressed_encrypt_sk_composition<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeedMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: GLWESecretToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos;
}

impl<BE: Backend> GLWESwitchingKeyCompressedEncryptSkComposition<BE> for Module<BE>
where
    Self: ModuleN + GGLWECompressedEncryptSk<BE> + GLWESecretPreparedFactory<BE> + VecZnxSwitchRing<BE>,
{
    fn glwe_switching_key_compressed_encrypt_sk_tmp_bytes_composition<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = BE::bytes_of_scalar_znx(self.n(), infos.rank_in().into());
        let lvl_1: usize = BE::bytes_of_scalar_znx(self.n(), infos.rank_out().into());
        let lvl_2: usize = self.glwe_secret_prepared_bytes_of(infos.rank_out());
        let lvl_3_encrypt: usize = self.gglwe_compressed_encrypt_sk_tmp_bytes(infos);
        lvl_0 + lvl_1 + lvl_2 + lvl_3_encrypt
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_switching_key_compressed_encrypt_sk_composition<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeedMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: GLWESecretToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        let sk_in = sk_in.to_backend_ref();
        let sk_out_ref = sk_out.to_backend_ref();

        assert!(sk_in.n().0 <= self.n() as u32);
        assert!(sk_out_ref.n().0 <= self.n() as u32);
        assert!(
            scratch.available() >= self.glwe_switching_key_compressed_encrypt_sk_tmp_bytes_composition(res),
            "scratch.available(): {} < GLWESwitchingKeyCompressedEncryptSk::glwe_switching_key_compressed_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.glwe_switching_key_compressed_encrypt_sk_tmp_bytes_composition(res)
        );

        let (mut sk_in_lifted, scratch_1) = scratch.borrow().take_scalar_znx_scratch(self.n(), sk_in.rank().into());
        let sk_in_backend_vec = scalar_znx_as_vec_znx_backend_ref_from_ref::<BE>(sk_in.data());
        for i in 0..sk_in.rank().into() {
            let mut sk_in_lifted_backend_vec = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(&mut sk_in_lifted);
            self.vec_znx_switch_ring(&mut sk_in_lifted_backend_vec, i, &sk_in_backend_vec, i);
        }

        let (mut sk_out_lifted, scratch_2) = scratch_1.take_glwe_secret_scratch(self.n().into(), sk_out_ref.rank());
        sk_out_lifted.dist = *sk_out.dist();
        let sk_out_backend_vec = scalar_znx_as_vec_znx_backend_ref_from_ref::<BE>(sk_out_ref.data());
        for i in 0..sk_out_ref.rank().into() {
            let mut sk_out_lifted_backend_vec = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(sk_out_lifted.data_mut());
            self.vec_znx_switch_ring(&mut sk_out_lifted_backend_vec, i, &sk_out_backend_vec, i);
        }

        let (mut sk_out_prepared, scratch_3) = scratch_2.take_glwe_secret_prepared_scratch(self, sk_out_ref.rank());
        self.glwe_secret_prepare(&mut sk_out_prepared, &sk_out_lifted);

        let (mut enc_scratch, _scratch_4) = scratch_3.split_at(self.gglwe_compressed_encrypt_sk_tmp_bytes(res));
        self.gglwe_compressed_encrypt_sk(
            res,
            &sk_in_lifted,
            &sk_out_prepared,
            seed_xa,
            enc_infos,
            source_xe,
            &mut enc_scratch,
        );

        *res.input_degree() = sk_in.n();
        *res.output_degree() = sk_out_ref.n();
    }
}

/// Forwards every method of [`GLWESwitchingKeyCompressedEncryptSkReference`] to its portable composition.
#[macro_export]
macro_rules! impl_glwe_switching_key_compressed_encrypt_sk_reference_full {
    ($be:ty) => {
        impl $crate::reference::encryption::GLWESwitchingKeyCompressedEncryptSkReference<$be> for ::poulpy_hal::layouts::Module<$be> {
    fn glwe_switching_key_compressed_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: $crate::layouts::GGLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWESwitchingKeyCompressedEncryptSkComposition<$be>>::glwe_switching_key_compressed_encrypt_sk_tmp_bytes_composition::<A>(self, infos)
        }

    fn glwe_switching_key_compressed_encrypt_sk_reference<R, S1, S2, E>(
        &self,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GGLWECompressedToBackendMut<$be> + $crate::layouts::GGLWECompressedSeedMut + $crate::layouts::GLWESwitchingKeyDegreesMut + $crate::layouts::GGLWEInfos,
        E: $crate::api::EncryptionInfos,
        S1: $crate::layouts::GLWESecretToBackendRef<$be> + $crate::layouts::GLWEInfos,
        S2: $crate::layouts::GLWESecretToBackendRef<$be> + $crate::GetDistribution + $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWESwitchingKeyCompressedEncryptSkComposition<$be>>::glwe_switching_key_compressed_encrypt_sk_composition::<R, S1, S2, E>(self, res, sk_in, sk_out, seed_xa, enc_infos, source_xe, scratch)
        }
        }
    };
}
