use poulpy_hal::{
    api::{ModuleN, VecZnxAutomorphism, VecZnxCopy, VecZnxZero},
    layouts::{
        Backend, Module, ScratchArena, scalar_znx_as_vec_znx_backend_mut_from_mut, scalar_znx_as_vec_znx_backend_ref_from_mut,
        scalar_znx_as_vec_znx_backend_ref_from_ref, vec_znx_backend_mut_from_mut,
    },
    source::Source,
};

use crate::api::GLWEBytesOf;
use crate::{
    EncryptionInfos, GGLWEEncryptSk, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GLWESecretToBackendRef, LWEInfos, LWESecretToBackendRef, Rank,
        prepared::GLWESecretPreparedFactory,
    },
};

/// Backend override contract; opt into its portable body with the matching forwarding macro.
pub trait GLWEToLWESwitchingKeyEncryptSkReference<BE: Backend> {
    fn glwe_to_lwe_key_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_to_lwe_key_encrypt_sk_reference<R, S1, S2, E>(
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

/// Independently callable portable composition for [`GLWEToLWESwitchingKeyEncryptSkReference`].
///
/// HAL bounds belong to this helper, not to the backend override contract.
pub trait GLWEToLWESwitchingKeyEncryptSkComposition<BE: Backend> {
    fn glwe_to_lwe_key_encrypt_sk_tmp_bytes_composition<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_to_lwe_key_encrypt_sk_composition<R, S1, S2, E>(
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

impl<BE: Backend> GLWEToLWESwitchingKeyEncryptSkComposition<BE> for Module<BE>
where
    Self: ModuleN + GGLWEEncryptSk<BE> + GLWESecretPreparedFactory<BE> + VecZnxAutomorphism<BE> + VecZnxCopy<BE> + VecZnxZero<BE>,
{
    fn glwe_to_lwe_key_encrypt_sk_tmp_bytes_composition<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = self.glwe_secret_prepared_bytes_of(infos.rank_in());
        let lvl_1_sk_lwe_as_glwe_src: usize = self.glwe_secret_bytes_of(self.n().into(), Rank(1));
        let lvl_2_sk_lwe_as_glwe: usize = self.glwe_secret_bytes_of(self.n().into(), Rank(1));
        let lvl_3_encrypt: usize = self.gglwe_encrypt_sk_tmp_bytes(infos);

        lvl_0 + lvl_1_sk_lwe_as_glwe_src + lvl_2_sk_lwe_as_glwe + lvl_3_encrypt
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_to_lwe_key_encrypt_sk_composition<R, S1, S2, E>(
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
            scratch.available() >= self.glwe_to_lwe_key_encrypt_sk_tmp_bytes_composition(res),
            "scratch.available(): {} < GLWEToLWESwitchingKeyEncryptSk::glwe_to_lwe_key_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.glwe_to_lwe_key_encrypt_sk_tmp_bytes_composition(res)
        );

        let scratch = scratch.borrow();
        let (mut sk_lwe_as_glwe_prep, scratch_1) = scratch.take_glwe_secret_prepared_scratch(self, Rank(1));
        let (mut sk_lwe_as_glwe_src, scratch_2) = scratch_1.take_glwe_secret_scratch(self.n().into(), Rank(1));
        let (mut sk_lwe_as_glwe, scratch_3) = scratch_2.take_glwe_secret_scratch(self.n().into(), Rank(1));

        sk_lwe_as_glwe_src.dist = sk_lwe.dist;
        sk_lwe_as_glwe.dist = sk_lwe.dist;
        {
            let mut sk_lwe_as_glwe_src_backend = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(sk_lwe_as_glwe_src.data_mut());
            let sk_lwe_backend = scalar_znx_as_vec_znx_backend_ref_from_ref::<BE>(sk_lwe.data());
            self.vec_znx_zero(&mut sk_lwe_as_glwe_src_backend, 0);
            self.vec_znx_copy(
                &mut vec_znx_backend_mut_from_mut::<BE>(&mut sk_lwe_as_glwe_src_backend).window_coeffs(0, sk_lwe.n().into()),
                0,
                &sk_lwe_backend,
                0,
            );
        }
        {
            let sk_lwe_as_glwe_src_backend = scalar_znx_as_vec_znx_backend_ref_from_mut::<BE>(sk_lwe_as_glwe_src.data());
            let mut sk_lwe_as_glwe_backend = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(sk_lwe_as_glwe.data_mut());
            self.vec_znx_automorphism(-1, &mut sk_lwe_as_glwe_backend, 0, &sk_lwe_as_glwe_src_backend, 0);
        }
        self.glwe_secret_prepare(&mut sk_lwe_as_glwe_prep, &sk_lwe_as_glwe);

        let (mut enc_scratch, _scratch_4) = scratch_3.split_at(self.gglwe_encrypt_sk_tmp_bytes(res));
        let sk_glwe_data_ref = sk_glwe.data();
        self.gglwe_encrypt_sk(
            res,
            &sk_glwe_data_ref,
            &sk_lwe_as_glwe_prep,
            enc_infos,
            source_xe,
            source_xa,
            &mut enc_scratch,
        );
    }
}

/// Forwards every method of [`GLWEToLWESwitchingKeyEncryptSkReference`] to its portable composition.
#[macro_export]
macro_rules! impl_glwe_to_lwe_switching_key_encrypt_sk_reference_full {
    ($be:ty) => {
        impl $crate::reference::encryption::GLWEToLWESwitchingKeyEncryptSkReference<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn glwe_to_lwe_key_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
            where
                A: $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEToLWESwitchingKeyEncryptSkComposition<
                    $be,
                >>::glwe_to_lwe_key_encrypt_sk_tmp_bytes_composition::<A>(self, infos)
            }

            fn glwe_to_lwe_key_encrypt_sk_reference<R, S1, S2, E>(
                &self,
                res: &mut R,
                sk_lwe: &S1,
                sk_glwe: &S2,
                enc_infos: &E,
                source_xe: &mut ::poulpy_hal::source::Source,
                source_xa: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                S1: $crate::layouts::LWESecretToBackendRef<$be>,
                S2: $crate::layouts::GLWESecretToBackendRef<$be>,
                E: $crate::api::EncryptionInfos,
                R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEToLWESwitchingKeyEncryptSkComposition<
                    $be,
                >>::glwe_to_lwe_key_encrypt_sk_composition::<R, S1, S2, E>(
                    self, res, sk_lwe, sk_glwe, enc_infos, source_xe, source_xa, scratch,
                )
            }
        }
    };
}
