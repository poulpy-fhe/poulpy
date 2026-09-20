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
        GGLWEInfos, GGLWEToBackendMut, GLWESecretPreparedFactory, GLWESecretPreparedToBackendRef, LWEInfos,
        LWESecretToBackendRef, Rank,
    },
};

/// Backend override contract; opt into its portable body with the matching forwarding macro.
pub trait LWEToGLWESwitchingKeyEncryptSkReference<BE: Backend> {
    fn lwe_to_glwe_key_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn lwe_to_glwe_key_encrypt_sk_reference<R, S1, S2, E>(
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

/// Independently callable portable composition for [`LWEToGLWESwitchingKeyEncryptSkReference`].
///
/// HAL bounds belong to this helper, not to the backend override contract.
pub trait LWEToGLWESwitchingKeyEncryptSkComposition<BE: Backend> {
    fn lwe_to_glwe_key_encrypt_sk_tmp_bytes_composition<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn lwe_to_glwe_key_encrypt_sk_composition<R, S1, S2, E>(
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

impl<BE: Backend> LWEToGLWESwitchingKeyEncryptSkComposition<BE> for Module<BE>
where
    Self: ModuleN + GGLWEEncryptSk<BE> + GLWESecretPreparedFactory<BE> + VecZnxAutomorphism<BE> + VecZnxCopy<BE> + VecZnxZero<BE>,
{
    fn lwe_to_glwe_key_encrypt_sk_tmp_bytes_composition<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(
            infos.rank_in(),
            Rank(1),
            "rank_in != 1 is not supported for LWEToGLWEKeyPrepared"
        );
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = self.glwe_secret_bytes_of(self.n().into(), Rank(1));
        let lvl_1: usize = self.glwe_secret_bytes_of(self.n().into(), Rank(1));
        let lvl_2_encrypt: usize = self.gglwe_encrypt_sk_tmp_bytes(infos);

        lvl_0 + lvl_1 + lvl_2_encrypt
    }

    #[allow(clippy::too_many_arguments)]
    fn lwe_to_glwe_key_encrypt_sk_composition<R, S1, S2, E>(
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
            scratch.available() >= self.lwe_to_glwe_key_encrypt_sk_tmp_bytes_composition(res),
            "scratch.available(): {} < LWEToGLWESwitchingKeyEncryptSk::lwe_to_glwe_key_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.lwe_to_glwe_key_encrypt_sk_tmp_bytes_composition(res)
        );

        let scratch = scratch.borrow();
        let (mut sk_lwe_as_glwe_src, scratch_1) = scratch.take_glwe_secret_scratch(self.n().into(), Rank(1));
        let (mut sk_lwe_as_glwe, scratch_2) = scratch_1.take_glwe_secret_scratch(self.n().into(), Rank(1));

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

        let (mut enc_scratch, _scratch_3) = scratch_2.split_at(self.gglwe_encrypt_sk_tmp_bytes(res));
        let sk_lwe_as_glwe_data = sk_lwe_as_glwe.data_mut();
        self.gglwe_encrypt_sk(
            res,
            &sk_lwe_as_glwe_data,
            sk_glwe,
            enc_infos,
            source_xe,
            source_xa,
            &mut enc_scratch,
        );
    }
}

/// Forwards every method of [`LWEToGLWESwitchingKeyEncryptSkReference`] to its portable composition.
#[macro_export]
macro_rules! impl_lwe_to_glwe_switching_key_encrypt_sk_reference_full {
    ($be:ty) => {
        impl $crate::reference::encryption::LWEToGLWESwitchingKeyEncryptSkReference<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn lwe_to_glwe_key_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
            where
                A: $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::LWEToGLWESwitchingKeyEncryptSkComposition<
                    $be,
                >>::lwe_to_glwe_key_encrypt_sk_tmp_bytes_composition::<A>(self, infos)
            }

            fn lwe_to_glwe_key_encrypt_sk_reference<R, S1, S2, E>(
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
                S2: $crate::layouts::GLWESecretPreparedToBackendRef<$be>,
                E: $crate::api::EncryptionInfos,
                R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::LWEToGLWESwitchingKeyEncryptSkComposition<
                    $be,
                >>::lwe_to_glwe_key_encrypt_sk_composition::<R, S1, S2, E>(
                    self, res, sk_lwe, sk_glwe, enc_infos, source_xe, source_xa, scratch,
                )
            }
        }
    };
}
