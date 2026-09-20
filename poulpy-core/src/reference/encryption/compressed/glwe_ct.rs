#![allow(clippy::too_many_arguments)]

use poulpy_hal::{
    api::VecZnxCopy,
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{
    EncryptionInfos, ScratchArenaTakeCore,
    encryption::{GLWEEncryptSk, GLWEEncryptSkInternal, glwe::GLWEMaskFillReference},
    layouts::{
        GLWECompressedSeedMut, GLWEInfos, GLWEToBackendRef, LWEInfos, compressed::GLWECompressedToBackendMut,
        prepared::GLWESecretPreparedToBackendRef,
    },
};

/// Backend override contract; opt into its portable body with the matching forwarding macro.
pub trait GLWECompressedEncryptSkReference<BE: Backend> {
    fn glwe_compressed_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_compressed_encrypt_sk_reference<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWECompressedToBackendMut<BE> + GLWECompressedSeedMut,
        P: GLWEToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>;
}

/// Independently callable portable composition for [`GLWECompressedEncryptSkReference`].
///
/// HAL bounds belong to this helper, not to the backend override contract.
pub trait GLWECompressedEncryptSkComposition<BE: Backend> {
    fn glwe_compressed_encrypt_sk_tmp_bytes_composition<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_compressed_encrypt_sk_composition<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWECompressedToBackendMut<BE> + GLWECompressedSeedMut,
        P: GLWEToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>;
}

impl<BE: Backend> GLWECompressedEncryptSkComposition<BE> for Module<BE>
where
    Self: GLWEEncryptSkInternal<BE> + GLWEEncryptSk<BE> + GLWEMaskFillReference<BE> + VecZnxCopy<BE>,
{
    fn glwe_compressed_encrypt_sk_tmp_bytes_composition<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());
        let full_ct = self.bytes_of_vec_znx(self.n(), infos.rank().as_usize() + 1, infos.size());
        full_ct + self.glwe_encrypt_sk_tmp_bytes(infos)
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_compressed_encrypt_sk_composition<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWECompressedToBackendMut<BE> + GLWECompressedSeedMut,
        P: GLWEToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
    {
        res.seed_mut().copy_from_slice(&seed_xa);

        {
            let mut res_backend = res.to_backend_mut();
            assert!(
                scratch.available() >= self.glwe_compressed_encrypt_sk_tmp_bytes_composition(&res_backend),
                "scratch.available(): {} < GLWECompressedEncryptSk::glwe_compressed_encrypt_sk_tmp_bytes: {}",
                scratch.available(),
                self.glwe_compressed_encrypt_sk_tmp_bytes_composition(&res_backend)
            );

            let (mut full_ct, mut scratch_1) = scratch.borrow().take_glwe_scratch(&res_backend);
            self.fill_glwe_mask_from_seed_reference(
                res_backend.base2k().into(),
                &mut full_ct,
                1,
                res_backend.rank().as_usize(),
                seed_xa,
            );
            self.glwe_encrypt_sk_internal(
                res_backend.base2k().into(),
                &mut full_ct.data,
                Some((pt.to_backend_ref(), 0)),
                sk,
                enc_infos,
                source_xe,
                &mut scratch_1,
            );
            let full_ct_ref = full_ct.to_backend_ref();
            self.vec_znx_copy(&mut res_backend.data, 0, &full_ct_ref.data, 0);
        }
    }
}

/// Forwards every method of [`GLWECompressedEncryptSkReference`] to its portable composition.
#[macro_export]
macro_rules! impl_glwe_compressed_encrypt_sk_reference_full {
    ($be:ty) => {
        impl $crate::reference::encryption::GLWECompressedEncryptSkReference<$be> for ::poulpy_hal::layouts::Module<$be> {
    fn glwe_compressed_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWECompressedEncryptSkComposition<$be>>::glwe_compressed_encrypt_sk_tmp_bytes_composition::<A>(self, infos)
        }

    fn glwe_compressed_encrypt_sk_reference<R, P, S, E>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GLWECompressedToBackendMut<$be> + $crate::layouts::GLWECompressedSeedMut,
        P: $crate::layouts::GLWEToBackendRef<$be>,
        E: $crate::api::EncryptionInfos,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWECompressedEncryptSkComposition<$be>>::glwe_compressed_encrypt_sk_composition::<R, P, S, E>(self, res, pt, sk, seed_xa, enc_infos, source_xe, scratch)
        }
        }
    };
}
