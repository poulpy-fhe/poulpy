#![allow(clippy::too_many_arguments)]

use poulpy_hal::{
    api::{VecZnxCopyBackend, VecZnxLshAssignBackend, VecZnxRshAssignBackend},
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{
    EncryptionInfos, ScratchArenaTakeCore,
    encryption::{GLWEEncryptSk, GLWEEncryptSkInternal, glwe::GLWEMaskFillDefault},
    layouts::{
        GLWECompressedSeedMut, GLWEInfos, GLWEToBackendRef, LWEInfos, compressed::GLWECompressedToBackendMut,
        prepared::GLWESecretPreparedToBackendRef,
    },
};

#[doc(hidden)]
pub trait GLWECompressedEncryptSkDefault<BE: Backend> {
    fn glwe_compressed_encrypt_sk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_compressed_encrypt_sk_default<R, P, S, E>(
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

impl<BE: Backend> GLWECompressedEncryptSkDefault<BE> for Module<BE>
where
    Self: GLWEEncryptSkInternal<BE>
        + GLWEEncryptSk<BE>
        + GLWEMaskFillDefault<BE>
        + VecZnxCopyBackend<BE>
        + VecZnxRshAssignBackend<BE>
        + VecZnxLshAssignBackend<BE>,
{
    fn glwe_compressed_encrypt_sk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());
        let full_ct = self.bytes_of_vec_znx(infos.rank().as_usize() + 1, infos.size());
        full_ct + self.glwe_encrypt_sk_tmp_bytes(infos)
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_compressed_encrypt_sk_default<R, P, S, E>(
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
                scratch.available() >= self.glwe_compressed_encrypt_sk_tmp_bytes_default(&res_backend),
                "scratch.available(): {} < GLWECompressedEncryptSk::glwe_compressed_encrypt_sk_tmp_bytes: {}",
                scratch.available(),
                self.glwe_compressed_encrypt_sk_tmp_bytes_default(&res_backend)
            );

            let (mut full_ct, mut scratch_1) = scratch.borrow().take_glwe_scratch(&res_backend);
            self.fill_glwe_mask_from_seed_default(
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
            crate::default::encryption::glwe::round_glwe_columns_to_k(self, &mut full_ct, 0..1, &mut scratch_1);
            let full_ct_ref = full_ct.to_backend_ref();
            self.vec_znx_copy_backend(&mut res_backend.data, 0, &full_ct_ref.data, 0);
        }
    }
}
