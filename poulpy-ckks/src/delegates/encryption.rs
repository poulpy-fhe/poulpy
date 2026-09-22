use crate::CKKSResult as Result;
use poulpy_core::EncryptionInfos;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{GLWEInfos, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    api::{CKKSDecryptOps, CKKSEncryptOps},
    oep::CKKSEncryptionImpl,
};

impl<BE: Backend + CKKSEncryptionImpl> CKKSEncryptOps<BE> for Module<BE> {
    fn ckks_encrypt_sk_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: CKKSCtBounds,
    {
        BE::ckks_encrypt_sk_tmp_bytes_impl(self, ct_infos)
    }

    #[allow(clippy::too_many_arguments)]
    fn ckks_encrypt_sk<Dct, Dpt, S, E: EncryptionInfos>(
        &self,
        ct: &mut Dct,
        pt: &Dpt,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        S: GLWESecretPreparedToBackendRef<BE>,
        Dct: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Dpt: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
    {
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_encrypt_sk", ct)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_plaintext("ckks_encrypt_sk", pt)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_degree("ckks_encrypt_sk", sk.to_backend_ref().n())?;
        BE::ckks_encrypt_sk_impl(self, ct, pt, sk, enc_infos, source_xe, source_xa, scratch)
    }
}

impl<BE: Backend + CKKSEncryptionImpl> CKKSDecryptOps<BE> for Module<BE> {
    fn ckks_decrypt_tmp_bytes<Pt, Ct>(&self, pt_infos: &Pt, ct_infos: &Ct) -> usize
    where
        Pt: CKKSInfos,
        Ct: CKKSCtBounds,
    {
        BE::ckks_decrypt_tmp_bytes_impl(self, pt_infos, ct_infos)
    }

    fn ckks_decrypt<Dpt, Dct, S>(&self, pt: &mut Dpt, ct: &Dct, sk: &S, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        Dpt: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + IntPolyInfos,
        Dct: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        crate::api::CKKSModuleInfos::ckks_ring(self).check_plaintext("ckks_decrypt", pt)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_decrypt", ct)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_degree("ckks_decrypt", sk.to_backend_ref().n())?;
        BE::ckks_decrypt_impl(self, pt, ct, sk, scratch)
    }
}
