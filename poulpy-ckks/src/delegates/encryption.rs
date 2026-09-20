use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{GLWEInfos, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef};
use poulpy_core::{EncryptionInfos, GLWEDecrypt, GLWEEncryptSk};
use poulpy_hal::{
    api::{VecZnxLsh, VecZnxLshTmpBytes, VecZnxRsh, VecZnxRshAdd, VecZnxRshTmpBytes},
    layouts::{Backend, HostDataMut, Module, ScratchArena},
    source::Source,
};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSDecryptOps, CKKSEncryptOps},
    oep::CKKSEncryptionImpl,
};

impl<BE: Backend + CKKSEncryptionImpl> CKKSEncryptOps<BE> for Module<BE>
where
    BE: poulpy_hal::oep::HalVecZnxImpl,
    Self: GLWEEncryptSk<BE> + VecZnxRshAdd<BE> + VecZnxRshTmpBytes,
{
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
        sk: &crate::layouts::CKKSKey<S>,
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
        crate::api::CKKSModuleInfos::ckks_ring(self).check("ckks_encrypt_sk", sk.key_ring())?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_encrypt_sk", ct)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_plaintext("ckks_encrypt_sk", pt)?;
        BE::ckks_encrypt_sk_impl(self, ct, pt, sk, enc_infos, source_xe, source_xa, scratch)
    }
}

// The `BE::OwnedBuf: HostDataMut` bound restricts this delegate to host
// backends; the `CKKSDecryptOps` trait itself carries no host bounds and a device
// backend provides its own impl.
impl<BE: Backend + CKKSEncryptionImpl> CKKSDecryptOps<BE> for Module<BE>
where
    BE: poulpy_hal::oep::HalVecZnxImpl,
    Self: GLWEDecrypt<BE>
        + VecZnxLsh<BE>
        + VecZnxLshTmpBytes
        + VecZnxRsh<BE>
        + VecZnxRshTmpBytes
        + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>,
    BE::OwnedBuf: HostDataMut,
{
    fn ckks_decrypt_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: CKKSCtBounds,
    {
        BE::ckks_decrypt_tmp_bytes_impl(self, ct_infos)
    }

    fn ckks_decrypt<Dpt, Dct, S>(
        &self,
        pt: &mut Dpt,
        ct: &Dct,
        sk: &crate::layouts::CKKSKey<S>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        Dpt: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + IntPolyInfos,
        Dct: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        crate::api::CKKSModuleInfos::ckks_ring(self).check("ckks_decrypt", sk.key_ring())?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_plaintext("ckks_decrypt", pt)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_decrypt", ct)?;
        BE::ckks_decrypt_impl(self, pt, ct, sk, scratch)
    }
}
