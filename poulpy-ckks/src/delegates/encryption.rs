use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWEInfos, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef};
use poulpy_core::{EncryptionInfos, GLWEDecrypt, GLWEEncryptSk};
use poulpy_hal::{
    api::{VecZnxLshBackend, VecZnxLshTmpBytes, VecZnxRshAddIntoBackend, VecZnxRshBackend, VecZnxRshTmpBytes},
    layouts::{Backend, HostDataMut, Module, ScratchArena},
    source::Source,
};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSDecryptOps, CKKSEncryptOps},
    oep::CKKSEncryptionImpl,
};

impl<BE: Backend + CKKSEncryptionImpl<BE>> CKKSEncryptOps<BE> for Module<BE>
where
    BE: poulpy_hal::oep::HalVecZnxImpl<BE>,
    Self: GLWEEncryptSk<BE> + VecZnxRshAddIntoBackend<BE> + VecZnxRshTmpBytes,
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
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        S: GLWESecretPreparedToBackendRef<BE>,
        Dct: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Dpt: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_encrypt_sk_impl(self, ct, pt, sk, enc_infos, source_xe, source_xa, scratch)
    }
}

// The `BE::OwnedBuf: HostDataMut` bound restricts this delegate to host
// backends; the `CKKSDecryptOps` trait itself carries no host bounds and a device
// backend provides its own impl.
impl<BE: Backend + CKKSEncryptionImpl<BE>> CKKSDecryptOps<BE> for Module<BE>
where
    BE: poulpy_hal::oep::HalVecZnxImpl<BE>,
    Self: GLWEDecrypt<BE>
        + VecZnxLshBackend<BE>
        + VecZnxLshTmpBytes
        + VecZnxRshBackend<BE>
        + VecZnxRshTmpBytes
        + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
    BE::OwnedBuf: HostDataMut,
{
    fn ckks_decrypt_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: CKKSCtBounds,
    {
        BE::ckks_decrypt_tmp_bytes_impl(self, ct_infos)
    }

    fn ckks_decrypt<Dpt, Dct, S>(&self, pt: &mut Dpt, ct: &Dct, sk: &S, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        Dpt: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Dct: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_decrypt_impl(self, pt, ct, sk, scratch)
    }
}
