use anyhow::Result;
use poulpy_core::layouts::{GLWEInfos, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef};
use poulpy_core::{EncryptionInfos, GLWEDecrypt, GLWEEncryptSk, ScratchArenaTakeCore};
use poulpy_hal::{
    api::{ScratchAvailable, VecZnxLshBackend, VecZnxLshTmpBytes, VecZnxRshAddIntoBackend, VecZnxRshBackend, VecZnxRshTmpBytes},
    layouts::{Backend, HostDataMut, Module, ScratchArena},
    source::Source,
};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSDecrypt, CKKSEncrypt},
    oep::CKKSEncryptionImpl,
};

impl<BE: Backend + CKKSEncryptionImpl<BE>> CKKSEncrypt<BE> for Module<BE>
where
    BE: poulpy_hal::oep::HalVecZnxImpl<BE>,
    Self: GLWEEncryptSk<BE> + VecZnxRshAddIntoBackend<BE> + VecZnxRshTmpBytes,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_encrypt_sk_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: CKKSCtBounds,
    {
        BE::ckks_encrypt_sk_tmp_bytes(self, ct_infos)
    }

    #[allow(clippy::too_many_arguments)]
    fn ckks_encrypt_sk<'s, Dct, Dpt, S, E: EncryptionInfos>(
        &self,
        ct: &mut Dct,
        pt: &Dpt,
        sk: &S,
        enc_infos: &E,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        S: GLWESecretPreparedToBackendRef<BE>,
        Dct: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Dpt: GLWEToBackendRef<BE> + CKKSCtBounds,
        BE: 's,
    {
        BE::ckks_encrypt_sk(self, ct, pt, sk, enc_infos, source_xa, source_xe, scratch)
    }
}

impl<BE: Backend + CKKSEncryptionImpl<BE>> CKKSDecrypt<BE> for Module<BE>
where
    BE: poulpy_hal::oep::HalVecZnxImpl<BE>,
    Self: GLWEDecrypt<BE>
        + VecZnxLshBackend<BE>
        + VecZnxLshTmpBytes
        + VecZnxRshBackend<BE>
        + VecZnxRshTmpBytes
        + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
    BE::OwnedBuf: HostDataMut,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_decrypt_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: CKKSCtBounds,
    {
        BE::ckks_decrypt_tmp_bytes(self, ct_infos)
    }

    fn ckks_decrypt<Dpt, Dct, S>(&self, pt: &mut Dpt, ct: &Dct, sk: &S, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        Dpt: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Dct: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_decrypt(self, pt, ct, sk, scratch)
    }
}
