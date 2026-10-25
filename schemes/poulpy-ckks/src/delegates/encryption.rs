use crate::CKKSResult as Result;
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
        BE::ckks_encrypt_sk_tmp_bytes(self, ct_infos)
    }

    #[allow(clippy::too_many_arguments)]
    fn ckks_encrypt_sk<Dct, Dpt, S>(
        &self,
        ct: &mut Dct,
        pt: &Dpt,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        S: GLWESecretPreparedToBackendRef<BE>,
        Dct: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Dpt: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
    {
        BE::ckks_encrypt_sk(self, ct, pt, sk, source_xe, source_xa, scratch)
    }
}

impl<BE: Backend + CKKSEncryptionImpl> CKKSDecryptOps<BE> for Module<BE> {
    fn ckks_decrypt_tmp_bytes<Pt, Ct>(&self, pt_infos: &Pt, ct_infos: &Ct) -> usize
    where
        Pt: CKKSInfos,
        Ct: CKKSCtBounds,
    {
        BE::ckks_decrypt_tmp_bytes(self, pt_infos, ct_infos)
    }

    fn ckks_decrypt<Dpt, Dct, S>(&self, pt: &mut Dpt, ct: &Dct, sk: &S, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        Dpt: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + IntPolyInfos,
        Dct: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_decrypt(self, pt, ct, sk, scratch)
    }
}
