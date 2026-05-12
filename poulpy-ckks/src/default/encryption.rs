use anyhow::Result;
use poulpy_core::layouts::{GLWEInfos, GLWEPlaintext, GLWESecretPreparedToBackendRef, GLWEToBackendMut, LWEInfos};
use poulpy_core::{EncryptionInfos, GLWEDecrypt, GLWEEncryptSk, ScratchArenaTakeCore};
use poulpy_hal::{
    api::{ScratchAvailable, VecZnxLshBackend, VecZnxLshTmpBytes, VecZnxRshAddIntoBackend, VecZnxRshBackend, VecZnxRshTmpBytes},
    layouts::{Backend, ScratchArena},
    source::Source,
};

use crate::GLWEToBackendRef;
use crate::{CKKSInfos, SetCKKSInfos, checked_log_budget_sub};

use super::CKKSPlaintextDefault;

pub trait CKKSEncryptionDefault<BE: Backend> {
    fn ckks_encrypt_sk_tmp_bytes_default<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos + CKKSInfos,
        Self: GLWEEncryptSk<BE> + VecZnxRshAddIntoBackend<BE> + VecZnxRshTmpBytes,
    {
        self.glwe_encrypt_sk_tmp_bytes(ct_infos).max(self.vec_znx_rsh_tmp_bytes())
    }

    #[allow(clippy::too_many_arguments)]
    fn ckks_encrypt_sk_default<'s, Dct, Dpt, S, E>(
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
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        Dct: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Dpt: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        Self: GLWEEncryptSk<BE> + VecZnxRshAddIntoBackend<BE> + CKKSPlaintextDefault<BE>,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        self.glwe_encrypt_zero_sk(ct, sk, enc_infos, source_xe, source_xa, scratch);
        ct.set_log_budget(checked_log_budget_sub(
            "ckks_encrypt_sk",
            enc_infos.noise_infos().k,
            pt.log_delta(),
        )?);
        ct.set_log_delta(pt.log_delta());
        self.ckks_add_pt_vec_into_default(ct, pt, scratch)
    }

    fn ckks_decrypt_tmp_bytes_default<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos + CKKSInfos,
        Self: GLWEDecrypt<BE>
            + VecZnxLshBackend<BE>
            + VecZnxLshTmpBytes
            + VecZnxRshBackend<BE>
            + VecZnxRshTmpBytes
            + CKKSPlaintextDefault<BE>,
    {
        GLWEPlaintext::<Vec<u8>>::bytes_of_from_infos(ct_infos)
            + self
                .glwe_decrypt_tmp_bytes(ct_infos)
                .max(self.ckks_extract_pt_tmp_bytes_default())
    }

    fn ckks_decrypt_default<Dpt, Dct, S>(&self, pt: &mut Dpt, ct: &Dct, sk: &S, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        Dpt: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Dct: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
        Self: GLWEDecrypt<BE> + CKKSPlaintextDefault<BE> + VecZnxLshBackend<BE> + VecZnxRshBackend<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        let (mut full_pt, mut scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(ct);
        self.glwe_decrypt(ct, &mut full_pt, sk, &mut scratch_1);

        CKKSPlaintextDefault::ckks_extract_pt_with_meta_default(self, pt, &full_pt, ct.meta(), &mut scratch_1)?;

        Ok(())
    }
}
