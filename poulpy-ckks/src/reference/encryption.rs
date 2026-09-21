use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{GLWEInfos, GLWESecretPreparedToBackendRef, GLWEToBackendMut};
use poulpy_core::{EncryptionInfos, GLWEDecrypt, GLWEEncryptSk, GLWENormalize, ScratchArenaTakeCore};
use poulpy_hal::{
    api::{VecZnxLsh, VecZnxLshAdd, VecZnxLshTmpBytes, VecZnxRsh, VecZnxRshAdd, VecZnxRshTmpBytes},
    layouts::{Backend, ScratchArena},
    source::Source,
};

use crate::GLWEToBackendRef;
use crate::{CKKSInfos, SetCKKSInfos, checked_log_budget_sub};

use super::CKKSPlaintextReference;
use poulpy_core::GLWEBytesOf;

pub trait CKKSEncryptionReference<BE: Backend> {
    fn ckks_encrypt_sk_tmp_bytes_reference<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos + CKKSInfos,
        Self: GLWEEncryptSk<BE> + GLWENormalize<BE> + VecZnxLshTmpBytes + VecZnxRshAdd<BE> + VecZnxRshTmpBytes,
    {
        self.glwe_encrypt_sk_tmp_bytes(ct_infos)
            .max(self.vec_znx_lsh_tmp_bytes(ct_infos.size()))
            .max(self.vec_znx_rsh_tmp_bytes(ct_infos.size()))
            .max(self.glwe_normalize_tmp_bytes())
    }

    #[allow(clippy::too_many_arguments)]
    fn ckks_encrypt_sk_reference<Dct, Dpt, S, E>(
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
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        Dct: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        Dpt: GLWEToBackendRef<BE> + CKKSInfos + IntPolyInfos,
        Self: GLWEEncryptSk<BE> + GLWENormalize<BE> + VecZnxLshAdd<BE> + VecZnxRshAdd<BE> + CKKSPlaintextReference<BE>,
    {
        self.glwe_encrypt_zero_sk(ct, sk, enc_infos, source_xe, source_xa, scratch);
        ct.set_log_budget(checked_log_budget_sub(
            "ckks_encrypt_sk",
            enc_infos.noise_infos().k,
            pt.log_delta(),
        )?);
        ct.set_log_delta(pt.log_delta());
        self.ckks_add_pt_vec_into_reference(ct, pt, scratch)?;
        // The raw limb-add above can leave digits one bit beyond the `base2k`
        // normalized range; a fresh encryption is typed `Normalized`, so
        // propagate the carries before returning (the crate's digit contract
        // for every DFT-domain op).
        self.glwe_normalize_assign(ct, scratch);
        Ok(())
    }

    fn ckks_decrypt_tmp_bytes_reference<Pt, Ct>(&self, pt_infos: &Pt, ct_infos: &Ct) -> usize
    where
        Self: GLWEBytesOf<BE>,
        Pt: CKKSInfos,
        Ct: GLWEInfos + CKKSInfos,
        Self:
            GLWEDecrypt<BE> + VecZnxLsh<BE> + VecZnxLshTmpBytes + VecZnxRsh<BE> + VecZnxRshTmpBytes + CKKSPlaintextReference<BE>,
    {
        self.glwe_plaintext_bytes_of_from_infos(ct_infos)
            + self
                .glwe_decrypt_tmp_bytes(ct_infos)
                // Extraction uses the destination's physical limb allocation,
                // which can exceed either object's effective precision.
                .max(self.ckks_extract_pt_tmp_bytes_reference(pt_infos.max_size()))
    }

    fn ckks_decrypt_reference<Dpt, Dct, S>(
        &self,
        pt: &mut Dpt,
        ct: &Dct,
        sk: &S,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        Dpt: GLWEToBackendMut<BE> + CKKSInfos + IntPolyInfos + SetCKKSInfos,
        Dct: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos,
        Self: GLWEDecrypt<BE> + CKKSPlaintextReference<BE> + VecZnxLsh<BE> + VecZnxRsh<BE>,
    {
        let (mut full_pt, mut scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(ct);
        self.glwe_decrypt(ct, &mut full_pt, sk, &mut scratch_1);

        CKKSPlaintextReference::ckks_extract_pt_with_meta_reference(self, pt, &full_pt, ct.meta(), &mut scratch_1)?;

        Ok(())
    }
}

impl<BE: Backend> CKKSEncryptionReference<BE> for poulpy_hal::layouts::Module<BE> {}
