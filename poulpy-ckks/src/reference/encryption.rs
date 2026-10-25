use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{GLWEInfos, GLWELayout, GLWESecretPreparedToBackendRef, GLWEToBackendMut, LWEInfos, Rank};
use poulpy_core::{ComponentNoise, GetDistribution};
use poulpy_core::{GLWEDecrypt, GLWEEncryptSk, GLWENormalize, ScratchArenaTakeCore};
use poulpy_hal::{
    api::{ModuleN, VecZnxLsh, VecZnxLshAdd, VecZnxLshTmpBytes, VecZnxRsh, VecZnxRshAdd, VecZnxRshTmpBytes, VecZnxSwitchRing},
    layouts::{Backend, ScratchArena},
    source::Source,
};

use crate::GLWEToBackendRef;
use crate::{CKKSCompositionError, CKKSInfos, SetCKKSInfos, checked_log_budget_sub, error::ensure_encryption_degrees};

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
    fn ckks_encrypt_sk_reference<Dct, Dpt, S>(
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
        Dct: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        Dpt: GLWEToBackendRef<BE> + CKKSInfos + IntPolyInfos,
        Self: ModuleN + GLWEEncryptSk<BE> + GLWENormalize<BE> + VecZnxLshAdd<BE> + VecZnxRshAdd<BE> + CKKSPlaintextReference<BE>,
    {
        ensure_encryption_degrees(
            "ckks_encrypt_sk",
            self.n(),
            ct.n().as_usize(),
            sk.to_backend_ref().n().as_usize(),
        )?;
        let log_budget = checked_log_budget_sub("ckks_encrypt_sk", ct.k().as_usize(), pt.log_delta())?;
        self.glwe_encrypt_zero_sk(ct, sk, source_xe, source_xa, scratch);
        ct.set_log_budget(log_budget);
        ct.set_log_delta(pt.log_delta());
        ct.set_slots(pt.slots());
        self.ckks_add_pt_vec_into_reference(ct, pt, scratch)?;
        // A plaintext is dense at its own degree.
        ct.set_log_sparsity(ct.log_n() - pt.log_n());
        // The raw limb-add above can leave digits one bit beyond the `base2k`
        // normalized range; a fresh encryption is typed `Normalized`, so
        // propagate the carries before returning (the crate's digit contract
        // for every DFT-domain op).
        self.glwe_normalize_assign(ct, scratch);
        let noise = ComponentNoise::from_secret_at(*sk.to_backend_ref().dist(), ct.k(), ct.to_backend_ref().rank().as_usize());
        ct.set_noise(Some(noise));
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
        let below = if pt_infos.n() < ct_infos.n() {
            BE::scratch_aligned(self.glwe_plaintext_bytes_of_from_infos(&below_layout(pt_infos, ct_infos)))
        } else {
            0
        };
        self.glwe_plaintext_bytes_of_from_infos(ct_infos)
            + below
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
        Self: ModuleN + GLWEDecrypt<BE> + CKKSPlaintextReference<BE> + VecZnxLsh<BE> + VecZnxRsh<BE> + VecZnxSwitchRing<BE>,
    {
        ensure_encryption_degrees("ckks_decrypt", self.n(), ct.n().as_usize(), sk.n().as_usize())?;
        let (ct_n, pt_n) = (ct.n().as_usize(), pt.n().as_usize());
        // Every `ct_n / pt_n`-th coefficient holds the message only when the slots repeat that often.
        if pt_n != ct_n && !(pt_n < ct_n && ct_n.is_multiple_of(pt_n) && (ct_n / pt_n).ilog2() as usize <= ct.log_sparsity()) {
            return Err(CKKSCompositionError::PlaintextDegreeMismatch {
                op: "ckks_decrypt",
                ct_n,
                pt_n,
            }
            .into());
        }
        let (mut full_pt, mut scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(ct);
        self.glwe_decrypt(ct, &mut full_pt, sk, &mut scratch_1);
        if pt_n == ct_n {
            return CKKSPlaintextReference::ckks_extract_pt_with_meta_reference(self, pt, &full_pt, ct.meta(), &mut scratch_1);
        }
        let (mut below, mut scratch_2) = scratch_1.take_glwe_plaintext_scratch(&below_layout(pt, ct));
        self.vec_znx_switch_ring(below.to_backend_mut().data_mut(), 0, full_pt.to_backend_ref().data(), 0);
        CKKSPlaintextReference::ckks_extract_pt_with_meta_reference(self, pt, &below, ct.meta(), &mut scratch_2)
    }
}

impl<BE: Backend> CKKSEncryptionReference<BE> for poulpy_hal::layouts::Module<BE> {}

/// The decrypted polynomial at the degree of `pt`.
fn below_layout<P: LWEInfos, C: LWEInfos>(pt: &P, ct: &C) -> GLWELayout {
    GLWELayout {
        n: pt.n(),
        base2k: ct.base2k(),
        k: ct.k(),
        rank: Rank(0),
    }
}
