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
use crate::{
    CKKSCompositionError, CKKSInfos, CKKSLayout, SetCKKSInfos, SlotLayout,
    api::CKKSRoundOps,
    checked_log_budget_sub, ckks_ensure, ensure_base2k_match, ensure_plaintext_alignment, ensure_plaintext_vec_operand,
    error::ensure_encryption_degrees,
    layouts::ScratchArenaTakeCKKS,
    shaping::{MAX_PASS_BITS, ShapingProfile},
};

use super::CKKSPlaintextVecReference;
use poulpy_core::GLWEBytesOf;

pub trait CKKSEncryptionReference<BE: Backend> {
    fn ckks_encrypt_sk_tmp_bytes_reference<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos + CKKSInfos,
        Self: GLWEEncryptSk<BE>
            + GLWENormalize<BE>
            + GLWEBytesOf<BE>
            + VecZnxLshTmpBytes
            + VecZnxRshAdd<BE>
            + VecZnxRshTmpBytes
            + CKKSRoundOps<BE>,
    {
        let plain = |infos: &GLWELayout| {
            self.glwe_encrypt_sk_tmp_bytes(infos)
                .max(self.vec_znx_lsh_tmp_bytes(infos.size()))
                .max(self.vec_znx_rsh_tmp_bytes(infos.size()))
                .max(self.glwe_normalize_tmp_bytes())
        };
        let layout = CKKSLayout {
            glwe_layout: ct_infos.glwe_layout(),
            meta: ct_infos.meta(),
        };
        // A replicated plaintext encrypts at most `MAX_PASS_BITS` wider, then rounds.
        let mut wide = layout;
        wide.glwe_layout.k = (ct_infos.k().as_usize() + MAX_PASS_BITS).into();
        plain(&layout.glwe_layout).max(
            self.glwe_bytes_of_from_infos(&wide.glwe_layout)
                + plain(&wide.glwe_layout).max(self.ckks_round_tmp_bytes(&layout, &wide)),
        )
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
        Dct: GLWEToBackendMut<BE> + GLWEInfos + CKKSInfos + SetCKKSInfos,
        Dpt: GLWEToBackendRef<BE> + CKKSInfos + IntPolyInfos,
        Self: ModuleN
            + GLWEEncryptSk<BE>
            + GLWENormalize<BE>
            + VecZnxLshAdd<BE>
            + VecZnxRshAdd<BE>
            + CKKSPlaintextVecReference<BE>
            + CKKSRoundOps<BE>,
    {
        ensure_encryption_degrees(
            "ckks_encrypt_sk",
            self.n(),
            ct.n().as_usize(),
            sk.to_backend_ref().n().as_usize(),
        )?;
        let log_budget = checked_log_budget_sub("ckks_encrypt_sk", ct.k().as_usize(), pt.log_delta())?;
        // The plaintext addition's checks at the budget the encryption stamps, before the first draw.
        ensure_plaintext_vec_operand(
            "ckks_encrypt_sk",
            ct.base2k().as_usize(),
            ct.n().as_usize(),
            log_budget,
            pt,
            BE::MIN_DEGREE,
        )?;
        let SlotLayout::Replicated(layout) = pt.slot_layout() else {
            self.ckks_encrypt_aligned(ct, pt, log_budget, pt.log_delta(), sk, source_xe, source_xa, scratch)?;
            let noise =
                ComponentNoise::from_secret_at(*sk.to_backend_ref().dist(), ct.k(), ct.to_backend_ref().rank().as_usize());
            ct.set_noise(Some(noise));
            return Ok(());
        };
        // Encrypting `λ` bits wider puts the fresh error under the bits that
        // shaped rounding then drops.
        let lambda = ShapingProfile::new(&layout).max_pass_bits();
        let mut wide = ct.glwe_layout();
        wide.k = (wide.k.as_usize() + lambda).into();
        let (mut tmp, mut scratch_1) = scratch.borrow().take_ckks_ciphertext_scratch(&wide, pt.meta());
        self.ckks_encrypt_aligned(
            &mut tmp,
            pt,
            log_budget,
            pt.log_delta() + lambda,
            sk,
            source_xe,
            source_xa,
            &mut scratch_1,
        )?;
        self.ckks_round_into(ct, &tmp, &mut scratch_1)
    }

    /// Encrypts `pt` into `ct` labelled `(log_budget, log_delta)`, the
    /// plaintext placed at `2^-log_budget`.
    #[allow(clippy::too_many_arguments)]
    fn ckks_encrypt_aligned<Dct, Dpt, S>(
        &self,
        ct: &mut Dct,
        pt: &Dpt,
        log_budget: usize,
        log_delta: usize,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        S: GLWESecretPreparedToBackendRef<BE>,
        Dct: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        Dpt: GLWEToBackendRef<BE> + CKKSInfos + IntPolyInfos,
        Self: GLWEEncryptSk<BE> + GLWENormalize<BE> + VecZnxLshAdd<BE> + VecZnxRshAdd<BE> + CKKSPlaintextVecReference<BE>,
    {
        self.glwe_encrypt_zero_sk(ct, sk, source_xe, source_xa, scratch);
        ct.set_log_budget(log_budget);
        ct.set_log_delta(log_delta);
        ct.set_slots(pt.slots());
        ct.set_slot_layout(pt.slot_layout());
        self.ckks_add_pt_vec_into_reference(ct, pt, scratch)?;
        // A plaintext is dense at its own degree.
        ct.set_log_sparsity(ct.log_n() - pt.log_n());
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
        Self: GLWEDecrypt<BE>
            + VecZnxLsh<BE>
            + VecZnxLshTmpBytes
            + VecZnxRsh<BE>
            + VecZnxRshTmpBytes
            + CKKSPlaintextVecReference<BE>,
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
        Self: ModuleN
            + GLWEDecrypt<BE>
            + GLWEBytesOf<BE>
            + CKKSPlaintextVecReference<BE>
            + VecZnxLsh<BE>
            + VecZnxLshTmpBytes
            + VecZnxRsh<BE>
            + VecZnxRshTmpBytes
            + VecZnxSwitchRing<BE>,
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
        // The extraction's checks, before the first secret-derived write.
        ensure_base2k_match("ckks_decrypt", ct.base2k().as_usize(), pt.base2k().as_usize())?;
        ensure_plaintext_alignment(
            "ckks_decrypt",
            ct.log_budget(),
            pt.log_delta(),
            pt.log_delta() + pt.log_budget(),
        )?;
        let tmp_bytes = self.ckks_decrypt_tmp_bytes_reference(&*pt, ct);
        ckks_ensure!(scratch.available() >= tmp_bytes, "ckks_decrypt: scratch below its query");
        {
            let (mut full_pt, mut scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(ct);
            self.glwe_decrypt(ct, &mut full_pt, sk, &mut scratch_1);
            if pt_n == ct_n {
                CKKSPlaintextVecReference::ckks_extract_pt_with_meta_reference(self, pt, &full_pt, ct.meta(), &mut scratch_1)?;
            } else {
                let (mut below, mut scratch_2) = scratch_1.take_glwe_plaintext_scratch(&below_layout(pt, ct));
                self.vec_znx_switch_ring(below.to_backend_mut().data_mut(), 0, full_pt.to_backend_ref().data(), 0);
                CKKSPlaintextVecReference::ckks_extract_pt_with_meta_reference(self, pt, &below, ct.meta(), &mut scratch_2)?;
            }
        }
        scratch.wipe(tmp_bytes);
        Ok(())
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
