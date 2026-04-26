use anyhow::Result;
use poulpy_core::{
    GLWEAdd, GLWENormalize, GLWEShift, ScratchTakeCore,
    layouts::{GLWEInfos, GLWEPlaintext, GLWEPlaintextLayout, LWEInfos},
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxRshAddInto, VecZnxRshTmpBytes},
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ZnxViewMut},
};

use crate::{
    CKKSInfos, CKKSMeta, checked_log_budget_sub, ensure_plaintext_alignment,
    layouts::{
        CKKSCiphertext,
        ciphertext::CKKSOffset,
        plaintext::{
            CKKSConstPlaintextConversion, CKKSPlaintextConversion, CKKSPlaintextCstRnx, CKKSPlaintextCstZnx, CKKSPlaintextVecRnx,
            CKKSPlaintextVecZnx,
        },
    },
    leveled::default::CKKSPlaintextZnxDefault,
};

pub(crate) trait CKKSAddDefault<BE: Backend> {
    fn ckks_add_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE> + GLWENormalize<BE>,
    {
        self.glwe_shift_tmp_bytes().max(self.glwe_normalize_tmp_bytes())
    }

    fn ckks_add_pt_vec_znx_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE> + GLWENormalize<BE> + VecZnxRshTmpBytes,
    {
        self.glwe_shift_tmp_bytes()
            .max(self.vec_znx_rsh_tmp_bytes())
            .max(self.glwe_normalize_tmp_bytes())
    }

    fn ckks_add_pt_vec_rnx_tmp_bytes_default<R, A>(&self, res: &R, _a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEShift<BE> + GLWENormalize<BE> + VecZnxRshTmpBytes,
    {
        let b_infos = GLWEPlaintextLayout {
            n: self.n().into(),
            base2k: res.base2k(),
            k: b.min_k(res.base2k()),
        };
        GLWEPlaintext::<Vec<u8>>::bytes_of_from_infos(&b_infos) + self.ckks_add_pt_vec_znx_tmp_bytes_default()
    }

    fn ckks_add_pt_const_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE> + GLWENormalize<BE>,
    {
        self.glwe_shift_tmp_bytes().max(self.glwe_normalize_tmp_bytes())
    }

    fn ckks_add_into_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEShift<BE> + GLWENormalize<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_add_into_unsafe_default(dst, a, b, scratch)?;
        self.glwe_normalize_assign(dst, scratch);
        Ok(())
    }

    fn ckks_add_into_unsafe_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let offset = dst.offset_binary(a, b);

        if offset == 0 && a.log_budget() == b.log_budget() {
            self.glwe_add_into(dst, a, b);
        } else if a.log_budget() <= b.log_budget() {
            self.glwe_lsh(dst, a, offset, scratch);
            self.glwe_lsh_add(dst, b, b.log_budget() - a.log_budget() + offset, scratch);
        } else {
            self.glwe_lsh(dst, b, offset, scratch);
            self.glwe_lsh_add(dst, a, a.log_budget() - b.log_budget() + offset, scratch);
        }

        let log_budget = checked_log_budget_sub("add", a.log_budget().min(b.log_budget()), offset)?;
        dst.meta.log_delta = a.log_delta().max(b.log_delta());
        dst.meta.log_budget = log_budget;
        Ok(())
    }

    fn ckks_add_assign_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEShift<BE> + GLWENormalize<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_add_assign_unsafe_default(dst, a, scratch)?;
        self.glwe_normalize_assign(dst, scratch);
        Ok(())
    }

    fn ckks_add_assign_unsafe_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let dst_log_budget = dst.log_budget();

        if dst_log_budget < a.log_budget() {
            self.glwe_lsh_add(dst, a, a.log_budget() - dst_log_budget, scratch);
        } else if dst_log_budget > a.log_budget() {
            self.glwe_lsh_assign(dst, dst_log_budget - a.log_budget(), scratch);
            self.glwe_add_assign(dst, a);
        } else {
            self.glwe_add_assign(dst, a);
        }

        dst.meta.log_budget = dst_log_budget.min(a.log_budget());
        Ok(())
    }

    fn ckks_add_pt_vec_znx_into_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshAddInto<BE> + GLWEShift<BE> + GLWENormalize<BE> + CKKSPlaintextZnxDefault<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_add_pt_vec_znx_into_unsafe_default(dst, a, pt_znx, scratch)?;
        self.glwe_normalize_assign(dst, scratch);
        Ok(())
    }

    fn ckks_add_pt_vec_znx_into_unsafe_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshAddInto<BE> + GLWEShift<BE> + CKKSPlaintextZnxDefault<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let offset = dst.offset_unary(a);
        self.glwe_lsh(dst, a, offset, scratch);
        dst.meta = a.meta();
        dst.meta.log_budget = checked_log_budget_sub("add_pt_vec_znx", a.log_budget(), offset)?;
        self.ckks_add_pt_vec_znx_assign_unsafe_default(dst, pt_znx, scratch)?;
        Ok(())
    }

    fn ckks_add_pt_vec_znx_assign_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshAddInto<BE> + GLWENormalize<BE> + CKKSPlaintextZnxDefault<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_add_pt_vec_znx_assign_unsafe_default(dst, pt_znx, scratch)?;
        self.glwe_normalize_assign(dst, scratch);
        Ok(())
    }

    fn ckks_add_pt_vec_znx_assign_unsafe_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshAddInto<BE> + CKKSPlaintextZnxDefault<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSPlaintextZnxDefault::ckks_add_pt_vec_znx_into_default(self, dst, pt_znx, scratch)?;
        Ok(())
    }

    fn ckks_add_pt_vec_rnx_into_default<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshAddInto<BE> + GLWEShift<BE> + GLWENormalize<BE> + CKKSPlaintextZnxDefault<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        self.ckks_add_pt_vec_rnx_into_unsafe_default(dst, a, pt_rnx, prec, scratch)?;
        self.glwe_normalize_assign(dst, scratch);
        Ok(())
    }

    fn ckks_add_pt_vec_rnx_into_unsafe_default<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshAddInto<BE> + GLWEShift<BE> + CKKSPlaintextZnxDefault<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        let (pt_glwe, scratch_1) = scratch.take_glwe_plaintext(&GLWEPlaintextLayout {
            n: self.n().into(),
            base2k: dst.base2k(),
            k: prec.min_k(dst.base2k()),
        });
        let mut pt_znx = CKKSPlaintextVecZnx::from_plaintext_with_meta(pt_glwe, prec);
        pt_rnx.to_znx(&mut pt_znx)?;
        CKKSAddDefault::ckks_add_pt_vec_znx_into_unsafe_default(self, dst, a, &pt_znx, scratch_1)?;
        Ok(())
    }

    fn ckks_add_pt_vec_rnx_assign_default<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshAddInto<BE> + GLWENormalize<BE> + CKKSPlaintextZnxDefault<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        self.ckks_add_pt_vec_rnx_assign_unsafe_default(dst, pt_rnx, prec, scratch)?;
        self.glwe_normalize_assign(dst, scratch);
        Ok(())
    }

    fn ckks_add_pt_vec_rnx_assign_unsafe_default<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshAddInto<BE> + CKKSPlaintextZnxDefault<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        let (pt_glwe, scratch_1) = scratch.take_glwe_plaintext(&GLWEPlaintextLayout {
            n: self.n().into(),
            base2k: dst.base2k(),
            k: prec.min_k(dst.base2k()),
        });
        let mut pt_znx = CKKSPlaintextVecZnx::from_plaintext_with_meta(pt_glwe, prec);
        pt_rnx.to_znx(&mut pt_znx)?;
        self.ckks_add_pt_vec_znx_assign_unsafe_default(dst, &pt_znx, scratch_1)?;
        Ok(())
    }

    fn ckks_add_pt_const_znx_into_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE> + GLWENormalize<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_add_pt_const_znx_into_unsafe_default(dst, a, cst_znx, scratch)?;
        self.glwe_normalize_assign(dst, scratch);
        Ok(())
    }

    fn ckks_add_pt_const_znx_into_unsafe_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let offset = dst.offset_unary(a);
        self.glwe_lsh(dst, a, offset, scratch);
        dst.meta = a.meta();
        dst.meta.log_budget = checked_log_budget_sub("add_const_znx", a.log_budget(), offset)?;
        self.ckks_add_pt_const_znx_assign_unsafe_default(dst, cst_znx, scratch)
    }

    fn ckks_add_pt_const_znx_assign_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWENormalize<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_add_pt_const_znx_assign_unsafe_default(dst, cst_znx, scratch)?;
        self.glwe_normalize_assign(dst, scratch);
        Ok(())
    }

    fn ckks_add_pt_const_znx_assign_unsafe_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_znx: &CKKSPlaintextCstZnx,
        _scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        if cst_znx.re().is_none() && cst_znx.im().is_none() {
            return Ok(());
        }

        let _offset = ensure_plaintext_alignment(
            "ckks_add_pt_const_znx_into",
            dst.log_budget(),
            cst_znx.log_delta(),
            cst_znx.effective_k(),
        )?;
        let n = dst.n().as_usize();
        if let Some(coeff) = cst_znx.re() {
            for (limb, digit) in coeff.iter().enumerate() {
                dst.data_mut().at_mut(0, limb)[0] += *digit;
            }
        }
        if let Some(coeff) = cst_znx.im() {
            for (limb, digit) in coeff.iter().enumerate() {
                dst.data_mut().at_mut(0, limb)[n / 2] += *digit;
            }
        }
        Ok(())
    }

    fn ckks_add_pt_const_rnx_into_default<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE> + GLWENormalize<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        self.ckks_add_pt_const_rnx_into_unsafe_default(dst, a, cst_rnx, prec, scratch)?;
        self.glwe_normalize_assign(dst, scratch);
        Ok(())
    }

    fn ckks_add_pt_const_rnx_into_unsafe_default<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        let offset = dst.offset_unary(a);
        if cst_rnx.re().is_none() && cst_rnx.im().is_none() {
            self.glwe_lsh(dst, a, offset, scratch);
            dst.meta = a.meta();
            dst.meta.log_budget = checked_log_budget_sub("add_const_rnx", a.log_budget(), offset)?;
            return Ok(());
        }

        let res_log_budget = checked_log_budget_sub("add_const_rnx", a.log_budget(), offset)?;
        let cst_znx = cst_rnx.to_znx_at_k(
            dst.base2k(),
            res_log_budget
                .checked_add(prec.log_delta)
                .expect("aligned precision overflow"),
            prec.log_delta,
        )?;
        self.ckks_add_pt_const_znx_into_unsafe_default(dst, a, &cst_znx, scratch)
    }

    fn ckks_add_pt_const_rnx_assign_default<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWENormalize<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        self.ckks_add_pt_const_rnx_assign_unsafe_default(dst, cst_rnx, prec, scratch)?;
        self.glwe_normalize_assign(dst, scratch);
        Ok(())
    }

    fn ckks_add_pt_const_rnx_assign_unsafe_default<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        if cst_rnx.re().is_none() && cst_rnx.im().is_none() {
            return Ok(());
        }

        let cst_znx = cst_rnx.to_znx_at_k(
            dst.base2k(),
            dst.log_budget()
                .checked_add(prec.log_delta)
                .expect("aligned precision overflow"),
            prec.log_delta,
        )?;
        self.ckks_add_pt_const_znx_assign_unsafe_default(dst, &cst_znx, scratch)
    }
}

impl<BE: Backend> CKKSAddDefault<BE> for Module<BE> {}
