use poulpy_core::layouts::{
    BSGSMeta, GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, SetBSGSMeta,
    prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_core::{
    BSGSConstAdd, BSGSPrecision, GLWEAdd, GLWECopy, GLWEMulConst, GLWENormalize, GLWEPolynomialEvaluation, GLWEShift,
    GLWETensoring, GLWEZero, GiantStepTensorBounds, ScratchArenaTakeCore,
};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Module, ScratchArena},
};

use crate::{
    SetCKKSInfos,
    api::{BSGSPolynomialInfos, BabyStep as BabyStepInfos, CKKSAddOps, CKKSImagOps, CKKSMulAddOps, CKKSMulOps, PowerBasisHelper},
    checked_log_budget_sub, checked_mul_ct_log_budget, checked_mul_pt_log_budget,
    layouts::{CKKSCiphertext, CKKSModuleAlloc},
    polynomial::ComplexBSGSPolynomial,
};
use anyhow::{Result, ensure};

struct EvaluatedBabyStep<D: poulpy_hal::layouts::Data> {
    degree: usize,
    value: CKKSCiphertext<D>,
}

impl<BE, D> BabyStepInfos<BE> for EvaluatedBabyStep<D>
where
    BE: Backend,
    D: poulpy_hal::layouts::Data,
    CKKSCiphertext<D>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
{
    type Value = CKKSCiphertext<D>;

    fn degree(&self) -> usize {
        self.degree
    }

    fn get(&self) -> &Self::Value {
        &self.value
    }

    fn get_mut(&mut self) -> &mut Self::Value {
        &mut self.value
    }
}

/// CKKS precision and plaintext-coupled ops for the core BSGS engine.
pub struct CKKSBSGSPrecision<'m, BE: Backend> {
    module: &'m Module<BE>,
}

impl<BE: Backend> BSGSPrecision<BE> for CKKSBSGSPrecision<'_, BE> {
    fn mul_ct_params<R, A, B>(&self, res: &R, a: &A, b: &B) -> Result<(usize, usize, usize)>
    where
        R: GLWEInfos + BSGSMeta,
        A: GLWEInfos + BSGSMeta,
        B: GLWEInfos + BSGSMeta,
    {
        let res_log_budget = checked_mul_ct_log_budget(
            "mul",
            a.bsgs_log_budget(),
            b.bsgs_log_budget(),
            a.bsgs_log_delta(),
            b.bsgs_log_delta(),
        )?;
        let res_log_delta = a.bsgs_log_delta().min(b.bsgs_log_delta());
        let res_offset = (res_log_budget + res_log_delta).saturating_sub(res.max_k().as_usize());
        let cnv_offset = a.bsgs_effective_k().max(b.bsgs_effective_k()) + res_offset;
        Ok((
            checked_log_budget_sub("mul", res_log_budget, res_offset)?,
            res_log_delta,
            cnv_offset,
        ))
    }

    fn mul_pt_params<R, A, P>(&self, res: &R, a: &A, pt: &P) -> Result<(usize, usize, usize)>
    where
        R: GLWEInfos + BSGSMeta,
        A: GLWEInfos + BSGSMeta,
        P: GLWEInfos + BSGSMeta,
    {
        let res_log_budget = checked_mul_pt_log_budget(
            "mul",
            a.bsgs_log_budget(),
            pt.bsgs_log_budget(),
            a.bsgs_log_delta(),
            pt.bsgs_log_delta(),
        )?;
        let res_log_delta = a.bsgs_log_delta();
        let res_offset = (res_log_budget + res_log_delta).saturating_sub(res.max_k().as_usize());
        let cnv_offset = pt.max_k().as_usize() + res_offset;
        Ok((
            checked_log_budget_sub("mul", res_log_budget, res_offset)?,
            res_log_delta,
            cnv_offset,
        ))
    }
}

impl<BE: Backend, R, P> BSGSConstAdd<BE, R, P> for CKKSBSGSPrecision<'_, BE>
where
    Module<BE>: CKKSAddOps<BE>,
    R: GLWEToBackendMut<BE> + crate::CKKSCtBounds + SetCKKSInfos,
    P: GLWEToBackendRef<BE> + crate::CKKSCtBounds,
{
    fn add_pt_const_assign(
        &self,
        res: &mut R,
        res_coeff: usize,
        coeffs: &P,
        idx: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()> {
        self.module.ckks_add_pt_const_assign(res, res_coeff, coeffs, idx, scratch)
    }
}

pub trait PolynomialEvaluationDefault<BE: Backend> {
    fn ckks_eval_poly_real_const_coeffs_from_power_basis_default<R, B, A, G, T>(
        &self,
        res: &mut R,
        poly: &B,
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GiantStepTensorBounds<BE>
            + GLWEMulConst<BE>
            + GLWEAdd<BE>
            + GLWEShift<BE>
            + GLWETensoring<BE>
            + GLWENormalize<BE>
            + GLWEZero<BE>
            + GLWECopy<BE>
            + CKKSAddOps<BE>
            + CKKSMulAddOps<BE>
            + CKKSModuleAlloc<BE>
            + Sized,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta + SetCKKSInfos + crate::CKKSCtBounds,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: crate::CKKSCtBounds,
        A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + crate::CKKSCtBounds,
        G: PowerBasisHelper<BE, A>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        for<'b> ScratchArena<'b, BE>: ScratchAvailable + ScratchArenaTakeCore<'b, BE>;

    fn ckks_eval_poly_complex_const_coeffs_from_power_basis_default<R, C, A, G, T>(
        &self,
        res: &mut R,
        poly: &ComplexBSGSPolynomial<C>,
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GiantStepTensorBounds<BE>
            + GLWEMulConst<BE>
            + GLWEAdd<BE>
            + GLWEShift<BE>
            + GLWETensoring<BE>
            + GLWENormalize<BE>
            + GLWEZero<BE>
            + GLWECopy<BE>
            + CKKSAddOps<BE>
            + CKKSImagOps<BE>
            + CKKSMulOps<BE>
            + CKKSMulAddOps<BE>
            + CKKSModuleAlloc<BE>
            + Sized,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta + SetCKKSInfos + crate::CKKSCtBounds,
        C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + crate::CKKSCtBounds,
        A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + crate::CKKSCtBounds,
        G: PowerBasisHelper<BE, A>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        for<'b> ScratchArena<'b, BE>: ScratchAvailable + ScratchArenaTakeCore<'b, BE>;
}

impl<BE: Backend> PolynomialEvaluationDefault<BE> for Module<BE> {
    fn ckks_eval_poly_real_const_coeffs_from_power_basis_default<R, B, A, G, T>(
        &self,
        res: &mut R,
        poly: &B,
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GiantStepTensorBounds<BE>
            + GLWEMulConst<BE>
            + GLWEAdd<BE>
            + GLWEShift<BE>
            + GLWETensoring<BE>
            + GLWENormalize<BE>
            + GLWEZero<BE>
            + GLWECopy<BE>
            + GLWEPolynomialEvaluation<BE>
            + CKKSAddOps<BE>
            + CKKSMulAddOps<BE>
            + CKKSModuleAlloc<BE>
            + Sized,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta + SetCKKSInfos + crate::CKKSCtBounds,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: crate::CKKSCtBounds,
        A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + crate::CKKSCtBounds,
        G: PowerBasisHelper<BE, A>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        for<'b> ScratchArena<'b, BE>: ScratchAvailable + ScratchArenaTakeCore<'b, BE>,
    {
        ensure!(
            poly.baby_steps() > 0,
            "ckks_eval_poly_real_const_coeffs_from_power_basis: polynomial must contain at least one baby step"
        );
        let poly_basis = poly.basis();
        let power_basis_basis = power_basis.basis();
        ensure!(
            poly_basis == power_basis_basis,
            "ckks_eval_poly_real_const_coeffs_from_power_basis: polynomial basis {poly_basis:?} does not match power basis {power_basis_basis:?}"
        );

        let n_baby = poly.baby_steps();
        let last_coeffs = poly.baby_step(n_baby - 1);
        let trailing_const_only = n_baby >= 2 && last_coeffs.n().as_usize() == 1;
        let fold_power = poly.degree();
        let can_fold = trailing_const_only && power_basis.has_power(fold_power);

        let n_to_process = if can_fold { n_baby - 1 } else { n_baby };
        let mut baby_steps = Vec::with_capacity(n_to_process);
        let parity = poly.parity();
        let x = power_basis.get(1)?;
        let precision = CKKSBSGSPrecision { module: self };
        for i in 0..n_to_process {
            let coeffs = poly.baby_step(i);
            let degree = coeffs.n().as_usize() - 1;
            let mut value = self.ckks_ciphertext_alloc_from_infos(x);
            value.set_meta(x.meta());
            self.glwe_eval_baby_step::<_, _, _, A, G>(
                &precision,
                &mut value,
                parity,
                coeffs,
                power_basis,
                &mut scratch.borrow(),
            )?;
            baby_steps.push(EvaluatedBabyStep { degree, value });
        }

        self.glwe_eval_giant_steps(&precision, res, &mut baby_steps, power_basis, tsk, &mut scratch.borrow())?;

        if can_fold {
            let xpow = power_basis.get(fold_power)?;
            self.ckks_mul_add_pt_const_into(res, xpow, last_coeffs, 0, scratch)?;
        }

        Ok(())
    }

    fn ckks_eval_poly_complex_const_coeffs_from_power_basis_default<R, C, A, G, T>(
        &self,
        res: &mut R,
        poly: &ComplexBSGSPolynomial<C>,
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GiantStepTensorBounds<BE>
            + GLWEMulConst<BE>
            + GLWEAdd<BE>
            + GLWEShift<BE>
            + GLWETensoring<BE>
            + GLWENormalize<BE>
            + GLWEZero<BE>
            + GLWECopy<BE>
            + GLWEPolynomialEvaluation<BE>
            + CKKSAddOps<BE>
            + CKKSImagOps<BE>
            + CKKSMulOps<BE>
            + CKKSMulAddOps<BE>
            + CKKSModuleAlloc<BE>
            + Sized,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta + SetCKKSInfos + crate::CKKSCtBounds,
        C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + crate::CKKSCtBounds,
        A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + crate::CKKSCtBounds,
        G: PowerBasisHelper<BE, A>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        for<'b> ScratchArena<'b, BE>: ScratchAvailable + ScratchArenaTakeCore<'b, BE>,
    {
        let poly_re = &poly.re;
        let poly_im = &poly.im;
        let n_baby = BSGSPolynomialInfos::<BE>::baby_steps(poly_re);
        ensure!(
            n_baby > 0,
            "ckks_eval_poly_complex_const_coeffs_from_power_basis: polynomial must contain at least one baby step"
        );
        ensure!(
            BSGSPolynomialInfos::<BE>::baby_steps(poly_im) == n_baby,
            "ckks_eval_poly_complex_const_coeffs_from_power_basis: real/imag baby-step schedules differ"
        );
        let poly_basis = BSGSPolynomialInfos::<BE>::basis(poly_re);
        let power_basis_basis = power_basis.basis();
        ensure!(
            poly_basis == power_basis_basis,
            "ckks_eval_poly_complex_const_coeffs_from_power_basis: polynomial basis {poly_basis:?} does not match power basis {power_basis_basis:?}"
        );

        // Fold the trailing lone constant via the highest power when present.
        let last_re = BSGSPolynomialInfos::<BE>::baby_step(poly_re, n_baby - 1);
        let trailing_const_only = n_baby >= 2 && last_re.n().as_usize() == 1;
        let fold_power = BSGSPolynomialInfos::<BE>::degree(poly_re);
        let can_fold = trailing_const_only && power_basis.has_power(fold_power);
        let n_to_process = if can_fold { n_baby - 1 } else { n_baby };

        // Per baby step: baby_i = eval(re_i) + i·eval(im_i). A single giant tree
        // over baby_i runs the relinearizations once.
        let parity = BSGSPolynomialInfos::<BE>::parity(poly_re);
        let x = power_basis.get(1)?;
        let precision = CKKSBSGSPrecision { module: self };
        let mut baby_steps = Vec::with_capacity(n_to_process);
        for i in 0..n_to_process {
            let re_coeffs = BSGSPolynomialInfos::<BE>::baby_step(poly_re, i);
            let im_coeffs = BSGSPolynomialInfos::<BE>::baby_step(poly_im, i);
            let degree = re_coeffs.n().as_usize() - 1;

            let mut value = self.ckks_ciphertext_alloc_from_infos(x);
            value.set_meta(x.meta());
            self.glwe_eval_baby_step::<_, _, _, A, G>(
                &precision,
                &mut value,
                parity,
                re_coeffs,
                power_basis,
                &mut scratch.borrow(),
            )?;

            let mut im_value = self.ckks_ciphertext_alloc_from_infos(x);
            im_value.set_meta(x.meta());
            self.glwe_eval_baby_step::<_, _, _, A, G>(
                &precision,
                &mut im_value,
                parity,
                im_coeffs,
                power_basis,
                &mut scratch.borrow(),
            )?;
            self.ckks_mul_i_assign(&mut im_value, &mut scratch.borrow())?;
            self.ckks_add_assign(&mut value, &im_value, &mut scratch.borrow())?;

            baby_steps.push(EvaluatedBabyStep { degree, value });
        }

        self.glwe_eval_giant_steps(&precision, res, &mut baby_steps, power_basis, tsk, &mut scratch.borrow())?;

        if can_fold {
            // res += a·x^fold + i·(b·x^fold), with a = last_re[0], b = last_im[0].
            let last_im = BSGSPolynomialInfos::<BE>::baby_step(poly_im, n_baby - 1);
            let xpow = power_basis.get(fold_power)?;
            self.ckks_mul_add_pt_const_into(res, xpow, last_re, 0, scratch)?;
            let mut im_fold = self.ckks_ciphertext_alloc_from_infos(res);
            self.ckks_mul_pt_const_into(&mut im_fold, xpow, last_im, 0, scratch)?;
            self.ckks_mul_i_assign(&mut im_fold, scratch)?;
            self.ckks_add_assign(res, &im_fold, scratch)?;
        }

        Ok(())
    }
}
