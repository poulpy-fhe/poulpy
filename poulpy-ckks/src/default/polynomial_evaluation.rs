use poulpy_core::layouts::{
    BSGSMeta, GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, SetBSGSMeta,
    prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_core::{BSGSBabyOps, BSGSGiantOps, GLWEPolynomialEvaluation, GLWEZero};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::CKKSCtBounds;
use crate::{
    SetCKKSInfos,
    api::{
        BSGSPolynomialInfos, BabyStep as BabyStepInfos, CKKSAddOps, CKKSCopyOps, CKKSImagOps, CKKSMulAddOps, CKKSMulOps,
        PowerBasisHelper,
    },
    layouts::{CKKSCiphertext, CKKSModuleAlloc, CKKSPreparedRight},
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

/// CKKS scale-aware operations for the scale-agnostic core BSGS engine.
///
/// Every method is a thin dispatch to the CKKS API ([`CKKSMulOps`], [`CKKSAddOps`],
/// [`CKKSCopyOps`]); the only non-dispatch bits are the accumulator seed
/// ([`BSGSBabyOps::init_accumulator`]) and the compactions, which have no single
/// existing API equivalent.
struct CKKSBSGSOps;

impl<BE: Backend, V, P, A> BSGSBabyOps<BE, V, P, A> for CKKSBSGSOps
where
    Module<BE>: CKKSAddOps<BE> + CKKSMulOps<BE> + CKKSMulAddOps<BE> + GLWEZero<BE>,
    V: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    P: GLWEToBackendRef<BE> + CKKSCtBounds,
    A: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
{
    fn init_accumulator(&self, module: &Module<BE>, res: &mut V, seed: &A, _scratch: &mut ScratchArena<'_, BE>) -> Result<()> {
        res.set_bsgs_log_budget(seed.bsgs_log_budget());
        res.set_bsgs_log_delta(seed.bsgs_log_delta());
        SetBSGSMeta::compact_in_place(res);
        module.glwe_zero(res);
        Ok(())
    }

    fn add_pt_const_assign(
        &self,
        module: &Module<BE>,
        res: &mut V,
        res_coeff: usize,
        coeffs: &P,
        idx: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()> {
        module.ckks_add_pt_const_assign(res, res_coeff, coeffs, idx, scratch)
    }

    fn mul_pt_const(
        &self,
        module: &Module<BE>,
        res: &mut V,
        a: &A,
        coeffs: &P,
        idx: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()> {
        module.ckks_mul_pt_const_into(res, a, coeffs, idx, scratch)
    }

    fn mul_add_pt_const(
        &self,
        module: &Module<BE>,
        res: &mut V,
        a: &A,
        coeffs: &P,
        idx: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()> {
        module.ckks_mul_add_pt_const_into(res, a, coeffs, idx, scratch)
    }

    fn compact(&self, res: &mut V) {
        SetBSGSMeta::compact_in_place(res);
    }
}

impl<BE: Backend, V, A, R> BSGSGiantOps<BE, V, A, R> for CKKSBSGSOps
where
    Module<BE>: CKKSMulOps<BE> + CKKSAddOps<BE> + CKKSCopyOps<BE>,
    V: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
{
    type Prepared = CKKSPreparedRight<BE>;

    fn prepare_right(&self, module: &Module<BE>, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<Self::Prepared> {
        module.ckks_prepare_right(a, scratch)
    }

    fn mul_prepared_assign<T>(
        &self,
        module: &Module<BE>,
        dst: &mut V,
        prepared: &Self::Prepared,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    {
        module.ckks_mul_prepared_assign(dst, prepared, tsk, scratch)
    }

    fn add_assign(&self, module: &Module<BE>, dst: &mut V, a: &V, scratch: &mut ScratchArena<'_, BE>) -> Result<()> {
        module.ckks_add_assign(dst, a, scratch)
    }

    fn copy(&self, module: &Module<BE>, res: &mut R, src: &V, scratch: &mut ScratchArena<'_, BE>) -> Result<()> {
        module.ckks_copy(res, src, scratch)?;
        SetBSGSMeta::compact_in_place(res);
        Ok(())
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
        Self: GLWEZero<BE>
            + CKKSAddOps<BE>
            + CKKSMulOps<BE>
            + CKKSMulAddOps<BE>
            + CKKSModuleAlloc<BE>
            + CKKSCopyOps<BE>
            + Sized,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta + SetCKKSInfos + CKKSCtBounds,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds,
        A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds,
        G: PowerBasisHelper<BE, A>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;

    fn ckks_eval_poly_complex_const_coeffs_from_power_basis_default<R, C, A, G, T>(
        &self,
        res: &mut R,
        poly: &ComplexBSGSPolynomial<C>,
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEZero<BE>
            + CKKSAddOps<BE>
            + CKKSImagOps<BE>
            + CKKSMulOps<BE>
            + CKKSMulAddOps<BE>
            + CKKSModuleAlloc<BE>
            + CKKSCopyOps<BE>
            + Sized,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta + SetCKKSInfos + CKKSCtBounds,
        C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds,
        A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds,
        G: PowerBasisHelper<BE, A>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;
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
        Self: GLWEZero<BE>
            + GLWEPolynomialEvaluation<BE>
            + CKKSAddOps<BE>
            + CKKSMulOps<BE>
            + CKKSMulAddOps<BE>
            + CKKSModuleAlloc<BE>
            + CKKSCopyOps<BE>
            + Sized,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta + SetCKKSInfos + CKKSCtBounds,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds,
        A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds,
        G: PowerBasisHelper<BE, A>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
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
        let precision = CKKSBSGSOps;
        for i in 0..n_to_process {
            let coeffs = poly.baby_step(i);
            let degree = coeffs.n().as_usize() - 1;
            let mut value = self.ckks_ciphertext_alloc_from_infos(x);
            value.set_meta(x.meta());
            self.glwe_eval_baby_step(
                &precision,
                &mut value,
                parity,
                coeffs,
                power_basis,
                &mut scratch.borrow(),
            )?;
            baby_steps.push(EvaluatedBabyStep { degree, value });
        }

        self.glwe_eval_giant_steps(&precision, res, &mut baby_steps, power_basis, tsk, &mut scratch.borrow())?; //TODO: ensure each giant-step intermediate state is compacted

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
        Self: GLWEZero<BE>
            + GLWEPolynomialEvaluation<BE>
            + CKKSAddOps<BE>
            + CKKSImagOps<BE>
            + CKKSMulOps<BE>
            + CKKSMulAddOps<BE>
            + CKKSModuleAlloc<BE>
            + CKKSCopyOps<BE>
            + Sized,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta + SetCKKSInfos + CKKSCtBounds,
        C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds,
        A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds,
        G: PowerBasisHelper<BE, A>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
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
        ensure!(
            BSGSPolynomialInfos::<BE>::degree(poly_im) == BSGSPolynomialInfos::<BE>::degree(poly_re),
            "ckks_eval_poly_complex_const_coeffs_from_power_basis: real/imag degrees differ"
        );
        ensure!(
            BSGSPolynomialInfos::<BE>::parity(poly_im) == BSGSPolynomialInfos::<BE>::parity(poly_re),
            "ckks_eval_poly_complex_const_coeffs_from_power_basis: real/imag parities differ"
        );
        ensure!(
            BSGSPolynomialInfos::<BE>::basis(poly_im) == BSGSPolynomialInfos::<BE>::basis(poly_re),
            "ckks_eval_poly_complex_const_coeffs_from_power_basis: real/imag bases differ"
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
        let precision = CKKSBSGSOps;
        let mut baby_steps = Vec::with_capacity(n_to_process);
        for i in 0..n_to_process {
            let re_coeffs = BSGSPolynomialInfos::<BE>::baby_step(poly_re, i);
            let im_coeffs = BSGSPolynomialInfos::<BE>::baby_step(poly_im, i);
            ensure!(
                im_coeffs.n() == re_coeffs.n(),
                "ckks_eval_poly_complex_const_coeffs_from_power_basis: real/imag baby-step {i} lengths differ"
            );
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
            ensure!(
                last_im.n() == last_re.n(),
                "ckks_eval_poly_complex_const_coeffs_from_power_basis: real/imag trailing baby-step lengths differ"
            );
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
