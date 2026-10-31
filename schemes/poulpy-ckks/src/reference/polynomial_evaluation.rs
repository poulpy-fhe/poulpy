use crate::{CKKSResult as Result, ckks_ensure_operand};
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{BSGSMeta, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, SetBSGSMeta};
use poulpy_core::oep::{ConstTerm, GLWEMulConstImpl};
use poulpy_core::{GLWEBSGSOps, GLWEPolynomialEvaluation, GLWEZero};
use poulpy_hal::{
    api::{CnvOps, VecZnxBigNormalize, VecZnxRsh, VecZnxRshTmpBytes},
    layouts::{Backend, Data, Module, PrepareHint, Ring, ScratchArena, ZnxWord, vec_znx_backend_ref_from_ref},
};

use crate::CKKSCtBounds;
use crate::{
    SetCKKSInfos,
    api::{
        BSGSPolynomialInfos, BabyStep as BabyStepInfos, Basis, CKKSAddOps, CKKSCopyOps, CKKSImagOps, CKKSMulAddOps, CKKSMulOps,
        CKKSRightOperandPreparedFactory, PowerBasisHelper,
    },
    layouts::{CKKSCiphertext, CKKSCiphertextAlloc, CKKSRightOperandPrepared},
    polynomial::ComplexBSGSPolynomial,
};

struct EvaluatedBabyStep<D: Data, W: ZnxWord, R: Ring> {
    degree: usize,
    value: CKKSCiphertext<D, W, R>,
}

impl<BE, D> BabyStepInfos<BE> for EvaluatedBabyStep<D, BE::ZnxWord, BE::Ring>
where
    BE: Backend,
    D: Data,
    CKKSCiphertext<D, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
{
    type Value = CKKSCiphertext<D, BE::ZnxWord, BE::Ring>;

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
/// [`CKKSCopyOps`]); the only non-dispatch bit is the accumulator seed, which
/// has no single existing API equivalent.
struct CKKSBSGSOps;

impl<BE: Backend, V, P, A, R> GLWEBSGSOps<BE, V, P, A, R> for CKKSBSGSOps
where
    BE: GLWEMulConstImpl,
    Module<BE>: CKKSAddOps<BE>
        + CKKSMulOps<BE>
        + CKKSRightOperandPreparedFactory<BE>
        + CKKSMulAddOps<BE>
        + CKKSCopyOps<BE>
        + GLWEZero<BE>,
    V: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
{
    type Prepared = CKKSRightOperandPrepared<BE>;

    fn init_accumulator(
        &self,
        module: &Module<BE>,
        res: &mut V,
        seed: &A,
        _scratch: &mut ScratchArena<'_, BE>,
    ) -> anyhow::Result<()> {
        res.set_log_delta(seed.log_delta());
        res.set_log_budget(seed.log_budget());
        res.set_slots(seed.slots());
        module.glwe_zero(res);
        Ok(())
    }

    fn eval_baby_linear_combination(
        &self,
        module: &Module<BE>,
        res: &mut V,
        terms: &[(&A, usize)],
        coeffs: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> anyhow::Result<bool> {
        use crate::mul_pt_params_raw;

        let Some(&(seed, _)) = terms.last() else {
            return Ok(false);
        };
        let kb: usize = res.base2k().into();
        let delta = seed.log_delta();
        if kb == 0 {
            return Ok(false);
        }

        let mut budget = seed.log_budget();
        let mut term_params = Vec::with_capacity(terms.len());
        for (t, _) in terms {
            if usize::from(t.base2k()) != kb || t.log_delta() != delta {
                return Ok(false);
            }
            let Ok((b_i, d_i, off_i)) = mul_pt_params_raw(
                "ckks_eval_poly_real_const_coeffs",
                budget + delta,
                t.log_delta(),
                t.log_budget(),
                coeffs.log_delta(),
                coeffs.log_budget(),
                coeffs.encoded_k().as_usize(),
            ) else {
                return Ok(false);
            };
            if d_i != delta || off_i < kb {
                return Ok(false);
            }
            budget = budget.min(b_i);
            term_params.push((b_i, off_i));
        }
        for (term_budget, offset) in &mut term_params {
            *offset += *term_budget - budget;
        }

        let mut sparsity = res.log_sparsity();
        let mut slots = seed.slots();
        for (t, _) in terms {
            sparsity = sparsity.min(t.log_sparsity());
            slots = slots.join(t.slots());
        }
        res.set_log_delta(delta);
        res.set_log_budget(budget);
        res.set_log_sparsity(sparsity);
        res.set_slots(slots);
        let pt_bk = coeffs.to_backend_ref();
        let pt_size = pt_bk.data().size();
        let res_size = res.to_backend_ref().data().size();
        // Splits a term's total bit shift into the sub-limb right-shift of the
        // constant (`r`) and the whole-limb convolution offset (`rho`).
        let shift_split = |off: usize| {
            let r = (kb - off % kb) % kb;
            (r, (off + r) / kb - 1)
        };
        let mut acc_size = res_size;
        for ((t, _), (_, off)) in terms.iter().zip(term_params.iter()) {
            let (r, rho) = shift_split(*off);
            let g_size = pt_size + usize::from(r != 0);
            let needed = (t.to_backend_ref().data().size() + g_size - 1).saturating_sub(rho);
            acc_size = acc_size.max(needed);
        }

        res.set_canonical(true);
        let term_refs: Vec<_> = terms.iter().map(|(t, _)| t.to_backend_ref()).collect();
        let combination: Vec<ConstTerm<'_, BE>> = terms
            .iter()
            .zip(term_params.iter())
            .zip(term_refs.iter())
            .map(|(((_, coeff_idx), (_, off)), t_bk)| {
                let (shift, cnv_offset) = shift_split(*off);
                ConstTerm {
                    a: vec_znx_backend_ref_from_ref::<BE>(t_bk.data()),
                    b: vec_znx_backend_ref_from_ref::<BE>(pt_bk.data()),
                    b_col: 0,
                    b_coeff: *coeff_idx,
                    shift,
                    cnv_offset,
                }
            })
            .collect();
        let mut res_bk = res.to_backend_mut();
        BE::glwe_const_combination(module, res_bk.data_mut(), kb, budget + delta, acc_size, &combination, scratch);
        Ok(true)
    }

    fn add_pt_const_assign(
        &self,
        module: &Module<BE>,
        res: &mut V,
        res_coeff: usize,
        coeffs: &P,
        idx: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> anyhow::Result<()> {
        module
            .ckks_add_pt_const_assign(res, res_coeff, coeffs, idx, scratch)
            .map_err(::anyhow::Error::from)
    }

    fn mul_pt_const(
        &self,
        module: &Module<BE>,
        res: &mut V,
        a: &A,
        coeffs: &P,
        idx: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> anyhow::Result<()> {
        module
            .ckks_mul_pt_const_into(res, a, coeffs, idx, scratch)
            .map_err(::anyhow::Error::from)
    }

    fn mul_add_pt_const(
        &self,
        module: &Module<BE>,
        res: &mut V,
        a: &A,
        coeffs: &P,
        idx: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> anyhow::Result<()> {
        module
            .ckks_mul_add_pt_const_into(res, a, coeffs, idx, scratch)
            .map_err(::anyhow::Error::from)
    }

    fn prepare_right(&self, module: &Module<BE>, a: &A, scratch: &mut ScratchArena<'_, BE>) -> anyhow::Result<Self::Prepared> {
        // `X^{gsp}` multiplies every pair of its giant-step level.
        module
            .ckks_right_operand_prepare_alloc(a, PrepareHint::Reuse, scratch)
            .map_err(::anyhow::Error::from)
    }

    fn mul_prepared_assign<H>(
        &self,
        module: &Module<BE>,
        dst: &mut V,
        prepared: &Self::Prepared,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> anyhow::Result<()>
    where
        H: GetTensorKey<BE>,
    {
        module
            .ckks_mul_prepared_assign(dst, prepared, tsk, scratch)
            .map_err(::anyhow::Error::from)
    }

    fn add_assign(&self, module: &Module<BE>, dst: &mut V, a: &V, scratch: &mut ScratchArena<'_, BE>) -> anyhow::Result<()> {
        module.ckks_add_assign(dst, a, scratch).map_err(::anyhow::Error::from)
    }

    fn copy(&self, module: &Module<BE>, res: &mut R, src: &V, scratch: &mut ScratchArena<'_, BE>) -> anyhow::Result<()> {
        module.ckks_copy(res, src, scratch)?;
        Ok(())
    }
}

pub trait CKKSPolynomialEvaluationReference<BE: Backend> {
    fn ckks_eval_poly_real_const_coeffs_from_power_basis_reference<R, B, A, G, H>(
        &self,
        res: &mut R,
        power_basis: &G,
        poly: &B,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEZero<BE>
            + CKKSAddOps<BE>
            + CKKSMulOps<BE>
            + CKKSRightOperandPreparedFactory<BE>
            + CKKSMulAddOps<BE>
            + CKKSCiphertextAlloc<BE>
            + CKKSCopyOps<BE>
            + CnvOps<BE>
            + VecZnxBigNormalize<BE>
            + VecZnxRsh<BE>
            + VecZnxRshTmpBytes
            + Sized,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta + SetCKKSInfos + CKKSCtBounds,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds,
        A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds,
        G: PowerBasisHelper<BE, A>,
        H: GetTensorKey<BE>;

    fn ckks_eval_poly_complex_const_coeffs_from_power_basis_reference<R, C, A, G, H>(
        &self,
        res: &mut R,
        power_basis: &G,
        poly: &ComplexBSGSPolynomial<C>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEZero<BE>
            + CKKSAddOps<BE>
            + CKKSImagOps<BE>
            + CKKSMulOps<BE>
            + CKKSRightOperandPreparedFactory<BE>
            + CKKSMulAddOps<BE>
            + CKKSCiphertextAlloc<BE>
            + CKKSCopyOps<BE>
            + CnvOps<BE>
            + VecZnxBigNormalize<BE>
            + VecZnxRsh<BE>
            + VecZnxRshTmpBytes
            + Sized,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta + SetCKKSInfos + CKKSCtBounds,
        C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds + IntPolyInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds,
        G: PowerBasisHelper<BE, A>,
        H: GetTensorKey<BE>;
}

impl<BE: Backend> CKKSPolynomialEvaluationReference<BE> for Module<BE>
where
    BE: GLWEMulConstImpl,
    Module<BE>: GLWEPolynomialEvaluation<BE>,
{
    fn ckks_eval_poly_real_const_coeffs_from_power_basis_reference<R, B, A, G, H>(
        &self,
        res: &mut R,
        power_basis: &G,
        poly: &B,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEZero<BE>
            + GLWEPolynomialEvaluation<BE>
            + CKKSAddOps<BE>
            + CKKSMulOps<BE>
            + CKKSRightOperandPreparedFactory<BE>
            + CKKSMulAddOps<BE>
            + CKKSCiphertextAlloc<BE>
            + CKKSCopyOps<BE>
            + CnvOps<BE>
            + VecZnxBigNormalize<BE>
            + VecZnxRsh<BE>
            + VecZnxRshTmpBytes
            + Sized,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta + SetCKKSInfos + CKKSCtBounds,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds,
        A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds,
        G: PowerBasisHelper<BE, A>,
        H: GetTensorKey<BE>,
    {
        const OP: &str = "ckks_eval_poly_real_const_coeffs_from_power_basis";
        ensure_baby_steps(OP, poly)?;
        ensure_power_basis(OP, poly.basis(), power_basis.basis())?;

        let n_baby = poly.baby_steps();
        let last_coeffs = poly.baby_step(n_baby - 1);
        let trailing_const_only = n_baby >= 2 && last_coeffs.n().as_usize() == 1;
        let fold_power = poly.degree();
        let can_fold = trailing_const_only && power_basis.has_power(fold_power);

        let n_to_process = if can_fold { n_baby - 1 } else { n_baby };
        // The evaluated baby steps are deliberately heap-allocated: their count
        // is polynomial-dependent (`√degree`-ish) and the giant-step fold
        // consumes them progressively, so a scratch carve would have to reserve
        // the full vector for the whole evaluation. Stage internals are
        // scratch-sized (see `ckks_eval_mod_tmp_bytes`).
        let mut baby_steps = Vec::with_capacity(n_to_process);
        let parity = poly.parity();
        let x = power_basis.get(1)?;
        let precision = CKKSBSGSOps;
        for i in 0..n_to_process {
            let coeffs = poly.baby_step(i);
            let degree = coeffs.n().as_usize() - 1;
            let mut value = self.ckks_ciphertext_alloc_from_infos(x);
            value.set_meta(x.meta());
            self.glwe_eval_baby_step(&precision, &mut value, parity, coeffs, power_basis, &mut scratch.borrow())?;
            baby_steps.push(EvaluatedBabyStep { degree, value });
        }

        self.glwe_eval_giant_steps::<_, R, _, _, B::Coeffs, A, G, H>(
            &precision,
            res,
            &mut baby_steps,
            power_basis,
            tsk,
            &mut scratch.borrow(),
        )?;

        if can_fold {
            let xpow = power_basis.get(fold_power)?;
            self.ckks_mul_add_pt_const_into(res, xpow, last_coeffs, 0, scratch)?;
        }

        Ok(())
    }

    fn ckks_eval_poly_complex_const_coeffs_from_power_basis_reference<R, C, A, G, H>(
        &self,
        res: &mut R,
        power_basis: &G,
        poly: &ComplexBSGSPolynomial<C>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEZero<BE>
            + GLWEPolynomialEvaluation<BE>
            + CKKSAddOps<BE>
            + CKKSImagOps<BE>
            + CKKSMulOps<BE>
            + CKKSRightOperandPreparedFactory<BE>
            + CKKSMulAddOps<BE>
            + CKKSCiphertextAlloc<BE>
            + CKKSCopyOps<BE>
            + CnvOps<BE>
            + VecZnxBigNormalize<BE>
            + VecZnxRsh<BE>
            + VecZnxRshTmpBytes
            + Sized,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta + SetCKKSInfos + CKKSCtBounds,
        C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds + IntPolyInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds,
        G: PowerBasisHelper<BE, A>,
        H: GetTensorKey<BE>,
    {
        const OP: &str = "ckks_eval_poly_complex_const_coeffs_from_power_basis";
        let poly_re = &poly.re;
        let poly_im = &poly.im;
        ensure_complex_schedules::<BE, C>(OP, poly)?;
        ensure_power_basis(OP, BSGSPolynomialInfos::<BE>::basis(poly_re), power_basis.basis())?;
        let n_baby = BSGSPolynomialInfos::<BE>::baby_steps(poly_re);

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
            let degree = re_coeffs.n().as_usize() - 1;

            let mut value = self.ckks_ciphertext_alloc_from_infos(x);
            value.set_meta(x.meta());
            self.glwe_eval_baby_step::<_, _, C, A, G>(
                &precision,
                &mut value,
                parity,
                re_coeffs,
                power_basis,
                &mut scratch.borrow(),
            )?;

            let mut im_value = self.ckks_ciphertext_alloc_from_infos(x);
            im_value.set_meta(x.meta());
            self.glwe_eval_baby_step::<_, _, C, A, G>(
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

        self.glwe_eval_giant_steps::<_, R, _, _, C, A, G, H>(
            &precision,
            res,
            &mut baby_steps,
            power_basis,
            tsk,
            &mut scratch.borrow(),
        )?;

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

/// `InvalidOperand` for `op` unless `poly` has a baby step.
pub(crate) fn ensure_baby_steps<BE: Backend, B: BSGSPolynomialInfos<BE>>(op: &'static str, poly: &B) -> Result<()> {
    ckks_ensure_operand!(op, poly.baby_steps() > 0, "the polynomial has no baby step");
    Ok(())
}

fn ensure_power_basis(op: &'static str, poly: Basis, power_basis: Basis) -> Result<()> {
    ckks_ensure_operand!(
        op,
        poly == power_basis,
        "polynomial basis {poly:?} does not match power basis {power_basis:?}"
    );
    Ok(())
}

/// `InvalidOperand` for `op` unless the real and imaginary parts of `poly` share a nonempty baby-step schedule, each
/// step of one length, and their degree, parity and basis.
pub(crate) fn ensure_complex_schedules<BE, C>(op: &'static str, poly: &ComplexBSGSPolynomial<C>) -> Result<()>
where
    BE: Backend,
    C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + IntPolyInfos,
{
    let (re, im) = (&poly.re, &poly.im);
    ensure_baby_steps::<BE, _>(op, re)?;
    let n_baby = BSGSPolynomialInfos::<BE>::baby_steps(re);
    ckks_ensure_operand!(
        op,
        BSGSPolynomialInfos::<BE>::baby_steps(im) == n_baby,
        "real/imag baby-step schedules differ"
    );
    ckks_ensure_operand!(
        op,
        BSGSPolynomialInfos::<BE>::degree(im) == BSGSPolynomialInfos::<BE>::degree(re),
        "real/imag degrees differ"
    );
    ckks_ensure_operand!(
        op,
        BSGSPolynomialInfos::<BE>::parity(im) == BSGSPolynomialInfos::<BE>::parity(re),
        "real/imag parities differ"
    );
    ckks_ensure_operand!(
        op,
        BSGSPolynomialInfos::<BE>::basis(im) == BSGSPolynomialInfos::<BE>::basis(re),
        "real/imag bases differ"
    );
    ckks_ensure_operand!(
        op,
        (0..n_baby).all(|i| BSGSPolynomialInfos::<BE>::baby_step(im, i).n() == BSGSPolynomialInfos::<BE>::baby_step(re, i).n()),
        "real/imag baby-step lengths differ"
    );
    Ok(())
}
