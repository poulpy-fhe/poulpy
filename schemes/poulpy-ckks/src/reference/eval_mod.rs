//! Homomorphic evaluation of the `x mod 1` reduction, the core non-linear step
//! of CKKS bootstrapping.
//!
//! This module holds the *evaluation* — the backend-generic reference
//! [`CKKSEvalModReference`] and the `eval_mod` pipeline. The parameterization
//! it consumes (the periodic-function approximation polynomials and their
//! encoding) lives in [`crate::layouts::eval_mod`]; see there for the maths and
//! the [`EvalMod`] structure. The public entry point is
//! [`CKKSEvalModOps`](crate::api::CKKSEvalModOps).

use crate::{CKKSResult as Result, checked_log_budget_sub, ckks_ensure_operand};
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::{
    GLWECopy, GLWENormalize,
    layouts::{
        BSGSMeta, GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, SetBSGSMeta, SetK,
        prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{CnvOps, CnvPVecBytesOf, VecZnxBigNormalizeTmpBytes, VecZnxRshTmpBytes},
    layouts::{Backend, ConjugateInvariant, Module, Ring, ScratchArena, Standard},
};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSComplexPolynomialEvaluationOps, CKKSCopyOps, CKKSMulOps, CKKSPolynomialEvaluationOps, CKKSPow2Ops,
        CKKSSubOps, PolynomialInputTransform,
    },
    layouts::{
        CKKSCiphertextOwned, CKKSCiphertextViewMut, ScratchArenaTakeCKKSCiphertext,
        eval_mod::{EvalMod, EvalModBsgs},
    },
    oep::derived::polynomial_evaluation::polynomial_input_in,
    polynomial::ComplexBSGSPolynomial,
    power_basis::{PowerBasis, populate_in},
    reference::polynomial_evaluation::ensure_baby_steps,
};

/// Backend-generic reference implementation of [`CKKSEvalModOps`].
///
/// Blanket-implemented for any [`Module<BE>`] whose backend provides the
/// constituent CKKS ops (polynomial evaluation, add/sub/mul/copy, allocation).
/// Backends wire this into the public [`CKKSEvalModOps`] trait through the
/// [`CKKSEvalModImpl`](crate::oep::CKKSEvalModImpl) OEP hook, which by default
/// forwards to [`Self::ckks_eval_mod_reference`].
///
/// [`CKKSEvalModOps`]: crate::api::CKKSEvalModOps
pub trait CKKSEvalModReference<BE: Backend> {
    /// Reference body of [`ckks_eval_mod`](crate::api::CKKSEvalModOps::ckks_eval_mod).
    fn ckks_eval_mod_reference<R, C, P, F, H>(
        &self,
        res: &mut R,
        ct: &mut C,
        params: &EvalMod<F, P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSPolynomialEvaluationOps<BE>
            + CKKSEvalModComplexOps<BE, BE::Ring>
            + CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + CKKSMulOps<BE>
            + CKKSCopyOps<BE>
            + Sized,
        BE: Backend,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
        H: GetTensorKey<BE>,
        CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos;

    /// Scratch of [`Self::ckks_eval_mod_reference`].
    fn ckks_eval_mod_tmp_bytes_reference<R, C, P, F, T>(&self, res: &R, ct: &C, params: &EvalMod<F, P>, tsk: &T) -> usize
    where
        BE: poulpy_core::oep::GLWEMulConstImpl,
        Self: CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + CKKSMulOps<BE>
            + CKKSPow2Ops<BE>
            + GLWENormalize<BE>
            + CKKSCopyOps<BE>
            + GLWECopy<BE>
            + CnvPVecBytesOf
            + CnvOps<BE>
            + VecZnxRshTmpBytes
            + VecZnxBigNormalizeTmpBytes,
        R: CKKSCtBounds,
        C: CKKSCtBounds,
        P: CKKSCtBounds,
        T: GGLWEInfos;
}

impl<BE: Backend> CKKSEvalModReference<BE> for Module<BE>
where
    Module<BE>: CKKSPolynomialEvaluationOps<BE>
        + CKKSEvalModComplexOps<BE, BE::Ring>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSMulOps<BE>
        + CKKSCopyOps<BE>
        + CKKSPow2Ops<BE>
        + GLWECopy<BE>
        + GLWENormalize<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
{
    fn ckks_eval_mod_reference<R, C, P, F, H>(
        &self,
        res: &mut R,
        ct: &mut C,
        params: &EvalMod<F, P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos,
        H: GetTensorKey<BE>,
    {
        eval_mod(self, res, ct, params, tsk, scratch)
    }

    fn ckks_eval_mod_tmp_bytes_reference<R, C, P, F, T>(&self, res: &R, ct: &C, params: &EvalMod<F, P>, tsk: &T) -> usize
    where
        BE: poulpy_core::oep::GLWEMulConstImpl,
        Self: CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + CKKSMulOps<BE>
            + CKKSPow2Ops<BE>
            + GLWENormalize<BE>
            + CKKSCopyOps<BE>
            + GLWECopy<BE>
            + CnvPVecBytesOf
            + CnvOps<BE>
            + VecZnxRshTmpBytes
            + VecZnxBigNormalizeTmpBytes,
        R: CKKSCtBounds,
        C: CKKSCtBounds,
        P: CKKSCtBounds,
        T: GGLWEInfos,
    {
        ckks_eval_mod_tmp_bytes_reference(self, res, ct, params, tsk)
    }
}

/// Refuses what EvalMod cannot evaluate on an input like `ct`.
fn check_eval_mod<BE, C, F, P>(op: &'static str, module: &Module<BE>, ct: &C, params: &EvalMod<F, P>) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSEvalModComplexOps<BE, BE::Ring>,
    C: CKKSCtBounds,
{
    if matches!(params.f_mod_bsgs, EvalModBsgs::Complex(_)) && !module.ckks_eval_mod_complex_supported() {
        return Err(complex_refused());
    }
    ckks_ensure_operand!(
        op,
        ct.rank().as_usize() == 1,
        "supports rank-1 ciphertexts only, got rank {}",
        ct.rank().as_usize()
    );
    checked_log_budget_sub(op, ct.log_budget(), params.consumed_bits())?;
    Ok(())
}

/// The evaluation [`EvalModPlan`](crate::layouts::eval_mod::EvalModPlan) describes, on `ct` when its allocation holds the
/// working width, on a carved copy otherwise. A range-extension step is `res ← 2·res² − dac` on the real path, the
/// `cos 2θ` identity, and `res ← res²` on the complex path.
fn eval_mod<R, C, P, F, BE, H>(
    module: &Module<BE>,
    res: &mut R,
    ct: &mut C,
    params: &EvalMod<F, P>,
    tsk: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSPolynomialEvaluationOps<BE>
        + CKKSEvalModComplexOps<BE, BE::Ring>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSMulOps<BE>
        + CKKSCopyOps<BE>
        + CKKSPow2Ops<BE>
        + GLWECopy<BE>
        + GLWENormalize<BE>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    C: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos,
    H: GetTensorKey<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    check_eval_mod("ckks_eval_mod", module, ct, params)?;
    let log_delta = ct.log_delta();
    let (work_layout, work_meta) = params.plan.work_layout(ct);
    if work_layout.k <= ct.max_k() {
        let mut input = CKKSCiphertextViewMut::of(ct);
        input.set_meta(work_meta);
        input.set_k(work_layout.k);
        eval_mod_base(module, res, input, params, tsk, scratch.borrow())?;
    } else {
        let (mut input, mut arena) = scratch.borrow().take_ckks_ciphertext_scratch(&work_layout, work_meta);
        module.glwe_copy(&mut input, &*ct, &mut arena);
        input.set_meta(work_meta);
        eval_mod_base(module, res, input, params, tsk, arena)?;
    }

    match &params.f_mod_bsgs {
        EvalModBsgs::Real(_) => {
            if let Some(consts) = params.range_extension_consts.as_ref() {
                for i in 0..params.plan.f_mod_log_interval_reduction {
                    scratch.scope(|scratch_local| -> Result<()> {
                        let (mut squared, mut nested) = scratch_local.take_ckks_ciphertext_like_scratch(&*res);
                        module.ckks_square_into(&mut squared, &*res, tsk, &mut nested)?;
                        module.ckks_double_into(res, &squared, &mut nested)
                    })?;
                    module.ckks_sub_pt_const_assign(res, 0, consts, i, scratch)?;
                }
            }

            if let Some(inv) = params.f_mod_inv_bsgs.as_ref() {
                // The inverse consumes the base result, so this is the only
                // stage that still needs a separate working copy.
                scratch.scope(|scratch_local| {
                    let (mut input, mut nested) = scratch_local.take_ckks_ciphertext_scratch(&work_layout, work_meta);
                    module.ckks_copy(&mut input, &*res, &mut nested)?;
                    module.ckks_eval_poly_real_const_coeffs(res, &input, inv, tsk, &mut nested)
                })?;
            }
        }
        EvalModBsgs::Complex(_) => {
            for _ in 0..params.plan.f_mod_log_interval_reduction {
                module.ckks_square_assign(res, tsk, scratch)?;
            }
        }
    }

    // A relabel, not a rescale: it undoes the one at entry, keeping the budget. When the plan scale is the wider one it
    // lowers `k`, which clears the canonical flag.
    if params.plan.f_mod_log_delta != log_delta {
        res.set_log_delta(log_delta);
    }

    Ok(())
}

/// The base polynomial on `input`, at the working layout, its powers carved from `scratch`.
fn eval_mod_base<'a, R, P, F, BE, H>(
    module: &Module<BE>,
    res: &mut R,
    mut input: CKKSCiphertextViewMut<'a, BE>,
    params: &EvalMod<F, P>,
    tsk: &H,
    mut scratch: ScratchArena<'a, BE>,
) -> Result<()>
where
    BE: Backend + 'a,
    Module<BE>: CKKSPolynomialEvaluationOps<BE>
        + CKKSEvalModComplexOps<BE, BE::Ring>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSMulOps<BE>
        + CKKSCopyOps<BE>
        + CKKSPow2Ops<BE>
        + GLWENormalize<BE>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos,
    H: GetTensorKey<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    match &params.f_mod_bsgs {
        EvalModBsgs::Real(bsgs) => {
            ensure_baby_steps("ckks_eval_poly_real_const_coeffs", bsgs)?;
            let transform = bsgs.input_transform();
            if transform == PolynomialInputTransform::Identity {
                let mut power_basis = PowerBasis::new(bsgs.basis(), input);
                let mut scratch = populate_in(
                    &mut power_basis,
                    bsgs.degree(),
                    bsgs.log_split(),
                    bsgs.parity(),
                    module,
                    tsk,
                    scratch,
                )?;
                return module.ckks_eval_poly_real_const_coeffs_from_power_basis(res, &power_basis, bsgs, tsk, &mut scratch);
            }
            if let Some(offset) = params.f_mod_input_offset.as_ref() {
                module.ckks_add_pt_const_assign(&mut input, 0, offset, 0, &mut scratch)?;
            }
            let (x1, scratch) =
                polynomial_input_in::<_, CKKSCiphertextViewMut<'a, BE>, _, _>(module, &input, transform, tsk, scratch)?;
            let mut power_basis = PowerBasis::new(bsgs.basis(), x1);
            let mut scratch = populate_in(
                &mut power_basis,
                bsgs.degree(),
                bsgs.log_split(),
                bsgs.parity(),
                module,
                tsk,
                scratch,
            )?;
            module.ckks_eval_poly_real_const_coeffs_from_power_basis(res, &power_basis, bsgs, tsk, &mut scratch)?;
            if matches!(
                transform,
                PolynomialInputTransform::SquareTimesInput | PolynomialInputTransform::ChebyshevT2TimesInput
            ) {
                module.ckks_mul_assign(res, &input, tsk, &mut scratch)?;
            }
            Ok(())
        }
        EvalModBsgs::Complex(bsgs) => module.ckks_eval_mod_complex(res, input, bsgs, tsk, scratch),
    }
}

/// The complex base polynomial of [`EvalModType::ExpCmplx`](crate::layouts::eval_mod::EvalModType::ExpCmplx)
/// and its squarings, on a module of ring `R`: refused by default, as on the
/// conjugate-invariant ring, whose slots are real.
///
/// Implemented once per ring for [`Module`], for EvalMod to dispatch on: not meant to be
/// called directly.
pub trait CKKSEvalModComplexOps<BE: Backend, R: Ring> {
    /// Whether the ring evaluates the complex base polynomial; EvalMod refuses `ExpCmplx`
    /// before any other check otherwise.
    fn ckks_eval_mod_complex_supported(&self) -> bool {
        false
    }

    fn ckks_eval_mod_complex<'a, Res, P, H>(
        &self,
        _res: &mut Res,
        _input: CKKSCiphertextViewMut<'a, BE>,
        _bsgs: &ComplexBSGSPolynomial<P>,
        _tsk: &H,
        _scratch: ScratchArena<'a, BE>,
    ) -> Result<()>
    where
        Res: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos,
        H: GetTensorKey<BE>,
    {
        Err(complex_refused())
    }
}

/// The refusal of `ExpCmplx` on a ring of real slots.
fn complex_refused() -> crate::CKKSError {
    crate::CKKSCompositionError::InvalidPlan {
        plan: "EvalModPlan",
        reason: "ExpCmplx needs complex slots, which the conjugate-invariant ring has not".into(),
    }
    .into()
}

impl<BE: Backend> CKKSEvalModComplexOps<BE, ConjugateInvariant> for Module<BE> {}

impl<BE: Backend> CKKSEvalModComplexOps<BE, Standard> for Module<BE>
where
    Module<BE>: CKKSComplexPolynomialEvaluationOps<BE>
        + CKKSMulOps<BE>
        + CKKSSubOps<BE>
        + CKKSPow2Ops<BE>
        + GLWECopy<BE>
        + GLWENormalize<BE>,
{
    fn ckks_eval_mod_complex_supported(&self) -> bool {
        true
    }

    fn ckks_eval_mod_complex<'a, Res, P, H>(
        &self,
        res: &mut Res,
        input: CKKSCiphertextViewMut<'a, BE>,
        bsgs: &ComplexBSGSPolynomial<P>,
        tsk: &H,
        scratch: ScratchArena<'a, BE>,
    ) -> Result<()>
    where
        Res: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos,
        H: GetTensorKey<BE>,
    {
        let mut scratch = scratch;
        if bsgs.re.input_transform() == PolynomialInputTransform::Identity
            && bsgs.im.input_transform() == PolynomialInputTransform::Identity
        {
            let mut power_basis = PowerBasis::new(bsgs.re.basis(), input);
            let mut scratch = populate_in(
                &mut power_basis,
                bsgs.re.degree(),
                bsgs.re.log_split(),
                bsgs.re.parity(),
                self,
                tsk,
                scratch,
            )?;
            self.ckks_eval_poly_complex_const_coeffs_from_power_basis(res, &power_basis, bsgs, tsk, &mut scratch)
        } else {
            self.ckks_eval_poly_complex_const_coeffs(res, &input, bsgs, tsk, &mut scratch)
        }
    }
}

pub use super::eval_mod_scratch::ckks_eval_mod_tmp_bytes_reference;
