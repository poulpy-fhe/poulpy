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
        BSGSMeta, GGLWEInfos, GLWEInfos, GLWELayout, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        SetBSGSMeta, SetK, prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{CnvOps, CnvPVecBytesOf, VecZnxBigNormalizeTmpBytes, VecZnxRshTmpBytes},
    layouts::{Backend, ConjugateInvariant, Module, Ring, ScratchArena, Standard},
};

use crate::{
    CKKSCtBounds, CKKSInfos, GLWEPlaintextMeta, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSComplexPolynomialEvaluationOps, CKKSCopyOps, CKKSMulOps, CKKSPolynomialEvaluationOps, CKKSPow2Ops,
        CKKSSubOps, PolynomialInputTransform,
    },
    layouts::{
        CKKSCiphertextAlloc, CKKSCiphertextOwned, ScratchArenaTakeCKKSCiphertext,
        eval_mod::{EvalMod, EvalModBsgs},
    },
    polynomial::ComplexBSGSPolynomial,
    power_basis::{CKKSPowerBasisGen, PowerBasis},
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
    /// Reference `x mod 1` evaluation: see [`crate::layouts::eval_mod`] for the
    /// base-polynomial / range-extension / inverse pipeline and the `eval_mod`
    /// function for the implementation.
    fn ckks_eval_mod_reference<R, C, P, F, H>(
        &self,
        res: &mut R,
        ct: &C,
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
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
        H: GetTensorKey<BE>,
        CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos;

    /// Reference body of [`ckks_eval_mod_owned`](crate::api::CKKSEvalModOps::ckks_eval_mod_owned).
    fn ckks_eval_mod_owned_reference<R, P, F, H>(
        &self,
        res: &mut R,
        input: CKKSCiphertextOwned<BE>,
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
        ct: &C,
        params: &EvalMod<F, P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos,
        H: GetTensorKey<BE>,
    {
        eval_mod(self, res, ct, params, tsk, scratch)
    }

    fn ckks_eval_mod_owned_reference<R, P, F, H>(
        &self,
        res: &mut R,
        input: CKKSCiphertextOwned<BE>,
        params: &EvalMod<F, P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos,
        H: GetTensorKey<BE>,
    {
        eval_mod_owned(self, res, input, params, tsk, scratch)
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

/// The evaluation [`EvalModPlan`](crate::layouts::eval_mod::EvalModPlan) describes, on a copy of `ct`.
fn eval_mod<R, C, P, F, BE, H>(
    module: &Module<BE>,
    res: &mut R,
    ct: &C,
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
    C: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos,
    H: GetTensorKey<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    check_eval_mod("ckks_eval_mod", module, ct, params)?;
    let (work_layout, work_meta) = params.plan.work_layout(ct);
    let input = eval_mod_input(module, ct, &work_layout, work_meta, scratch);
    eval_mod_on_input(module, res, input, ct.log_delta(), params, tsk, scratch)
}

/// The evaluation [`EvalModPlan`](crate::layouts::eval_mod::EvalModPlan) describes, on `input`, relabelled in place.
fn eval_mod_owned<R, P, F, BE, H>(
    module: &Module<BE>,
    res: &mut R,
    mut input: CKKSCiphertextOwned<BE>,
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
    P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos,
    H: GetTensorKey<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    const OP: &str = "ckks_eval_mod_owned";
    check_eval_mod(OP, module, &input, params)?;
    let (work_layout, work_meta) = params.plan.work_layout(&input);
    ckks_ensure_operand!(
        OP,
        work_layout.k <= input.max_k(),
        "takes an input whose allocation holds the working width of EvalMod"
    );
    let log_delta = input.log_delta();
    input.set_meta(work_meta);
    input.set_k(work_layout.k);
    eval_mod_on_input(module, res, input, log_delta, params, tsk, scratch)
}

/// EvalMod on `input`, at its working layout, the output labelled at `log_delta`. A range-extension step is
/// `res ← 2·res² − dac` on the real path, the `cos 2θ` identity, and `res ← res²` on the complex path.
fn eval_mod_on_input<R, P, F, BE, H>(
    module: &Module<BE>,
    res: &mut R,
    mut input: CKKSCiphertextOwned<BE>,
    log_delta: usize,
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
    P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos,
    H: GetTensorKey<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    let (work_layout, work_meta) = (input.glwe_layout(), input.meta());
    match &params.f_mod_bsgs {
        EvalModBsgs::Real(bsgs) => {
            if bsgs.input_transform() == PolynomialInputTransform::Identity {
                let mut power_basis = PowerBasis::new(bsgs.basis(), input);
                power_basis.populate(bsgs.degree(), bsgs.log_split(), bsgs.parity(), module, tsk, scratch)?;
                module.ckks_eval_poly_real_const_coeffs_from_power_basis(res, &power_basis, bsgs, tsk, scratch)?;
            } else {
                if let Some(offset) = params.f_mod_input_offset.as_ref() {
                    module.ckks_add_pt_const_assign(&mut input, 0, offset, 0, scratch)?;
                }
                module.ckks_eval_poly_real_const_coeffs(res, &input, bsgs, tsk, scratch)?;
            }

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
        EvalModBsgs::Complex(bsgs) => {
            module.ckks_eval_mod_complex(res, input, bsgs, params.plan.f_mod_log_interval_reduction, tsk, scratch)?
        }
    }

    // A relabel, not a rescale: it undoes the one at entry, keeping the budget. When the plan scale is the wider one it
    // lowers `k`, which clears the canonical flag.
    if params.plan.f_mod_log_delta != log_delta {
        res.set_log_delta(log_delta);
    }

    Ok(())
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

    fn ckks_eval_mod_complex<Res, P, H>(
        &self,
        _res: &mut Res,
        _input: CKKSCiphertextOwned<BE>,
        _bsgs: &ComplexBSGSPolynomial<P>,
        _reductions: usize,
        _tsk: &H,
        _scratch: &mut ScratchArena<'_, BE>,
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

    fn ckks_eval_mod_complex<Res, P, H>(
        &self,
        res: &mut Res,
        input: CKKSCiphertextOwned<BE>,
        bsgs: &ComplexBSGSPolynomial<P>,
        reductions: usize,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Res: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos,
        H: GetTensorKey<BE>,
    {
        if bsgs.re.input_transform() == PolynomialInputTransform::Identity
            && bsgs.im.input_transform() == PolynomialInputTransform::Identity
        {
            let mut power_basis = PowerBasis::new(bsgs.re.basis(), input);
            power_basis.populate(bsgs.re.degree(), bsgs.re.log_split(), bsgs.re.parity(), self, tsk, scratch)?;
            self.ckks_eval_poly_complex_const_coeffs_from_power_basis(res, &power_basis, bsgs, tsk, scratch)?;
        } else {
            self.ckks_eval_poly_complex_const_coeffs(res, &input, bsgs, tsk, scratch)?;
        }
        for _ in 0..reductions {
            self.ckks_square_assign(res, tsk, scratch)?;
        }
        Ok(())
    }
}

/// Allocates the single owned EvalMod input that becomes power-basis element 1.
/// Copying directly from `ct` avoids the scratch copy followed by the generic
/// polynomial evaluator's second owned copy.
fn eval_mod_input<BE, C>(
    module: &Module<BE>,
    ct: &C,
    layout: &GLWELayout,
    meta: GLWEPlaintextMeta,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSCiphertextOwned<BE>
where
    BE: Backend,
    Module<BE>: CKKSCiphertextAlloc<BE> + GLWECopy<BE>,
    C: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    let mut input = module.ckks_ciphertext_alloc(layout.base2k, layout.k);
    module.glwe_copy(&mut input, ct, scratch);
    input.set_meta(meta);
    input
}

pub use super::eval_mod_scratch::ckks_eval_mod_tmp_bytes_reference;
