//! Homomorphic evaluation of the `x mod 1` reduction, the core non-linear step
//! of CKKS bootstrapping.
//!
//! This module holds the *evaluation* — the backend-generic reference
//! [`CKKSEvalModOpsReference`] and the `eval_mod` pipeline. The parameterization
//! it consumes (the periodic-function approximation polynomials and their
//! encoding) lives in [`crate::layouts::eval_mod`]; see there for the maths and
//! the [`EvalMod`] structure. The public entry point is
//! [`CKKSEvalModOps`](crate::api::CKKSEvalModOps).

use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::{
    GLWECopy, GLWENormalize,
    layouts::{
        BSGSMeta, GGLWEInfos, GLWELayout, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, Rank, SetBSGSMeta,
        prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::layouts::{Backend, ConjugateInvariant, Module, Ring, ScratchArena, Standard};

use crate::{
    CKKSCtBounds, GLWEPlaintextMeta, Scale, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSComplexPolynomialEvaluationOps, CKKSCopyOps, CKKSMulOps, CKKSPolynomialEvaluationOps, CKKSPow2Ops,
        CKKSSubOps, PolynomialInputTransform,
    },
    layouts::{
        CKKSCiphertextOwned, CKKSModuleAlloc, ScratchArenaTakeCKKS,
        eval_mod::{EvalMod, EvalModBsgs},
    },
    polynomial::ComplexBSGSPolynomial,
    power_basis::{PowerBasis, PowerBasisGen},
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
pub trait CKKSEvalModOpsReference<BE: Backend> {
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
            + CKKSModuleAlloc<BE>
            + Sized,
        BE: Backend,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
        H: GetTensorKey<BE>,
        CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos;
}

impl<BE: Backend> CKKSEvalModOpsReference<BE> for Module<BE>
where
    Module<BE>: CKKSPolynomialEvaluationOps<BE>
        + CKKSEvalModComplexOps<BE, BE::Ring>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSMulOps<BE>
        + CKKSCopyOps<BE>
        + CKKSModuleAlloc<BE>
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
}

/// The `x mod 1` pipeline (see [`crate::layouts::eval_mod`] for the maths).
///
/// 1. Verify `ct` still has `params.eval_depth() · log_delta` bits of `log_budget` —
///    the multiplicative levels the evaluation will consume — and copy it into a
///    working ciphertext.
/// 2. Evaluate the base `f` polynomial by BSGS into `res`.
/// 3. Apply the `f_mod_log_interval_reduction` range-extension steps. For the
///    trigonometric families these are `res ← 2·res² − dac` on the real path (the
///    `cos 2θ` identity with the encoded constant `dac`) and `res ← res²` on the
///    complex path.
/// 4. If configured, compose the inverse `f⁻¹` polynomial in place.
///
/// `res` receives the result; `tsk` is the relinearization (tensor) key for the
/// squarings, and `scratch` supplies the working memory sized by
/// [`CKKSEvalModOps::ckks_eval_mod_tmp_bytes`](crate::api::CKKSEvalModOps::ckks_eval_mod_tmp_bytes).
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
        + CKKSModuleAlloc<BE>
        + CKKSPow2Ops<BE>
        + GLWECopy<BE>
        + GLWENormalize<BE>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    C: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos,
    H: GetTensorKey<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    // EvalMod runs at its own plan scale `f_mod_log_delta`: reinterpret the
    // working ciphertext to it on entry, then return the result to the input
    // scale. `consumed_bits()` accounts for the arithmetic at the plan scale;
    // if the plan scale is higher than the input scale, returning to the input
    // scale also drops that extra precision from the externally visible budget.
    // Intermediates are allocated rank-1 (`ckks_ciphertext_alloc`); reject
    // higher-rank inputs instead of silently mis-shaping them.
    if matches!(params.f_mod_bsgs, EvalModBsgs::Complex(_)) && !module.ckks_eval_mod_complex_supported() {
        return Err(complex_refused());
    }
    ckks_ensure!(
        ct.rank().as_usize() == 1,
        "ckks_eval_mod supports rank-1 ciphertexts only, got rank {}",
        ct.rank().as_usize()
    );
    let s_in = ct.log_delta();
    let s_eval = params.plan.f_mod_log_delta;
    let s_budget = ct.log_budget();

    let required = params.consumed_bits();
    ckks_ensure!(
        ct.log_budget() >= required,
        "ckks_eval_mod: input log_budget {got} < {required} bits required (consumed at scale {s_eval})",
        got = ct.log_budget(),
    );

    let work_layout = GLWELayout {
        n: ct.n(),
        base2k: ct.base2k(),
        k: (s_budget + s_eval).into(),
        rank: Rank(1),
    };
    let work_meta = GLWEPlaintextMeta {
        scale: Scale::Log(s_eval),
        slots: ct.slots(),
        log_sparsity: ct.log_sparsity(),
    };

    match &params.f_mod_bsgs {
        EvalModBsgs::Real(bsgs) => {
            if bsgs.input_transform() == PolynomialInputTransform::Identity {
                // The generic one-shot evaluator would copy this input again
                // before building its power basis. Hand ownership over directly.
                let x1 = eval_mod_input(module, ct, &work_layout, work_meta, scratch);
                let mut power_basis = PowerBasis::new(bsgs.basis(), x1);
                power_basis.populate(bsgs.degree(), bsgs.log_split(), bsgs.parity(), module, tsk, scratch)?;
                module.ckks_eval_poly_real_const_coeffs_from_power_basis(res, &power_basis, bsgs, tsk, scratch)?;
            } else {
                scratch.scope(|scratch_local| {
                    let (mut input, mut nested) = scratch_local.take_ckks_ciphertext_scratch(&work_layout, work_meta);
                    module.glwe_copy(&mut input, ct, &mut nested);
                    if let Some(offset) = params.f_mod_input_offset.as_ref() {
                        module.ckks_add_pt_const_assign(&mut input, 0, offset, 0, &mut nested)?;
                    }
                    module.ckks_eval_poly_real_const_coeffs(res, &input, bsgs, tsk, &mut nested)
                })?;
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
            let reductions = params.plan.f_mod_log_interval_reduction;
            module.ckks_eval_mod_complex(res, ct, bsgs, reductions, &work_layout, work_meta, tsk, scratch)?
        }
    }

    // Restore the input scale on the result. This is a pure metadata relabel
    // (`set_log_delta`), not a rescale: entry raised the scale `s_in -> s_eval`
    // without spending budget (`set_log_budget(s_budget)` on an MSB-aligned copy,
    // which reinterprets the value at `2^-(s_eval - s_in)`); relabelling back to
    // `s_in` here undoes exactly that, so the scale round-trip is budget-neutral
    // and the only consumption is the EvalMod arithmetic, which `consumed_bits()`
    // accounts for in full. When the plan scale is the wider one the relabel
    // lowers `k` by `s_eval - s_in`, which clears the canonical flag.
    if s_eval != s_in {
        res.set_log_delta(s_in);
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

    #[expect(
        clippy::too_many_arguments,
        reason = "the base polynomial takes EvalMod's working layout beside its operands"
    )]
    fn ckks_eval_mod_complex<Res, C, P, H>(
        &self,
        _res: &mut Res,
        _ct: &C,
        _bsgs: &ComplexBSGSPolynomial<P>,
        _reductions: usize,
        _work_layout: &GLWELayout,
        _work_meta: GLWEPlaintextMeta,
        _tsk: &H,
        _scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Res: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
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
        + CKKSModuleAlloc<BE>
        + GLWECopy<BE>
        + GLWENormalize<BE>,
{
    fn ckks_eval_mod_complex_supported(&self) -> bool {
        true
    }

    fn ckks_eval_mod_complex<Res, C, P, H>(
        &self,
        res: &mut Res,
        ct: &C,
        bsgs: &ComplexBSGSPolynomial<P>,
        reductions: usize,
        work_layout: &GLWELayout,
        work_meta: GLWEPlaintextMeta,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Res: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos,
        H: GetTensorKey<BE>,
    {
        if bsgs.re.input_transform() == PolynomialInputTransform::Identity
            && bsgs.im.input_transform() == PolynomialInputTransform::Identity
        {
            let x1 = eval_mod_input(self, ct, work_layout, work_meta, scratch);
            let mut power_basis = PowerBasis::new(bsgs.re.basis(), x1);
            power_basis.populate(bsgs.re.degree(), bsgs.re.log_split(), bsgs.re.parity(), self, tsk, scratch)?;
            self.ckks_eval_poly_complex_const_coeffs_from_power_basis(res, &power_basis, bsgs, tsk, scratch)?;
        } else {
            scratch.scope(|scratch_local| {
                let (mut input, mut nested) = scratch_local.take_ckks_ciphertext_scratch(work_layout, work_meta);
                self.glwe_copy(&mut input, ct, &mut nested);
                self.ckks_eval_poly_complex_const_coeffs(res, &input, bsgs, tsk, &mut nested)
            })?;
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
    Module<BE>: CKKSModuleAlloc<BE> + GLWECopy<BE>,
    C: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    let mut input = module.ckks_ciphertext_alloc(layout.base2k, layout.k);
    module.glwe_copy(&mut input, ct, scratch);
    input.set_meta(meta);
    input
}

pub use super::eval_mod_scratch::ckks_eval_mod_tmp_bytes_reference;
