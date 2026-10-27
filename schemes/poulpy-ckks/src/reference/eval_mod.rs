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
    GLWECopy, GLWENormalize, GLWETensoring,
    layouts::{
        BSGSMeta, GGLWEInfos, GLWELayout, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, Rank, SetBSGSMeta,
        prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::layouts::{Backend, ConjugateInvariant, Module, Ring, ScratchArena, Standard};

use crate::{
    CKKSCtBounds, CKKSInfos, GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotLayout,
    api::{
        CKKSAddOps, CKKSComplexPolynomialEvaluationOps, CKKSCopyOps, CKKSMulOps, CKKSPolynomialEvaluationOps, CKKSPow2Ops,
        CKKSRoundOps, CKKSSubOps, PolynomialInputTransform,
    },
    layouts::{
        CKKSCiphertextOwned, CKKSMaskedConstants, CKKSModuleAlloc, ScratchArenaTakeCKKS,
        eval_mod::{EvalMod, EvalModBsgs},
    },
    polynomial::ComplexBSGSPolynomial,
    power_basis::{PowerBasis, PowerBasisGen},
    reference::{
        mul::square_wide_into,
        polynomial_evaluation::{const_add_assign, const_sub_assign},
    },
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
            + CKKSModuleAlloc<BE>
            + Sized,
        BE: Backend,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta + CKKSMaskedConstants<BE>,
        H: GetTensorKey<BE>,
        CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos;
}

impl<BE: Backend> CKKSEvalModReference<BE> for Module<BE>
where
    Module<BE>: CKKSPolynomialEvaluationOps<BE>
        + CKKSEvalModComplexOps<BE, BE::Ring>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSMulOps<BE>
        + CKKSCopyOps<BE>
        + CKKSModuleAlloc<BE>
        + CKKSPow2Ops<BE>
        + CKKSRoundOps<BE>
        + GLWETensoring<BE>
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
        P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos + CKKSMaskedConstants<BE>,
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
        + CKKSRoundOps<BE>
        + GLWETensoring<BE>
        + GLWECopy<BE>
        + GLWENormalize<BE>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    C: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos + CKKSMaskedConstants<BE>,
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
    ckks_ensure_operand!(
        "ckks_eval_mod",
        ct.rank().as_usize() == 1,
        "supports rank-1 ciphertexts only, got rank {}",
        ct.rank().as_usize()
    );
    let s_in = ct.log_delta();
    let s_eval = params.plan.f_mod_log_delta;
    let s_budget = ct.log_budget();
    checked_log_budget_sub("ckks_eval_mod", s_budget, params.consumed_bits())?;

    let replicated = matches!(ct.slot_layout(), SlotLayout::Replicated(_));
    // The last squaring of a replicated layout rounds straight to the input
    // scale when no inverse follows it.
    let fused = replicated && params.f_mod_inv_bsgs.is_none() && 2 * s_eval >= s_in;
    let mut returned = false;

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
        slot_layout: ct.slot_layout(),
    };

    match &params.f_mod_bsgs {
        EvalModBsgs::Real(bsgs) => {
            if bsgs.input_transform() == PolynomialInputTransform::Identity {
                // The generic one-shot evaluator would copy this input again
                // before building its power basis. Hand ownership over directly.
                let x1 = eval_mod_input(module, ct, &work_layout, work_meta, scratch)?;
                let mut power_basis = PowerBasis::new(bsgs.basis(), x1);
                power_basis.populate(bsgs.degree(), bsgs.log_split(), bsgs.parity(), module, tsk, scratch)?;
                module.ckks_eval_poly_real_const_coeffs_from_power_basis(res, &power_basis, bsgs, tsk, scratch)?;
            } else {
                scratch.scope(|scratch_local| {
                    let (mut input, mut nested) = scratch_local.take_ckks_ciphertext_scratch(&work_layout, work_meta);
                    enter_plan_scale(module, &mut input, ct, work_meta, &mut nested)?;
                    if let Some(offset) = params.f_mod_input_offset.as_ref() {
                        const_add_assign(module, &mut input, offset, 0, &mut nested)?;
                    }
                    module.ckks_eval_poly_real_const_coeffs(res, &input, bsgs, tsk, &mut nested)
                })?;
            }

            if let Some(consts) = params.range_extension_consts.as_ref() {
                let steps = params.plan.f_mod_log_interval_reduction;
                for i in 0..steps {
                    if fused && i + 1 == steps {
                        last_square(module, res, s_in, Some((consts, i)), tsk, scratch)?;
                        returned = true;
                        continue;
                    }
                    scratch.scope(|scratch_local| -> Result<()> {
                        let (mut squared, mut nested) = scratch_local.take_ckks_ciphertext_like_scratch(&*res);
                        module.ckks_square_into(&mut squared, &*res, tsk, &mut nested)?;
                        module.ckks_double_into(res, &squared, &mut nested)
                    })?;
                    const_sub_assign(module, res, consts, i, scratch)?;
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
            // A fused last squaring is left to `last_square`, which rounds it with shaping.
            let steps = params.plan.f_mod_log_interval_reduction;
            let fused_last = fused && steps > 0;
            let reductions = steps - usize::from(fused_last);
            module.ckks_eval_mod_complex(res, ct, bsgs, reductions, &work_layout, work_meta, tsk, scratch)?;
            if fused_last {
                last_square::<_, _, P, _>(module, res, s_in, None, tsk, scratch)?;
                returned = true;
            }
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
    // A replicated layout returns to the input scale with a shaped rounding.
    if !returned && s_eval > s_in && replicated {
        scratch.scope(|scratch_local| -> Result<()> {
            let (mut wide, mut nested) = scratch_local.take_ckks_ciphertext_like_scratch(&*res);
            module.ckks_copy(&mut wide, &*res, &mut nested)?;
            res.set_k((wide.log_budget() + s_in).into());
            module.ckks_round_into(res, &wide, &mut nested)
        })?;
    } else if !returned && s_eval != s_in {
        res.set_log_delta(s_in);
    }

    Ok(())
}

/// The last range-extension step `res ← 2·res² − c·π_S` (`res ← res²` when
/// `step` is `None`) of a replicated layout, written at scale
/// `2^log_delta_out`. The square is kept `log_delta` bits under its scale,
/// where the doubling and the masked constant add exactly, so the division by
/// the square's scale and the return to `log_delta_out` are one shaped
/// rounding.
fn last_square<BE, R, P, H>(
    module: &Module<BE>,
    res: &mut R,
    log_delta_out: usize,
    step: Option<(&P, usize)>,
    tsk: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSSubOps<BE> + CKKSPow2Ops<BE> + CKKSRoundOps<BE> + GLWETensoring<BE>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + CKKSMaskedConstants<BE>,
    H: GetTensorKey<BE>,
{
    let lambda = res.log_delta();
    let wide_layout = GLWELayout {
        n: res.n(),
        base2k: res.base2k(),
        k: res.k(),
        rank: Rank(1),
    };
    scratch.scope(|scratch_local| -> Result<()> {
        let (mut wide, mut nested) = scratch_local.take_ckks_ciphertext_scratch(&wide_layout, res.meta());
        square_wide_into(module, &mut wide, &*res, tsk, lambda, &mut nested)?;
        if let Some((consts, i)) = step {
            module.ckks_mul_pow2_assign(1, &mut wide, &mut nested)?;
            const_sub_assign(module, &mut wide, consts, i, &mut nested)?;
        }
        res.set_k((wide.log_budget() + log_delta_out).into());
        module.ckks_round_into(res, &wide, &mut nested)
    })
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
        P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos + CKKSMaskedConstants<BE>,
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
        + CKKSRoundOps<BE>
        + GLWECopy<BE>
        + GLWENormalize<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
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
        P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos + CKKSMaskedConstants<BE>,
        H: GetTensorKey<BE>,
    {
        if bsgs.re.input_transform() == PolynomialInputTransform::Identity
            && bsgs.im.input_transform() == PolynomialInputTransform::Identity
        {
            let x1 = eval_mod_input(self, ct, work_layout, work_meta, scratch)?;
            let mut power_basis = PowerBasis::new(bsgs.re.basis(), x1);
            power_basis.populate(bsgs.re.degree(), bsgs.re.log_split(), bsgs.re.parity(), self, tsk, scratch)?;
            self.ckks_eval_poly_complex_const_coeffs_from_power_basis(res, &power_basis, bsgs, tsk, scratch)?;
        } else {
            scratch.scope(|scratch_local| {
                let (mut input, mut nested) = scratch_local.take_ckks_ciphertext_scratch(work_layout, work_meta);
                enter_plan_scale(self, &mut input, ct, work_meta, &mut nested)?;
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
) -> Result<CKKSCiphertextOwned<BE>>
where
    BE: Backend,
    Module<BE>: CKKSModuleAlloc<BE> + GLWECopy<BE> + CKKSRoundOps<BE>,
    C: GLWEToBackendRef<BE> + CKKSCtBounds,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
{
    let mut input = module.ckks_ciphertext_alloc(layout.base2k, layout.k);
    enter_plan_scale(module, &mut input, ct, meta, scratch)?;
    Ok(input)
}

/// Copies `ct` into `dst` at the plan scale of `meta`. A replicated layout
/// that drops bits to reach it rounds with shaping.
fn enter_plan_scale<BE, D, C>(
    module: &Module<BE>,
    dst: &mut D,
    ct: &C,
    meta: GLWEPlaintextMeta,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: GLWECopy<BE> + CKKSRoundOps<BE>,
    D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    C: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    if matches!(ct.slot_layout(), SlotLayout::Replicated(_)) && ct.k() > dst.k() {
        module.ckks_round_into(dst, ct, scratch)?;
    } else {
        module.glwe_copy(dst, ct, scratch);
    }
    dst.set_meta(meta);
    Ok(())
}

pub use super::eval_mod_scratch::ckks_eval_mod_tmp_bytes_reference;
