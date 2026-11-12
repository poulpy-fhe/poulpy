//! Homomorphic evaluation of the `x mod 1` reduction, the core non-linear step
//! of CKKS bootstrapping.
//!
//! This module holds the *evaluation* — the backend-generic reference
//! [`CKKSEvalModReference`] and the `eval_mod` pipeline. The parameterization
//! it consumes (the periodic-function approximation polynomials and their
//! encoding) lives in [`crate::layouts::eval_mod`]; see there for the maths and
//! the [`EvalMod`] structure. The public entry point is
//! [`CKKSEvalModOps`](crate::api::CKKSEvalModOps).

use crate::{CKKSResult as Result, checked_log_budget_sub, ckks_ensure, ckks_ensure_operand};
use poulpy_core::layouts::{GetTensorKey, IntPolyInfos};
use poulpy_core::oep::GLWEMulConstImpl;
use poulpy_core::{
    GLWEBytesOf, GLWECopy, GLWENormalize, GLWETensoring,
    layouts::{
        BSGSMeta, GGLWEInfos, GLWELayout, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank, SetBSGSMeta,
        SetK, prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{CnvOps, VecZnxBigNormalizeTmpBytes, VecZnxIdftApplyTmpBytes, VecZnxRshTmpBytes},
    layouts::{Backend, ConjugateInvariant, Module, Ring, ScratchArena, Standard},
};

use crate::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos, SlotLayout,
    api::{
        CKKSAddOps, CKKSComplexPolynomialEvaluationOps, CKKSCopyOps, CKKSMulAddOps, CKKSMulOps, CKKSPolynomialEvaluationOps,
        CKKSPow2Ops, CKKSRoundOps, CKKSSubOps, PolynomialInputTransform,
    },
    layouts::{
        CKKSCiphertextOwned, CKKSCiphertextViewMut, CKKSMaskedConstants, ScratchArenaTakeCKKSCiphertext,
        eval_mod::{EvalMod, EvalModBsgs},
    },
    oep::derived::polynomial_evaluation::{eval_poly_real_in, eval_poly_real_on_power},
    polynomial::{BSGSPolynomial, ComplexBSGSPolynomial},
    power_basis::{PowerBasis, populate_in},
    reference::{
        mul::{square_wide_into, tensor_key_at},
        polynomial_evaluation::{const_add_assign, const_sub_assign, ensure_baby_steps},
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
pub trait CKKSEvalModReference<BE: Backend, Rg: Ring = Standard> {
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
        Self: CKKSPolynomialEvaluationOps<BE, Rg>
            + CKKSEvalModComplexOps<BE, Rg>
            + CKKSAddOps<BE, Rg>
            + CKKSSubOps<BE, Rg>
            + CKKSMulOps<BE, Rg>
            + CKKSCopyOps<BE, Rg>
            + CnvOps<BE, Rg>
            + VecZnxRshTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VecZnxIdftApplyTmpBytes
            + GLWEBytesOf<BE>
            + CKKSRoundOps<BE, Rg>
            + CKKSMulAddOps<BE, Rg>
            + Sized,
        BE: GLWEMulConstImpl<Rg>,
        R: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, Rg> + IntPolyInfos + CKKSCtBounds + BSGSMeta + CKKSMaskedConstants<BE, Rg>,
        H: GetTensorKey<BE, Rg>,
        CKKSCiphertextOwned<BE, Rg>: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos;

    /// Scratch of [`Self::ckks_eval_mod_reference`].
    fn ckks_eval_mod_tmp_bytes_reference<R, C, P, F, T>(&self, res: &R, ct: &C, params: &EvalMod<F, P>, tsk: &T) -> usize
    where
        BE: GLWEMulConstImpl<Rg>,
        Self: CKKSAddOps<BE, Rg>
            + CKKSSubOps<BE, Rg>
            + CKKSMulOps<BE, Rg>
            + CKKSPow2Ops<BE, Rg>
            + GLWENormalize<BE, Rg>
            + CKKSCopyOps<BE, Rg>
            + GLWECopy<BE, Rg>
            + CnvOps<BE, Rg>
            + VecZnxRshTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VecZnxIdftApplyTmpBytes
            + GLWEBytesOf<BE>
            + CKKSRoundOps<BE, Rg>
            + CKKSMulAddOps<BE, Rg>,
        R: CKKSCtBounds,
        C: CKKSCtBounds,
        P: CKKSCtBounds,
        T: GGLWEInfos;
}

impl<BE: Backend, Rg: Ring> CKKSEvalModReference<BE, Rg> for Module<BE>
where
    Module<BE>: CKKSPolynomialEvaluationOps<BE, Rg>
        + CKKSEvalModComplexOps<BE, Rg>
        + CKKSAddOps<BE, Rg>
        + CKKSSubOps<BE, Rg>
        + CKKSMulOps<BE, Rg>
        + CKKSCopyOps<BE, Rg>
        + CKKSPow2Ops<BE, Rg>
        + CKKSRoundOps<BE, Rg>
        + GLWETensoring<BE, Rg>
        + GLWECopy<BE, Rg>
        + GLWENormalize<BE, Rg>,
    CKKSCiphertextOwned<BE, Rg>: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE, Rg>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE, Rg>,
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
        Self: CnvOps<BE, Rg>
            + VecZnxRshTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VecZnxIdftApplyTmpBytes
            + GLWEBytesOf<BE>
            + CKKSRoundOps<BE, Rg>
            + CKKSMulAddOps<BE, Rg>,
        BE: GLWEMulConstImpl<Rg>,
        R: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + BSGSMeta + IntPolyInfos + CKKSMaskedConstants<BE, Rg>,
        H: GetTensorKey<BE, Rg>,
    {
        crate::ensure_operand_degrees(OP, self, [res.n(), ct.n()])?;
        check_eval_mod(self, ct, params)?;
        // An evaluation without a product has no key to size its query with, and needs none.
        if relinearizes(params) {
            let widest = params.plan.relinearization_k(ct.log_budget());
            let key = tensor_key_at::<BE, _, Rg>(OP, tsk, widest.into())?;
            let bytes = ckks_eval_mod_tmp_bytes_reference(self, res, ct, params, &key);
            ckks_ensure!(
                scratch.available() >= bytes,
                "{OP} needs a scratch arena of {bytes} bytes, got {}",
                scratch.available()
            );
        }
        eval_mod(self, res, ct, params, tsk, scratch)
    }

    fn ckks_eval_mod_tmp_bytes_reference<R, C, P, F, T>(&self, res: &R, ct: &C, params: &EvalMod<F, P>, tsk: &T) -> usize
    where
        BE: GLWEMulConstImpl<Rg>,
        Self: CKKSAddOps<BE, Rg>
            + CKKSSubOps<BE, Rg>
            + CKKSMulOps<BE, Rg>
            + CKKSPow2Ops<BE, Rg>
            + GLWENormalize<BE, Rg>
            + CKKSCopyOps<BE, Rg>
            + GLWECopy<BE, Rg>
            + CnvOps<BE, Rg>
            + VecZnxRshTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VecZnxIdftApplyTmpBytes
            + GLWEBytesOf<BE>
            + CKKSRoundOps<BE, Rg>
            + CKKSMulAddOps<BE, Rg>,
        R: CKKSCtBounds,
        C: CKKSCtBounds,
        P: CKKSCtBounds,
        T: GGLWEInfos,
    {
        ckks_eval_mod_tmp_bytes_reference(self, res, ct, params, tsk)
    }
}

/// The name of [`ckks_eval_mod`](crate::api::CKKSEvalModOps::ckks_eval_mod) in its refusals.
const OP: &str = "ckks_eval_mod";

/// Refuses what EvalMod cannot evaluate on an input like `ct`.
fn check_eval_mod<BE, C, F, P, Rg: Ring>(module: &Module<BE>, ct: &C, params: &EvalMod<F, P>) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSEvalModComplexOps<BE, Rg>,
    C: CKKSCtBounds,
{
    if matches!(params.f_mod_bsgs, EvalModBsgs::Complex(_)) && !module.ckks_eval_mod_complex_supported() {
        return Err(complex_refused());
    }
    ckks_ensure_operand!(
        OP,
        ct.rank().as_usize() == 1,
        "supports rank-1 ciphertexts only, got rank {}",
        ct.rank().as_usize()
    );
    checked_log_budget_sub(OP, ct.log_budget(), params.consumed_bits())?;
    Ok(())
}

/// Whether evaluating `params` takes a ciphertext product: a polynomial of degree two or more, or with an input
/// transform, a range-extension step, or such an inverse.
fn relinearizes<F, P>(params: &EvalMod<F, P>) -> bool {
    let products = |poly: &BSGSPolynomial<P>| poly.degree() > 1 || poly.input_transform() != PolynomialInputTransform::Identity;
    let (base, squares) = match &params.f_mod_bsgs {
        EvalModBsgs::Real(bsgs) => (products(bsgs), params.range_extension_consts.is_some()),
        EvalModBsgs::Complex(bsgs) => (products(&bsgs.re) || products(&bsgs.im), true),
    };
    base || (squares && params.plan.f_mod_log_interval_reduction > 0) || params.f_mod_inv_bsgs.as_ref().is_some_and(products)
}

/// The evaluation [`EvalModPlan`](crate::layouts::eval_mod::EvalModPlan) describes, on `ct` when its allocation holds the
/// working width, on a carved copy otherwise. A range-extension step is `res ← 2·res² − dac` on the real path, the
/// `cos 2θ` identity, and `res ← res²` on the complex path.
fn eval_mod<R, C, P, F, BE, H, Rg: Ring>(
    module: &Module<BE>,
    res: &mut R,
    ct: &mut C,
    params: &EvalMod<F, P>,
    tsk: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSPolynomialEvaluationOps<BE, Rg>
        + CKKSEvalModComplexOps<BE, Rg>
        + CKKSAddOps<BE, Rg>
        + CKKSSubOps<BE, Rg>
        + CKKSMulOps<BE, Rg>
        + CKKSCopyOps<BE, Rg>
        + CKKSPow2Ops<BE, Rg>
        + CKKSRoundOps<BE, Rg>
        + GLWETensoring<BE, Rg>
        + GLWECopy<BE, Rg>
        + GLWENormalize<BE, Rg>,
    R: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    C: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    P: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + BSGSMeta + IntPolyInfos + CKKSMaskedConstants<BE, Rg>,
    H: GetTensorKey<BE, Rg>,
    CKKSCiphertextOwned<BE, Rg>: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
{
    let log_delta = ct.log_delta();
    let (work_layout, work_meta) = params.plan.work_layout(ct);
    let replicated = matches!(ct.slot_layout(), SlotLayout::Replicated(_));
    // The last squaring of a replicated layout rounds straight to the input
    // scale when no inverse follows it.
    let fused = replicated && params.f_mod_inv_bsgs.is_none() && 2 * params.plan.f_mod_log_delta >= log_delta;
    let mut returned = false;
    if replicated && work_layout.k < ct.k() {
        // A replicated layout drops bits to the plan scale with a shaped rounding.
        let (mut input, mut arena) = scratch.borrow().take_ckks_ciphertext_scratch(&work_layout, work_meta);
        module.ckks_round(&mut input, &*ct, &mut arena)?;
        input.set_meta(work_meta);
        eval_mod_base(module, res, input, params, tsk, arena)?;
    } else if work_layout.k <= ct.max_k() {
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
                let steps = params.plan.f_mod_log_interval_reduction;
                for i in 0..steps {
                    if fused && i + 1 == steps {
                        last_square(module, res, log_delta, Some((consts, i)), tsk, scratch)?;
                        returned = true;
                        continue;
                    }
                    scratch.scope(|scratch_local| -> Result<()> {
                        let (mut squared, mut nested) = scratch_local.take_ckks_ciphertext_like_scratch(&*res);
                        module.ckks_square_into(&mut squared, &*res, tsk, &mut nested)?;
                        module.ckks_double_into(res, &squared, &mut nested)
                    })?;
                    const_sub_assign(OP, module, res, consts, i, scratch)?;
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
            // A fused last squaring is left to `last_square`, which rounds it with shaping.
            let steps = params.plan.f_mod_log_interval_reduction;
            let fused_last = fused && steps > 0;
            for _ in 0..steps - usize::from(fused_last) {
                module.ckks_square_assign(res, tsk, scratch)?;
            }
            if fused_last {
                last_square::<_, _, P, _, _>(module, res, log_delta, None, tsk, scratch)?;
                returned = true;
            }
        }
    }

    // A relabel, not a rescale: it undoes the one at entry, keeping the budget. When the plan scale is the wider one it
    // lowers `k`, which clears the canonical flag. A replicated layout returns to the input scale with a shaped rounding.
    let s_eval = params.plan.f_mod_log_delta;
    if !returned && s_eval > log_delta && replicated {
        scratch.scope(|scratch_local| -> Result<()> {
            let (mut wide, mut nested) = scratch_local.take_ckks_ciphertext_like_scratch(&*res);
            module.ckks_copy(&mut wide, &*res, &mut nested)?;
            res.set_k((wide.log_budget() + log_delta).into());
            module.ckks_round(res, &wide, &mut nested)
        })?;
    } else if !returned && s_eval != log_delta {
        res.set_log_delta(log_delta);
    }

    Ok(())
}

/// The last range-extension step `res ← 2·res² − c·π_S` (`res ← res²` when
/// `step` is `None`) of a replicated layout, written at scale
/// `2^log_delta_out`. The square is kept `log_delta` bits under its scale,
/// where the doubling and the masked constant add exactly, so the division by
/// the square's scale and the return to `log_delta_out` are one shaped
/// rounding.
fn last_square<BE, R, P, H, Rg: Ring>(
    module: &Module<BE>,
    res: &mut R,
    log_delta_out: usize,
    step: Option<(&P, usize)>,
    tsk: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSSubOps<BE, Rg> + CKKSPow2Ops<BE, Rg> + CKKSRoundOps<BE, Rg> + GLWETensoring<BE, Rg>,
    R: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
    P: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + IntPolyInfos + CKKSMaskedConstants<BE, Rg>,
    H: GetTensorKey<BE, Rg>,
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
        square_wide_into(OP, module, &mut wide, &*res, tsk, lambda, &mut nested)?;
        if let Some((consts, i)) = step {
            module.ckks_mul_pow2_assign(1, &mut wide, &mut nested)?;
            const_sub_assign(OP, module, &mut wide, consts, i, &mut nested)?;
        }
        res.set_k((wide.log_budget() + log_delta_out).into());
        module.ckks_round(res, &wide, &mut nested)
    })
}

/// The base polynomial on `input`, at the working layout, its powers carved from `scratch`.
fn eval_mod_base<'a, R, P, F, BE, H, Rg: Ring>(
    module: &Module<BE>,
    res: &mut R,
    mut input: CKKSCiphertextViewMut<'a, BE, Rg>,
    params: &EvalMod<F, P>,
    tsk: &H,
    mut scratch: ScratchArena<'a, BE>,
) -> Result<()>
where
    BE: Backend + 'a,
    Module<BE>: CKKSPolynomialEvaluationOps<BE, Rg>
        + CKKSEvalModComplexOps<BE, Rg>
        + CKKSAddOps<BE, Rg>
        + CKKSSubOps<BE, Rg>
        + CKKSMulOps<BE, Rg>
        + CKKSCopyOps<BE, Rg>
        + CKKSPow2Ops<BE, Rg>
        + GLWENormalize<BE, Rg>,
    R: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    P: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + BSGSMeta + IntPolyInfos + CKKSMaskedConstants<BE, Rg>,
    H: GetTensorKey<BE, Rg>,
    CKKSCiphertextOwned<BE, Rg>: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
{
    match &params.f_mod_bsgs {
        // The identity transform takes `input` itself as the first power, with no copy.
        EvalModBsgs::Real(bsgs) if bsgs.input_transform() == PolynomialInputTransform::Identity => {
            ensure_baby_steps(OP, bsgs)?;
            eval_poly_real_on_power(module, res, input, bsgs, tsk, scratch).map(|_| ())
        }
        EvalModBsgs::Real(bsgs) => {
            if let Some(offset) = params.f_mod_input_offset.as_ref() {
                const_add_assign(OP, module, &mut input, offset, 0, &mut scratch)?;
            }
            eval_poly_real_in::<_, CKKSCiphertextViewMut<'a, BE, Rg>, _, _, _, _, Rg>(OP, module, res, &input, bsgs, tsk, scratch)
        }
        EvalModBsgs::Complex(bsgs) => module.ckks_eval_mod_complex(res, input, bsgs, tsk, scratch),
    }
}

/// The complex base polynomial of [`EvalModType::ExpCmplx`](crate::layouts::eval_mod::EvalModType::ExpCmplx)
/// and its squarings, on ring `Rg`: refused by default, as on the
/// conjugate-invariant ring, whose slots are real.
///
/// Implemented once per ring for [`Module`], for EvalMod to dispatch on: not meant to be
/// called directly.
pub trait CKKSEvalModComplexOps<BE: Backend, Rg: Ring> {
    /// Whether the ring evaluates the complex base polynomial; EvalMod refuses `ExpCmplx`
    /// before any other check otherwise.
    fn ckks_eval_mod_complex_supported(&self) -> bool {
        false
    }

    fn ckks_eval_mod_complex<'a, Res, P, H>(
        &self,
        _res: &mut Res,
        _input: CKKSCiphertextViewMut<'a, BE, Rg>,
        _bsgs: &ComplexBSGSPolynomial<P>,
        _tsk: &H,
        _scratch: ScratchArena<'a, BE>,
    ) -> Result<()>
    where
        Res: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        P: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + BSGSMeta + IntPolyInfos + CKKSMaskedConstants<BE, Rg>,
        H: GetTensorKey<BE, Rg>,
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
        P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos + CKKSMaskedConstants<BE>,
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
