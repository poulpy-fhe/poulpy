//! Homomorphic evaluation of the `x mod 1` reduction, the core non-linear step
//! of CKKS bootstrapping.
//!
//! This module holds the *evaluation* — the backend-generic reference
//! [`CKKSEvalModOpsDefault`] and the `eval_mod` pipeline. The parameterization
//! it consumes (the periodic-function approximation polynomials and their
//! encoding) lives in [`crate::layouts::eval_mod`]; see there for the maths and
//! the [`EvalMod`] structure. The public entry point is
//! [`CKKSEvalModOps`](crate::api::CKKSEvalModOps).

use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::{
    GLWECopy,
    layouts::{
        BSGSMeta, GGLWEInfos, GLWELayout, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, Rank, SetBSGSMeta,
        prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSMeta, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps, CKKSMulOps, CKKSPolynomialEvaluationOps, CKKSPow2Ops, CKKSSubOps},
    layouts::{
        CKKSCiphertext, CKKSModuleAlloc, ScratchArenaTakeCKKS,
        eval_mod::{EvalMod, EvalModBsgs},
    },
};

/// Backend-generic reference implementation of [`CKKSEvalModOps`].
///
/// Blanket-implemented for any [`Module<BE>`] whose backend provides the
/// constituent CKKS ops (polynomial evaluation, add/sub/mul/copy, allocation).
/// Backends wire this into the public [`CKKSEvalModOps`] trait through the
/// [`CKKSEvalModImpl`](crate::oep::CKKSEvalModImpl) OEP hook, which by default
/// forwards to [`Self::ckks_eval_mod_default`].
///
/// [`CKKSEvalModOps`]: crate::api::CKKSEvalModOps
pub trait CKKSEvalModOpsDefault<BE: Backend> {
    /// Reference `x mod 1` evaluation: see [`crate::layouts::eval_mod`] for the
    /// base-polynomial / range-extension / inverse pipeline and the `eval_mod`
    /// function for the implementation.
    fn ckks_eval_mod_default<R, C, P, F>(
        &self,
        res: &mut R,
        ct: &C,
        params: &EvalMod<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSPolynomialEvaluationOps<BE>
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
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos;
}

impl<BE: Backend> CKKSEvalModOpsDefault<BE> for Module<BE>
where
    Module<BE>: CKKSPolynomialEvaluationOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSMulOps<BE>
        + CKKSCopyOps<BE>
        + CKKSModuleAlloc<BE>
        + CKKSPow2Ops<BE>
        + GLWECopy<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
{
    fn ckks_eval_mod_default<R, C, P, F>(
        &self,
        res: &mut R,
        ct: &C,
        params: &EvalMod<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos,
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
fn eval_mod<R, C, P, F, BE>(
    module: &Module<BE>,
    res: &mut R,
    ct: &C,
    params: &EvalMod<F, P>,
    tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSPolynomialEvaluationOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSMulOps<BE>
        + CKKSCopyOps<BE>
        + CKKSModuleAlloc<BE>
        + CKKSPow2Ops<BE>
        + GLWECopy<BE>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    C: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    // EvalMod runs at its own plan scale `f_mod_log_delta`: reinterpret the
    // working ciphertext to it on entry, then return the result to the input
    // scale. `consumed_bits()` accounts for the arithmetic at the plan scale;
    // if the plan scale is higher than the input scale, returning to the input
    // scale also drops that extra precision from the externally visible budget.
    // Intermediates are allocated rank-1 (`ckks_ciphertext_alloc`); reject
    // higher-rank inputs instead of silently mis-shaping them.
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

    // The working copy at the plan scale is carved from scratch (accounted for
    // by `ckks_eval_mod_tmp_bytes`), not heap-allocated: it lives for the whole
    // evaluation (the inverse stage reuses it), so its bytes are charged on top
    // of every nested stage. `glwe_copy` zero-fills the destination limbs
    // beyond the source, so the dirty scratch region is fully defined.
    scratch.scope(|scratch_local| {
        let (mut t1, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(
            &GLWELayout {
                n: ct.n(),
                base2k: ct.base2k(),
                k: (s_budget + s_eval).into(),
                rank: Rank(1),
            },
            CKKSMeta {
                log_delta: s_eval,
                log_sparsity: ct.log_sparsity(),
            },
        );
        module.glwe_copy(&mut t1, ct);

        match &params.f_mod_bsgs {
            EvalModBsgs::Real(bsgs) => {
                module.ckks_eval_poly_real_const_coeffs(res, &t1, bsgs, tsk, &mut scratch_local)?;

                if let Some(consts) = params.range_extension_consts.as_ref() {
                    for i in 0..params.plan.f_mod_log_interval_reduction {
                        module.ckks_square_assign(res, tsk, &mut scratch_local)?;
                        module.ckks_mul_pow2_assign(res, 1, &mut scratch_local)?;
                        module.ckks_sub_pt_const_assign(res, 0, consts, i, &mut scratch_local)?;
                    }
                }

                if let Some(inv) = params.f_mod_inv_bsgs.as_ref() {
                    module.ckks_copy(&mut t1, &*res, &mut scratch_local)?;
                    module.ckks_eval_poly_real_const_coeffs(res, &t1, inv, tsk, &mut scratch_local)?;
                }
            }
            EvalModBsgs::Complex(bsgs) => {
                module.ckks_eval_poly_complex_const_coeffs(res, &t1, bsgs, tsk, &mut scratch_local)?;
                for _ in 0..params.plan.f_mod_log_interval_reduction {
                    module.ckks_square_assign(res, tsk, &mut scratch_local)?;
                }
            }
        }
        Result::Ok(())
    })?;

    // Restore the input scale on the result. This is a pure metadata relabel
    // (`set_log_delta`), not a rescale: entry raised the scale `s_in -> s_eval`
    // without spending budget (`set_log_budget(s_budget)` on an MSB-aligned copy,
    // which reinterprets the value at `2^-(s_eval - s_in)`); relabelling back to
    // `s_in` here undoes exactly that, so the scale round-trip is budget-neutral
    // and the only consumption is the EvalMod arithmetic, which `consumed_bits()`
    // accounts for in full.
    if s_eval != s_in {
        res.set_log_delta(s_in);
    }

    Ok(())
}
