//! Homomorphic evaluation of the `x mod 1` reduction, the core non-linear step
//! of CKKS bootstrapping.
//!
//! This module holds the *evaluation* — the backend-generic reference
//! [`CKKSEvalModOpsDefault`] and the `eval_mod` pipeline. The parameterization
//! it consumes (the periodic-function approximation polynomials and their
//! encoding) lives in [`crate::layouts::eval_mod`]; see there for the maths and
//! the [`EvalMod`] structure. The public entry point is
//! [`CKKSEvalModOps`](crate::api::CKKSEvalModOps).

use anyhow::{Result, ensure};
use poulpy_core::ScratchArenaTakeCore;
use poulpy_core::layouts::{
    BSGSMeta, GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta,
    prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Module, ScratchArena},
};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps, CKKSMulOps, CKKSSubOps, PolynomialEvaluation},
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
        Self: PolynomialEvaluation<BE>
            + CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + CKKSMulOps<BE>
            + CKKSCopyOps<BE>
            + CKKSModuleAlloc<BE>
            + Sized,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        for<'b> ScratchArena<'b, BE>: ScratchAvailable + ScratchArenaTakeCore<'b, BE>;
}

impl<BE: Backend> CKKSEvalModOpsDefault<BE> for Module<BE>
where
    Module<BE>:
        PolynomialEvaluation<BE> + CKKSAddOps<BE> + CKKSSubOps<BE> + CKKSMulOps<BE> + CKKSCopyOps<BE> + CKKSModuleAlloc<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    for<'b> ScratchArena<'b, BE>: ScratchAvailable + ScratchArenaTakeCore<'b, BE>,
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
        P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
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
fn eval_mod<R, C, P, F, BE: Backend>(
    module: &Module<BE>,
    res: &mut R,
    ct: &C,
    params: &EvalMod<F, P>,
    tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    Module<BE>:
        PolynomialEvaluation<BE> + CKKSAddOps<BE> + CKKSSubOps<BE> + CKKSMulOps<BE> + CKKSCopyOps<BE> + CKKSModuleAlloc<BE>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    C: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    for<'b> ScratchArena<'b, BE>: ScratchAvailable + ScratchArenaTakeCore<'b, BE>,
{
    let depth = params.eval_depth();
    let required = depth * ct.log_delta();
    ensure!(
        ct.log_budget() >= required,
        "ckks_eval_mod: input log_budget {got} < {required} bits required ({depth} levels × log_delta {ld})",
        got = ct.log_budget(),
        ld = ct.log_delta(),
    );

    let mut t1 = module.ckks_ciphertext_alloc_from_infos(ct);
    module.ckks_copy(&mut t1, ct, scratch)?;

    match &params.f_mod_bsgs {
        EvalModBsgs::Real(bsgs) => {
            module.ckks_eval_poly_real_const_coeffs(res, &t1, bsgs, tsk, scratch)?;

            let consts = params.range_extension_consts.as_ref();
            for i in 0..params.plan.f_mod_log_interval_reduction {
                // Step `i` subtracts coefficient `i` of the packed constants plaintext.
                let dac = consts.expect("range_extension_consts present when range extension is used");
                scratch.scope(|local| -> Result<()> {
                    let (mut work, mut local) = local.take_compact_ckks_ciphertext_scratch(&*res);
                    module.ckks_copy(&mut work, &*res, &mut local)?;
                    module.ckks_square_assign(&mut work, tsk, &mut local)?;
                    module.ckks_add_into(res, &work, &work, &mut local)?;
                    module.ckks_sub_pt_const_assign(res, 0, dac, i, &mut local)?;
                    Ok(())
                })?;
            }

            if let Some(inv) = params.f_mod_inv_bsgs.as_ref() {
                let compact_k = res.effective_k();
                let mut t1_inv = module.ckks_ciphertext_alloc(res.base2k(), compact_k.into());
                t1_inv.set_meta(res.meta());
                module.ckks_copy(&mut t1_inv, &*res, scratch)?;
                module.ckks_eval_poly_real_const_coeffs(res, &t1_inv, inv, tsk, scratch)?;
            }
        }
        EvalModBsgs::Complex(bsgs) => {
            module.ckks_eval_poly_complex_const_coeffs(res, &t1, bsgs, tsk, scratch)?;

            for _ in 0..params.plan.f_mod_log_interval_reduction {
                scratch.scope(|local| -> Result<()> {
                    let (mut work, mut local) = local.take_compact_ckks_ciphertext_scratch(&*res);
                    module.ckks_copy(&mut work, &*res, &mut local)?;
                    module.ckks_square_into(res, &work, tsk, &mut local)?;
                    Ok(())
                })?;
            }
        }
    }

    Ok(())
}
