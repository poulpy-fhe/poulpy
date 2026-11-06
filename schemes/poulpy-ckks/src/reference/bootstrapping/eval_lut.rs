//! LUT evaluation as a bootstrapping step, in place of EvalMod.

use poulpy_core::{
    GLWEBytesOf, GLWECopy, GLWENormalize,
    layouts::{
        BSGSMeta, GGLWEInfos, GLWELayout, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, LWEInfos,
        SetBSGSMeta, SetK, prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, CKKSResult as Result, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSAllOpsTmpBytes, CKKSComplexPolynomialEvaluationOps, CKKSConjugateOps, CKKSCopyOps, CKKSEvalModOps,
        CKKSImagOps, CKKSMulOps, CKKSNegOps, CKKSPolynomialEvaluationOps, CKKSPow2Ops, CKKSSubOps,
    },
    checked_log_budget_sub, ckks_ensure, ckks_ensure_operand,
    eval_lut::{ckks_eval_lut, ckks_eval_lut_binary, ckks_eval_lut_from_basis, ckks_lut_power_basis},
    layouts::{
        BootstrappingKeys, BootstrappingKeysLayout, BootstrappingLinearKeys, CKKSCITwin, CKKSCiphertextOwned, CKKSEvalModInfos,
        CKKSPlaintextOwned, EncodedLut, EncodedLutKind, ScratchArenaTakeCKKSCiphertext,
        bootstrapping::bootstrap_layouts,
        eval_mod::EvalModPlan,
        validation::{ensure_lut_batch, eval_mod_standard, remove_guard_bits, required_tensor_key},
    },
    reference::mul::tensor_key_at,
};

use super::{ckks_bootstrap_real_part, ckks_split_of, recombine_halves};

/// The name of [`ckks_eval_lut`](crate::api::CKKSBootstrappingOps::ckks_eval_lut) in its refusals.
const EVAL_LUT_OP: &str = "ckks_eval_lut";

/// Backend-generic reference of LUT evaluation as a bootstrapping step, selected by
/// [`impl_ckks_bootstrapping_reference_full`](crate::oep::impl_ckks_bootstrapping_reference_full).
pub trait CKKSEvalLutReference<BE: CKKSCITwin>: std::borrow::Borrow<Module<BE>> {
    /// Scratch upper bound for [`Self::ckks_eval_lut_reference`]: the halves and accumulator beside one
    /// LUT-evaluation temporary, so the bound does not grow with `luts.len()`.
    fn ckks_eval_lut_tmp_bytes_reference<R, A, P, F>(
        &self,
        res: &R,
        a: &A,
        plan: &P,
        luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        Module<BE>: GLWEBytesOf<BE>,
        Module<BE>: CKKSAllOpsTmpBytes<BE> + CKKSEvalModOps<BE> + CKKSMulOps<BE> + CKKSAddOps<BE> + CKKSSubOps<BE>,
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSEvalModInfos<BE, F>,
    {
        let module: &Module<BE> = self.borrow();
        let (boot_layout, a_layout) = bootstrap_layouts(res, a);
        let boot_bytes = BE::scratch_aligned(module.glwe_bytes_of_from_infos(&boot_layout));
        let a_bytes = BE::scratch_aligned(module.glwe_bytes_of_from_infos(&a_layout));
        // The two halves of `a` and the accumulator, beside one LUT-evaluation temporary.
        let carved = 2 * a_bytes + 2 * boot_bytes;
        let tsk = &keys_layout.tensor_key;
        let mut nested = 0;
        if let Some(eval_mod) = plan
            .eval_mod_standard()
            .filter(|_| luts.iter().any(EncodedLut::requires_eval_mod))
        {
            nested = nested
                .max(module.ckks_eval_mod_tmp_bytes(&boot_layout, &boot_layout, eval_mod, tsk))
                .max(module.ckks_eval_mod_tmp_bytes(res, &boot_layout, eval_mod, tsk));
        }
        for atk in keys_layout.rotation_key_layouts() {
            for lut in luts {
                nested = nested
                    .max(module.ckks_all_ops_with_atk_tmp_bytes(&boot_layout, lut.coeffs(), tsk, atk))
                    .max(module.ckks_all_ops_with_atk_tmp_bytes(&a_layout, lut.coeffs(), tsk, atk));
            }
        }
        // A double-angle chain squares into a temporary of its output's layout.
        if luts
            .iter()
            .any(|lut| matches!(lut.kind(), EncodedLutKind::Binary { chain: Some(_), .. }))
        {
            let size = res.max_size();
            nested = nested.max(
                BE::scratch_aligned(module.glwe_bytes_of_from_infos(&GLWELayout {
                    k: res.max_k(),
                    ..res.glwe_layout()
                })) + module
                    .ckks_square_tmp_bytes(res, res, tsk)
                    .max(module.ckks_add_pt_const_tmp_bytes(size))
                    .max(module.ckks_sub_pt_const_tmp_bytes(size)),
            );
        }
        carved + nested
    }

    /// Reference body of [`ckks_eval_lut`](crate::api::CKKSBootstrappingOps::ckks_eval_lut).
    fn ckks_eval_lut_reference<R, A, P, F, K>(
        &self,
        res: &mut [R],
        a: &A,
        plan: &P,
        luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEBytesOf<BE>
            + CKKSAllOpsTmpBytes<BE>
            + CKKSConjugateOps<BE>
            + CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + CKKSImagOps<BE>
            + CKKSEvalModOps<BE>
            + CKKSPolynomialEvaluationOps<BE>
            + CKKSComplexPolynomialEvaluationOps<BE>
            + CKKSCopyOps<BE>
            + CKKSMulOps<BE>
            + CKKSPow2Ops<BE>
            + CKKSNegOps<BE>
            + GLWECopy<BE>
            + GLWENormalize<BE>,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: CKKSEvalModInfos<BE, F>,
        K: BootstrappingKeys<BE>,
        CKKSCiphertextOwned<BE>:
            GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        let module: &Module<BE> = self.borrow();
        const OP: &str = EVAL_LUT_OP;
        ensure_lut_batch(OP, module.n(), a, res, luts, plan)?;
        let log_delta = remove_guard_bits(OP, a.log_delta(), plan.c2s_guard_bits())?;
        let tensor_key = required_tensor_key(OP, keys.tensor_key())?;
        // Every conjugation of the step (real part, split, LUT symmetrization) switches at a width at most `a.k()`.
        keys.get_automorphism_key(-1, a.k())
            .map_err(|source| CKKSCompositionError::MissingAutomorphismKey {
                op: OP,
                rotation: 0,
                k: a.k().into(),
                source,
            })?;
        let a_k = a.k().as_usize();
        for lut in luts {
            check_lut_capacity(OP, a.log_budget(), log_delta, plan.eval_mod_plan(), lut, tensor_key)?;
            if let Some((_, k)) = plan.eval_lut_output(OP, lut, a_k)? {
                let res_k = res[0].k().as_usize();
                ckks_ensure_operand!(OP, res_k >= k, "takes outputs of at least {k} bits, got {res_k}");
            }
        }
        // The evaluation runs at the width of `res`: a narrower output can fail after it was written.
        ckks_ensure_operand!(OP, a.k() <= res[0].k(), "takes an input no wider than its output");
        if let Some(keys_layout) = keys.keys_layout() {
            let bytes = module.ckks_eval_lut_tmp_bytes_reference(&res[0], a, plan, luts, keys_layout);
            ckks_ensure!(
                scratch.available() >= bytes,
                "{OP} needs a scratch arena of {bytes} bytes, got {}",
                scratch.available()
            );
        }
        let shared = luts.len() > 1 && luts.iter().all(|lut| lut.general_series().is_some());
        let boot_layout = bootstrap_layouts(&res[0], a).0.glwe_layout;
        let message_slots = plan.message_slots().unwrap_or(a.slots());
        scratch.scope(|scratch| {
            if message_slots.is_real() && plan.real_part_on_real_slots() {
                let (mut r0, mut scratch) = scratch.take_ckks_ciphertext_scratch(a, a.meta());
                ckks_bootstrap_real_part(module, &mut r0, a, keys, &mut scratch)?;
                r0.set_log_delta(log_delta);
                return eval_lut_batch(module, res, &mut r0, plan, luts, keys, tensor_key, shared, &mut scratch);
            }

            let (mut r0, scratch) = scratch.take_ckks_ciphertext_scratch(a, a.meta());
            let (mut i0, scratch) = scratch.take_ckks_ciphertext_scratch(a, a.meta());
            let (mut acc, mut scratch) = scratch.take_ckks_ciphertext_scratch(&boot_layout, a.meta());
            ckks_split_of(module, &mut r0, &mut i0, a, keys, &mut scratch)?;
            r0.set_log_delta(log_delta);
            i0.set_log_delta(log_delta);

            eval_lut_batch(module, res, &mut r0, plan, luts, keys, tensor_key, shared, &mut scratch)?;

            // The accumulator takes each LUT's imaginary half and folds it into its output before the next LUT, which
            // keeps the carved working set independent of the batch size. Multiplication produces its result at the
            // destination's requested `k`, so the accumulator is restored to the full width before each reuse;
            // otherwise the first LUT's consumption would clamp every later one.
            let acc_meta = acc.meta();
            if shared {
                let basis = ckks_lut_power_basis(
                    module,
                    &mut i0,
                    &boot_layout,
                    eval_mod_standard(OP, plan)?,
                    luts,
                    tensor_key,
                    &mut scratch,
                )?;
                for (ct_out, lut) in res.iter_mut().zip(luts) {
                    acc.set_meta(acc_meta);
                    acc.set_k(boot_layout.k);
                    ckks_eval_lut_from_basis(module, &mut acc, lut, &basis, keys, tensor_key, &mut scratch)?;
                    recombine_halves(module, ct_out, &mut acc, &mut scratch)?;
                }
            } else {
                for (ct_out, lut) in res.iter_mut().zip(luts) {
                    acc.set_meta(acc_meta);
                    acc.set_k(boot_layout.k);
                    ckks_eval_encoded_lut(module, &mut acc, &i0, plan, lut, keys, tensor_key, &mut scratch)?;
                    recombine_halves(module, ct_out, &mut acc, &mut scratch)?;
                }
            }
            Result::Ok(())
        })?;
        for (out, lut) in res.iter_mut().zip(luts) {
            if let Some((meta, k)) = plan.eval_lut_output(OP, lut, a_k)? {
                ckks_ensure!(
                    out.k().as_usize() >= k,
                    "{OP} produced k={}, below required {k}",
                    out.k().as_usize()
                );
                out.set_meta(meta);
                out.set_k(k.into());
            }
        }
        Ok(())
    }
}

impl<BE: CKKSCITwin> CKKSEvalLutReference<BE> for Module<BE> {}

/// What `lut` takes of an input of `log_budget` bits of budget at scale `log_delta`, the guard bits gone: the tensor key
/// of its widest product and the budget of its evaluation, so that neither refuses past a write.
fn check_lut_capacity<BE, T>(
    op: &'static str,
    log_budget: usize,
    log_delta: usize,
    eval_mod: &EvalModPlan,
    lut: &EncodedLut<CKKSPlaintextOwned<BE>>,
    tensor_key: &T,
) -> Result<()>
where
    BE: Backend,
    T: GetTensorKey<BE>,
{
    let (widest, left) = if lut.requires_eval_mod() {
        (
            eval_mod.relinearization_k(log_budget),
            checked_log_budget_sub(op, log_budget, eval_mod.consumed_bits())?,
        )
    } else {
        (log_budget + log_delta, log_budget)
    };
    tensor_key_at::<BE, _>(op, tensor_key, widest.into())?;
    checked_log_budget_sub(op, left, lut.consumed_bits(log_delta))?;
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "the LUT evaluation reads the rotation keys and the tensor key of the plan's ring"
)]
fn ckks_eval_encoded_lut<BE, F, P, K, T, C, R>(
    module: &Module<BE>,
    ct_out: &mut R,
    ct_in: &C,
    plan: &P,
    lut: &EncodedLut<CKKSPlaintextOwned<BE>>,
    keys: &K,
    tensor_key: &T,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSCITwin,
    P: CKKSEvalModInfos<BE, F>,
    Module<BE>: CKKSConjugateOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSEvalModOps<BE>
        + CKKSPolynomialEvaluationOps<BE>
        + CKKSComplexPolynomialEvaluationOps<BE>
        + CKKSMulOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSNegOps<BE>
        + GLWECopy<BE>
        + GLWENormalize<BE>,
    K: BootstrappingLinearKeys<BE>,
    T: GetTensorKey<BE>,
    C: GLWEToBackendRef<BE> + CKKSCtBounds,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    match lut.kind() {
        EncodedLutKind::General(series) => ckks_eval_lut(
            module,
            ct_out,
            ct_in,
            eval_mod_standard(EVAL_LUT_OP, plan)?,
            series,
            keys,
            tensor_key,
            scratch,
        )?,
        EncodedLutKind::Binary { cos, chain } => {
            ckks_eval_lut_binary(module, ct_out, ct_in, cos, chain.as_ref(), tensor_key, scratch)?
        }
    }
    Ok(())
}

/// Evaluates every LUT of a batch against one transformed half, writing into
/// `outs`. Equal-arity general LUTs share one power basis; a batch containing a
/// binary LUT falls back to evaluating each LUT against the shared input.
#[allow(clippy::too_many_arguments)]
fn eval_lut_batch<BE, F, P, K, T, C, R>(
    module: &Module<BE>,
    outs: &mut [R],
    ct: &mut C,
    plan: &P,
    luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
    keys: &K,
    tensor_key: &T,
    shared: bool,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSCITwin,
    P: CKKSEvalModInfos<BE, F>,
    Module<BE>: CKKSConjugateOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSEvalModOps<BE>
        + CKKSPolynomialEvaluationOps<BE>
        + CKKSComplexPolynomialEvaluationOps<BE>
        + CKKSCopyOps<BE>
        + CKKSMulOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSNegOps<BE>
        + GLWECopy<BE>
        + GLWENormalize<BE>,
    K: BootstrappingLinearKeys<BE>,
    T: GetTensorKey<BE>,
    C: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    if shared {
        // One EvalMod reads the input, which leaves it to the basis.
        let layout = GLWELayout {
            n: outs[0].n(),
            base2k: outs[0].base2k(),
            k: outs[0].k(),
            rank: outs[0].rank(),
        };
        let basis = ckks_lut_power_basis(
            module,
            ct,
            &layout,
            eval_mod_standard(EVAL_LUT_OP, plan)?,
            luts,
            tensor_key,
            scratch,
        )?;
        for (out, lut) in outs.iter_mut().zip(luts) {
            ckks_eval_lut_from_basis(module, out, lut, &basis, keys, tensor_key, scratch)?;
        }
        return Ok(());
    }
    for (out, lut) in outs.iter_mut().zip(luts) {
        ckks_eval_encoded_lut(module, out, &*ct, plan, lut, keys, tensor_key, scratch)?;
    }
    Ok(())
}
