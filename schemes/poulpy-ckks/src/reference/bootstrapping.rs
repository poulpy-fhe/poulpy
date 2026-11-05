use crate::{
    CKKSCompositionError, CKKSResult as Result, checked_log_budget_sub, ckks_ensure, ckks_ensure_operand,
    layouts::eval_mod::EvalModPlan,
    reference::{linear_transformation::PlaintextDiagonalKeys, mul::tensor_key_at},
};
use poulpy_core::layouts::GLWEPlaintextInfos;
use poulpy_core::{
    GLWECIEmbed, GLWECITrace, GLWECopy, GLWEKeyswitch, GLWERotate, GLWEShift, GLWESwitchDegree, LinearTransformationPrepared,
    layouts::{
        BSGSMeta, GGLWEInfos, GLWEInfos, GLWELayout, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef,
        GetAutomorphismKey, GetTensorKey, LWEInfos, LinearTransformation, LinearTransformationPlan, Rank, SetBSGSMeta, SetK,
        TorusPrecision,
        prepared::{GGLWEPreparedBackendRef, GGLWEPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
    },
};
use poulpy_hal::{
    execution::{TaskExecutor, worker_scratch_bytes},
    layouts::{Backend, Module, ScratchArena},
};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotsKind,
    api::{
        CKKSAddOps, CKKSAllOpsTmpBytes, CKKSComplexPolynomialEvaluationOps, CKKSConjugateOps, CKKSCopyOps, CKKSEvalModOps,
        CKKSImagOps, CKKSLinearTransformationOps, CKKSMulOps, CKKSNegOps, CKKSPolynomialEvaluationOps, CKKSPow2Ops, CKKSSubOps,
    },
    eval_lut::{ckks_eval_lut, ckks_eval_lut_binary, ckks_eval_lut_from_basis, ckks_lut_power_basis},
    layouts::{
        BootstrappingKeys, BootstrappingKeysLayout, BootstrappingLinearKeys, CKKSCITwin, CKKSCiphertextOwned,
        CKKSCiphertextViewRef, CKKSEvalModInfos, CKKSModRaiseInfos, CKKSPlaintextOwned, CKKSRightOperandPrepared, EncodedLut,
        EncodedLutKind, EvalMod, EvalModCIPrepared, ScratchArenaTakeCKKSCiphertext,
        bootstrapping::bootstrap_layouts,
        validation::{
            check_conversion_key_widths, check_eval_mod_on_ci, check_first_factor_raise, check_offset_raise, ensure_ci_module,
            ensure_encapsulation, ensure_eval_round_interval, ensure_lut_batch, ensure_mod_up_widens, ensure_one_base2k,
            ensure_one_degree, ensure_rank_one, eval_mod_standard, offset_key_window, remove_guard_bits, required_tensor_key,
        },
    },
    oep::{CKKSEncapsulatedModUpImpl, CKKSModUpFirstFactorImpl, derived::dft::ckks_split_real_imag},
};
use poulpy_core::{
    GLWEBytesOf, GLWELinearTransformationBabyStepsBytesOf, GLWENormalize, ScratchArenaTakeGLWE,
    ScratchArenaTakeGLWELinearTransformationBabySteps,
};

/// The name of [`ckks_bootstrap_eval_mod`](crate::api::CKKSBootstrappingOps::ckks_bootstrap_eval_mod) in its refusals.
const EVAL_MOD_OP: &str = "ckks_bootstrap_eval_mod";

/// The name of [`ckks_eval_lut`](crate::api::CKKSBootstrappingOps::ckks_eval_lut) in its refusals.
const EVAL_LUT_OP: &str = "ckks_eval_lut";

/// The proxy for the plaintext ops: EvalMod's encoded polynomial coefficients, the
/// widest plaintext the pipeline multiplies by.
fn eval_mod_coeffs_layout<C1, C2>(ct_out: &C1, ct_in: &C2, plan: &EvalModPlan) -> CKKSLayout
where
    C1: CKKSCtBounds,
    C2: CKKSCtBounds,
{
    let coeffs_meta = plan.coeffs_meta;
    CKKSLayout {
        glwe_layout: GLWELayout {
            n: ct_out.n(),
            base2k: ct_in.base2k(),
            k: coeffs_meta.k,
            rank: Rank(1),
        },
        meta: coeffs_meta.meta,
    }
}

/// Scratch of [`eval_mod_real`] and [`eval_mod_complex`], the larger of the two: the
/// imaginary half and output, then the conjugation and the split, or EvalMod, on two
/// worker arenas on a parallel executor.
fn eval_mod_standard_tmp_bytes<BE, R, A, P, F>(
    module: &Module<BE>,
    res: &R,
    a: &A,
    plan: &P,
    keys_layout: &BootstrappingKeysLayout,
) -> usize
where
    BE: CKKSCITwin,
    Module<BE>: GLWEBytesOf<BE> + CKKSAllOpsTmpBytes<BE> + CKKSEvalModOps<BE>,
    R: CKKSCtBounds,
    A: CKKSCtBounds,
    P: CKKSEvalModInfos<BE, F>,
{
    let (boot_layout, a_layout) = bootstrap_layouts(res, a);
    let coeffs_layout = eval_mod_coeffs_layout(res, a, plan.eval_mod_plan());
    let tsk = &keys_layout.tensor_key;
    // EvalMod writes into a ciphertext of `res`'s width and into `res` itself, which
    // may be allocated wider than its `k`.
    let eval_mod = plan.eval_mod_standard().map_or(0, |eval_mod| {
        module
            .ckks_eval_mod_tmp_bytes(&boot_layout, &a_layout, eval_mod, tsk)
            .max(module.ckks_eval_mod_tmp_bytes(res, &a_layout, eval_mod, tsk))
    });
    let eval_mod = if <BE::TaskExecutor as TaskExecutor>::IS_PARALLEL {
        2 * worker_scratch_bytes::<BE>(eval_mod)
    } else {
        eval_mod
    };
    let mut linear = 0;
    for atk in keys_layout.rotation_key_layouts() {
        linear = linear
            .max(module.ckks_all_ops_with_atk_tmp_bytes(res, &coeffs_layout, tsk, atk))
            .max(module.ckks_all_ops_with_atk_tmp_bytes(&boot_layout, &coeffs_layout, tsk, atk))
            .max(module.ckks_all_ops_with_atk_tmp_bytes(&a_layout, &coeffs_layout, tsk, atk));
    }
    let a_bytes = BE::scratch_aligned(module.glwe_bytes_of_from_infos(&a_layout));
    let boot_bytes = BE::scratch_aligned(module.glwe_bytes_of_from_infos(&boot_layout));
    a_bytes + boot_bytes + eval_mod.max(a_bytes + linear)
}

/// Scratch of the raise of `in_layout` to `boot_layout`: the lift and the shift, then under encapsulation, between
/// two switches, or with the first CoeffsToSlots factor, on a few small ciphertexts as wide as its keys switch.
fn raise_tmp_bytes<BE, P>(
    module: &Module<BE>,
    boot_layout: &CKKSLayout,
    in_layout: &CKKSLayout,
    plan: &P,
    keys_layout: &BootstrappingKeysLayout,
) -> usize
where
    BE: Backend + CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
    Module<BE>: GLWEShift<BE> + CKKSPow2Ops<BE>,
    P: CKKSModRaiseInfos<BE>,
{
    let mut bytes = module.glwe_shift_tmp_bytes(boot_layout.size());
    if plan.mod_raise_plan().lift.is_some() {
        bytes = bytes.max(module.ckks_mul_pow2_tmp_bytes(in_layout.size()));
    }
    if let Some((encaps, sparse_to_dense)) = keys_layout
        .encapsulation
        .as_ref()
        .and_then(|encaps| encaps.sparse_to_dense.as_ref().map(|s2d| (encaps, s2d)))
    {
        bytes = bytes.max(BE::ckks_encapsulated_mod_up_tmp_bytes(
            module,
            boot_layout,
            in_layout,
            &encaps.dense_to_sparse,
            sparse_to_dense,
        ));
    }
    if let (Some(factor), Some(key)) = (plan.first_factor(), &keys_layout.first_factor_key) {
        bytes = bytes.max(BE::ckks_mod_up_first_factor_tmp_bytes(
            module,
            boot_layout,
            in_layout,
            factor,
            keys_layout.encapsulation.as_ref().map(|encaps| &encaps.dense_to_sparse),
            key,
        ));
    }
    bytes
}

/// Backend-generic reference of the CKKS bootstrapping pipeline, selected by
/// [`impl_ckks_bootstrapping_reference_full`](crate::oep::impl_ckks_bootstrapping_reference_full).
pub trait CKKSBootstrappingReference<BE: CKKSCITwin>: std::borrow::Borrow<Module<BE>> {
    /// Scratch upper bound for [`Self::ckks_eval_lut_reference`]: the imaginary half and accumulator beside one
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
        // The imaginary half of `a` and the accumulator, beside one LUT-evaluation temporary.
        let carved = a_bytes + 2 * boot_bytes;
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

    /// Scratch upper bound for [`Self::ckks_mod_raise_reference`].
    fn ckks_mod_raise_tmp_bytes_reference<R, A, P>(
        &self,
        res: &R,
        a: &A,
        plan: &P,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        BE: CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
        Module<BE>: GLWEShift<BE> + CKKSPow2Ops<BE>,
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSModRaiseInfos<BE>,
    {
        let module: &Module<BE> = self.borrow();
        let (boot_layout, in_layout) = bootstrap_layouts(res, a);
        raise_tmp_bytes(module, &boot_layout, &in_layout, plan, keys_layout)
    }

    /// Whole raise step: on C2S-first, lift to the plan's message ratio; (encapsulate) ModUp with the second lift
    /// or the guard bits fused into its shift; the first CoeffsToSlots factor when the plan leaves it to the raise;
    /// then the relabeling of `res`. `a` is consumed.
    fn ckks_mod_raise_reference<R, A, P, K>(
        &self,
        res: &mut R,
        a: &mut A,
        plan: &P,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
        Module<BE>: GLWECopy<BE> + GLWEShift<BE> + GLWEKeyswitch<BE> + CKKSPow2Ops<BE>,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: CKKSModRaiseInfos<BE>,
        K: BootstrappingLinearKeys<BE>,
    {
        let module: &Module<BE> = self.borrow();
        const OP: &str = "ckks_mod_raise";
        let raise = plan.mod_raise_plan();
        ensure_rank_one(OP, a, res)?;
        ensure_one_degree(OP, module.n(), a, res)?;
        ensure_one_base2k(OP, a, res)?;
        ensure_encapsulation(OP, keys, raise.sparse_secret_hamming_weight)?;
        ckks_ensure_operand!(
            OP,
            !raise.first_factor_on_raise || plan.first_factor().is_some(),
            "the plan moves the first CoeffsToSlots factor onto the raise, and no prepared factor comes with it"
        );
        // C2S-first derives both lifts from EvalMod, and the first one writes `a`: the checks run on what it will leave.
        let (lift_bits, scale_up) = match &raise.lift {
            Some(lift) => {
                let lift_bits = lift.input_scale_up(a.k().as_usize(), a.log_delta())?;
                (lift_bits, lift.raised_scale_up(a.log_delta() + lift_bits)?)
            }
            None => (0, raise.c2s_guard_bits),
        };
        check_raise_widths(res, a, scale_up, plan.first_factor(), keys)?;
        if let Some(keys_layout) = keys.keys_layout() {
            let bytes = module.ckks_mod_raise_tmp_bytes_reference(res, a, plan, keys_layout);
            ckks_ensure!(
                scratch.available() >= bytes,
                "{OP} needs a scratch arena of {bytes} bytes, got {}",
                scratch.available()
            );
        }
        let (slots, log_sparsity) = (a.slots(), a.log_sparsity());
        let log_modulus_in = a.k().as_usize();
        mod_raise_body(module, res, a, lift_bits, scale_up, plan.first_factor(), keys, scratch)?;
        let scale = match &raise.lift {
            Some(lift) => res.log_delta() + lift.log_msg_ratio,
            None => log_modulus_in + raise.c2s_guard_bits,
        };
        res.set_meta(GLWEPlaintextMeta {
            scale: Scale::Log(scale),
            slots,
            log_sparsity,
        });
        Ok(())
    }

    /// Scratch upper bound for [`Self::ckks_bootstrap_eval_mod_reference`].
    fn ckks_bootstrap_eval_mod_tmp_bytes_reference<R, A, P, F>(
        &self,
        res: &R,
        a: &A,
        plan: &P,
        ci: Option<&Module<BE::CI>>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        Module<BE>: GLWEBytesOf<BE>
            + GLWELinearTransformationBabyStepsBytesOf<BE>
            + CKKSAllOpsTmpBytes<BE>
            + CKKSEvalModOps<BE>
            + CKKSPow2Ops<BE>
            + CKKSSubOps<BE>
            + CKKSAddOps<BE>
            + CKKSMulOps<BE>
            + CKKSLinearTransformationOps<BE>
            + GLWEKeyswitch<BE>
            + GLWECITrace<BE>
            + GLWERotate<BE>,
        Module<BE::CI>: CKKSEvalModOps<BE::CI>,
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSEvalModInfos<BE, F>,
    {
        let module: &Module<BE> = self.borrow();
        match (ci, plan.eval_mod_ci()) {
            (Some(ci), Some(eval_mod_ci)) => {
                let (boot_layout, _) = bootstrap_layouts(res, a);
                eval_mod_on_ci_tmp_bytes(module, ci, &boot_layout, eval_mod_ci, keys_layout)
            }
            _ => eval_mod_standard_tmp_bytes(module, res, a, plan, keys_layout),
        }
    }

    /// EvalMod of a bootstrap on `a`, the output of CoeffsToSlots, as the plan and the slots of the message say:
    /// one [`ckks_eval_mod`](CKKSEvalModOps::ckks_eval_mod) on the real part, one on each half with their recombination,
    /// or one on the conjugate-invariant ring between the conversions there and back. Under EvalRound+ it returns
    /// `EvalMod(x) - K·x` and keeps the guard bits and its working label, which go with the bypass; otherwise it calls
    /// [`stamp_eval_mod_output`](CKKSEvalModInfos::stamp_eval_mod_output).
    fn ckks_bootstrap_eval_mod_reference<R, A, P, F, K>(
        &self,
        res: &mut R,
        a: &mut A,
        plan: &P,
        ci: Option<&Module<BE::CI>>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        F: Sync,
        Module<BE>: GLWEBytesOf<BE>
            + GLWELinearTransformationBabyStepsBytesOf<BE>
            + CKKSAllOpsTmpBytes<BE>
            + CKKSConjugateOps<BE>
            + CKKSImagOps<BE>
            + CKKSEvalModOps<BE>
            + CKKSPow2Ops<BE>
            + CKKSSubOps<BE>
            + CKKSAddOps<BE>
            + CKKSMulOps<BE>
            + CKKSLinearTransformationOps<BE>
            + GLWEKeyswitch<BE>
            + GLWECITrace<BE>
            + GLWECIEmbed<BE>
            + GLWERotate<BE>
            + GLWESwitchDegree<BE>,
        Module<BE::CI>: CKKSEvalModOps<BE::CI>,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + Send,
        A: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: CKKSEvalModInfos<BE, F>,
        K: BootstrappingKeys<BE> + Sync,
        K::TensorKey: Sync,
    {
        let module: &Module<BE> = self.borrow();
        const OP: &str = EVAL_MOD_OP;
        ensure_ci_module(OP, ci, plan.eval_mod_ci().is_some())?;
        ensure_rank_one(OP, a, res)?;
        ensure_one_degree(OP, module.n(), a, res)?;
        ensure_one_base2k(OP, a, res)?;
        remove_guard_bits(OP, a.log_delta(), plan.c2s_guard_bits())?;
        let (standard, on_ci) = match (ci, plan.eval_mod_ci()) {
            (Some(ci), Some(eval_mod_ci)) => {
                let ci_keys = check_eval_mod_on_ci(OP, module, ci, &*a, keys)?;
                ckks_ensure_operand!(OP, a.k() <= res.k(), "takes an input no wider than its output");
                (None, Some((ci, eval_mod_ci, ci_keys)))
            }
            _ => {
                let eval_mod = eval_mod_standard(OP, plan)?;
                let tensor_key = required_tensor_key(OP, keys.tensor_key())?;
                let widest = plan.eval_mod_plan().relinearization_k(a.log_budget());
                tensor_key_at::<BE, _>(OP, tensor_key, widest.into())?;
                // EvalMod checks this too, on a route that has already written `a`.
                checked_log_budget_sub(OP, a.log_budget(), eval_mod.consumed_bits())?;
                (Some((eval_mod, tensor_key)), None)
            }
        };
        ensure_eval_round_interval(plan)?;
        if let Some(keys_layout) = keys.keys_layout() {
            let bytes = module.ckks_bootstrap_eval_mod_tmp_bytes_reference(res, &*a, plan, ci, keys_layout);
            ckks_ensure!(
                scratch.available() >= bytes,
                "{OP} needs a scratch arena of {bytes} bytes, got {}",
                scratch.available()
            );
        }
        if let Some((eval_mod, tensor_key)) = standard {
            if plan.message_slots().unwrap_or(a.slots()).is_real() && plan.one_eval_mod_on_real_slots() {
                eval_mod_real(module, res, a, eval_mod, tensor_key, keys, scratch)?;
            } else {
                eval_mod_complex(module, res, a, plan, eval_mod, tensor_key, keys, scratch)?;
            }
        } else if let Some((ci, eval_mod_ci, ci_keys)) = on_ci {
            eval_mod_on_ci(module, ci, res, &*a, plan, eval_mod_ci, ci_keys, keys, scratch)?;
        }
        if !plan.eval_round_plus() {
            plan.stamp_eval_mod_output(res)?;
        }
        Ok(())
    }

    /// Backend-generic reference for [`ckks_eval_lut`](crate::api::CKKSBootstrappingOps::ckks_eval_lut): on a real message
    /// of a plan that reads the real part of real slots, that part of `a`; otherwise its split, with the imaginary half
    /// folded into each output as it is produced. The guard bits leave the halves before the exponential, and each output takes
    /// its [`eval_lut_output`](CKKSEvalModInfos::eval_lut_output) at the end.
    fn ckks_eval_lut_reference<R, A, P, F, K>(
        &self,
        res: &mut [R],
        a: &mut A,
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
            + GLWENormalize<BE>,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        A: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: CKKSEvalModInfos<BE, F>,
        K: BootstrappingKeys<BE>,
        CKKSCiphertextOwned<BE>:
            GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        let module: &Module<BE> = self.borrow();
        const OP: &str = EVAL_LUT_OP;
        ensure_lut_batch(OP, module.n(), &*a, res, luts, plan)?;
        let log_delta = remove_guard_bits(OP, a.log_delta(), plan.c2s_guard_bits())?;
        let tensor_key = required_tensor_key(OP, keys.tensor_key())?;
        // Every conjugation of the step (real part, split, LUT symmetrization) switches at a width at most `a.k()`.
        keys.rotation_keys()
            .get_automorphism_key(-1, a.k())
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
            let bytes = module.ckks_eval_lut_tmp_bytes_reference(&res[0], &*a, plan, luts, keys_layout);
            ckks_ensure!(
                scratch.available() >= bytes,
                "{OP} needs a scratch arena of {bytes} bytes, got {}",
                scratch.available()
            );
        }
        let shared = luts.len() > 1 && luts.iter().all(|lut| lut.general_series().is_some());
        let boot_layout = bootstrap_layouts(&res[0], &*a).0.glwe_layout;
        let message_slots = plan.message_slots().unwrap_or(a.slots());
        scratch.scope(|scratch| {
            if message_slots.is_real() && plan.real_part_on_real_slots() {
                let (mut conjugate, mut scratch) = scratch.take_ckks_ciphertext_scratch(&*a, a.meta());
                ckks_bootstrap_real_part(module, a, &mut conjugate, keys, &mut scratch)?;
                a.set_log_delta(log_delta);
                return eval_lut_batch(module, res, &*a, plan, luts, keys, tensor_key, shared, &mut scratch);
            }

            let (mut i0, scratch) = scratch.take_ckks_ciphertext_scratch(&*a, a.meta());
            let (mut acc, mut scratch) = scratch.take_ckks_ciphertext_scratch(&boot_layout, a.meta());
            ckks_split_real_imag(module, a, &mut i0, keys.rotation_keys(), &mut scratch)?;
            a.set_log_delta(log_delta);
            i0.set_log_delta(log_delta);

            eval_lut_batch(module, res, &*a, plan, luts, keys, tensor_key, shared, &mut scratch)?;

            // The accumulator takes each LUT's imaginary half and folds it into its output before the next LUT, which
            // keeps the carved working set independent of the batch size. Multiplication produces its result at the
            // destination's requested `k`, so the accumulator is restored to the full width before each reuse;
            // otherwise the first LUT's consumption would clamp every later one.
            let acc_meta = acc.meta();
            if shared {
                let basis = ckks_lut_power_basis(
                    module,
                    &i0,
                    &boot_layout,
                    eval_mod_standard(OP, plan)?,
                    luts,
                    tensor_key,
                    &mut scratch,
                )?;
                for (ct_out, lut) in res.iter_mut().zip(luts) {
                    acc.set_meta(acc_meta);
                    acc.set_k(boot_layout.k);
                    ckks_eval_lut_from_basis(module, &mut acc, lut, &basis, keys.rotation_keys(), tensor_key, &mut scratch)?;
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

impl<BE: CKKSCITwin> CKKSBootstrappingReference<BE> for Module<BE> {}

/// `scale_up` lifts the message that many bits above its natural magnitude,
/// fused into the widening shift rather than costing a second pass.
fn ckks_mod_up_scaled_reference<BE, Dst, Src>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &Src,
    scale_up: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSEncapsulatedModUpImpl,
    Module<BE>: GLWECopy<BE> + GLWEShift<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSInfos,
{
    // `dst` is the (freshly-allocated) widened target: ModUp raises the modulus
    // to the destination's requested modulus, so the widening width is `dst.k()`.
    // `dst.k()` is meta-derived and is `0` on a fresh ciphertext, which would
    // spuriously fail the "must widen" check below.
    let k_large: usize = dst.k().as_usize();
    let k_small: usize = src.k().as_usize();

    ensure_mod_up_widens("ckks_mod_raise", k_large, k_small, scale_up)?;

    // MSB-align `src` into the wider `dst`: the value occupies the top
    // `k_small` bits and the freshly-introduced low-order limbs are zero.
    module.glwe_copy(dst, src, scratch);

    // Shift the digits down to their natural integer magnitude. This is the
    // modulus raise: the raised-from modulus `q = 2^k_small` becomes an
    // explicit, un-reduced multiple `I(X)·q` living in the `[0, 2^k_large)`
    // window, which EvalMod subsequently removes. (A zero shift, when the
    // input already fills the storage, is a no-op.) Stopping `scale_up` bits
    // short of the natural magnitude is the fused lift: the value lands
    // multiplied by `2^scale_up`, absorbed by the matching `log_delta`.
    module.glwe_rsh_assign(k_large - k_small - scale_up, dst, scratch);

    // `log_delta` picks up the fused lift, so the encoded value is unchanged;
    // the remaining headroom now spans the full raised modulus, so the torus
    // width `k` becomes `k_large` and `log_budget = k_large - log_delta`.
    dst.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(src.log_delta() + scale_up),
        slots: src.slots(),
        log_sparsity: src.log_sparsity(),
    });
    dst.set_k(k_large.into());

    Ok(())
}

/// What [`CKKSBootstrappingReference::ckks_mod_raise_reference`] refuses on widths: `src`, lifted by `scale_up` fused
/// into ModUp, into `dst`, under the encapsulation keys and the first-factor keys of `keys`.
fn check_raise_widths<BE, Dst, Src, K>(
    dst: &Dst,
    src: &Src,
    scale_up: usize,
    first_factor: Option<&LinearTransformation<CKKSRightOperandPrepared<BE>>>,
    keys: &K,
) -> Result<()>
where
    BE: Backend,
    K: BootstrappingLinearKeys<BE>,
    Dst: CKKSCtBounds,
    Src: CKKSCtBounds,
{
    const OP: &str = "ckks_mod_raise";
    let (k_large, k_small) = (dst.k().as_usize(), src.k().as_usize());
    if let Some(factor) = first_factor {
        if let Some((dense_to_sparse, _)) = keys.encapsulation_keys() {
            ensure_encapsulation_key_covers(src.k(), dense_to_sparse)?;
        }
        ensure_mod_up_widens(OP, k_large, k_small, scale_up)?;
        check_first_factor_raise(
            OP,
            dst.base2k().as_usize(),
            dst.n().as_usize().ilog2() as usize,
            k_large,
            k_small + scale_up,
            factor,
            keys,
        )?;
        return Ok(());
    }
    match keys.encapsulation_keys() {
        Some((dense_to_sparse, Some(sparse_to_dense))) => {
            check_encapsulated_raise(dst, src, scale_up, dense_to_sparse, sparse_to_dense)
        }
        Some((_, None)) => Err(CKKSCompositionError::InvalidOperand {
            op: OP,
            reason:
                "the encapsulated ModUp needs a sparse-to-dense key, which a key set for the first factor on the raise has not"
                    .into(),
        }
        .into()),
        None => ensure_mod_up_widens(OP, k_large, k_small, scale_up),
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the raise takes its lift, its scale and the first factor as separate operands"
)]
fn mod_raise_body<BE, Dst, Src, K>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &mut Src,
    lift_bits: usize,
    scale_up: usize,
    first_factor: Option<&LinearTransformation<CKKSRightOperandPrepared<BE>>>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
    Module<BE>: GLWECopy<BE> + GLWEShift<BE> + GLWEKeyswitch<BE> + CKKSPow2Ops<BE>,
    K: BootstrappingLinearKeys<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    if lift_bits != 0 {
        let log_delta: usize = src.log_delta();
        module.ckks_mul_pow2_assign(lift_bits, src, scratch)?;
        // Not `set_log_delta`: it preserves `log_budget` and grows `k`, which
        // cancels the shift. The lift spends budget at a fixed modulus.
        let mut meta = src.meta();
        meta.scale = Scale::Log(log_delta + lift_bits);
        src.set_meta(meta);
    }

    // A first CoeffsToSlots factor left to the raise is applied here, on the
    // raised ciphertext while it is small; its switches replace sparse-to-dense.
    if let Some(factor) = first_factor {
        return BE::ckks_mod_up_first_factor(module, dst, src, scale_up, factor, keys, scratch);
    }

    match keys.encapsulation_keys() {
        Some((dense_to_sparse, Some(sparse_to_dense))) => BE::ckks_encapsulated_mod_up(
            module,
            dst,
            src,
            scale_up,
            &dense_to_sparse.to_backend_ref(),
            &sparse_to_dense.to_backend_ref(),
            scratch,
        ),
        _ => ckks_mod_up_scaled_reference(module, dst, src, scale_up, scratch),
    }
}

fn recombine_halves<BE, R1, R2>(
    module: &Module<BE>,
    res_re: &mut R1,
    res_im: &mut R2,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSImagOps<BE> + CKKSAddOps<BE>,
    R1: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    R2: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    module.ckks_mul_i_assign(res_im, scratch)?;
    module.ckks_add_assign(res_re, &*res_im, scratch)?;
    Ok(())
}

/// `ct = z + conj(z) = 2·Re(z)` from `z` in `ct`: the input polynomial's coefficients.
fn ckks_bootstrap_real_part<BE, K, R1, R2>(
    module: &Module<BE>,
    ct: &mut R1,
    conjugate: &mut R2,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSConjugateOps<BE> + CKKSAddOps<BE>,
    K: BootstrappingLinearKeys<BE>,
    R1: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    R2: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    module.ckks_conjugate_into(conjugate, &*ct, keys.rotation_keys(), scratch)?;
    module.ckks_add_assign(ct, &*conjugate, scratch)?;
    ct.set_slots(SlotsKind::Real);
    Ok(())
}

/// `eval_mod` on `r0` into `res_real` and on `i0` into `res_imag`, on a parallel executor at once.
#[allow(clippy::too_many_arguments)]
fn ckks_bootstrap_eval_mod_halves<BE, F, T, C, R1, R2>(
    module: &Module<BE>,
    r0: &C,
    i0: &C,
    res_real: &mut R1,
    res_imag: &mut R2,
    eval_mod: &EvalMod<F, CKKSPlaintextOwned<BE>>,
    tensor_key: &T,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSEvalModOps<BE>,
    F: Sync,
    T: GetTensorKey<BE> + Sync,
    C: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
    R1: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + Send,
    R2: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + Send,
{
    if !<BE::TaskExecutor as TaskExecutor>::IS_PARALLEL {
        module.ckks_eval_mod(res_real, r0, eval_mod, tensor_key, scratch)?;
        module.ckks_eval_mod(res_imag, i0, eval_mod, tensor_key, scratch)?;
        return Ok(());
    }

    // The two halves share what the layout-sized query reserved for them
    // (`ckks_bootstrap_eval_mod_tmp_bytes`): two worker arenas of EvalMod's scratch.
    let task_bytes = scratch.available() / 2 / BE::SCRATCH_ALIGN * BE::SCRATCH_ALIGN;
    let (arenas, _) = scratch.borrow().split(2, task_bytes);
    let mut arenas = arenas.into_iter();
    let mut scratch_real = arenas.next().unwrap();
    let mut scratch_imag = arenas.next().unwrap();
    let (real, imag) = <BE::TaskExecutor as TaskExecutor>::join(
        || module.ckks_eval_mod(res_real, r0, eval_mod, tensor_key, &mut scratch_real),
        || module.ckks_eval_mod(res_imag, i0, eval_mod, tensor_key, &mut scratch_imag),
    );
    real?;
    imag?;
    Ok(())
}

/// One EvalMod on the real part of `a`, in place, on the real slots of an S2C-first plan without a bypass.
fn eval_mod_real<BE, F, R, A, T, K>(
    module: &Module<BE>,
    res: &mut R,
    a: &mut A,
    eval_mod: &EvalMod<F, CKKSPlaintextOwned<BE>>,
    tensor_key: &T,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSCITwin,
    Module<BE>: CKKSConjugateOps<BE> + CKKSAddOps<BE> + CKKSEvalModOps<BE>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    A: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    T: GetTensorKey<BE>,
    K: BootstrappingKeys<BE>,
{
    let (mut conjugate, mut scratch) = scratch.borrow().take_ckks_ciphertext_scratch(&*a, a.meta());
    ckks_bootstrap_real_part(module, a, &mut conjugate, keys, &mut scratch)?;
    module.ckks_eval_mod(res, &*a, eval_mod, tensor_key, &mut scratch)
}

/// The split of `a`, in place, one EvalMod per half and their recombination. Under EvalRound+ each
/// half also takes `-K·x`, on the very input `x` of its EvalMod, so that the error of the
/// low-precision CoeffsToSlots cancels against the bypass the caller adds back.
#[expect(
    clippy::too_many_arguments,
    reason = "the route reads the plan beside the EvalMod and the tensor key its entry resolved"
)]
fn eval_mod_complex<BE, F, R, A, P, T, K>(
    module: &Module<BE>,
    res: &mut R,
    a: &mut A,
    plan: &P,
    eval_mod: &EvalMod<F, CKKSPlaintextOwned<BE>>,
    tensor_key: &T,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSCITwin,
    F: Sync,
    Module<BE>: CKKSConjugateOps<BE> + CKKSAddOps<BE> + CKKSSubOps<BE> + CKKSImagOps<BE> + CKKSPow2Ops<BE> + CKKSEvalModOps<BE>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + Send,
    A: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    P: CKKSEvalModInfos<BE, F>,
    T: GetTensorKey<BE> + Sync,
    K: BootstrappingKeys<BE>,
{
    let (mut i0, scratch) = scratch.borrow().take_ckks_ciphertext_scratch(&*a, a.meta());
    let (mut imag, mut scratch) = scratch.take_ckks_ciphertext_scratch(res, res.meta());
    ckks_split_real_imag(module, a, &mut i0, keys.rotation_keys(), &mut scratch)?;
    {
        let (r0_ref, i0_ref) = (CKKSCiphertextViewRef::of(&*a), i0.to_backend_view_ref());
        ckks_bootstrap_eval_mod_halves(module, &r0_ref, &i0_ref, res, &mut imag, eval_mod, tensor_key, &mut scratch)?;
    }
    if plan.eval_round_plus() {
        let log2_k = plan.eval_mod_plan().f_mod_interval.trailing_zeros() as usize;
        module.ckks_mul_pow2_assign(log2_k, a, &mut scratch)?;
        module.ckks_mul_pow2_assign(log2_k, &mut i0, &mut scratch)?;
        module.ckks_sub_assign(res, &*a, &mut scratch)?;
        module.ckks_sub_assign(&mut imag, &i0, &mut scratch)?;
    }
    recombine_halves(module, res, &mut imag, &mut scratch)
}

/// Scratch of [`eval_mod_on_ci`] at the width of `boot_layout`, on a layout with conversion keys:
/// the degree-`2N` ciphertext beside the conversions, the traced and the reduced ciphertext
/// beside EvalMod.
fn eval_mod_on_ci_tmp_bytes<BE, F>(
    module: &Module<BE>,
    ci: &Module<BE::CI>,
    boot_layout: &CKKSLayout,
    eval_mod_ci: &EvalModCIPrepared<BE, F>,
    keys_layout: &BootstrappingKeysLayout,
) -> usize
where
    BE: CKKSCITwin,
    Module<BE>: GLWEBytesOf<BE>
        + GLWELinearTransformationBabyStepsBytesOf<BE>
        + CKKSPow2Ops<BE>
        + GLWEKeyswitch<BE>
        + GLWECITrace<BE>
        + CKKSMulOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSLinearTransformationOps<BE>
        + GLWERotate<BE>,
    Module<BE::CI>: CKKSEvalModOps<BE::CI>,
{
    // A layout without conversions fails at the bootstrap.
    let Some((trace_key, embed_key)) = &keys_layout.conversion_keys else {
        return 0;
    };
    let embedded = CKKSLayout {
        glwe_layout: GLWELayout {
            n: (2 * boot_layout.n().as_usize()).into(),
            ..boot_layout.glwe_layout
        },
        ..*boot_layout
    };
    let embedded_bytes = BE::scratch_aligned(module.glwe_bytes_of_from_infos(&embedded));
    let trace = embedded_bytes
        + module
            .glwe_keyswitch_tmp_bytes(&embedded, &embedded, trace_key)
            .max(module.glwe_ci_trace_tmp_bytes(boot_layout, &embedded));
    let pre = match &eval_mod_ci.pre.0 {
        crate::layouts::bootstrapping::CIPreForm::Product(u) => module.ckks_mul_pt_vec_tmp_bytes(&embedded, &embedded, u),
        // The baby steps of `even` and the outputs of the two folded factors beside
        // their evaluation, then the odd one's embedding beside its rotation.
        crate::layouts::bootstrapping::CIPreForm::Folded { plans, .. } => {
            // The baby steps of `even`, which `odd` shares, and each factor on its own schedule,
            // which counts the doubling and the giant-step tree of power-of-two keys.
            let [even_plan, odd_plan] = &**plans;
            let diagonal = crate::api::ckks_plaintext_diagonal_layout(boot_layout);
            let mut factors = 0;
            for atk in keys_layout.rotation_key_layouts() {
                factors = factors
                    .max(module.ckks_linear_transformation_baby_steps_prepare_tmp_bytes(boot_layout, even_plan, atk))
                    .max(module.ckks_linear_transformation_eval_tmp_bytes(boot_layout, &diagonal, even_plan, atk, atk))
                    .max(module.ckks_linear_transformation_eval_tmp_bytes(boot_layout, &diagonal, odd_plan, atk, atk));
            }
            module.glwe_linear_transformation_baby_steps_bytes_of(even_plan, boot_layout)
                + 2 * BE::scratch_aligned(module.glwe_bytes_of_from_infos(boot_layout))
                + factors.max(embedded_bytes + module.glwe_rotate_tmp_bytes().max(module.ckks_add_tmp_bytes(embedded.size())))
        }
    };
    let conversions = pre
        .max(module.ckks_mul_pt_vec_tmp_bytes(&embedded, &embedded, &eval_mod_ci.post))
        .max(module.glwe_keyswitch_tmp_bytes(&embedded, &embedded, embed_key))
        .max(module.ckks_mul_pow2_tmp_bytes(boot_layout.size()))
        .max(module.ckks_sub_tmp_bytes(boot_layout.size()))
        .max(trace);
    let eval_mod = ci.ckks_eval_mod_tmp_bytes(boot_layout, boot_layout, &eval_mod_ci.eval_mod, &keys_layout.tensor_key);
    let ct_bytes = BE::scratch_aligned(module.glwe_bytes_of_from_infos(boot_layout));
    embedded_bytes + 2 * ct_bytes + conversions.max(eval_mod)
}

/// `u·z(X²)` into `embedded`, of degree `2N`, from the input `w` of the last
/// CoeffsToSlots factor folded with `u` ([`CIPrePrepared::Folded`]): `a(X²) + X·b(X²)`,
/// with `a` and `b` the even and odd factors on `w`, which share its baby steps.
fn ci_folded_pre<BE, R, W, H>(
    module: &Module<BE>,
    embedded: &mut R,
    w: &W,
    [even, odd]: [&LinearTransformationPrepared<BE>; 2],
    plan: &LinearTransformationPlan,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSLinearTransformationOps<BE> + GLWERotate<BE> + CKKSAddOps<BE> + GLWESwitchDegree<BE>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    W: GLWEToBackendRef<BE> + CKKSCtBounds,
    H: GetAutomorphismKey<BE>,
{
    let (mut babies, mut scratch) = scratch
        .borrow()
        .take_glwe_linear_transformation_baby_steps_scratch(module, plan, w);
    module.ckks_linear_transformation_baby_steps_prepare(&mut babies, w, keys, &mut scratch)?;
    let (mut a, scratch_1) = scratch.borrow().take_ckks_ciphertext_scratch(w, w.meta());
    let (mut b, mut scratch_1) = scratch_1.take_ckks_ciphertext_scratch(w, w.meta());
    let lt_keys = PlaintextDiagonalKeys(keys);
    module.ckks_linear_transformation_eval_into(&mut a, w, &babies, even, &lt_keys, &mut scratch_1)?;
    module.ckks_linear_transformation_eval_into(&mut b, w, &babies, odd, &lt_keys, &mut scratch_1)?;
    let (mut odd_wide, mut scratch_2) = scratch_1.take_ckks_ciphertext_scratch(embedded, b.meta());
    odd_wide.set_k(b.k());
    module.glwe_switch_degree(&mut odd_wide, &b);
    // `X·b(X²)` fills the odd coefficients, which `a(X²)` leaves at zero.
    module.glwe_rotate_assign(1, &mut odd_wide, &mut scratch_2);
    embedded.set_meta(a.meta());
    embedded.set_k(a.k());
    module.glwe_switch_degree(embedded, &a);
    module.ckks_add_assign(embedded, &odd_wide, &mut scratch_2)?;
    Ok(())
}

/// One EvalMod for all `N` coefficients, on the conjugate-invariant ring: `a` to degree `2N`, its
/// product by `u`, the trace onto the CI ring, EvalMod on `ci`, and back by the embedding and `v`.
/// The two rings share their buffers, so `module` and `ci` work on one arena. Under EvalRound+ the
/// result is `EvalMod(x) - K·x`.
#[expect(
    clippy::too_many_arguments,
    reason = "the route reads a module per ring and the CI part of the plan"
)]
fn eval_mod_on_ci<BE, F, R, A, P, K>(
    module: &Module<BE>,
    ci: &Module<BE::CI>,
    res: &mut R,
    a: &A,
    plan: &P,
    eval_mod_ci: &EvalModCIPrepared<BE, F>,
    (tensor_key, trace_key, embed_key): (&K::CITensorKey, &K::TraceKey, &K::EmbedKey),
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSCITwin,
    Module<BE>: GLWEKeyswitch<BE>
        + GLWECITrace<BE>
        + GLWECIEmbed<BE>
        + GLWERotate<BE>
        + GLWESwitchDegree<BE>
        + CKKSSubOps<BE>
        + CKKSLinearTransformationOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSMulOps<BE>
        + CKKSAddOps<BE>,
    Module<BE::CI>: CKKSEvalModOps<BE::CI>,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: CKKSEvalModInfos<BE, F>,
    K: BootstrappingKeys<BE>,
{
    let boot_layout = bootstrap_layouts(&*res, a).0.glwe_layout;
    let wide_layout = GLWELayout {
        n: (2 * boot_layout.n.as_usize()).into(),
        ..boot_layout
    };
    let (mut embedded, mut scratch) = scratch.borrow().take_ckks_ciphertext_scratch(&wide_layout, a.meta());

    // `z(X²)` times `u`, traced onto the CI ring without halving: `2·Re(u·z)`, the
    // split EvalMod input.
    match &eval_mod_ci.pre.0 {
        crate::layouts::bootstrapping::CIPreForm::Product(u) => {
            embedded.set_k(a.k());
            module.glwe_switch_degree(&mut embedded, a);
            module.ckks_mul_pt_vec_assign(&mut embedded, u, &mut scratch)?;
        }
        crate::layouts::bootstrapping::CIPreForm::Folded { even, odd, plans } => ci_folded_pre(
            module,
            &mut embedded,
            a,
            [even, odd],
            &plans[0],
            keys.rotation_keys(),
            &mut scratch,
        )?,
    }
    let widths = plan.eval_mod_plan().conversion_widths(embedded.k().as_usize());
    check_conversion_key_widths(EVAL_MOD_OP, trace_key, embed_key, widths)?;
    let meta = GLWEPlaintextMeta {
        slots: SlotsKind::Real,
        ..embedded.meta()
    };
    let (mut traced, scratch) = scratch.take_ckks_ciphertext_scratch(&boot_layout, meta);
    let (reduced, mut scratch) = scratch.take_ckks_ciphertext_scratch(&boot_layout, meta);
    traced.set_k(embedded.k());
    {
        let (mut switched, mut scratch) = scratch.borrow().take_ckks_ciphertext_scratch(&embedded, embedded.meta());
        module.glwe_keyswitch(&mut switched, &embedded, &trace_key.to_backend_ref(), &mut scratch);
        module.glwe_ci_trace(&mut traced, &switched, &mut scratch);
    }

    let traced = traced.rebind::<BE::CI>();
    let mut reduced = reduced.rebind::<BE::CI>();
    reduced.set_k(traced.k());
    ci.ckks_eval_mod(
        &mut reduced,
        &traced,
        &eval_mod_ci.eval_mod,
        tensor_key,
        &mut scratch.borrow().into_backend::<BE::CI>(),
    )?;
    let mut reduced = reduced.rebind::<BE>();
    // EvalRound+: `EvalMod(x) − K·x` on the very input `x` of EvalMod, so that the
    // error of the low-precision CoeffsToSlots cancels; the caller adds the bypass
    // back. Both are coefficient-wise, the same on either ring.
    if plan.eval_round_plus() {
        let mut traced = traced.rebind::<BE>();
        let log2_k = plan.eval_mod_plan().f_mod_interval.trailing_zeros() as usize;
        module.ckks_mul_pow2_assign(log2_k, &mut traced, &mut scratch)?;
        module.ckks_sub_assign(&mut reduced, &traced, &mut scratch)?;
    }

    // Back to degree `2N` and times `v`, which carries the factor 2 of the
    // conjugate pair: `r_j + i·r_{j+N/2}` on the even coefficients, under `s(X²)`.
    embedded.set_meta(GLWEPlaintextMeta {
        slots: SlotsKind::Real,
        ..reduced.meta()
    });
    embedded.set_k(reduced.k());
    module.glwe_ci_embed(&mut embedded, &reduced);
    module.glwe_keyswitch_assign(&mut embedded, &embed_key.to_backend_ref(), &mut scratch);
    module.ckks_mul_pt_vec_assign(&mut embedded, &eval_mod_ci.post, &mut scratch)?;
    res.set_k(embedded.k());
    module.glwe_switch_degree(res, &embedded);
    res.set_meta(embedded.meta());
    Ok(())
}

/// Reference SSE pipeline: switch to the sparse secret before ModUp to bound
/// wrap-around, then restore the dense secret afterward.
#[doc(hidden)]
pub fn ckks_encapsulated_mod_up_reference<BE, Dst, Src>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &mut Src,
    scale_up: usize,
    dense_to_sparse: &GGLWEPreparedBackendRef<'_, BE>,
    sparse_to_dense: &GGLWEPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSEncapsulatedModUpImpl,
    Module<BE>: GLWECopy<BE> + GLWEShift<BE> + GLWEKeyswitch<BE> + CKKSPow2Ops<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    // The switch keeps the width of `src`: the raise's checks run before it.
    check_encapsulated_raise(dst, src, scale_up, dense_to_sparse, sparse_to_dense)?;
    module.glwe_keyswitch_assign(src, &dense_to_sparse.to_backend_ref(), scratch);
    if sparse_to_dense.k_offset().as_usize() > 0 {
        return ckks_mod_up_offset_reference(module, dst, src, scale_up, sparse_to_dense, scratch);
    }
    // The lift is fused into ModUp, so the message is already at its final scale
    // when sparse-to-dense adds its noise.
    ckks_mod_up_scaled_reference(module, dst, src, scale_up, scratch)?;
    module.glwe_keyswitch_assign(dst, &sparse_to_dense.to_backend_ref(), scratch);
    Ok(())
}

/// ModUp and the sparse-to-dense switch as one key switch, with a key whose
/// rows sit `k_offset` bits lower: the switch reads the digits of `src` as
/// balanced integers and lowers them, which is the raise, integer multiple of
/// the modulus included. The raise lowers `src` by `dst.k - src.k - scale_up`
/// bits; a key of a smaller offset takes the rest as a shift of `src` inside
/// its window first, exact since the window holds every bit of it.
fn ckks_mod_up_offset_reference<BE, Dst, Src>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &Src,
    scale_up: usize,
    sparse_to_dense: &GGLWEPreparedBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: GLWECopy<BE> + GLWEShift<BE> + GLWEKeyswitch<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    Src: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
{
    let k_large: usize = dst.k().as_usize();
    let k_small: usize = src.k().as_usize();
    let (shift, window) = check_offset_raise("ckks_mod_raise", k_large, k_small, scale_up, sparse_to_dense)?;
    let k_offset = sparse_to_dense.k_offset().as_usize();
    let key = sparse_to_dense.to_backend_ref();
    if shift == k_offset {
        module.glwe_keyswitch(dst, src, &key, scratch);
    } else {
        let layout = GLWELayout {
            n: src.n(),
            base2k: src.base2k(),
            k: window.into(),
            rank: src.rank(),
        };
        scratch.scope(|scratch_inner| {
            let (mut lowered, mut scratch_inner) = scratch_inner.take_glwe_scratch(&layout);
            module.glwe_copy(&mut lowered, src, &mut scratch_inner);
            module.glwe_rsh_assign(shift - k_offset, &mut lowered, &mut scratch_inner);
            module.glwe_keyswitch(dst, &lowered, &key, &mut scratch_inner);
        });
    }
    dst.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(src.log_delta() + scale_up),
        slots: src.slots(),
        log_sparsity: src.log_sparsity(),
    });
    Ok(())
}

/// `InvalidOperand` unless the encapsulation key `key` takes an input of `k` bits.
fn ensure_encapsulation_key_covers<K: GGLWEInfos>(k: TorusPrecision, key: &K) -> Result<()> {
    ckks_ensure_operand!(
        "ckks_mod_raise",
        k <= key.gglwe_layout().max_input_k(),
        "the input is wider than an encapsulation key's max_input_k"
    );
    Ok(())
}

/// Refuses encapsulation keys too narrow for their inputs before any write: `src`
/// for the dense-to-sparse switch, `dst` for an offset-free sparse-to-dense one, and
/// the raise of `src` by `scale_up` into `dst`, a sparse-to-dense key with a row
/// offset being checked against that raise, by [`check_offset_raise`].
fn check_encapsulated_raise<Dst: CKKSCtBounds, Src: CKKSCtBounds, D2S: GGLWEInfos, S2D: GGLWEInfos>(
    dst: &Dst,
    src: &Src,
    scale_up: usize,
    dense_to_sparse: &D2S,
    sparse_to_dense: &S2D,
) -> Result<()> {
    const OP: &str = "ckks_mod_raise";
    ensure_encapsulation_key_covers(src.k(), dense_to_sparse)?;
    let (k_large, k_small) = (dst.k().as_usize(), src.k().as_usize());
    if sparse_to_dense.k_offset().as_usize() > 0 {
        check_offset_raise(OP, k_large, k_small, scale_up, sparse_to_dense)?;
    } else {
        ensure_encapsulation_key_covers(dst.k(), sparse_to_dense)?;
        ensure_mod_up_widens(OP, k_large, k_small, scale_up)?;
    }
    Ok(())
}

/// Scratch bound for [`ckks_encapsulated_mod_up_reference`].
#[doc(hidden)]
pub fn ckks_encapsulated_mod_up_tmp_bytes_reference<BE, Dst, Src, D2S, S2D>(
    module: &Module<BE>,
    dst_infos: &Dst,
    src_infos: &Src,
    dense_to_sparse_infos: &D2S,
    sparse_to_dense_infos: &S2D,
) -> usize
where
    BE: Backend,
    Module<BE>: GLWEShift<BE> + GLWEKeyswitch<BE>,
    Dst: CKKSCtBounds,
    Src: CKKSCtBounds,
    D2S: GGLWEInfos,
    S2D: GGLWEInfos,
{
    let d2s = module.glwe_keyswitch_tmp_bytes(src_infos, src_infos, dense_to_sparse_infos);
    if sparse_to_dense_infos.k_offset().as_usize() > 0 {
        // The input of the switch, `src` or its shift inside the key's window.
        let window = GLWELayout {
            n: src_infos.n(),
            base2k: src_infos.base2k(),
            k: offset_key_window(dst_infos.max_k().as_usize(), sparse_to_dense_infos)
                .max(src_infos.k().as_usize())
                .into(),
            rank: src_infos.rank(),
        };
        let switch = module.glwe_keyswitch_tmp_bytes(dst_infos, &window, sparse_to_dense_infos);
        let shifted = module.glwe_bytes_of_from_infos(&window) + module.glwe_shift_tmp_bytes(window.size()).max(switch);
        return d2s.max(switch).max(shifted);
    }
    d2s.max(module.glwe_shift_tmp_bytes(dst_infos.size().max(src_infos.size())))
        .max(module.glwe_keyswitch_tmp_bytes(dst_infos, dst_infos, sparse_to_dense_infos))
}

/// Reference bodies of the required [`CKKSEncapsulatedModUpImpl`] methods, which
/// [`impl_ckks_encapsulated_mod_up_reference_full!`](crate::impl_ckks_encapsulated_mod_up_reference_full) forwards to.
pub trait CKKSEncapsulatedModUpReference<BE: Backend> {
    fn ckks_encapsulated_mod_up_tmp_bytes_reference<Dst, Src, D2S, S2D>(
        &self,
        dst_infos: &Dst,
        src_infos: &Src,
        dense_to_sparse_infos: &D2S,
        sparse_to_dense_infos: &S2D,
    ) -> usize
    where
        Module<BE>: GLWEShift<BE> + GLWEKeyswitch<BE>,
        Dst: CKKSCtBounds,
        Src: CKKSCtBounds,
        D2S: GGLWEInfos,
        S2D: GGLWEInfos;

    fn ckks_encapsulated_mod_up_reference<Dst, Src>(
        &self,
        dst: &mut Dst,
        src: &mut Src,
        scale_up: usize,
        dense_to_sparse: &GGLWEPreparedBackendRef<'_, BE>,
        sparse_to_dense: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE> + GLWEShift<BE> + GLWEKeyswitch<BE> + CKKSPow2Ops<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos;
}

impl<BE: Backend + CKKSEncapsulatedModUpImpl> CKKSEncapsulatedModUpReference<BE> for Module<BE> {
    fn ckks_encapsulated_mod_up_tmp_bytes_reference<Dst, Src, D2S, S2D>(
        &self,
        dst_infos: &Dst,
        src_infos: &Src,
        dense_to_sparse_infos: &D2S,
        sparse_to_dense_infos: &S2D,
    ) -> usize
    where
        Module<BE>: GLWEShift<BE> + GLWEKeyswitch<BE>,
        Dst: CKKSCtBounds,
        Src: CKKSCtBounds,
        D2S: GGLWEInfos,
        S2D: GGLWEInfos,
    {
        ckks_encapsulated_mod_up_tmp_bytes_reference(self, dst_infos, src_infos, dense_to_sparse_infos, sparse_to_dense_infos)
    }

    fn ckks_encapsulated_mod_up_reference<Dst, Src>(
        &self,
        dst: &mut Dst,
        src: &mut Src,
        scale_up: usize,
        dense_to_sparse: &GGLWEPreparedBackendRef<'_, BE>,
        sparse_to_dense: &GGLWEPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE> + GLWEShift<BE> + GLWEKeyswitch<BE> + CKKSPow2Ops<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        ckks_encapsulated_mod_up_reference(self, dst, src, scale_up, dense_to_sparse, sparse_to_dense, scratch)
    }
}

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
            keys.rotation_keys(),
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
    ct: &C,
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
        + GLWENormalize<BE>,
    K: BootstrappingLinearKeys<BE>,
    T: GetTensorKey<BE>,
    C: GLWEToBackendRef<BE> + CKKSCtBounds,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    if shared {
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
            ckks_eval_lut_from_basis(module, out, lut, &basis, keys.rotation_keys(), tensor_key, scratch)?;
        }
        return Ok(());
    }
    for (out, lut) in outs.iter_mut().zip(luts) {
        ckks_eval_encoded_lut(module, out, ct, plan, lut, keys, tensor_key, scratch)?;
    }
    Ok(())
}
