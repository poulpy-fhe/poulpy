use crate::{
    CKKSCompositionError, CKKSResult as Result, checked_log_budget_sub, ckks_ensure, ckks_ensure_operand,
    layouts::eval_mod::EvalModPlan,
};
use poulpy_core::layouts::{GLWEPlaintextInfos, SetGLWEPlaintextInfos};
use poulpy_core::{
    GLWECIEmbed, GLWECITrace, GLWECopy, GLWEKeyswitch, GLWERotate, GLWEShift, GLWESwitchDegree, LinearTransformationPrepared,
    layouts::{
        BSGSMeta, GGLWEInfos, GLWEInfos, GLWELayout, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef,
        GetAutomorphismKey, GetTensorKey, LWEInfos, LinearTransformation, Rank, SetBSGSMeta, SetK, TorusPrecision,
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
        CKKSAddOps, CKKSAllOpsTmpBytes, CKKSBootstrappingOps, CKKSComplexPolynomialEvaluationOps, CKKSConjugateOps, CKKSCopyOps,
        CKKSDFTOps, CKKSEvalModOps, CKKSImagOps, CKKSLinearTransformationOps, CKKSMulOps, CKKSNegOps,
        CKKSPolynomialEvaluationOps, CKKSPow2Ops, CKKSSubOps,
    },
    eval_lut::{ckks_eval_lut, ckks_eval_lut_binary, ckks_eval_lut_from_basis, ckks_lut_power_basis},
    layouts::{
        BootstrappingContextPrepared, BootstrappingKeys, BootstrappingKeysLayout, BootstrappingLinearKeys, BootstrappingPipeline,
        CKKSCITwin, CKKSCiphertextOwned, CKKSEvalModInfos, CKKSModRaiseInfos, CKKSPlaintextOwned, CKKSRightOperandPrepared,
        EncodedLut, EncodedLutKind, EvalMod, EvalModCIPrepared, EvalModRing, EvalModType, ScratchArenaTakeCKKSCiphertext,
    },
    oep::{CKKSBootstrappingImpl, CKKSEncapsulatedModUpImpl, CKKSModUpFirstFactorImpl, derived::dft::ckks_split_real_imag},
};
use poulpy_core::{
    GLWEBytesOf, GLWELinearTransformationBabyStepsBytesOf, GLWENormalize, ScratchArenaTakeGLWE,
    ScratchArenaTakeGLWELinearTransformationBabySteps,
};

/// The name of [`CKKSBootstrappingOps::ckks_bootstrap_eval_mod`] in its refusals.
const EVAL_MOD_OP: &str = "ckks_bootstrap_eval_mod";

/// [`CKKSBootstrappingReference::ckks_bootstrap_tmp_bytes_reference`]: the terms
/// of the rotation keys are maximized over their layouts, those of EvalMod come
/// from the tensor key alone. The raised ciphertext and the EvalRound+ bypass sit
/// beside every stage, a copy of the input beside the stages up to the raise.
fn bootstrap_tmp_bytes_for<BE, C1, C2, F>(
    module: &Module<BE>,
    ct_out: &C1,
    ct_in: &C2,
    ctx: &BootstrappingContextPrepared<BE, F>,
    ci: Option<&Module<BE::CI>>,
    keys_layout: &BootstrappingKeysLayout,
) -> usize
where
    BE: Backend + CKKSBootstrappingImpl + CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
    Module<BE>: GLWEBytesOf<BE>,
    Module<BE>:
        CKKSAllOpsTmpBytes<BE> + CKKSDFTOps<BE> + CKKSEvalModOps<BE> + GLWEKeyswitch<BE> + GLWEShift<BE> + CKKSPow2Ops<BE>,
    C1: CKKSCtBounds,
    C2: CKKSCtBounds,
    CKKSCiphertextOwned<BE>: CKKSCtBounds,
{
    let (boot_layout, in_layout) = bootstrap_layouts(ct_out, ct_in);
    let boot_ct_bytes = BE::scratch_aligned(module.glwe_bytes_of_from_infos(&boot_layout));
    let in_ct_bytes = BE::scratch_aligned(module.glwe_bytes_of_from_infos(&in_layout));
    let coeffs_layout = eval_mod_coeffs_layout(ct_out, ct_in, ctx.eval_mod_plan());
    let tsk = &keys_layout.tensor_key;

    // SlotsToCoeffs runs in place on the copy of the input before the raise in the
    // S2C-first pipeline. Otherwise it writes `ct_out` at the end, whose allocation
    // can be wider than its `k`.
    let out_layout = CKKSLayout {
        glwe_layout: GLWELayout {
            k: ct_out.max_k(),
            ..boot_layout.glwe_layout
        },
        meta: boot_layout.meta,
    };
    let (s2c_dst, s2c_src) = match ctx.pipeline() {
        BootstrappingPipeline::C2SFirst => (&out_layout, &boot_layout),
        BootstrappingPipeline::S2CFirst => (&in_layout, &in_layout),
    };
    let mut before_raise = raise_tmp_bytes(module, &boot_layout, &in_layout, ctx, keys_layout);
    let mut after_raise = module.ckks_bootstrap_eval_mod_tmp_bytes(ct_out, &boot_layout, ctx, ci, keys_layout);
    // Every rotation key shape against the widest ciphertext: a narrow key has a
    // wider digit, which can need more scratch than the wide key.
    for atk in &keys_layout.rotation_key_layouts() {
        before_raise = before_raise.max(module.ckks_all_ops_with_atk_tmp_bytes(&in_layout, &coeffs_layout, tsk, atk));
        after_raise = after_raise
            .max(module.ckks_all_ops_with_atk_tmp_bytes(&boot_layout, &coeffs_layout, tsk, atk))
            .max(module.ckks_dft_tmp_bytes(&boot_layout, &boot_layout, ctx.coeffs_to_slots(), atk));
        let s2c = module.ckks_dft_tmp_bytes(s2c_dst, s2c_src, ctx.slots_to_coeffs(), atk);
        match ctx.pipeline() {
            BootstrappingPipeline::C2SFirst => after_raise = after_raise.max(s2c),
            BootstrappingPipeline::S2CFirst => before_raise = before_raise.max(s2c),
        }
        if let Some(bypass) = ctx.coeffs_to_slots_bypass() {
            after_raise = after_raise.max(module.ckks_dft_tmp_bytes(&boot_layout, &boot_layout, bypass, atk));
        }
    }

    let carved = if ctx.coeffs_to_slots_bypass().is_some() {
        2 * boot_ct_bytes
    } else {
        boot_ct_bytes
    };
    carved + after_raise.max(in_ct_bytes + before_raise)
}

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
/// halves and the imaginary output, then the conjugation and the split, or EvalMod,
/// on two worker arenas on a parallel executor.
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
    for atk in &keys_layout.rotation_key_layouts() {
        linear = linear
            .max(module.ckks_all_ops_with_atk_tmp_bytes(&boot_layout, &coeffs_layout, tsk, atk))
            .max(module.ckks_all_ops_with_atk_tmp_bytes(&a_layout, &coeffs_layout, tsk, atk));
    }
    let a_bytes = BE::scratch_aligned(module.glwe_bytes_of_from_infos(&a_layout));
    let boot_bytes = BE::scratch_aligned(module.glwe_bytes_of_from_infos(&boot_layout));
    2 * a_bytes + boot_bytes + eval_mod.max(a_bytes + linear)
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
    BE: Backend + CKKSBootstrappingImpl + CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
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

/// Fails before any work unless every key covers the widest ciphertext the
/// bootstrap switches with it, and with `InvalidOperand` when the working width
/// or the input is narrower than the bits the stages running on it consume.
///
/// The bootstrap runs at `ct_out.k()`, which need not be the width the keys
/// were laid out for. Each stage's key lookup also refuses a key too narrow for
/// its width; this check only moves the failure ahead of the computation. Keys
/// with a row offset (sparse-to-dense, first factor) are laid out for one raise:
/// they are checked against the raise this bootstrap runs. `op` names the
/// bootstrap in the errors.
fn check_key_widths<BE, M, F, K, T, C1, C2>(
    op: &'static str,
    ct_out: &C1,
    ct_in: &C2,
    ctx: &BootstrappingContextPrepared<BE, F>,
    keys: &K,
    tensor_key: &T,
) -> Result<()>
where
    BE: Backend,
    M: Backend,
    K: BootstrappingLinearKeys<BE>,
    T: GetTensorKey<M>,
    C1: CKKSCtBounds,
    C2: CKKSCtBounds,
{
    let log_n = ct_out.n().as_usize().trailing_zeros() as usize;
    let (bootstrap_k, input_k) = (ct_out.k().as_usize(), ct_in.k().as_usize());
    let Some(widths) = ctx.stage_widths().key_widths(log_n, bootstrap_k, input_k, ct_in.log_delta()) else {
        let reason = format!(
            "a bootstrap at {bootstrap_k} bits of an input of {input_k} bits is narrower than the bits its stages consume"
        );
        return Err(CKKSCompositionError::InvalidOperand { op, reason }.into());
    };
    for (p, k) in widths.rotation_keys {
        let k = TorusPrecision(k as u32);
        keys.rotation_keys()
            .get_automorphism_key(p, k)
            .map_err(|source| CKKSCompositionError::MissingAutomorphismKey {
                op,
                rotation: p,
                k: k.into(),
                source,
            })?;
    }
    let k = TorusPrecision(widths.tensor_key as u32);
    tensor_key
        .get_tensor_key(k)
        .map_err(|source| CKKSCompositionError::MissingRelinearizationKey { op, k: k.into(), source })?;
    check_raise_keys(op, ct_out, ct_in, ctx, keys)
}

/// Keys with a row offset are laid out for the raise: fails before any work
/// unless the first-factor keys, or a sparse-to-dense key with an offset, fit
/// the raise of `ct_in` into `ct_out.k()` bits.
pub(crate) fn check_raise_keys<BE, F, K, C1, C2>(
    op: &'static str,
    ct_out: &C1,
    ct_in: &C2,
    ctx: &BootstrappingContextPrepared<BE, F>,
    keys: &K,
) -> Result<()>
where
    BE: Backend,
    K: BootstrappingLinearKeys<BE>,
    C1: CKKSCtBounds,
    C2: CKKSCtBounds,
{
    let bootstrap_k = ct_out.k().as_usize();
    let (k_small, raised) = ctx.stage_widths().raise(ct_in.k().as_usize());
    if let Some(factor) = ctx.coeffs_to_slots_first_factor() {
        crate::reference::first_factor::check_first_factor_raise(
            op,
            ct_out.base2k().as_usize(),
            ct_out.n().as_usize().ilog2() as usize,
            bootstrap_k,
            raised,
            factor,
            keys,
        )?;
    } else if let Some((_, Some(sparse_to_dense))) = keys.encapsulation_keys()
        && sparse_to_dense.k_offset().as_usize() > 0
    {
        check_offset_raise(op, bootstrap_k, k_small, raised - k_small, sparse_to_dense)?;
    }
    Ok(())
}

fn bootstrap_layouts<C1, C2>(ct_out: &C1, ct_in: &C2) -> (CKKSLayout, CKKSLayout)
where
    C1: CKKSCtBounds,
    C2: CKKSCtBounds,
{
    let base2k = ct_in.base2k();
    let boot_layout = CKKSLayout {
        glwe_layout: GLWELayout {
            n: ct_out.n(),
            base2k,
            k: ct_out.k(),
            rank: Rank(1),
        },
        // `log_delta = 0` maximizes `log_budget`, which upper-bounds the
        // EvalMod working width.
        meta: GLWEPlaintextMeta::default(),
    };
    let in_layout = CKKSLayout {
        glwe_layout: GLWELayout {
            n: ct_in.n(),
            base2k,
            k: ct_in.k(),
            rank: Rank(1),
        },
        meta: GLWEPlaintextMeta::default(),
    };
    (boot_layout, in_layout)
}

/// Backend-generic reference of the CKKS bootstrapping pipeline, selected by
/// [`impl_ckks_bootstrapping_reference_full`](crate::oep::impl_ckks_bootstrapping_reference_full).
pub trait CKKSBootstrappingReference<BE: CKKSCITwin> {
    /// Scratch upper bound for [`Self::ckks_bootstrap_reference`].
    fn ckks_bootstrap_tmp_bytes_reference<C1, C2, F>(
        &self,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContextPrepared<BE, F>,
        ci: Option<&Module<BE::CI>>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        Module<BE>: GLWEBytesOf<BE>,
        Module<BE>:
            CKKSAllOpsTmpBytes<BE> + CKKSDFTOps<BE> + CKKSEvalModOps<BE> + GLWEKeyswitch<BE> + GLWEShift<BE> + CKKSPow2Ops<BE>,
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
        CKKSCiphertextOwned<BE>: CKKSCtBounds;

    /// Scratch upper bound for [`Self::ckks_functional_bootstrap_reference`], for one
    /// LUT or a batch: the batch folds each imaginary half as it is produced,
    /// so the carved working set does not grow with `luts.len()`.
    fn ckks_functional_bootstrap_tmp_bytes_reference<C1, C2, F>(
        &self,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContextPrepared<BE, F>,
        luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        Module<BE>: GLWEBytesOf<BE>,
        Module<BE>: CKKSAllOpsTmpBytes<BE>
            + CKKSDFTOps<BE>
            + CKKSEvalModOps<BE>
            + GLWEKeyswitch<BE>
            + CKKSMulOps<BE>
            + CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + GLWEShift<BE>
            + CKKSPow2Ops<BE>,
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
        CKKSCiphertextOwned<BE>: CKKSCtBounds;

    /// Scratch upper bound for [`Self::ckks_mod_raise_reference`].
    fn ckks_mod_raise_tmp_bytes_reference<R, A, P>(
        &self,
        res: &R,
        a: &A,
        plan: &P,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        Module<BE>: GLWEShift<BE> + CKKSPow2Ops<BE>,
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSModRaiseInfos<BE>;

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
        Module<BE>: GLWECopy<BE> + GLWEShift<BE> + GLWEKeyswitch<BE> + CKKSPow2Ops<BE>,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: CKKSModRaiseInfos<BE>,
        K: BootstrappingLinearKeys<BE>;

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
        P: CKKSEvalModInfos<BE, F>;

    /// EvalMod of a bootstrap on `a`, the output of CoeffsToSlots, as the plan and the slots of `a` say:
    /// one [`ckks_eval_mod`](CKKSEvalModOps::ckks_eval_mod) on the real part, one on each half with their recombination,
    /// or one on the conjugate-invariant ring between the conversions there and back. Under EvalRound+ it returns
    /// `EvalMod(x) - K·x` and keeps the guard bits, which go with the bypass; otherwise it removes them.
    fn ckks_bootstrap_eval_mod_reference<R, A, P, F, K>(
        &self,
        res: &mut R,
        a: &A,
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
            + CKKSCopyOps<BE>
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
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: CKKSEvalModInfos<BE, F>,
        K: BootstrappingKeys<BE> + Sync,
        K::TensorKey: Sync;

    /// Refreshes one standard ciphertext. The pipeline is selected from the context.
    #[allow(clippy::too_many_arguments)]
    fn ckks_bootstrap_reference<F, K>(
        &self,
        ct_out: &mut CKKSCiphertextOwned<BE>,
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingContextPrepared<BE, F>,
        ci: Option<&Module<BE::CI>>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        F: Sync,
        Module<BE>: GLWECopy<BE>
            + GLWEShift<BE>
            + GLWEKeyswitch<BE>
            + CKKSCopyOps<BE>
            + CKKSPow2Ops<BE>
            + CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + CKKSConjugateOps<BE>
            + CKKSImagOps<BE>
            + CKKSDFTOps<BE>
            + CKKSEvalModOps<BE>
            + GLWENormalize<BE>,
        K: BootstrappingKeys<BE> + Sync,
        K::TensorKey: Sync,
        CKKSCiphertextOwned<BE>:
            GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos;

    /// Backend-generic reference for [`CKKSBootstrappingOps::ckks_functional_bootstrap`].
    ///
    /// One LUT or many: the batch shares the S2C, ModUp and CoeffsToSlots stages,
    /// and general LUTs additionally share the power basis of each transformed
    /// half. The slot kind of `ct_in` selects the pipeline: real slots skip the
    /// imaginary branch entirely.
    fn ckks_functional_bootstrap_reference<F, K>(
        &self,
        ct_outs: &mut [CKKSCiphertextOwned<BE>],
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingContextPrepared<BE, F>,
        luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE>
            + GLWEShift<BE>
            + GLWEKeyswitch<BE>
            + CKKSDFTOps<BE>
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
        K: BootstrappingKeys<BE>,
        CKKSCiphertextOwned<BE>:
            GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos;
}

impl<BE: Backend + CKKSBootstrappingImpl + CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl> CKKSBootstrappingReference<BE>
    for Module<BE>
{
    fn ckks_bootstrap_tmp_bytes_reference<C1, C2, F>(
        &self,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContextPrepared<BE, F>,
        ci: Option<&Module<BE::CI>>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        Module<BE>: GLWEBytesOf<BE>,
        Module<BE>:
            CKKSAllOpsTmpBytes<BE> + CKKSDFTOps<BE> + CKKSEvalModOps<BE> + GLWEKeyswitch<BE> + GLWEShift<BE> + CKKSPow2Ops<BE>,
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
        CKKSCiphertextOwned<BE>: CKKSCtBounds,
    {
        bootstrap_tmp_bytes_for(self, ct_out, ct_in, ctx, ci, keys_layout)
    }

    fn ckks_functional_bootstrap_tmp_bytes_reference<C1, C2, F>(
        &self,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContextPrepared<BE, F>,
        luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        Module<BE>: GLWEBytesOf<BE>,
        Module<BE>: CKKSAllOpsTmpBytes<BE>
            + CKKSDFTOps<BE>
            + CKKSEvalModOps<BE>
            + GLWEKeyswitch<BE>
            + CKKSMulOps<BE>
            + CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + GLWEShift<BE>
            + CKKSPow2Ops<BE>,
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
        CKKSCiphertextOwned<BE>: CKKSCtBounds,
    {
        let base = self.ckks_bootstrap_tmp_bytes_reference(ct_out, ct_in, ctx, None, keys_layout);
        let (boot_layout, in_layout) = bootstrap_layouts(ct_out, ct_in);
        let boot_bytes = self.glwe_bytes_of_from_infos(&boot_layout);
        let in_bytes = self.glwe_bytes_of_from_infos(&in_layout);
        // The complex path holds `ct_raised`, `r0` and `i0` while carving one
        // LUT-evaluation temporary. Direct S2C instead holds one boot-width
        // ciphertext beside its input-width buffer.
        let carved = (4 * boot_bytes).max(boot_bytes + in_bytes);
        let tsk = &keys_layout.tensor_key;
        let mut nested = s2c_first_linear_tmp_bytes(self, ct_out, ct_in, ctx, keys_layout);
        if let Some(eval_mod) = ctx.eval_mod().filter(|_| luts.iter().any(EncodedLut::requires_eval_mod)) {
            nested = nested
                .max(self.ckks_eval_mod_tmp_bytes(&boot_layout, &boot_layout, eval_mod, tsk))
                .max(self.ckks_eval_mod_tmp_bytes(ct_out, &boot_layout, eval_mod, tsk));
        }
        for atk in &keys_layout.rotation_key_layouts() {
            for lut in luts {
                nested = nested
                    .max(self.ckks_all_ops_with_atk_tmp_bytes(&boot_layout, lut.coeffs(), tsk, atk))
                    .max(self.ckks_all_ops_with_atk_tmp_bytes(&in_layout, lut.coeffs(), tsk, atk));
            }
        }
        // A double-angle chain squares into a temporary of its output's layout.
        if luts
            .iter()
            .any(|lut| matches!(lut.kind(), EncodedLutKind::Binary { chain: Some(_), .. }))
        {
            let size = ct_out.max_size();
            nested = nested.max(
                BE::scratch_aligned(self.glwe_bytes_of_from_infos(&GLWELayout {
                    k: ct_out.max_k(),
                    ..ct_out.glwe_layout()
                })) + self
                    .ckks_square_tmp_bytes(ct_out, ct_out, tsk)
                    .max(self.ckks_add_pt_const_tmp_bytes(size))
                    .max(self.ckks_sub_pt_const_tmp_bytes(size)),
            );
        }
        base.max(carved + nested)
    }

    fn ckks_mod_raise_tmp_bytes_reference<R, A, P>(
        &self,
        res: &R,
        a: &A,
        plan: &P,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        Module<BE>: GLWEShift<BE> + CKKSPow2Ops<BE>,
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSModRaiseInfos<BE>,
    {
        let (boot_layout, in_layout) = bootstrap_layouts(res, a);
        raise_tmp_bytes(self, &boot_layout, &in_layout, plan, keys_layout)
    }

    fn ckks_mod_raise_reference<R, A, P, K>(
        &self,
        res: &mut R,
        a: &mut A,
        plan: &P,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE> + GLWEShift<BE> + GLWEKeyswitch<BE> + CKKSPow2Ops<BE>,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: CKKSModRaiseInfos<BE>,
        K: BootstrappingLinearKeys<BE>,
    {
        const OP: &str = "ckks_mod_raise";
        let raise = plan.mod_raise_plan();
        // The pipeline carves rank-1 intermediates, so reject higher-rank inputs
        // rather than silently copying into a rank-1 scratch ciphertext.
        ckks_ensure_operand!(
            OP,
            a.rank().as_usize() == 1 && res.rank().as_usize() == 1,
            "supports rank-1 ciphertexts only, got rank {} -> {}",
            a.rank().as_usize(),
            res.rank().as_usize()
        );
        ckks_ensure_operand!(
            OP,
            keys.encapsulation_keys().is_some() == raise.sparse_secret_hamming_weight.is_some(),
            "bootstrapping key encapsulation does not match the compiled recipe"
        );
        if let Some(keys_layout) = keys.keys_layout() {
            let bytes = self.ckks_mod_raise_tmp_bytes_reference(res, a, plan, &keys_layout);
            ckks_ensure!(
                scratch.available() >= bytes,
                "{OP} needs a scratch arena of {bytes} bytes, got {}",
                scratch.available()
            );
        }
        let (slots, log_sparsity) = (a.slots(), a.log_sparsity());
        let log_modulus_in = a.k().as_usize();
        mod_raise_body(
            self,
            res,
            a,
            raise.lift.as_ref(),
            raise.c2s_guard_bits,
            plan.first_factor(),
            keys,
            scratch,
        )?;
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
        match (ci, plan.eval_mod_ci()) {
            (Some(ci), Some(eval_mod_ci)) => {
                let (boot_layout, _) = bootstrap_layouts(res, a);
                eval_mod_on_ci_tmp_bytes(self, ci, &boot_layout, eval_mod_ci, keys_layout)
            }
            _ => eval_mod_standard_tmp_bytes(self, res, a, plan, keys_layout),
        }
    }

    fn ckks_bootstrap_eval_mod_reference<R, A, P, F, K>(
        &self,
        res: &mut R,
        a: &A,
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
            + CKKSCopyOps<BE>
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
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: CKKSEvalModInfos<BE, F>,
        K: BootstrappingKeys<BE> + Sync,
        K::TensorKey: Sync,
    {
        const OP: &str = EVAL_MOD_OP;
        ckks_ensure_operand!(
            OP,
            ci.is_some() == plan.eval_mod_ci().is_some(),
            "a CI module goes with a plan of EvalMod on the CI ring, and only with one"
        );
        ckks_ensure_operand!(
            OP,
            a.rank().as_usize() == 1 && res.rank().as_usize() == 1,
            "supports rank-1 ciphertexts only, got rank {} -> {}",
            a.rank().as_usize(),
            res.rank().as_usize()
        );
        ensure_eval_round_interval(plan)?;
        if let Some(keys_layout) = keys.keys_layout() {
            let bytes = self.ckks_bootstrap_eval_mod_tmp_bytes_reference(res, a, plan, ci, &keys_layout);
            ckks_ensure!(
                scratch.available() >= bytes,
                "{OP} needs a scratch arena of {bytes} bytes, got {}",
                scratch.available()
            );
        }
        match (ci, plan.eval_mod_ci()) {
            (Some(ci), Some(eval_mod_ci)) => eval_mod_on_ci(self, ci, res, a, plan, eval_mod_ci, keys, scratch)?,
            _ if a.slots().is_real() && plan.one_eval_mod_on_real_slots() => eval_mod_real(self, res, a, plan, keys, scratch)?,
            _ => eval_mod_complex(self, res, a, plan, keys, scratch)?,
        }
        if !plan.eval_round_plus() {
            res.set_log_delta(res.log_delta() - plan.c2s_guard_bits());
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn ckks_bootstrap_reference<F, K>(
        &self,
        ct_out: &mut CKKSCiphertextOwned<BE>,
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingContextPrepared<BE, F>,
        ci: Option<&Module<BE::CI>>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        F: Sync,
        Module<BE>: GLWECopy<BE>
            + GLWEShift<BE>
            + GLWEKeyswitch<BE>
            + CKKSCopyOps<BE>
            + CKKSPow2Ops<BE>
            + CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + CKKSConjugateOps<BE>
            + CKKSImagOps<BE>
            + CKKSDFTOps<BE>
            + CKKSEvalModOps<BE>
            + GLWENormalize<BE>,
        K: BootstrappingKeys<BE> + Sync,
        K::TensorKey: Sync,
        CKKSCiphertextOwned<BE>:
            GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        const OP: &str = "ckks_bootstrap";
        ckks_ensure_operand!(
            OP,
            ctx.functional_message_modulus().is_none() && ctx.eval_mod_ring() == EvalModRing::Standard,
            "requires an identity bootstrapping context with EvalMod on the standard ring"
        );
        ckks_ensure_operand!(
            OP,
            ci.is_some() == ctx.eval_mod_ci().is_some(),
            "a CI module goes with a plan of EvalMod on the CI ring, and only with one"
        );
        // All pipeline intermediates are rank-1 working ciphertexts carved from
        // scratch (accounted for by `ckks_bootstrap_tmp_bytes`); reject
        // higher-rank inputs up front.
        ckks_ensure_operand!(
            OP,
            ct_in.rank().as_usize() == 1 && ct_out.rank().as_usize() == 1,
            "supports rank-1 ciphertexts only, got rank {} -> {}",
            ct_in.rank().as_usize(),
            ct_out.rank().as_usize()
        );

        let encapsulation_keys = keys.encapsulation_keys();
        ckks_ensure_operand!(
            OP,
            encapsulation_keys.is_some() == ctx.sparse_secret_hamming_weight().is_some(),
            "bootstrapping key encapsulation does not match the compiled recipe (expected {}, got {})",
            ctx.sparse_secret_hamming_weight().is_some(),
            encapsulation_keys.is_some()
        );
        let tensor_key = required_tensor_key(OP, keys.tensor_key())?;
        check_key_widths(OP, ct_out, ct_in, ctx, keys, tensor_key)?;

        let boot_layout = GLWELayout {
            n: ct_out.n(),
            base2k: ct_in.base2k(),
            k: ct_out.k(),
            rank: Rank(1),
        };
        scratch.scope(|scratch| {
            let (mut z, mut scratch) = scratch.take_ckks_ciphertext_scratch(&boot_layout, ct_in.meta());
            match ctx.pipeline() {
                BootstrappingPipeline::S2CFirst => {
                    ckks_bootstrap_s2c_mod_up(self, &mut z, ct_in, ctx, keys, &mut scratch)?;
                    coeffs_to_slots_and_eval_mod(self, ct_out, &mut z, ct_in.slots(), ctx, ci, keys, &mut scratch)
                }
                BootstrappingPipeline::C2SFirst => {
                    // (encapsulate) denseToSparse → ModUp → sparseToDense. With encapsulation the
                    // integer wrap-around `I·q` exposed by ModUp is bounded by the *sparse* secret's
                    // Hamming weight (https://eprint.iacr.org/2022/024).
                    // The raise consumes its input: it works on a copy, released once `z` holds the raised ciphertext.
                    scratch.scope(|scratch_raise| {
                        let (mut ct0, mut scratch_raise) = scratch_raise.take_ckks_ciphertext_scratch(ct_in, ct_in.meta());
                        self.ckks_copy(&mut ct0, ct_in, &mut scratch_raise)?;
                        self.ckks_mod_raise(&mut z, &mut ct0, ctx, keys, &mut scratch_raise)
                    })?;
                    coeffs_to_slots_and_eval_mod(self, ct_out, &mut z, ct_in.slots(), ctx, ci, keys, &mut scratch)?;
                    self.ckks_slots_to_coeffs_assign(ct_out, ctx, keys.rotation_keys(), &mut scratch)
                }
            }
        })?;
        match ctx.pipeline() {
            // The guard bits are gone: relabel at the output width.
            BootstrappingPipeline::S2CFirst => ct_out.set_meta(GLWEPlaintextMeta {
                scale: Scale::Log(ct_in.log_delta()),
                slots: ct_in.slots(),
                log_sparsity: ct_in.log_sparsity(),
            }),
            BootstrappingPipeline::C2SFirst => {
                ct_out.set_log_delta(ct_in.log_delta());
                ct_out.set_slots(ct_in.slots());
            }
        }
        Ok(())
    }

    fn ckks_functional_bootstrap_reference<F, K>(
        &self,
        ct_outs: &mut [CKKSCiphertextOwned<BE>],
        ct_in: &CKKSCiphertextOwned<BE>,
        ctx: &BootstrappingContextPrepared<BE, F>,
        luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE>
            + GLWEShift<BE>
            + GLWEKeyswitch<BE>
            + CKKSDFTOps<BE>
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
        K: BootstrappingKeys<BE>,
        CKKSCiphertextOwned<BE>:
            GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        const OP: &str = "ckks_functional_bootstrap";
        ckks_ensure_operand!(OP, !luts.is_empty(), "functional bootstrap requires at least one LUT");
        ckks_ensure_operand!(
            OP,
            ct_outs.len() == luts.len(),
            "functional bootstrap ct_outs/luts length mismatch ({} vs {})",
            ct_outs.len(),
            luts.len()
        );
        ckks_ensure_operand!(
            OP,
            ctx.pipeline() == BootstrappingPipeline::S2CFirst,
            "functional bootstrapping requires an S2C-first context"
        );
        let log_msg_ratio = luts[0].log_msg_ratio();
        ckks_ensure_operand!(
            OP,
            luts.iter().all(|lut| lut.message_modulus() == luts[0].message_modulus()),
            "functional bootstrap LUTs must have the same message modulus"
        );
        if luts.iter().any(EncodedLut::requires_eval_mod) {
            ensure_unit_circle_exp_context(OP, ctx)?;
        }
        let head = &ct_outs[0];
        let (out_n, out_k) = (head.n(), head.k());
        ckks_ensure_operand!(
            OP,
            ct_in.rank().as_usize() == 1
                && ct_outs.iter().all(|ct| {
                    ct.rank().as_usize() == 1 && ct.n() == head.n() && ct.base2k() == head.base2k() && ct.k() == head.k()
                }),
            "functional bootstrapping outputs must share one rank-1 layout"
        );
        ckks_ensure_operand!(
            OP,
            keys.encapsulation_keys().is_some() == ctx.sparse_secret_hamming_weight().is_some(),
            "bootstrapping key encapsulation does not match the compiled recipe"
        );
        let tensor_key = required_tensor_key(OP, keys.tensor_key())?;
        ensure_functional_message_ratio(OP, ct_in, ctx.slots_to_coeffs().consumed_bits(), log_msg_ratio)?;
        // Keys laid out for the raise fail here rather than after SlotsToCoeffs.
        check_raise_keys(OP, head, ct_in, ctx, keys)?;
        ckks_ensure_operand!(
            OP,
            ctx.functional_message_modulus().unwrap_or(1usize << log_msg_ratio) == luts[0].message_modulus(),
            "functional bootstrap context must be configured for message modulus {}",
            luts[0].message_modulus()
        );
        let output_contracts = ct_outs
            .iter()
            .zip(luts)
            .map(|(ct_out, lut)| functional_output_contract(OP, ct_in, ctx, lut, ct_out.k().as_usize()))
            .collect::<Result<Vec<_>>>()?;

        let shared = luts.len() > 1 && luts.iter().all(|lut| lut.general_series().is_some());
        let boot_layout = GLWELayout {
            n: out_n,
            base2k: ct_in.base2k(),
            k: out_k,
            rank: Rank(1),
        };
        scratch.scope(|scratch_local| {
            let (mut ct_raised, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_in.meta());
            ckks_bootstrap_s2c_mod_up(self, &mut ct_raised, ct_in, ctx, keys, &mut scratch_local)?;

            if ct_in.slots().is_real() {
                let (mut r0, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_raised.meta());
                ckks_bootstrap_coeffs_to_slots_real(self, &mut ct_raised, &mut r0, ctx, keys, &mut scratch_local)?;
                ct_raised.set_log_delta(ct_raised.log_delta() - ctx.c2s_guard_bits());
                return eval_lut_batch(
                    self,
                    ct_outs,
                    &ct_raised,
                    ctx,
                    luts,
                    keys,
                    tensor_key,
                    shared,
                    &mut scratch_local,
                );
            }

            let (mut r0, scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_raised.meta());
            let (mut i0, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&boot_layout, ct_raised.meta());
            ckks_bootstrap_coeffs_to_slots(self, &ct_raised, &mut r0, &mut i0, ctx, keys, &mut scratch_local)?;
            r0.set_log_delta(r0.log_delta() - ctx.c2s_guard_bits());
            i0.set_log_delta(i0.log_delta() - ctx.c2s_guard_bits());

            eval_lut_batch(self, ct_outs, &r0, ctx, luts, keys, tensor_key, shared, &mut scratch_local)?;

            // `ct_raised` is dead after CoeffsToSlots, so it serves as the single
            // imaginary accumulator: each LUT's imaginary half is folded into its
            // output before the next one is evaluated, which keeps the carved
            // working set independent of the batch size. Multiplication produces
            // its result at the destination's requested `k`, so the accumulator is
            // restored to the full boot width before each reuse; otherwise the
            // first LUT's consumption would clamp every later one.
            let im_meta = ct_raised.meta();
            if shared {
                let basis = ckks_lut_power_basis(
                    self,
                    &i0,
                    &boot_layout,
                    ctx.standard_eval_mod("ckks_functional_bootstrap")?,
                    luts,
                    tensor_key,
                    &mut scratch_local,
                )?;
                for (ct_out, lut) in ct_outs.iter_mut().zip(luts) {
                    ct_raised.set_meta(im_meta);
                    ct_raised.set_k(boot_layout.k);
                    ckks_eval_lut_from_basis(
                        self,
                        &mut ct_raised,
                        lut,
                        &basis,
                        keys.rotation_keys(),
                        tensor_key,
                        &mut scratch_local,
                    )?;
                    recombine_halves(self, ct_out, &mut ct_raised, &mut scratch_local)?;
                }
            } else {
                for (ct_out, lut) in ct_outs.iter_mut().zip(luts) {
                    ct_raised.set_meta(im_meta);
                    ct_raised.set_k(boot_layout.k);
                    ckks_eval_encoded_lut(self, &mut ct_raised, &i0, ctx, lut, keys, tensor_key, &mut scratch_local)?;
                    recombine_halves(self, ct_out, &mut ct_raised, &mut scratch_local)?;
                }
            }
            Result::Ok(())
        })?;
        for (ct_out, (meta, expected_k)) in ct_outs.iter_mut().zip(output_contracts) {
            ct_out.set_meta(meta);
            ckks_ensure!(
                ct_out.k().as_usize() >= expected_k,
                "functional bootstrap produced k={}, below required {expected_k}",
                ct_out.k().as_usize()
            );
            ct_out.set_k(expected_k.into());
        }
        Ok(())
    }
}

/// `scale_up` lifts the message that many bits above its natural magnitude,
/// fused into the widening shift rather than costing a second pass.
pub(crate) fn ckks_mod_up_scaled_reference<BE, Dst, Src>(
    op: &'static str,
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

    ensure_mod_up_widens(op, k_large, k_small, scale_up)?;

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

#[expect(
    clippy::too_many_arguments,
    reason = "the raise takes its lift, its scale and the first factor as separate operands"
)]
fn mod_raise_body<BE, Dst, Src, K>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &mut Src,
    lift: Option<&EvalModPlan>,
    mut scale_up: usize,
    first_factor: Option<&LinearTransformation<CKKSRightOperandPrepared<BE>>>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSBootstrappingImpl + CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
    Module<BE>: GLWECopy<BE> + GLWEShift<BE> + GLWEKeyswitch<BE> + CKKSPow2Ops<BE>,
    K: BootstrappingLinearKeys<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    // C2S-first derives both lifts from EvalMod; S2C-first supplies only
    // the internal guard bits, fused into ModUp's shift.
    if let Some(plan) = lift {
        let k: usize = src.k().as_usize();
        let log_delta: usize = src.log_delta();
        let scale_up_in = plan.input_scale_up(k, log_delta)?;
        if scale_up_in != 0 {
            module.ckks_mul_pow2_assign(scale_up_in, src, scratch)?;
            // Not `set_log_delta`: it preserves `log_budget` and grows `k`, which
            // cancels the shift. The lift spends budget at a fixed modulus.
            let mut meta = src.meta();
            meta.scale = Scale::Log(log_delta + scale_up_in);
            src.set_meta(meta);
        }
        scale_up = plan.raised_scale_up(src.log_delta())?;
    }

    // A first CoeffsToSlots factor left to the raise is applied here, on the
    // raised ciphertext while it is small; its switches replace sparse-to-dense.
    if let Some(factor) = first_factor {
        return BE::ckks_mod_up_first_factor(module, dst, src, scale_up, factor, keys, scratch);
    }

    match keys.encapsulation_keys() {
        Some((dense_to_sparse, sparse_to_dense)) => {
            ckks_ensure_operand!(
                "ckks_mod_raise",
                sparse_to_dense.is_some(),
                "the encapsulated ModUp needs a sparse-to-dense key, which a key set for the first factor on the raise has not"
            );
            let sparse_to_dense = sparse_to_dense.unwrap();
            BE::ckks_encapsulated_mod_up(
                module,
                dst,
                src,
                scale_up,
                &dense_to_sparse.to_backend_ref(),
                &sparse_to_dense.to_backend_ref(),
                scratch,
            )?;
        }
        None => ckks_mod_up_scaled_reference("ckks_mod_raise", module, dst, src, scale_up, scratch)?,
    }
    Ok(())
}

fn recombine_halves<BE, R1, R2>(
    module: &Module<BE>,
    res_re: &mut R1,
    res_im: &mut R2,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSEncapsulatedModUpImpl,
    Module<BE>: CKKSImagOps<BE> + CKKSAddOps<BE>,
    R1: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    R2: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    module.ckks_mul_i_assign(res_im, scratch)?;
    module.ckks_add_assign(res_re, &*res_im, scratch)?;
    Ok(())
}

fn ckks_bootstrap_coeffs_to_slots<BE, F, K, C, R>(
    module: &Module<BE>,
    ct: &C,
    r0: &mut R,
    i0: &mut R,
    ctx: &BootstrappingContextPrepared<BE, F>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSEncapsulatedModUpImpl,
    Module<BE>: CKKSDFTOps<BE>,
    K: BootstrappingLinearKeys<BE>,
    C: GLWEToBackendRef<BE> + CKKSCtBounds,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    module.ckks_coeffs_to_slots_split(r0, i0, ct, ctx.coeffs_to_slots(), keys.rotation_keys(), scratch)
}

fn ckks_bootstrap_coeffs_to_slots_real<BE, F, K, R1, R2>(
    module: &Module<BE>,
    ct: &mut R1,
    conjugate: &mut R2,
    ctx: &BootstrappingContextPrepared<BE, F>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSEncapsulatedModUpImpl,
    Module<BE>: CKKSDFTOps<BE> + CKKSConjugateOps<BE> + CKKSAddOps<BE>,
    K: BootstrappingLinearKeys<BE>,
    R1: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    R2: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    module.ckks_dft_matrix_eval_assign(ct, ctx.coeffs_to_slots(), keys.rotation_keys(), scratch)?;
    ckks_bootstrap_real_part(module, ct, conjugate, keys, scratch)
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
    BE: Backend + CKKSEncapsulatedModUpImpl,
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

/// `key`, the tensor key of the plan's ring, or the refusal of `op`.
fn required_tensor_key<'a, T>(op: &'static str, key: Option<&'a T>) -> Result<&'a T> {
    key.ok_or_else(|| {
        CKKSCompositionError::InvalidOperand {
            op,
            reason: "the keys hold no tensor key of the plan's ring".into(),
        }
        .into()
    })
}

/// The plan constructor rejects this, but re-check before the power-of-two shift of EvalRound+.
fn ensure_eval_round_interval<BE, F, P>(plan: &P) -> Result<()>
where
    BE: CKKSCITwin,
    P: CKKSEvalModInfos<BE, F>,
{
    if plan.eval_round_plus() {
        ckks_ensure!(
            plan.eval_mod_plan().f_mod_interval.is_power_of_two(),
            "EvalRound+ requires a power-of-two f_mod_interval, got {}",
            plan.eval_mod_plan().f_mod_interval
        );
    }
    Ok(())
}

/// EvalMod on the standard ring, which the two standard routes run.
fn eval_mod_standard<BE, F, P>(plan: &P) -> Result<&EvalMod<F, CKKSPlaintextOwned<BE>>>
where
    BE: CKKSCITwin,
    P: CKKSEvalModInfos<BE, F>,
{
    plan.eval_mod_standard().ok_or_else(|| {
        CKKSCompositionError::InvalidOperand {
            op: EVAL_MOD_OP,
            reason: "the plan runs no EvalMod on the standard ring".into(),
        }
        .into()
    })
}

/// One EvalMod on the real part of `a`, on the real slots of an S2C-first plan without a bypass.
fn eval_mod_real<BE, F, R, A, P, K>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    plan: &P,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSCITwin,
    Module<BE>: CKKSCopyOps<BE> + CKKSConjugateOps<BE> + CKKSAddOps<BE> + CKKSEvalModOps<BE>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: CKKSEvalModInfos<BE, F>,
    K: BootstrappingKeys<BE>,
{
    let eval_mod = eval_mod_standard(plan)?;
    let tensor_key = required_tensor_key(EVAL_MOD_OP, keys.tensor_key())?;
    let (mut z, scratch) = scratch.borrow().take_ckks_ciphertext_scratch(a, a.meta());
    let (mut conjugate, mut scratch) = scratch.take_ckks_ciphertext_scratch(a, a.meta());
    module.ckks_copy(&mut z, a, &mut scratch)?;
    ckks_bootstrap_real_part(module, &mut z, &mut conjugate, keys, &mut scratch)?;
    module.ckks_eval_mod(res, &z, eval_mod, tensor_key, &mut scratch)
}

/// The split of `a`, one EvalMod per half and their recombination. Under EvalRound+ each
/// half also takes `-K·x`, on the very input `x` of its EvalMod, so that the error of the
/// low-precision CoeffsToSlots cancels against the bypass the caller adds back.
fn eval_mod_complex<BE, F, R, A, P, K>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    plan: &P,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSCITwin + CKKSEncapsulatedModUpImpl,
    F: Sync,
    Module<BE>: CKKSCopyOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSImagOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSEvalModOps<BE>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + Send,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: CKKSEvalModInfos<BE, F>,
    K: BootstrappingKeys<BE>,
    K::TensorKey: Sync,
{
    let eval_mod = eval_mod_standard(plan)?;
    let tensor_key = required_tensor_key(EVAL_MOD_OP, keys.tensor_key())?;
    let (mut r0, scratch) = scratch.borrow().take_ckks_ciphertext_scratch(a, a.meta());
    let (mut i0, scratch) = scratch.take_ckks_ciphertext_scratch(a, a.meta());
    let (mut imag, mut scratch) = scratch.take_ckks_ciphertext_scratch(res, res.meta());
    module.ckks_copy(&mut r0, a, &mut scratch)?;
    ckks_split_real_imag(module, &mut r0, &mut i0, keys.rotation_keys(), &mut scratch)?;
    {
        let (r0_ref, i0_ref) = (r0.to_backend_view_ref(), i0.to_backend_view_ref());
        ckks_bootstrap_eval_mod_halves(module, &r0_ref, &i0_ref, res, &mut imag, eval_mod, tensor_key, &mut scratch)?;
    }
    if plan.eval_round_plus() {
        let log2_k = plan.eval_mod_plan().f_mod_interval.trailing_zeros() as usize;
        module.ckks_mul_pow2_assign(log2_k, &mut r0, &mut scratch)?;
        module.ckks_mul_pow2_assign(log2_k, &mut i0, &mut scratch)?;
        module.ckks_sub_assign(res, &r0, &mut scratch)?;
        module.ckks_sub_assign(&mut imag, &i0, &mut scratch)?;
    }
    recombine_halves(module, res, &mut imag, &mut scratch)
}

/// CoeffsToSlots on `z` and EvalMod into `res`, with the EvalRound+ bypass when the plan has one.
/// `slots` is the kind of the message: the transforms label their outputs complex, and the step routes on it.
#[expect(
    clippy::too_many_arguments,
    reason = "the composition hands the arguments of its two steps down, with the slots they lose on the way"
)]
fn coeffs_to_slots_and_eval_mod<BE, F, K, R, Z>(
    module: &Module<BE>,
    res: &mut R,
    z: &mut Z,
    slots: SlotsKind,
    ctx: &BootstrappingContextPrepared<BE, F>,
    ci: Option<&Module<BE::CI>>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSBootstrappingImpl,
    F: Sync,
    Module<BE>: CKKSDFTOps<BE> + CKKSCopyOps<BE> + CKKSPow2Ops<BE> + CKKSAddOps<BE>,
    K: BootstrappingKeys<BE> + Sync,
    K::TensorKey: Sync,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + Send,
    Z: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    let Some(bypass) = ctx.coeffs_to_slots_bypass() else {
        module.ckks_coeffs_to_slots_assign(z, ctx, keys.rotation_keys(), scratch)?;
        z.set_slots(slots);
        return module.ckks_bootstrap_eval_mod(res, &*z, ctx, ci, keys, scratch);
    };
    let (mut z_hp, mut scratch) = scratch.borrow().take_ckks_ciphertext_scratch(&*z, z.meta());
    module.ckks_copy(&mut z_hp, &*z, &mut scratch)?;
    module.ckks_dft_matrix_eval_assign(&mut z_hp, bypass, keys.rotation_keys(), &mut scratch)?;
    module.ckks_coeffs_to_slots_assign(z, ctx, keys.rotation_keys(), &mut scratch)?;
    module.ckks_bootstrap_eval_mod(res, &*z, ctx, ci, keys, &mut scratch)?;
    // `2·z_hp`, the bypass on the packed halves, which the split doubles as `2·Re` and `2·Im`.
    module.ckks_mul_pow2_assign(1, &mut z_hp, &mut scratch)?;
    module.ckks_add_assign(&mut z_hp, &*res, &mut scratch)?;
    module.ckks_copy(res, &z_hp, &mut scratch)?;
    res.set_log_delta(res.log_delta() - ctx.c2s_guard_bits());
    Ok(())
}

fn ckks_bootstrap_s2c_mod_up<BE, F, K, R, Src>(
    module: &Module<BE>,
    ct_raised: &mut R,
    ct_in: &Src,
    ctx: &BootstrappingContextPrepared<BE, F>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSBootstrappingImpl + CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
    Module<BE>: GLWECopy<BE> + GLWEShift<BE> + GLWEKeyswitch<BE> + CKKSCopyOps<BE> + CKKSPow2Ops<BE> + CKKSDFTOps<BE>,
    K: BootstrappingLinearKeys<BE>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
{
    let input_layout = GLWELayout {
        n: ct_in.n(),
        base2k: ct_in.base2k(),
        k: ct_in.k(),
        rank: Rank(1),
    };

    scratch.scope(|scratch_inner| {
        let (mut ct_coeffs, mut scratch_inner) = scratch_inner.take_ckks_ciphertext_scratch(&input_layout, ct_in.meta());
        // A `Split` decode matrix is numerically identical to the standard
        // matrix after the two halves are recombined. Preserve the split
        // path's `2 * ct_in`, then intentionally use the format-agnostic
        // evaluator directly.
        module.ckks_double_into(&mut ct_coeffs, ct_in, &mut scratch_inner)?;
        module.ckks_dft_matrix_eval_assign(
            &mut ct_coeffs,
            ctx.slots_to_coeffs(),
            keys.rotation_keys(),
            &mut scratch_inner,
        )?;

        module.ckks_mod_raise(ct_raised, &mut ct_coeffs, ctx, keys, &mut scratch_inner)?;
        Result::Ok(())
    })
}

/// Scratch of [`ckks_bootstrap_s2c_mod_up`] into a ciphertext like `ct_raised`
/// followed by CoeffsToSlots on it, with every rotation key shape against the
/// widest ciphertext: the linear stages of an S2C-first bootstrap.
fn s2c_first_linear_tmp_bytes<BE, C1, C2, F>(
    module: &Module<BE>,
    ct_raised: &C1,
    ct_in: &C2,
    ctx: &BootstrappingContextPrepared<BE, F>,
    keys_layout: &BootstrappingKeysLayout,
) -> usize
where
    BE: Backend + CKKSBootstrappingImpl + CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
    Module<BE>: GLWEBytesOf<BE> + CKKSDFTOps<BE> + CKKSPow2Ops<BE> + GLWEShift<BE> + GLWEKeyswitch<BE>,
    C1: CKKSCtBounds,
    C2: CKKSCtBounds,
{
    let (boot_layout, in_layout) = bootstrap_layouts(ct_raised, ct_in);
    // The raise: a shift, or the raise of the keys.
    let raise =
        module
            .glwe_shift_tmp_bytes(boot_layout.size())
            .max(raise_tmp_bytes(module, &boot_layout, &in_layout, ctx, keys_layout));
    // SlotsToCoeffs and the raise work on a copy of the input, CoeffsToSlots on
    // the raised ciphertext.
    let in_ct_bytes = BE::scratch_aligned(module.glwe_bytes_of_from_infos(&in_layout));
    let mut bytes = 0;
    for atk in &keys_layout.rotation_key_layouts() {
        let s2c_mod_up = module
            .ckks_dft_tmp_bytes(&in_layout, &in_layout, ctx.slots_to_coeffs(), atk)
            .max(module.ckks_double_tmp_bytes(in_layout.size()))
            .max(raise);
        let c2s = module.ckks_dft_tmp_bytes(&boot_layout, &boot_layout, ctx.coeffs_to_slots(), atk);
        bytes = bytes.max(in_ct_bytes + s2c_mod_up).max(c2s);
    }
    bytes
}

/// Scratch of [`ckks_bootstrap_ci_reference`], `(scratch, scratch_ci)`: the raised
/// ciphertext beside the linear stages, and EvalMod with its conversions, which run on `scratch_ci`.
pub fn ckks_bootstrap_ci_tmp_bytes_reference<BE, C1, C2, F>(
    module: &Module<BE>,
    ci: &Module<BE::CI>,
    ct_out: &C1,
    ct_in: &C2,
    ctx: &BootstrappingContextPrepared<BE, F>,
    keys_layout: &BootstrappingKeysLayout,
) -> (usize, usize)
where
    BE: Backend + CKKSBootstrappingImpl + CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
    Module<BE>: GLWEBytesOf<BE>
        + GLWELinearTransformationBabyStepsBytesOf<BE>
        + CKKSDFTOps<BE>
        + CKKSPow2Ops<BE>
        + GLWEShift<BE>
        + GLWEKeyswitch<BE>
        + GLWECITrace<BE>
        + CKKSMulOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSCopyOps<BE>
        + CKKSLinearTransformationOps<BE>
        + GLWERotate<BE>,
    Module<BE::CI>: CKKSEvalModOps<BE::CI>,
    C1: CKKSCtBounds,
    C2: CKKSCtBounds,
{
    // A context or a layout without conversions fails at the bootstrap.
    let Some(eval_mod_ci) = ctx.eval_mod_ci().filter(|_| keys_layout.conversion_keys.is_some()) else {
        return (0, 0);
    };
    let (boot_layout, _) = bootstrap_layouts(ct_out, ct_in);
    let mut linear = s2c_first_linear_tmp_bytes(module, &boot_layout, ct_in, ctx, keys_layout);
    let ct_bytes = BE::scratch_aligned(module.glwe_bytes_of_from_infos(&boot_layout));
    // EvalRound+ holds the high-precision CoeffsToSlots beside the rest of the bootstrap.
    let bypass = match ctx.coeffs_to_slots_bypass() {
        Some(bypass) => {
            for atk in &keys_layout.rotation_key_layouts() {
                linear = linear.max(module.ckks_dft_tmp_bytes(&boot_layout, &boot_layout, bypass, atk));
            }
            linear = linear
                .max(module.ckks_mul_pow2_tmp_bytes(boot_layout.size()))
                .max(module.ckks_add_tmp_bytes(boot_layout.size()))
                .max(module.ckks_copy_tmp_bytes(&boot_layout, &boot_layout));
            ct_bytes
        }
        None => 0,
    };
    (
        ct_bytes + bypass + linear,
        eval_mod_on_ci_tmp_bytes(module, ci, &boot_layout, eval_mod_ci, keys_layout),
    )
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
        crate::layouts::bootstrapping::CIPreForm::Folded { even, odd } => {
            // The baby steps of `even`, which `odd` shares, and each factor on its own schedule,
            // which counts the doubling and the giant-step tree of power-of-two keys.
            let (even_plan, odd_plan) = (even.index(), odd.index());
            let diagonal = crate::api::ckks_plaintext_diagonal_layout(boot_layout);
            let mut factors = 0;
            for atk in &keys_layout.rotation_key_layouts() {
                factors = factors
                    .max(module.ckks_linear_transformation_baby_steps_prepare_tmp_bytes(boot_layout, &even_plan, atk))
                    .max(module.ckks_linear_transformation_eval_tmp_bytes(boot_layout, &diagonal, &even_plan, atk, atk))
                    .max(module.ckks_linear_transformation_eval_tmp_bytes(boot_layout, &diagonal, &odd_plan, atk, atk));
            }
            module.glwe_linear_transformation_baby_steps_bytes_of(&even_plan, boot_layout)
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
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSLinearTransformationOps<BE> + GLWERotate<BE> + CKKSAddOps<BE> + GLWESwitchDegree<BE>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    W: GLWEToBackendRef<BE> + CKKSCtBounds,
    H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
{
    let plan = even.index();
    let (mut babies, mut scratch) = scratch
        .borrow()
        .take_glwe_linear_transformation_baby_steps_scratch(module, &plan, w);
    module.ckks_linear_transformation_baby_steps_prepare(&mut babies, w, keys, &mut scratch)?;
    let (mut a, scratch_1) = scratch.borrow().take_ckks_ciphertext_scratch(w, w.meta());
    let (mut b, mut scratch_1) = scratch_1.take_ckks_ciphertext_scratch(w, w.meta());
    module.ckks_linear_transformation_eval_into(&mut a, w, &babies, even, keys, &mut scratch_1)?;
    module.ckks_linear_transformation_eval_into(&mut b, w, &babies, odd, keys, &mut scratch_1)?;
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

/// The S2C-first bootstrap with EvalMod once on the conjugate-invariant ring
/// ([`CKKSBootstrappingCIOps::ckks_bootstrap_ci_into`](crate::api::CKKSBootstrappingCIOps::ckks_bootstrap_ci_into)):
/// SlotsToCoeffs, the raise and CoeffsToSlots without the split on `module`, of
/// degree `N`, and the conversions at degree `2N`, on `module`; EvalMod on `ci`.
#[expect(
    clippy::too_many_arguments,
    reason = "the bootstrap takes a module and a scratch arena per ring"
)]
pub fn ckks_bootstrap_ci_reference<BE, Dst, Src, F, K>(
    module: &Module<BE>,
    ci: &Module<BE::CI>,
    ct_out: &mut Dst,
    ct_in: &Src,
    ctx: &BootstrappingContextPrepared<BE, F>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
    scratch_ci: &mut ScratchArena<'_, BE::CI>,
) -> Result<()>
where
    BE: CKKSBootstrappingImpl + CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl,
    Module<BE>: GLWECopy<BE>
        + GLWEShift<BE>
        + GLWEKeyswitch<BE>
        + GLWECITrace<BE>
        + GLWECIEmbed<BE>
        + GLWERotate<BE>
        + CKKSSubOps<BE>
        + CKKSLinearTransformationOps<BE>
        + CKKSCopyOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSMulOps<BE>
        + CKKSAddOps<BE>
        + CKKSDFTOps<BE>
        + GLWESwitchDegree<BE>,
    Module<BE::CI>: CKKSEvalModOps<BE::CI>,
    K: BootstrappingKeys<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
{
    let op = "ckks_bootstrap_ci_into";
    let refuse = |reason: &'static str| -> crate::CKKSError {
        CKKSCompositionError::InvalidOperand {
            op,
            reason: reason.into(),
        }
        .into()
    };
    let (Some(eval_mod_ci), Some(_)) = (ctx.eval_mod_ci(), keys.conversion_keys()) else {
        return Err(refuse(
            "takes a context of a plan on the conjugate-invariant ring and keys holding its conversion keys",
        ));
    };
    let Some(tensor_key) = keys.ci_tensor_key() else {
        return Err(refuse("the keys hold no tensor key of the plan's ring"));
    };
    if !(module.n() >= 2 * ct_in.n().as_usize()
        && ci.n() == ct_in.n().as_usize()
        && ct_out.n() == ct_in.n()
        && ct_in.rank().as_usize() == 1
        && ct_out.rank().as_usize() == 1)
    {
        return Err(refuse(
            "takes a module of degree at least 2N, a CI module of degree N and rank-1 ciphertexts of degree N",
        ));
    }
    if keys.encapsulation_keys().is_some() != ctx.sparse_secret_hamming_weight().is_some() {
        return Err(refuse(
            "the bootstrapping key encapsulation does not match the compiled recipe",
        ));
    }
    check_key_widths(op, ct_out, ct_in, ctx, keys, tensor_key)?;
    let (trace_key, embed_key) = check_conversion_key_shapes(op, ct_in, keys)?;
    let widths = ctx.stage_widths().conversions(ct_out.k().as_usize()).unwrap_or_default();
    check_conversion_key_widths(op, trace_key, embed_key, widths)?;
    ensure_eval_round_interval(ctx)?;
    if let Some(keys_layout) = keys.keys_layout() {
        let (bytes, bytes_ci) = ckks_bootstrap_ci_tmp_bytes_reference(module, ci, ct_out, ct_in, ctx, &keys_layout);
        ckks_ensure!(
            scratch.available() >= bytes && scratch_ci.available() >= bytes_ci,
            "{op} needs scratch arenas of {bytes} and {bytes_ci} bytes, got {} and {}",
            scratch.available(),
            scratch_ci.available()
        );
    }

    let boot_layout = GLWELayout {
        n: ct_out.n(),
        base2k: ct_in.base2k(),
        k: ct_out.k(),
        rank: Rank(1),
    };
    scratch.scope(|scratch| {
        let (mut z, mut scratch) = scratch.take_ckks_ciphertext_scratch(&boot_layout, ct_in.meta());
        ckks_bootstrap_s2c_mod_up(module, &mut z, ct_in, ctx, keys, &mut scratch)?;
        // EvalRound+: the high-precision CoeffsToSlots of the raised ciphertext, without the split.
        let (mut z_hp, mut scratch) = match ctx.coeffs_to_slots_bypass() {
            Some(bypass) => {
                let (mut z_hp, mut scratch) = scratch.borrow().take_ckks_ciphertext_scratch(&boot_layout, z.meta());
                module.ckks_copy(&mut z_hp, &z, &mut scratch)?;
                module.ckks_dft_matrix_eval_assign(&mut z_hp, bypass, keys.rotation_keys(), &mut scratch)?;
                (Some(z_hp), scratch)
            }
            None => (None, scratch.borrow()),
        };
        // CoeffsToSlots without the split: the complex slots `z_j = c_j + i·c_{j+N/2}`.
        module.ckks_dft_matrix_eval_assign(&mut z, ctx.coeffs_to_slots(), keys.rotation_keys(), &mut scratch)?;
        eval_mod_on_ci(
            module,
            ci,
            ct_out,
            &z,
            ctx,
            eval_mod_ci,
            keys,
            &mut scratch_ci.borrow().into_backend::<BE>(),
        )?;

        // EvalRound+: `2·z_hp`, the bypass on the packed halves, which the split
        // CoeffsToSlots of the standard ring doubles as `2·Re` and `2·Im`.
        if let Some(z_hp) = z_hp.as_mut() {
            module.ckks_mul_pow2_assign(1, z_hp, &mut scratch)?;
            module.ckks_add_assign(z_hp, ct_out, &mut scratch)?;
            module.ckks_copy(ct_out, z_hp, &mut scratch)?;
        }
        Result::Ok(())
    })?;
    // Keep the C2S guard bits through EvalMod, then relabel at the output width.
    ct_out.set_log_delta(ct_out.log_delta() - ctx.c2s_guard_bits());
    ct_out.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(ct_in.log_delta()),
        slots: ct_in.slots(),
        log_sparsity: ct_in.log_sparsity(),
    });
    Ok(())
}

/// Refuses conversion keys of another degree, radix or rank than the ciphertexts they switch,
/// or absent: they go through no key source, which would check their shapes and coverage.
fn check_conversion_key_shapes<'a, BE, K, A>(op: &'static str, a: &A, keys: &'a K) -> Result<(&'a K::TraceKey, &'a K::EmbedKey)>
where
    BE: CKKSCITwin,
    K: BootstrappingKeys<BE>,
    A: CKKSCtBounds,
{
    let Some((trace_key, embed_key)) = keys.conversion_keys() else {
        let reason = "the keys hold no conversion keys of the plan's ring".into();
        return Err(CKKSCompositionError::InvalidOperand { op, reason }.into());
    };
    let n_wide = 2 * a.n().as_usize();
    for (name, key, view) in [
        ("trace key", trace_key.gglwe_layout(), trace_key.to_backend_ref()),
        ("embed key", embed_key.gglwe_layout(), embed_key.to_backend_ref()),
    ] {
        crate::layouts::validation::validate_gadget_backend_view(name, &key, &view, n_wide, a.base2k(), 0).map_err(|error| {
            CKKSCompositionError::InvalidOperand {
                op,
                reason: error.to_string(),
            }
        })?;
    }
    Ok((trace_key, embed_key))
}

/// Refuses conversion keys narrower than the ciphertexts they switch, of `traced` bits to the
/// conjugate-invariant ring and of `embedded` bits back.
fn check_conversion_key_widths<T, E>(
    op: &'static str,
    trace_key: &T,
    embed_key: &E,
    (traced, embedded): (usize, usize),
) -> Result<()>
where
    T: GGLWEInfos,
    E: GGLWEInfos,
{
    for (key, k) in [(trace_key.gglwe_layout(), traced), (embed_key.gglwe_layout(), embedded)] {
        let covered = key.max_input_k().as_usize();
        if k > covered {
            let reason = format!("a conversion key covers inputs of {covered} bits, asked for {k}");
            return Err(CKKSCompositionError::InvalidOperand { op, reason }.into());
        }
    }
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
    const OP: &str = EVAL_MOD_OP;
    ckks_ensure_operand!(
        OP,
        module.n() >= 2 * a.n().as_usize() && ci.n() == a.n().as_usize() && res.n() == a.n(),
        "takes a module of degree at least 2N, a CI module of degree N and ciphertexts of degree N"
    );
    let tensor_key = required_tensor_key(OP, keys.ci_tensor_key())?;
    let (trace_key, embed_key) = check_conversion_key_shapes(OP, a, keys)?;

    let boot_layout = GLWELayout {
        n: res.n(),
        base2k: a.base2k(),
        k: res.k(),
        rank: Rank(1),
    };
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
        crate::layouts::bootstrapping::CIPreForm::Folded { even, odd } => {
            ci_folded_pre(module, &mut embedded, a, [even, odd], keys.rotation_keys(), &mut scratch)?
        }
    }
    let traced_k = embedded.k().as_usize();
    let widths = (traced_k, traced_k.saturating_sub(plan.eval_mod_plan().consumed_bits()));
    check_conversion_key_widths(OP, trace_key, embed_key, widths)?;
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
    const OP: &str = "ckks_mod_raise";
    ensure_encapsulation_covers(dst, src, dense_to_sparse, sparse_to_dense)?;
    // The switch keeps the width of `src`: the raise's checks run before it.
    let (k_large, k_small) = (dst.k().as_usize(), src.k().as_usize());
    if sparse_to_dense.k_offset().as_usize() > 0 {
        check_offset_raise(OP, k_large, k_small, scale_up, sparse_to_dense)?;
    } else {
        ensure_mod_up_widens(OP, k_large, k_small, scale_up)?;
    }
    module.glwe_keyswitch_assign(src, &dense_to_sparse.to_backend_ref(), scratch);
    if sparse_to_dense.k_offset().as_usize() > 0 {
        return ckks_mod_up_offset_reference(module, dst, src, scale_up, sparse_to_dense, scratch);
    }
    // The lift is fused into ModUp, so the message is already at its final scale
    // when sparse-to-dense adds its noise.
    ckks_mod_up_scaled_reference(OP, module, dst, src, scale_up, scratch)?;
    module.glwe_keyswitch_assign(dst, &sparse_to_dense.to_backend_ref(), scratch);
    Ok(())
}

/// Bits of input a sparse-to-dense key with a row offset takes for a raise
/// into `dst_k` bits: those below its offset, which its rows must reach.
fn offset_key_window<K: GGLWEInfos>(dst_k: usize, key: &K) -> usize {
    dst_k.saturating_sub(key.k_offset().as_usize())
}

/// `InvalidOperand` for `op` unless the raise of a `k_small`-bit ciphertext lifted by `scale_up` into `k_large` bits
/// widens it.
fn ensure_mod_up_widens(op: &'static str, k_large: usize, k_small: usize, scale_up: usize) -> Result<()> {
    ckks_ensure_operand!(
        op,
        k_large >= k_small + scale_up,
        "dst.k ({k_large}) < src.k ({k_small}) + scale_up ({scale_up}); ModUp must widen, not shrink, the modulus"
    );
    Ok(())
}

/// Checks that a sparse-to-dense key with a row offset serves the raise of a
/// `k_small`-bit ciphertext lifted by `scale_up` into `k_large` bits, and
/// returns the raise's shift and the bits the key's window then holds.
pub(crate) fn check_offset_raise<K: GGLWEInfos>(
    op: &'static str,
    k_large: usize,
    k_small: usize,
    scale_up: usize,
    key: &K,
) -> Result<(usize, usize)> {
    ensure_mod_up_widens(op, k_large, k_small, scale_up)?;
    let shift = k_large - k_small - scale_up;
    let k_offset = key.k_offset().as_usize();
    ckks_ensure_operand!(
        op,
        k_offset <= shift,
        "the sparse-to-dense key lowers its input by {k_offset} bits, more than the raise's {shift}"
    );
    let window = offset_key_window(k_large, key) - scale_up;
    let max_input_k = key.gglwe_layout().max_input_k().as_usize();
    let reach = k_offset + key.gglwe_layout().gadget_k().as_usize();
    ckks_ensure_operand!(
        op,
        window <= max_input_k && k_large <= reach,
        "the sparse-to-dense key takes {max_input_k} bits of input and reaches bit {reach}; the raise needs {window} bits up to bit {k_large}"
    );
    Ok((shift, window))
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

/// Refuses encapsulation keys too narrow for their inputs before any write: `src`
/// for the dense-to-sparse switch, `dst` for an offset-free sparse-to-dense one. A
/// sparse-to-dense key with a row offset is checked against its raise, by
/// [`check_offset_raise`].
fn ensure_encapsulation_covers<Dst: CKKSCtBounds, Src: CKKSCtBounds, D2S: GGLWEInfos, S2D: GGLWEInfos>(
    dst: &Dst,
    src: &Src,
    dense_to_sparse: &D2S,
    sparse_to_dense: &S2D,
) -> Result<()> {
    let sparse_to_dense_covers =
        sparse_to_dense.k_offset().as_usize() > 0 || dst.k() <= sparse_to_dense.gglwe_layout().max_input_k();
    if src.k() > dense_to_sparse.gglwe_layout().max_input_k() || !sparse_to_dense_covers {
        return Err(CKKSCompositionError::InvalidOperand {
            op: "ckks_mod_raise",
            reason: "the input is wider than an encapsulation key's max_input_k".into(),
        }
        .into());
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

/// Reference bodies of the required [`CKKSBootstrappingCIImpl`](crate::oep::CKKSBootstrappingCIImpl) methods, which
/// [`impl_ckks_bootstrapping_ci_reference_full!`](crate::impl_ckks_bootstrapping_ci_reference_full) forwards to.
pub trait CKKSBootstrappingCIReference<BE: CKKSCITwin> {
    fn ckks_bootstrap_ci_tmp_bytes_reference<C1, C2, F>(
        &self,
        ci: &Module<BE::CI>,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContextPrepared<BE, F>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> (usize, usize)
    where
        Module<BE>: GLWEBytesOf<BE>
            + GLWELinearTransformationBabyStepsBytesOf<BE>
            + CKKSDFTOps<BE>
            + CKKSPow2Ops<BE>
            + GLWEShift<BE>
            + GLWEKeyswitch<BE>
            + GLWECITrace<BE>
            + CKKSMulOps<BE>
            + CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + CKKSCopyOps<BE>
            + CKKSLinearTransformationOps<BE>
            + GLWERotate<BE>,
        Module<BE::CI>: CKKSEvalModOps<BE::CI>,
        C1: CKKSCtBounds,
        C2: CKKSCtBounds;

    #[expect(
        clippy::too_many_arguments,
        reason = "the bootstrap takes a module and a scratch arena per ring"
    )]
    fn ckks_bootstrap_ci_reference<Dst, Src, F, K>(
        &self,
        ci: &Module<BE::CI>,
        ct_out: &mut Dst,
        ct_in: &Src,
        ctx: &BootstrappingContextPrepared<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
        scratch_ci: &mut ScratchArena<'_, BE::CI>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE>
            + GLWEShift<BE>
            + GLWEKeyswitch<BE>
            + GLWECITrace<BE>
            + GLWECIEmbed<BE>
            + GLWERotate<BE>
            + CKKSSubOps<BE>
            + CKKSLinearTransformationOps<BE>
            + CKKSCopyOps<BE>
            + CKKSPow2Ops<BE>
            + CKKSMulOps<BE>
            + CKKSAddOps<BE>
            + CKKSDFTOps<BE>
            + GLWESwitchDegree<BE>,
        Module<BE::CI>: CKKSEvalModOps<BE::CI>,
        K: BootstrappingKeys<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        CKKSCiphertextOwned<BE>:
            GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta;
}

impl<BE: Backend + CKKSBootstrappingImpl + CKKSEncapsulatedModUpImpl + CKKSModUpFirstFactorImpl> CKKSBootstrappingCIReference<BE>
    for Module<BE>
{
    fn ckks_bootstrap_ci_tmp_bytes_reference<C1, C2, F>(
        &self,
        ci: &Module<BE::CI>,
        ct_out: &C1,
        ct_in: &C2,
        ctx: &BootstrappingContextPrepared<BE, F>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> (usize, usize)
    where
        Module<BE>: GLWEBytesOf<BE>
            + GLWELinearTransformationBabyStepsBytesOf<BE>
            + CKKSDFTOps<BE>
            + CKKSPow2Ops<BE>
            + GLWEShift<BE>
            + GLWEKeyswitch<BE>
            + GLWECITrace<BE>
            + CKKSMulOps<BE>
            + CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + CKKSCopyOps<BE>
            + CKKSLinearTransformationOps<BE>
            + GLWERotate<BE>,
        Module<BE::CI>: CKKSEvalModOps<BE::CI>,
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
    {
        ckks_bootstrap_ci_tmp_bytes_reference(self, ci, ct_out, ct_in, ctx, keys_layout)
    }

    fn ckks_bootstrap_ci_reference<Dst, Src, F, K>(
        &self,
        ci: &Module<BE::CI>,
        ct_out: &mut Dst,
        ct_in: &Src,
        ctx: &BootstrappingContextPrepared<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
        scratch_ci: &mut ScratchArena<'_, BE::CI>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE>
            + GLWEShift<BE>
            + GLWEKeyswitch<BE>
            + GLWECITrace<BE>
            + GLWECIEmbed<BE>
            + GLWERotate<BE>
            + CKKSSubOps<BE>
            + CKKSLinearTransformationOps<BE>
            + CKKSCopyOps<BE>
            + CKKSPow2Ops<BE>
            + CKKSMulOps<BE>
            + CKKSAddOps<BE>
            + CKKSDFTOps<BE>
            + GLWESwitchDegree<BE>,
        Module<BE::CI>: CKKSEvalModOps<BE::CI>,
        K: BootstrappingKeys<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        CKKSCiphertextOwned<BE>:
            GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + BSGSMeta,
    {
        ckks_bootstrap_ci_reference(self, ci, ct_out, ct_in, ctx, keys, scratch, scratch_ci)
    }
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

fn ensure_unit_circle_exp_context<BE: Backend, F>(op: &'static str, ctx: &BootstrappingContextPrepared<BE, F>) -> Result<()> {
    ckks_ensure_operand!(
        op,
        ctx.eval_mod_plan().eval_mod_type == EvalModType::ExpCmplx,
        "general functional bootstrapping requires an ExpCmplx EvalMod context"
    );
    ckks_ensure_operand!(
        op,
        ctx.eval_mod_plan().scaling == Some(std::f64::consts::TAU),
        "general functional bootstrapping requires a unit-circle ExpCmplx context (scaling = 2π)"
    );
    Ok(())
}

fn functional_output_contract<BE: Backend, F>(
    op: &'static str,
    ct_in: &CKKSCiphertextOwned<BE>,
    ctx: &BootstrappingContextPrepared<BE, F>,
    lut: &EncodedLut<CKKSPlaintextOwned<BE>>,
    bootstrap_k: usize,
) -> Result<(GLWEPlaintextMeta, usize)> {
    let log_delta = ct_in
        .log_delta()
        .checked_add(lut.log_msg_ratio())
        .ok_or_else(|| crate::CKKSError::Internal(anyhow::anyhow!("functional bootstrap output scale overflow")))?;
    let eval_mod = if lut.requires_eval_mod() {
        ctx.eval_mod_plan().consumed_bits()
    } else {
        0
    };
    let consumed = ctx
        .coeffs_to_slots()
        .consumed_bits()
        .checked_add(ctx.c2s_guard_bits())
        .and_then(|bits| bits.checked_add(eval_mod))
        .and_then(|bits| bits.checked_add(lut.consumed_bits(log_delta)))
        .ok_or_else(|| crate::CKKSError::Internal(anyhow::anyhow!("functional bootstrap budget overflow")))?;
    ckks_ensure_operand!(
        op,
        bootstrap_k >= consumed,
        "functional bootstrap output width {bootstrap_k} is smaller than its {consumed}-bit post-ModUp cost"
    );
    let output_k = bootstrap_k - consumed;
    ckks_ensure_operand!(
        op,
        output_k >= log_delta,
        "functional bootstrap leaves output k={output_k} below log_delta={log_delta}"
    );
    Ok((
        GLWEPlaintextMeta {
            scale: Scale::Log(log_delta),
            slots: ct_in.slots(),
            log_sparsity: ct_in.log_sparsity(),
        },
        output_k,
    ))
}

#[expect(
    clippy::too_many_arguments,
    reason = "the LUT evaluation reads the rotation keys and the tensor key of the plan's ring"
)]
fn ckks_eval_encoded_lut<BE, F, K, T, C, R>(
    module: &Module<BE>,
    ct_out: &mut R,
    ct_in: &C,
    ctx: &BootstrappingContextPrepared<BE, F>,
    lut: &EncodedLut<CKKSPlaintextOwned<BE>>,
    keys: &K,
    tensor_key: &T,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
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
            ctx.standard_eval_mod("ckks_functional_bootstrap")?,
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
fn eval_lut_batch<BE, F, K, T, C>(
    module: &Module<BE>,
    outs: &mut [CKKSCiphertextOwned<BE>],
    ct: &C,
    ctx: &BootstrappingContextPrepared<BE, F>,
    luts: &[EncodedLut<CKKSPlaintextOwned<BE>>],
    keys: &K,
    tensor_key: &T,
    shared: bool,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
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
            ctx.standard_eval_mod("ckks_functional_bootstrap")?,
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
        ckks_eval_encoded_lut(module, out, ct, ctx, lut, keys, tensor_key, scratch)?;
    }
    Ok(())
}

fn ensure_functional_message_ratio<C>(op: &'static str, ct_in: &C, s2c_consumed_bits: usize, expected: usize) -> Result<()>
where
    C: CKKSInfos,
{
    let actual = checked_log_budget_sub(op, ct_in.log_budget(), s2c_consumed_bits)?;
    ckks_ensure_operand!(
        op,
        actual == expected,
        "functional bootstrap LUT requires log_msg_ratio {expected}, got {actual}"
    );
    Ok(())
}
