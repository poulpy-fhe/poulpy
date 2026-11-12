//! EvalMod as a bootstrapping step: its routes on the standard ring and the choice between the two rings.

use poulpy_core::{
    GLWEBytesOf, GLWECIEmbed, GLWECITrace, GLWEKeyswitch, GLWELinearTransformationBabyStepsBytesOf, GLWERotate, GLWESwitchDegree,
    layouts::{GLWELayout, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, LWEInfos, Rank, SetBSGSMeta, SetK},
};
use poulpy_hal::{
    execution::{TaskExecutor, worker_scratch_bytes},
    layouts::{Backend, ConjugateInvariant, Module, ScratchArena, Standard},
};

use crate::{
    CKKSCtBounds, CKKSLayout, CKKSResult as Result, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSAllOpsTmpBytes, CKKSConjugateOps, CKKSEvalModOps, CKKSImagOps, CKKSLinearTransformationOps, CKKSMulOps,
        CKKSPow2Ops, CKKSSubOps,
    },
    checked_log_budget_sub, ckks_ensure, ckks_ensure_operand,
    layouts::{
        BootstrappingKeys, BootstrappingKeysLayout, CKKSEvalModInfos, CKKSPlaintextOwned, EvalMod,
        ScratchArenaTakeCKKSCiphertext,
        bootstrapping::bootstrap_layouts,
        eval_mod::EvalModPlan,
        validation::{
            check_conversion_key_widths, check_eval_mod_on_ci, ensure_eval_round_interval, ensure_one_base2k, ensure_one_degree,
            ensure_rank_one, eval_mod_standard, remove_guard_bits, required_tensor_key,
        },
    },
    oep::derived::dft::ckks_split_halves,
    reference::mul::tensor_key_at,
};

use super::{
    ckks_bootstrap_real_part,
    eval_mod_ci::{eval_mod_on_ci, eval_mod_on_ci_tmp_bytes},
    recombine_halves,
};

/// The name of [`ckks_bootstrap_eval_mod`](crate::api::CKKSBootstrappingOps::ckks_bootstrap_eval_mod) in its refusals.
const EVAL_MOD_OP: &str = "ckks_bootstrap_eval_mod";

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

/// Scratch of [`eval_mod_real`] and [`eval_mod_complex`], the larger of the two: the inputs of EvalMod and the imaginary
/// output beside EvalMod, on two worker arenas on a parallel executor, or beside the conjugation, the split or the
/// subtraction of EvalRound+.
fn eval_mod_standard_tmp_bytes<BE, R, A, P, F>(
    module: &Module<BE>,
    res: &R,
    a: &A,
    plan: &P,
    keys_layout: &BootstrappingKeysLayout,
) -> usize
where
    BE: Backend,
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
    let inputs = plan.eval_mod_standard().map(|eval_mod| {
        let (_, meta) = eval_mod.plan.work_layout(&a_layout);
        CKKSLayout {
            glwe_layout: eval_mod.plan.input_layout(&a_layout),
            meta,
        }
    });
    let eval_mod = plan.eval_mod_standard().zip(inputs.as_ref()).map_or(0, |(eval_mod, input)| {
        module
            .ckks_eval_mod_tmp_bytes(&boot_layout, input, eval_mod, tsk)
            .max(module.ckks_eval_mod_tmp_bytes(res, input, eval_mod, tsk))
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
    let input_bytes = inputs.map_or(0, |input| BE::scratch_aligned(module.glwe_bytes_of_from_infos(&input)));
    2 * input_bytes + boot_bytes + eval_mod.max(a_bytes + linear)
}

/// Backend-generic reference of EvalMod as a bootstrapping step, selected by
/// [`impl_ckks_bootstrapping_reference_full`](crate::oep::impl_ckks_bootstrapping_reference_full).
pub trait CKKSBootstrapEvalModReference<BE: Backend>: std::borrow::Borrow<Module<BE>> {
    /// Scratch upper bound for [`Self::ckks_bootstrap_eval_mod_reference`].
    fn ckks_bootstrap_eval_mod_tmp_bytes_reference<R, A, P, F>(
        &self,
        res: &R,
        a: &A,
        plan: &P,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        Module<BE>: GLWEBytesOf<BE>
            + GLWELinearTransformationBabyStepsBytesOf<BE>
            + CKKSAllOpsTmpBytes<BE>
            + CKKSEvalModOps<BE>
            + CKKSAddOps<BE>
            + CKKSMulOps<BE>
            + CKKSLinearTransformationOps<BE>
            + GLWEKeyswitch<BE>
            + GLWECITrace<BE>
            + GLWERotate<BE>,
        Module<BE>:
            CKKSEvalModOps<BE, ConjugateInvariant> + CKKSPow2Ops<BE, ConjugateInvariant> + CKKSSubOps<BE, ConjugateInvariant>,
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSEvalModInfos<BE, F>,
    {
        let module: &Module<BE> = self.borrow();
        match plan.eval_mod_ci() {
            Some(eval_mod_ci) => {
                let (boot_layout, _) = bootstrap_layouts(res, a);
                eval_mod_on_ci_tmp_bytes(module, &boot_layout, eval_mod_ci, plan.eval_round_plus(), keys_layout)
            }
            None => eval_mod_standard_tmp_bytes(module, res, a, plan, keys_layout),
        }
    }

    /// Reference body of [`ckks_bootstrap_eval_mod`](crate::api::CKKSBootstrappingOps::ckks_bootstrap_eval_mod).
    fn ckks_bootstrap_eval_mod_reference<R, A, P, F, K>(
        &self,
        res: &mut R,
        a: &A,
        plan: &P,
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
        Module<BE>:
            CKKSEvalModOps<BE, ConjugateInvariant> + CKKSPow2Ops<BE, ConjugateInvariant> + CKKSSubOps<BE, ConjugateInvariant>,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + Send,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: CKKSEvalModInfos<BE, F>,
        K: BootstrappingKeys<BE> + Sync,
        K::TensorKey: Sync,
    {
        let module: &Module<BE> = self.borrow();
        const OP: &str = EVAL_MOD_OP;
        ensure_rank_one(OP, a, res)?;
        ensure_one_degree(OP, module.n(), a, res)?;
        ensure_one_base2k(OP, a, res)?;
        remove_guard_bits(OP, a.log_delta(), plan.c2s_guard_bits())?;
        let (standard, on_ci) = match plan.eval_mod_ci() {
            Some(eval_mod_ci) => {
                let ci_keys = check_eval_mod_on_ci(OP, module, a, keys)?;
                ckks_ensure_operand!(OP, a.k() <= res.k(), "takes an input no wider than its output");
                let (tensor_key, trace_key, embed_key) = ci_keys;
                // The conversion to the conjugate-invariant ring consumes its scale before EvalMod, at the scale of `a`.
                let traced = a.k().as_usize().saturating_sub(eval_mod_ci.pre_consumed_bits());
                check_conversion_key_widths(OP, trace_key, embed_key, plan.eval_mod_plan().conversion_widths(traced))?;
                let widest = plan.eval_mod_plan().relinearization_k(traced.saturating_sub(a.log_delta()));
                tensor_key_at::<BE, _, ConjugateInvariant>(OP, tensor_key, widest.into())?;
                (None, Some((eval_mod_ci, ci_keys)))
            }
            None => {
                let eval_mod = eval_mod_standard(OP, plan)?;
                let tensor_key = required_tensor_key(OP, keys.tensor_key())?;
                let widest = plan.eval_mod_plan().relinearization_k(a.log_budget());
                tensor_key_at::<BE, _, Standard>(OP, tensor_key, widest.into())?;
                // EvalMod checks this too, past the conjugation.
                checked_log_budget_sub(OP, a.log_budget(), eval_mod.consumed_bits())?;
                (Some((eval_mod, tensor_key)), None)
            }
        };
        ensure_eval_round_interval(plan)?;
        if let Some(keys_layout) = keys.keys_layout() {
            let bytes = module.ckks_bootstrap_eval_mod_tmp_bytes_reference(res, a, plan, keys_layout);
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
        } else if let Some((eval_mod_ci, ci_keys)) = on_ci {
            eval_mod_on_ci(module, res, a, plan, eval_mod_ci, ci_keys, keys, scratch)?;
        }
        if !plan.eval_round_plus() {
            plan.stamp_eval_mod_output(OP, res)?;
        }
        Ok(())
    }
}

impl<BE: Backend> CKKSBootstrapEvalModReference<BE> for Module<BE> {}

/// `eval_mod` on `r0` into `res_real` and on `i0` into `res_imag`, on a parallel executor at once.
#[allow(clippy::too_many_arguments)]
fn ckks_bootstrap_eval_mod_halves<BE, F, T, C, R1, R2>(
    module: &Module<BE>,
    r0: &mut C,
    i0: &mut C,
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
    C: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + Send,
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

/// One EvalMod on the real part of `a`, on the real slots of an S2C-first plan without a bypass.
fn eval_mod_real<BE, F, R, A, T, K>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    eval_mod: &EvalMod<F, CKKSPlaintextOwned<BE>>,
    tensor_key: &T,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSConjugateOps<BE> + CKKSAddOps<BE> + CKKSEvalModOps<BE>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    T: GetTensorKey<BE>,
    K: BootstrappingKeys<BE>,
{
    let layout = eval_mod.plan.input_layout(a);
    let (mut input, mut scratch) = scratch.borrow().take_ckks_ciphertext_scratch(&layout, a.meta());
    input.set_k(a.k());
    ckks_bootstrap_real_part(module, &mut input, a, keys, &mut scratch)?;
    module.ckks_eval_mod(res, &mut input, eval_mod, tensor_key, &mut scratch)
}

/// The split of `a`, one EvalMod per half and their recombination, from which EvalRound+ takes `K·(re + i·im) = 2K·a`.
#[expect(
    clippy::too_many_arguments,
    reason = "the route reads the plan beside the EvalMod and the tensor key its entry resolved"
)]
fn eval_mod_complex<BE, F, R, A, P, T, K>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    plan: &P,
    eval_mod: &EvalMod<F, CKKSPlaintextOwned<BE>>,
    tensor_key: &T,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    F: Sync,
    Module<BE>: CKKSConjugateOps<BE> + CKKSAddOps<BE> + CKKSSubOps<BE> + CKKSImagOps<BE> + CKKSPow2Ops<BE> + CKKSEvalModOps<BE>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + Send,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: CKKSEvalModInfos<BE, F>,
    T: GetTensorKey<BE> + Sync,
    K: BootstrappingKeys<BE>,
{
    let layout = eval_mod.plan.input_layout(a);
    let (mut r0, scratch) = scratch.borrow().take_ckks_ciphertext_scratch(&layout, a.meta());
    let (mut i0, scratch) = scratch.take_ckks_ciphertext_scratch(&layout, a.meta());
    let (mut imag, mut scratch) = scratch.take_ckks_ciphertext_scratch(res, res.meta());
    r0.set_k(a.k());
    i0.set_k(a.k());
    ckks_split_halves(module, &mut r0, &mut i0, a, keys, &mut scratch)?;
    ckks_bootstrap_eval_mod_halves(module, &mut r0, &mut i0, res, &mut imag, eval_mod, tensor_key, &mut scratch)?;
    recombine_halves(module, res, &mut imag, &mut scratch)?;
    if plan.eval_round_plus() {
        let bits = 1 + plan.eval_mod_plan().f_mod_interval.trailing_zeros() as usize;
        let (mut scaled, mut scratch) = scratch.borrow().take_ckks_ciphertext_scratch(a, a.meta());
        module.ckks_mul_pow2_into(bits, &mut scaled, a, &mut scratch)?;
        module.ckks_sub_assign(res, &scaled, &mut scratch)?;
    }
    Ok(())
}
