//! Scheme-agnostic Baby-Step / Giant-Step polynomial-evaluation engine.
//!
//! Owns the BSGS schedule, parity loop and giant-step folding, composing core
//! GLWE primitives. Per-operation precision integers and the
//! plaintext-coefficient addition are supplied by the scheme through
//! [`BSGSPrecision`].

use anyhow::{Result, ensure};
use poulpy_hal::{
    api::{
        CnvPVecBytesOf, Convolution, ModuleN, ScratchArenaTakeBasic, VecZnxAddAssignBackend, VecZnxBigBytesOf,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopyBackend, VecZnxDftBytesOf, VecZnxIdftApplyTmpA,
        VecZnxNegateBackend, VecZnxSubAssignBackend,
    },
    layouts::{Backend, ScratchArena},
};

use poulpy_hal::layouts::Module;

use crate::{
    GLWEAdd, GLWECopy, GLWENormalize, GLWEShift, GLWETensoring, GLWEZero, ScratchArenaTakeCore,
    default::operations::{glwe_prepare_right, glwe_tensor_apply_prepared_right},
    layouts::{
        BSGSMeta, BabyStep, GGLWEInfos, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Parity,
        PowerBasisHelper, SetBSGSMeta, prepared::GLWETensorKeyPreparedToBackendRef,
    },
    oep::PolynomialEvaluationDefault,
};

/// HAL bounds required to run the hoisted prepared-right tensor product.
pub trait GiantStepTensorBounds<BE: Backend>:
    Sized
    + ModuleN
    + CnvPVecBytesOf
    + VecZnxDftBytesOf
    + VecZnxBigBytesOf
    + VecZnxIdftApplyTmpA<BE>
    + VecZnxBigNormalize<BE>
    + Convolution<BE>
    + VecZnxSubAssignBackend<BE>
    + VecZnxAddAssignBackend<BE>
    + VecZnxBigNormalizeTmpBytes
    + VecZnxCopyBackend<BE>
    + VecZnxNegateBackend<BE>
{
}

impl<BE: Backend, M> GiantStepTensorBounds<BE> for M where
    M: Sized
        + ModuleN
        + CnvPVecBytesOf
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + Convolution<BE>
        + VecZnxSubAssignBackend<BE>
        + VecZnxAddAssignBackend<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxCopyBackend<BE>
        + VecZnxNegateBackend<BE>
{
}

/// Scheme-supplied per-operation precision integers for the engine-owned
/// giant-step `ct × ct` multiply.
pub trait BSGSPrecision<BE: Backend> {
    /// Returns `(log_budget, log_delta, cnv_offset)` for `res = a * b` (ct × ct).
    fn mul_ct_params<R, A, B>(&self, res: &R, a: &A, b: &B) -> Result<(usize, usize, usize)>
    where
        R: GLWEInfos + BSGSMeta,
        A: GLWEInfos + BSGSMeta,
        B: GLWEInfos + BSGSMeta;
}

/// Scheme-supplied baby-step coefficient operations.
///
/// The engine only sequences these calls; the scheme owns the scratch buffers,
/// precision bookkeeping, and normalization. This keeps the BSGS engine free of
/// any local ciphertext type — it operates purely through scheme-provided values
/// implementing the [`BSGSMeta`]/[`SetBSGSMeta`] traits.
///
/// `R` is the accumulator (baby-step value), `P` the encoded coefficients, and
/// `A` the power-basis entry type. `A` is a trait parameter (not a method generic)
/// so the scheme can constrain it to its own ciphertext bounds in the impl.
pub trait BSGSCoeffOps<BE: Backend, R, P, A> {
    /// Computes `res[res_coeff] += coeffs[idx]`, normalizing `res`.
    fn add_pt_const_assign(
        &self,
        module: &Module<BE>,
        res: &mut R,
        res_coeff: usize,
        coeffs: &P,
        idx: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    /// Computes `res = a · coeffs[idx]` (ct × pt), setting `res`'s precision metadata.
    fn mul_pt_const(
        &self,
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        coeffs: &P,
        idx: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    /// Computes `res += a · coeffs[idx]` (ct × pt), keeping `res` normalized.
    fn mul_add_pt_const(
        &self,
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        coeffs: &P,
        idx: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;
}

/// Evaluates a single baby step into `res`.
///
/// The `ct × pt` term products are delegated to the scheme via [`BSGSCoeffOps`]
/// (which owns their scratch and normalization); the engine only zeroes `res`,
/// sequences the terms, and compacts at the boundaries.
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_baby_step<PR, R, C, A, G, BE: Backend>(
    module: &Module<BE>,
    precision: &PR,
    res: &mut R,
    parity: Parity,
    coeffs: &C,
    power_basis: &G,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    Module<BE>: GLWEZero<BE>,
    PR: BSGSCoeffOps<BE, R, C, A>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta,
    C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
    A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
    G: PowerBasisHelper<BE, A>,
{
    let degree = coeffs.n().as_usize() - 1;

    let (first, step) = match parity {
        Parity::Even => (2, 2),
        Parity::Odd => (1, 2),
        Parity::Full => (1, 1),
    };

    // Seed `res` from the *highest* power this baby step uses (the lowest-budget
    // operand): the inner product `Σ cᵢ·xⁱ` is bounded by that term's budget, so
    // every `ct×pt` writes (and accumulates) at the final compact limb count
    // rather than starting at `x¹`'s width and only compacting at the end. A
    // constant-only baby step (no power term) falls back to `x¹`.
    let init_power = (first..=degree).step_by(step).last().unwrap_or(1);
    let x = power_basis.get(init_power)?;
    res.set_bsgs_log_budget(x.bsgs_log_budget());
    res.set_bsgs_log_delta(x.bsgs_log_delta());
    res.compact_in_place();
    module.glwe_zero(res);

    let mut has_value = false;
    if parity != Parity::Odd {
        precision.add_pt_const_assign(module, res, 0, coeffs, 0, scratch)?;
        has_value = true;
    }

    for i in (first..=degree).step_by(step) {
        let xpow = power_basis.get(i)?;
        if has_value {
            precision.mul_add_pt_const(module, res, xpow, coeffs, i, scratch)?;
        } else {
            precision.mul_pt_const(module, res, xpow, coeffs, i, scratch)?;
            has_value = true;
        }
    }

    res.compact_in_place();

    Ok(())
}

/// Computes `res += a` with budget alignment, without normalizing `res`.
fn add_assign_unnormalized<M, R, A, BE: Backend>(module: &M, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>)
where
    M: GLWEAdd<BE> + GLWEShift<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos + SetBSGSMeta,
    A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
{
    let res_log_budget = res.bsgs_log_budget();
    let a_log_budget = a.bsgs_log_budget();

    if res_log_budget < a_log_budget {
        module.glwe_lsh_add(res, a, a_log_budget - res_log_budget, scratch);
    } else if res_log_budget > a_log_budget {
        module.glwe_lsh_assign(res, res_log_budget - a_log_budget, scratch);
        module.glwe_add_assign(res, a);
    } else {
        module.glwe_add_assign(res, a);
    }

    res.set_bsgs_log_budget(res_log_budget.min(a_log_budget));
    res.set_bsgs_log_delta(res.bsgs_log_delta().min(a.bsgs_log_delta()));
}

/// Folds the evaluated baby steps into `res` using the giant-step schedule.
pub(crate) fn eval_giant_steps<M, R, B, A, G, T, BE: Backend>(
    module: &M,
    precision: &impl BSGSPrecision<BE>,
    res: &mut R,
    baby_steps: &mut [B],
    power_basis: &G,
    tsk: &T,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    M: GiantStepTensorBounds<BE> + GLWEAdd<BE> + GLWEShift<BE> + GLWETensoring<BE> + GLWENormalize<BE> + crate::GLWECopy<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos + SetBSGSMeta,
    B: BabyStep<BE>,
    A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
    G: PowerBasisHelper<BE, A>,
    T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
{
    ensure!(
        !baby_steps.is_empty(),
        "eval_giant_steps: polynomial must contain at least one baby step"
    );

    let mut active: Vec<(usize, usize)> = baby_steps
        .iter()
        .enumerate()
        .map(|(index, step)| (step.degree(), index))
        .collect();

    while active.len() > 1 {
        let mut next = Vec::with_capacity(active.len().div_ceil(2));
        let mut pairs: Vec<(usize, usize, usize)> = Vec::with_capacity(active.len() / 2);
        let mut i = 0;
        while i < active.len() {
            let is_last = i + 1 == active.len();
            if !is_last && active[i].0 == active[i + 1].0 {
                let gsp = giant_step_power(active[i].0);
                pairs.push((gsp, active[i].1, active[i + 1].1));
                next.push((2 * gsp - 1, active[i + 1].1));
                i += 2;
            } else if is_last && i > 0 {
                let degree = next.last().map(|(degree, _)| *degree).unwrap_or(active[i].0);
                next.push((degree, active[i].1));
                i += 1;
            } else {
                next.push(active[i]);
                i += 1;
            }
        }

        // Process pairs left-to-right, hoisting the prepared `X^{gsp}` across
        // consecutive pairs that share the same giant-step power.
        let mut p = 0;
        while p < pairs.len() {
            let gsp = pairs[p].0;
            let mut run_end = p + 1;
            while run_end < pairs.len() && pairs[run_end].0 == gsp {
                run_end += 1;
            }
            eval_monomial_run(
                module,
                precision,
                baby_steps,
                &pairs[p..run_end],
                power_basis.get(gsp)?,
                tsk,
                scratch,
            )?;
            p = run_end;
        }

        active = next;
    }

    let evaluated = baby_steps.last().expect("non-empty baby step vector");
    module.glwe_copy(res, evaluated.get());
    res.set_bsgs_log_budget(evaluated.get().bsgs_log_budget());
    res.set_bsgs_log_delta(evaluated.get().bsgs_log_delta());
    // Return a compacted result: `res` is the caller's (full-width) buffer, but the
    // evaluated value only spans `effective_k`. Compacting here means consumers
    // (e.g. the EvalMod squarings, which require a compacted square input) get a
    // tight ciphertext without an extra copy.
    res.compact_in_place();

    Ok(())
}

/// Scratch bytes consumed by the giant-step engine on top of the per-pair
/// mul/add scratch: the prepared hoisted `X^{gsp}` right operand, kept alive
/// across a run. The per-pair operands are no longer copied into compact scratch
/// — `b` is compacted in place and `a` is read directly — so this is just the
/// hoisted right operand. `X^{gsp}` is prepared straight from the power basis, as
/// `glwe_prepare_right` reads only its top effective-precision limbs.
pub fn glwe_eval_giant_steps_extra_tmp_bytes(hoisted_right_bytes: usize) -> usize {
    hoisted_right_bytes
}

/// Evaluates a run of `b = b * xpow + a` pairs that share the same `xpow`.
///
/// The compacted `xpow` is prepared once into a scratch `CnvPVecR` and reused as
/// the right tensor operand across every pair in the run.
#[allow(clippy::too_many_arguments)]
fn eval_monomial_run<M, B, A, T, BE: Backend>(
    module: &M,
    precision: &impl BSGSPrecision<BE>,
    baby_steps: &mut [B],
    pairs: &[(usize, usize, usize)],
    xpow: &A,
    tsk: &T,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    M: GiantStepTensorBounds<BE> + GLWEAdd<BE> + GLWEShift<BE> + GLWETensoring<BE> + GLWENormalize<BE> + crate::GLWECopy<BE>,
    B: BabyStep<BE>,
    A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
    T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
{
    scratch.scope(|run_scratch| {
        // Hoist: prepare `xpow` into a reusable right operand. `glwe_prepare_right`
        // reads only the top `xpow_size` (effective_k) limbs, so the operand does not
        // need to be pre-compacted into its own scratch buffer.
        let cols = xpow.rank().as_usize() + 1;
        let xpow_effective_k = xpow.bsgs_effective_k();
        let xpow_size = xpow_effective_k.div_ceil(xpow.base2k().as_usize());

        let (mut xpow_prep, mut run_scratch) = run_scratch.take_cnv_pvec_right_scratch(module, cols, xpow_size);
        run_scratch = run_scratch.apply_mut(|scratch_prep| {
            glwe_prepare_right(module, &mut xpow_prep, xpow, xpow_effective_k, scratch_prep);
        });

        for &(_, low_idx, high_idx) in pairs {
            ensure!(low_idx != high_idx, "eval_giant_steps: baby-step pair aliases itself");
            run_scratch.scope(|mut pair_scratch| {
                let (a, b) = if low_idx < high_idx {
                    let (low_steps, high_steps) = baby_steps.split_at_mut(high_idx);
                    (low_steps[low_idx].get(), high_steps[0].get_mut())
                } else {
                    let (high_steps, low_steps) = baby_steps.split_at_mut(low_idx);
                    (low_steps[0].get(), high_steps[high_idx].get_mut())
                };

                // `b·Xᵍˢᵖ` (ct×ct) compacts `b` to the consumed budget in place
                // (see `mul_assign_prepared`); `a` is read at its `effective_k` by
                // the add (the MSB prefix), so it needs no compaction. The
                // incoming `b` is already compact — a baby step (compacted by
                // `eval_baby_step`) or a prior round's compacted result.
                mul_assign_prepared(module, precision, b, xpow, &xpow_prep, xpow_size, tsk, &mut pair_scratch)?;
                add_assign(module, b, a, &mut pair_scratch);
                Result::<()>::Ok(())
            })?;
        }
        Ok(())
    })
}

/// Computes `dst *= a` (ct × ct) reusing the caller-prepared right operand `a_prep`.
///
/// `a_prep` is the prepared `CnvPVecR` of `a` and `a_size` its limb count, so the
/// tensor product feeds `mul_ct_params` the same operand metadata as `a`.
#[allow(clippy::too_many_arguments)]
fn mul_assign_prepared<M, V, A, AP, T, BE: Backend>(
    module: &M,
    precision: &impl BSGSPrecision<BE>,
    dst: &mut V,
    a: &A,
    a_prep: &AP,
    a_size: usize,
    tsk: &T,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    M: GiantStepTensorBounds<BE> + GLWETensoring<BE>,
    V: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta,
    A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
    AP: poulpy_hal::layouts::CnvPVecRToBackendRef<BE>,
    T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
{
    let (log_budget, log_delta, cnv_offset) = precision.mul_ct_params(dst, dst, a)?;

    // Size from `a`'s effective_k rather than its full `max_k`: the tensor product only
    // consumes the top effective_k limbs of `a` (via the prepared right operand), so a
    // non-compacted `a` must not inflate the intermediate buffer.
    let tensor_layout = GLWELayout {
        n: dst.n(),
        base2k: dst.base2k(),
        k: dst.max_k().max(a.bsgs_effective_k().into()),
        rank: dst.rank(),
    };
    let scratch_local = scratch.borrow();
    let (mut tmp, mut scratch_local) = scratch_local.take_glwe_tensor_scratch(&tensor_layout);
    let dst_effective_k = dst.bsgs_effective_k();
    glwe_tensor_apply_prepared_right(
        module,
        cnv_offset,
        &mut tmp,
        &*dst,
        dst_effective_k,
        a_prep,
        a_size,
        &mut scratch_local,
    );
    module.glwe_tensor_relinearize(dst, &tmp, tsk, tmp.size() + tsk.dsize().as_usize(), &mut scratch_local);

    dst.set_bsgs_log_budget(log_budget);
    dst.set_bsgs_log_delta(log_delta);
    // Compact the ct×ct result to the consumed budget: the relinearize wrote it
    // at the (larger) operand storage, so dropping the now sub-precision low limbs
    // keeps the next round's multiply — and the trailing copy into `res` — tight.
    dst.compact_in_place();
    Ok(())
}

/// Computes `dst += a` with budget alignment, normalizing `dst`.
fn add_assign<M, V, A, BE: Backend>(module: &M, dst: &mut V, a: &A, scratch: &mut ScratchArena<'_, BE>)
where
    M: GLWEAdd<BE> + GLWEShift<BE> + GLWENormalize<BE>,
    V: GLWEToBackendMut<BE> + GLWEInfos + SetBSGSMeta,
    A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
{
    add_assign_unnormalized(module, dst, a, scratch);
    module.glwe_normalize_assign(dst, scratch);
}

fn giant_step_power(degree: usize) -> usize {
    (degree + 1).next_power_of_two()
}

impl<BE: Backend> PolynomialEvaluationDefault<BE> for Module<BE> {
    fn glwe_eval_baby_step_default<PR, R, C, A, G>(
        &self,
        precision: &PR,
        res: &mut R,
        parity: Parity,
        coeffs: &C,
        power_basis: &G,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEZero<BE> + Sized,
        PR: BSGSCoeffOps<BE, R, C, A>,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
        A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
        G: PowerBasisHelper<BE, A>,
    {
        eval_baby_step::<PR, R, C, A, G, BE>(self, precision, res, parity, coeffs, power_basis, scratch)
    }

    fn glwe_eval_giant_steps_default<PR, R, B, A, G, T>(
        &self,
        precision: &PR,
        res: &mut R,
        baby_steps: &mut [B],
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GiantStepTensorBounds<BE>
            + GLWEAdd<BE>
            + GLWEShift<BE>
            + GLWETensoring<BE>
            + GLWENormalize<BE>
            + GLWECopy<BE>
            + Sized,
        PR: BSGSPrecision<BE>,
        R: GLWEToBackendMut<BE> + GLWEInfos + SetBSGSMeta,
        B: BabyStep<BE>,
        A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    {
        eval_giant_steps::<_, R, B, A, G, T, BE>(self, precision, res, baby_steps, power_basis, tsk, scratch)
    }
}
