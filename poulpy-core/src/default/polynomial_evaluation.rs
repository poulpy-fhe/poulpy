//! Scheme-agnostic Baby-Step / Giant-Step polynomial-evaluation engine.
//!
//! Owns the BSGS schedule, parity loop and giant-step folding **only** — the
//! combinatorial structure of the evaluation. It has no concept of scale: every
//! arithmetic operation (the `ct×pt` baby-step terms, the hoisted `ct×ct`
//! giant-step multiply, the `ct+ct` add, the accumulator seed and the final
//! copy) is supplied by the scheme through [`BSGSOps`], which owns all precision
//! bookkeeping and normalization.

use anyhow::{Result, ensure};
use poulpy_hal::{
    api::{
        CnvPVecBytesOf, Convolution, ModuleN, VecZnxAddAssignBackend, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopyBackend, VecZnxDftBytesOf, VecZnxIdftApplyTmpA, VecZnxNegateBackend,
        VecZnxSubAssignBackend,
    },
    layouts::{Backend, Module, ScratchArena},
};

use crate::{
    layouts::{
        BabyStep, GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, Parity, PowerBasisHelper,
        prepared::{GGLWEPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
    },
    oep::PolynomialEvaluationDefault,
};

/// HAL bounds required to run the hoisted prepared-right tensor product.
///
/// Retained as a convenience bundle for the **scheme** implementations of
/// [`BSGSOps::mul_prepared_assign`] (the engine itself no longer touches
/// these primitives).
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

pub trait BSGSOps<BE, V, P, A, R = V>
where
    BE: Backend,
    V: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    P: GLWEToBackendRef<BE>,
    A: GLWEToBackendRef<BE>,
    R: GLWEToBackendMut<BE>,
{
    /// Backend-resident prepared right multiply operand, reusable across a
    /// giant-step level.
    type Prepared;

    /// Initializes the accumulator `res` from `seed`'s precision: sets `res`'s
    /// metadata to that of `seed` and zeroes its data.
    fn init_accumulator(&self, module: &Module<BE>, res: &mut V, seed: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>;

    /// Computes `res[res_coeff] += coeffs[idx]`, normalizing `res`.
    fn add_pt_const_assign(
        &self,
        module: &Module<BE>,
        res: &mut V,
        res_coeff: usize,
        coeffs: &P,
        idx: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    /// Tries to evaluate `res = Σ coeffs[idx]·term` in one pass. Returning
    /// `false` leaves `res` untouched and selects the default operation sequence.
    fn eval_baby_linear_combination(
        &self,
        _module: &Module<BE>,
        _res: &mut V,
        _terms: &[(&A, usize)],
        _coeffs: &P,
        _scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<bool> {
        Ok(false)
    }

    /// Computes `res = a · coeffs[idx]` (ct × pt), setting `res`'s precision metadata.
    fn mul_pt_const(
        &self,
        module: &Module<BE>,
        res: &mut V,
        a: &A,
        coeffs: &P,
        idx: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    /// Computes `res += a · coeffs[idx]` (ct × pt), keeping `res` normalized.
    fn mul_add_pt_const(
        &self,
        module: &Module<BE>,
        res: &mut V,
        a: &A,
        coeffs: &P,
        idx: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    /// Prepares `a` as a reusable right operand for [`Self::mul_prepared_assign`].
    fn prepare_right(&self, module: &Module<BE>, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<Self::Prepared>;

    /// Computes `dst *= prepared` (ct × ct), relinearizing with `tsk` and
    /// stamping the consumed budget on the result.
    fn mul_prepared_assign<T>(
        &self,
        module: &Module<BE>,
        dst: &mut V,
        prepared: &Self::Prepared,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>;

    /// Computes `dst += a` with budget alignment, normalizing `dst`.
    fn add_assign(&self, module: &Module<BE>, dst: &mut V, a: &V, scratch: &mut ScratchArena<'_, BE>) -> Result<()>;

    /// Computes `res = src`, stamping `res` with `src`'s effective precision.
    fn copy(&self, module: &Module<BE>, res: &mut R, src: &V, scratch: &mut ScratchArena<'_, BE>) -> Result<()>;
}

/// Evaluates a single baby step into `res`.
///
/// All arithmetic is delegated to the scheme via [`BSGSOps`]; the engine only
/// computes the parity schedule, seeds the accumulator from the *highest* power
/// (the lowest-budget operand, so every term writes at the final result width)
/// and sequences the terms.
pub(crate) fn eval_baby_step<BE: Backend, Ops, V, P, G, A>(
    module: &Module<BE>,
    ops: &Ops,
    res: &mut V,
    parity: Parity,
    coeffs: &P,
    power_basis: &G,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    Ops: BSGSOps<BE, V, P, A, V>,
    V: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    P: GLWEToBackendRef<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE>,
    G: PowerBasisHelper<BE, A>,
{
    let degree = coeffs.n().as_usize() - 1;

    let (first, step) = match parity {
        Parity::Even => (2, 2),
        Parity::Odd => (1, 2),
        Parity::Full => (1, 1),
    };

    let init_power = (first..=degree).step_by(step).last().unwrap_or(1);

    let mut terms: Vec<(&A, usize)> = Vec::with_capacity(degree / step + 1);
    for i in (first..=degree).step_by(step) {
        terms.push((power_basis.get(i)?, i));
    }
    if !terms.is_empty() && ops.eval_baby_linear_combination(module, res, &terms, coeffs, scratch)? {
        if parity != Parity::Odd {
            ops.add_pt_const_assign(module, res, 0, coeffs, 0, scratch)?;
        }
        return Ok(());
    }

    ops.init_accumulator(module, res, power_basis.get(init_power)?, scratch)?;

    if parity != Parity::Odd {
        ops.add_pt_const_assign(module, res, 0, coeffs, 0, scratch)?;
    }

    for (xpow, i) in &terms {
        ops.mul_add_pt_const(module, res, xpow, coeffs, *i, scratch)?;
    }

    Ok(())
}

/// Folds the evaluated baby steps into `res` using the giant-step schedule.
///
/// The engine owns the schedule and the hoisting (`X^{gsp}` is prepared once per
/// level via [`BSGSOps::prepare_right`] and reused across the level's
/// sibling pairs); the per-pair `ct×ct`/`ct+ct` arithmetic and the final copy are
/// delegated to the scheme.
pub(crate) fn eval_giant_steps<R, B, V, P, A, G, T, BE: Backend, Ops>(
    module: &Module<BE>,
    ops: &Ops,
    res: &mut R,
    baby_steps: &mut [B],
    power_basis: &G,
    tsk: &T,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    Ops: BSGSOps<BE, V, P, A, R>,
    R: GLWEToBackendMut<BE>,
    B: BabyStep<BE, Value = V>,
    V: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    P: GLWEToBackendRef<BE>,
    A: GLWEToBackendRef<BE>,
    G: PowerBasisHelper<BE, A>,
    T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
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
        // consecutive pairs that share the same giant-step power: one
        // `prepare_right` per run, reused across every `mul_prepared_assign`.
        let mut p = 0;
        while p < pairs.len() {
            let gsp = pairs[p].0;
            let mut run_end = p + 1;
            while run_end < pairs.len() && pairs[run_end].0 == gsp {
                run_end += 1;
            }

            let prepared = ops.prepare_right(module, power_basis.get(gsp)?, scratch)?;
            for &(_, low_idx, high_idx) in &pairs[p..run_end] {
                ensure!(low_idx != high_idx, "eval_giant_steps: baby-step pair aliases itself");
                let (a, b) = if low_idx < high_idx {
                    let (low_steps, high_steps) = baby_steps.split_at_mut(high_idx);
                    (low_steps[low_idx].get(), high_steps[0].get_mut())
                } else {
                    let (high_steps, low_steps) = baby_steps.split_at_mut(low_idx);
                    (low_steps[0].get(), high_steps[high_idx].get_mut())
                };

                // `b·Xᵍˢᵖ` (ct×ct, the scheme stamps `b` with the consumed
                // budget); then `b += a`.
                ops.mul_prepared_assign(module, b, &prepared, tsk, scratch)?;
                ops.add_assign(module, b, a, scratch)?;
            }
            p = run_end;
        }

        active = next;
    }

    let evaluated = baby_steps.last().expect("non-empty baby step vector");
    ops.copy(module, res, evaluated.get(), scratch)?;

    Ok(())
}

fn giant_step_power(degree: usize) -> usize {
    (degree + 1).next_power_of_two()
}

impl<BE: Backend> PolynomialEvaluationDefault<BE> for Module<BE> {
    fn glwe_eval_baby_step_default<Ops, R, P, A, G>(
        &self,
        ops: &Ops,
        res: &mut R,
        parity: Parity,
        coeffs: &P,
        power_basis: &G,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Ops: BSGSOps<BE, R, P, A, R>,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
        P: GLWEToBackendRef<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE>,
        G: PowerBasisHelper<BE, A>,
    {
        eval_baby_step::<BE, Ops, R, P, G, A>(self, ops, res, parity, coeffs, power_basis, scratch)
    }

    fn glwe_eval_giant_steps_default<Ops, R, B, V, P, A, G, T>(
        &self,
        ops: &Ops,
        res: &mut R,
        baby_steps: &mut [B],
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Ops: BSGSOps<BE, V, P, A, R>,
        R: GLWEToBackendMut<BE>,
        B: BabyStep<BE, Value = V>,
        V: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
        P: GLWEToBackendRef<BE>,
        A: GLWEToBackendRef<BE>,
        G: PowerBasisHelper<BE, A>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        eval_giant_steps::<R, B, V, P, A, G, T, BE, Ops>(self, ops, res, baby_steps, power_basis, tsk, scratch)
    }
}
