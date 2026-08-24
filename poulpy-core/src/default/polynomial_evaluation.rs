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

/// One [`BSGSOps::mul_prepared_assign`] in a dependency-frontier batch.
pub struct PreparedMulAssignItem<D, P> {
    pub dst: D,
    pub prepared: P,
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

    /// Computes `res += Σ a·coeffs[idx]` over `terms`, in order, each step
    /// identical to [`Self::mul_add_pt_const`].
    ///
    /// One ordered batch boundary, not a dot product: every term keeps its own
    /// convolution offset, rounding, budget alignment and carry normalization,
    /// and `res`'s metadata evolves term by term. A backend may fuse the steps
    /// but not reassociate them. An empty slice is a no-op; a singleton is one
    /// [`Self::mul_add_pt_const`].
    fn mul_add_pt_consts(
        &self,
        module: &Module<BE>,
        res: &mut V,
        terms: &[(&A, usize)],
        coeffs: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()> {
        for &(a, idx) in terms {
            self.mul_add_pt_const(module, res, a, coeffs, idx, scratch)?;
        }
        Ok(())
    }

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

    /// Independent [`Self::mul_prepared_assign`] calls, in item order.
    ///
    /// The giant-step engine hands over the whole ready frontier of a level, so
    /// a scheme can dispatch one batch instead of a call per pair. Destinations
    /// are distinct (`&mut`); the same prepared operand may repeat, which is the
    /// common case when a level's sibling pairs share `X^{gsp}`.
    fn mul_prepared_assign_batch<T>(
        &self,
        module: &Module<BE>,
        items: &mut [PreparedMulAssignItem<&mut V, &Self::Prepared>],
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        for item in items.iter_mut() {
            self.mul_prepared_assign(module, &mut *item.dst, item.prepared, tsk, scratch)?;
        }
        Ok(())
    }

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
    ops.init_accumulator(module, res, power_basis.get(init_power)?, scratch)?;

    if parity != Parity::Odd {
        ops.add_pt_const_assign(module, res, 0, coeffs, 0, scratch)?;
    }

    let terms: Vec<(&A, usize)> = (first..=degree)
        .step_by(step)
        .map(|i| power_basis.get(i).map(|xpow| (xpow, i)))
        .collect::<Result<_>>()?;
    ops.mul_add_pt_consts(module, res, &terms, coeffs, scratch)?;

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

    let degrees: Vec<usize> = baby_steps.iter().map(|step| step.degree()).collect();

    for pairs in giant_step_schedule(&degrees) {
        for pair in &pairs {
            ensure!(pair.low != pair.high, "eval_giant_steps: baby-step pair aliases itself");
        }

        // The level's pairs are mutually independent: hoist each distinct
        // `X^{gsp}` once, hand the whole ready frontier to the scheme as one
        // batch, then fold the `+= a` tail in pair order.
        let mut prepared: Vec<(usize, Ops::Prepared)> = Vec::new();
        for pair in &pairs {
            if !prepared.iter().any(|(g, _)| *g == pair.gsp) {
                prepared.push((pair.gsp, ops.prepare_right(module, power_basis.get(pair.gsp)?, scratch)?));
            }
        }

        // Each baby step is the high operand of at most one pair, so the
        // destinations are distinct; `rank` restores pair order after the
        // index-ordered `iter_mut`.
        let mut rank: Vec<usize> = vec![usize::MAX; baby_steps.len()];
        for (position, pair) in pairs.iter().enumerate() {
            rank[pair.high] = position;
        }
        let mut frontier: Vec<(usize, PreparedMulAssignItem<&mut V, &Ops::Prepared>)> = baby_steps
            .iter_mut()
            .enumerate()
            .filter(|(index, _)| rank[*index] != usize::MAX)
            .map(|(index, step)| {
                let position = rank[index];
                let gsp = pairs[position].gsp;
                let prep = &prepared
                    .iter()
                    .find(|(g, _)| *g == gsp)
                    .expect("every pair's giant-step power was prepared")
                    .1;
                (
                    position,
                    PreparedMulAssignItem {
                        dst: step.get_mut(),
                        prepared: prep,
                    },
                )
            })
            .collect();
        frontier.sort_by_key(|(position, _)| *position);
        let mut frontier: Vec<PreparedMulAssignItem<&mut V, &Ops::Prepared>> =
            frontier.into_iter().map(|(_, item)| item).collect();
        // `b·Xᵍˢᵖ` (ct×ct, the scheme stamps `b` with the consumed budget).
        ops.mul_prepared_assign_batch(module, &mut frontier, tsk, scratch)?;
        drop(frontier);

        for pair in &pairs {
            let (a, b) = if pair.low < pair.high {
                let (low_steps, high_steps) = baby_steps.split_at_mut(pair.high);
                (low_steps[pair.low].get(), high_steps[0].get_mut())
            } else {
                let (high_steps, low_steps) = baby_steps.split_at_mut(pair.low);
                (low_steps[0].get(), high_steps[pair.high].get_mut())
            };
            ops.add_assign(module, b, a, scratch)?;
        }
    }

    let evaluated = baby_steps.last().expect("non-empty baby step vector");
    ops.copy(module, res, evaluated.get(), scratch)?;

    Ok(())
}

fn giant_step_power(degree: usize) -> usize {
    (degree + 1).next_power_of_two()
}

/// One giant-step fold: `baby[high] = baby[high]·X^gsp + baby[low]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GiantStepPair {
    pub gsp: usize,
    pub low: usize,
    pub high: usize,
}

/// Giant-step fold schedule for baby steps of the given degrees: one pair list
/// per level, outermost first. The pairs within a level are independent, which
/// is what lets the engine dispatch a level as one batch; the result ends up in
/// the last baby step.
///
/// Shared with the lockstep EvalMod driver, which merges the levels of two
/// branches into one frontier, so the schedule has exactly one definition.
pub fn giant_step_schedule(degrees: &[usize]) -> Vec<Vec<GiantStepPair>> {
    let mut active: Vec<(usize, usize)> = degrees.iter().copied().enumerate().map(|(i, d)| (d, i)).collect();
    let mut levels: Vec<Vec<GiantStepPair>> = Vec::new();

    while active.len() > 1 {
        let mut next = Vec::with_capacity(active.len().div_ceil(2));
        let mut pairs: Vec<GiantStepPair> = Vec::with_capacity(active.len() / 2);
        let mut i = 0;
        while i < active.len() {
            let is_last = i + 1 == active.len();
            if !is_last && active[i].0 == active[i + 1].0 {
                let gsp = giant_step_power(active[i].0);
                pairs.push(GiantStepPair {
                    gsp,
                    low: active[i].1,
                    high: active[i + 1].1,
                });
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
        levels.push(pairs);
        active = next;
    }

    levels
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
