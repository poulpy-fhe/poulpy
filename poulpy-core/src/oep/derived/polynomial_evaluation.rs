//! Derived BSGS schedules over caller-supplied arithmetic policy operations.
//!
//! The schedule composes `GLWEBSGSOps` rather than HAL kernels. The policy owns
//! scheme-specific precision and normalization; default backend hooks preserve
//! that policy dispatch, including fused baby steps and prepared giant products.

use crate::{
    api::GLWEBSGSOps,
    layouts::{BabyStep, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, Parity, PowerBasisHelper},
};
use anyhow::{Result, ensure};
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena};

/// Evaluates a single baby step into `res`.
///
/// All arithmetic is delegated to the scheme via [`GLWEBSGSOps`]; the engine only
/// computes the parity schedule, seeds the accumulator from the *highest* power
/// (the lowest-budget operand, so every term writes at the final result width)
/// and sequences the terms.
pub(crate) fn glwe_eval_baby_step_derived<BE: Backend, Ops, V, P, G, A, Rg: Ring>(
    module: &Module<BE>,
    ops: &Ops,
    res: &mut V,
    parity: Parity,
    coeffs: &P,
    power_basis: &G,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    Ops: GLWEBSGSOps<BE, V, P, A, V, Rg>,
    V: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg>,
    P: GLWEToBackendRef<BE, Rg> + GLWEInfos,
    A: GLWEToBackendRef<BE, Rg>,
    G: PowerBasisHelper<BE, A, Rg>,
{
    let degree = coeffs.n().as_usize() - 1;

    let mut terms = Vec::new();
    for i in parity.baby_powers(degree) {
        terms.push((power_basis.get(i)?, i));
    }
    if !terms.is_empty() && ops.eval_baby_linear_combination(module, res, &terms, coeffs, scratch)? {
        if parity != Parity::Odd {
            ops.add_pt_const_assign(module, res, 0, coeffs, 0, scratch)?;
        }
        res.set_noise(None);
        return Ok(());
    }

    let seed_power = terms.last().map_or(1, |&(_, i)| i);
    ops.init_accumulator(module, res, power_basis.get(seed_power)?, scratch)?;

    if parity != Parity::Odd {
        ops.add_pt_const_assign(module, res, 0, coeffs, 0, scratch)?;
    }

    for (xpow, i) in &terms {
        ops.mul_add_pt_const(module, res, xpow, coeffs, *i, scratch)?;
    }

    res.set_noise(None);
    Ok(())
}

/// Folds the evaluated baby steps into `res` using the giant-step schedule.
///
/// The engine owns the schedule and the hoisting (`X^{gsp}` is prepared once per
/// level via [`GLWEBSGSOps::prepare_right`] and reused across the level's
/// sibling pairs); the per-pair `ct×ct`/`ct+ct` arithmetic and the final copy are
/// delegated to the scheme.
pub(crate) fn glwe_eval_giant_steps_derived<R, B, V, P, A, G, H, BE: Backend, Ops, Rg: Ring>(
    module: &Module<BE>,
    ops: &Ops,
    res: &mut R,
    baby_steps: &mut [B],
    power_basis: &G,
    tsk: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    Ops: GLWEBSGSOps<BE, V, P, A, R, Rg>,
    R: GLWEToBackendMut<BE, Rg>,
    B: BabyStep<BE, Rg, Value = V>,
    V: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg>,
    P: GLWEToBackendRef<BE, Rg>,
    A: GLWEToBackendRef<BE, Rg>,
    G: PowerBasisHelper<BE, A, Rg>,
    H: GetTensorKey<BE, Rg>,
{
    ensure!(
        !baby_steps.is_empty(),
        "eval_giant_steps: polynomial must contain at least one baby step"
    );

    let degrees: Vec<_> = baby_steps.iter().map(|s| s.degree()).collect();
    for run in crate::layouts::giant_runs(&degrees)? {
        let prepared = ops.prepare_right(module, power_basis.get(run.power)?, scratch)?;
        for (low, high) in run.pairs {
            // The planner emits active blocks in increasing index order and writes their
            // combination into the high block, preserving this order for the next level.
            let (left, right) = baby_steps.split_at_mut(high);
            let b = right[0].get_mut();
            ops.mul_prepared_assign(module, b, &prepared, tsk, scratch)?;
            ops.add_assign(module, b, left[low].get(), scratch)?;
        }
    }

    let evaluated = baby_steps.last().expect("non-empty baby step vector");
    ops.copy(module, res, evaluated.get(), scratch)?;
    res.set_noise(None);

    Ok(())
}
