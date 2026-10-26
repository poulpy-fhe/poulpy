use crate::layouts::GetTensorKey;
use anyhow::Result;

use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    api::BSGSOps,
    layouts::{BabyStep, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, Parity, PowerBasisHelper},
};

/// Backend-provided Baby-Step / Giant-Step polynomial-evaluation phases.
///
/// The default phases are derived schedules over the caller's [`BSGSOps`]
/// policy. They preserve that policy's dispatch and do not implement HAL kernels.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait PolynomialEvaluationImpl: Backend {
    /// Default body: `glwe_eval_baby_step_derived`, the parity schedule of one baby step over the caller's
    /// [`BSGSOps`], seeding the accumulator from the highest power.
    #[allow(clippy::too_many_arguments)]
    fn glwe_eval_baby_step<Ops, V, P, A, G>(
        module: &Module<Self>,
        ops: &Ops,
        res: &mut V,
        parity: Parity,
        coeffs: &P,
        power_basis: &G,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Ops: BSGSOps<Self, V, P, A, V>,
        V: GLWEToBackendMut<Self> + GLWEToBackendRef<Self>,
        P: GLWEToBackendRef<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self>,
        G: PowerBasisHelper<Self, A>,
    {
        crate::oep::derived::polynomial_evaluation::glwe_eval_baby_step_derived::<Self, Ops, V, P, G, A>(
            module,
            ops,
            res,
            parity,
            coeffs,
            power_basis,
            scratch,
        )
    }

    /// Default body: `glwe_eval_giant_steps_derived`, the giant-step schedule over the caller's [`BSGSOps`],
    /// preparing each level's giant-step power once and reusing it across the level's pairs, under any
    /// [`PrepareHint`](poulpy_hal::layouts::PrepareHint) of its prepared operands.
    #[allow(clippy::too_many_arguments)]
    fn glwe_eval_giant_steps<Ops, R, B, V, P, A, G, H>(
        module: &Module<Self>,
        ops: &Ops,
        res: &mut R,
        baby_steps: &mut [B],
        power_basis: &G,
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Ops: BSGSOps<Self, V, P, A, R>,
        R: GLWEToBackendMut<Self>,
        B: BabyStep<Self, Value = V>,
        V: GLWEToBackendMut<Self> + GLWEToBackendRef<Self>,
        P: GLWEToBackendRef<Self>,
        A: GLWEToBackendRef<Self>,
        G: PowerBasisHelper<Self, A>,
        H: GetTensorKey<Self>,
    {
        crate::oep::derived::polynomial_evaluation::glwe_eval_giant_steps_derived::<R, B, V, P, A, G, H, Self, Ops>(
            module,
            ops,
            res,
            baby_steps,
            power_basis,
            tsk,
            scratch,
        )
    }
}

/// Selects the derived BSGS schedules while preserving the caller's arithmetic policy.
#[macro_export]
macro_rules! impl_polynomial_evaluation_derived_full {
    ($be:ty) => {
        unsafe impl $crate::oep::PolynomialEvaluationImpl for $be {}
    };
}
