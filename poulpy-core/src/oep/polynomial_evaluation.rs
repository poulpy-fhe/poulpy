use crate::layouts::GetTensorKey;
use anyhow::Result;

use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    layouts::{BabyStep, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, Parity, PowerBasisHelper},
    reference::polynomial_evaluation::BSGSOps,
};

/// Backend-provided Baby-Step / Giant-Step polynomial-evaluation phases.
///
/// The default phases are derived schedules over the caller's [`BSGSOps`]
/// policy. They preserve that policy's dispatch and do not implement HAL kernels.
///
/// # Safety
/// Implementations must preserve the BSGS schedule semantics and the precision
/// metadata contract expected by the scheme-supplied operations.
pub unsafe trait PolynomialEvaluationImpl: Backend {
    #[allow(clippy::too_many_arguments)]
    fn glwe_eval_baby_step<Ops, R, P, A, G>(
        module: &Module<Self>,
        ops: &Ops,
        res: &mut R,
        parity: Parity,
        coeffs: &P,
        power_basis: &G,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Ops: BSGSOps<Self, R, P, A, R>,
        R: GLWEToBackendMut<Self> + GLWEToBackendRef<Self>,
        P: GLWEToBackendRef<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self>,
        G: PowerBasisHelper<Self, A>,
    {
        crate::oep::derived::polynomial_evaluation::glwe_eval_baby_step_derived::<Self, Ops, R, P, G, A>(
            module,
            ops,
            res,
            parity,
            coeffs,
            power_basis,
            scratch,
        )
    }

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
