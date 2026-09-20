use crate::layouts::GetTensorKey;
use anyhow::Result;

/// Error result used by the portable BSGS schedule and its backend implementations.
pub use anyhow::Result as PolynomialEvaluationResult;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    layouts::{BabyStep, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, Parity, PowerBasisHelper},
    reference::polynomial_evaluation::BSGSOps,
};

/// Backend-provided Baby-Step / Giant-Step polynomial-evaluation phases.
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
        G: PowerBasisHelper<Self, A>;

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
        H: GetTensorKey<Self>;
}

/// Override surface carrying the reference BSGS phase implementations.
pub trait PolynomialEvaluationReference<BE: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn glwe_eval_baby_step_reference<Ops, R, P, A, G>(
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
        G: PowerBasisHelper<BE, A>;

    fn glwe_eval_giant_steps_reference<Ops, R, B, V, P, A, G, H>(
        &self,
        ops: &Ops,
        res: &mut R,
        baby_steps: &mut [B],
        power_basis: &G,
        tsk: &H,
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
        H: GetTensorKey<BE>;
}

unsafe impl<BE: Backend> PolynomialEvaluationImpl for BE
where
    Module<BE>: PolynomialEvaluationReference<BE>,
{
    fn glwe_eval_baby_step<Ops, R, P, A, G>(
        module: &Module<BE>,
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
        module.glwe_eval_baby_step_reference::<Ops, R, P, A, G>(ops, res, parity, coeffs, power_basis, scratch)
    }

    fn glwe_eval_giant_steps<Ops, R, B, V, P, A, G, H>(
        module: &Module<BE>,
        ops: &Ops,
        res: &mut R,
        baby_steps: &mut [B],
        power_basis: &G,
        tsk: &H,
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
        H: GetTensorKey<BE>,
    {
        module.glwe_eval_giant_steps_reference::<Ops, R, B, V, P, A, G, H>(ops, res, baby_steps, power_basis, tsk, scratch)
    }
}

/// Forwards the BSGS phases to the portable schedule, leaving scheme arithmetic in `BSGSOps`.
#[macro_export]
macro_rules! impl_polynomial_evaluation_reference_full {
    ($be:ty) => {
        impl $crate::oep::PolynomialEvaluationReference<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn glwe_eval_baby_step_reference<Ops, R, P, A, G>(
                &self,
                ops: &Ops,
                res: &mut R,
                parity: $crate::layouts::Parity,
                coeffs: &P,
                power_basis: &G,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) -> $crate::oep::PolynomialEvaluationResult<()>
            where
                Ops: $crate::reference::polynomial_evaluation::BSGSOps<$be, R, P, A, R>,
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEToBackendRef<$be>,
                P: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be>,
                G: $crate::layouts::PowerBasisHelper<$be, A>,
            {
                $crate::reference::polynomial_evaluation::eval_baby_step::<$be, Ops, R, P, G, A>(
                    self,
                    ops,
                    res,
                    parity,
                    coeffs,
                    power_basis,
                    scratch,
                )
            }
            fn glwe_eval_giant_steps_reference<Ops, R, B, V, P, A, G, H>(
                &self,
                ops: &Ops,
                res: &mut R,
                baby_steps: &mut [B],
                power_basis: &G,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) -> $crate::oep::PolynomialEvaluationResult<()>
            where
                Ops: $crate::reference::polynomial_evaluation::BSGSOps<$be, V, P, A, R>,
                R: $crate::layouts::GLWEToBackendMut<$be>,
                B: $crate::layouts::BabyStep<$be, Value = V>,
                V: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEToBackendRef<$be>,
                P: $crate::layouts::GLWEToBackendRef<$be>,
                A: $crate::layouts::GLWEToBackendRef<$be>,
                G: $crate::layouts::PowerBasisHelper<$be, A>,
                H: $crate::layouts::GetTensorKey<$be>,
            {
                $crate::reference::polynomial_evaluation::eval_giant_steps::<R, B, V, P, A, G, H, $be, Ops>(
                    self,
                    ops,
                    res,
                    baby_steps,
                    power_basis,
                    tsk,
                    scratch,
                )
            }
        }
    };
}
