use anyhow::Result;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    BSGSBabyOps, BSGSGiantOps,
    layouts::{BabyStep, GGLWEInfos, GLWEInfos, Parity, PowerBasisHelper, prepared::GLWETensorKeyPreparedToBackendRef},
};

/// Backend-provided Baby-Step / Giant-Step polynomial-evaluation phases.
///
/// # Safety
/// Implementations must preserve the BSGS schedule semantics and the precision
/// metadata contract expected by the scheme-supplied `precision` provider.
pub unsafe trait PolynomialEvaluationImpl<BE: Backend>: Backend {
    #[allow(clippy::too_many_arguments)]
    fn glwe_eval_baby_step<PR, V, P, A, G>(
        module: &Module<BE>,
        precision: &PR,
        res: &mut V,
        parity: Parity,
        coeffs: &P,
        power_basis: &G,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        PR: BSGSBabyOps<BE, V, P, A>,
        P: GLWEInfos,
        G: PowerBasisHelper<BE, A>;

    #[allow(clippy::too_many_arguments)]
    fn glwe_eval_giant_steps<PR, R, B, V, A, G, T>(
        module: &Module<BE>,
        precision: &PR,
        res: &mut R,
        baby_steps: &mut [B],
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        PR: BSGSGiantOps<BE, V, A, R>,
        B: BabyStep<BE, Value = V>,
        G: PowerBasisHelper<BE, A>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;
}

/// Override surface carrying the reference BSGS phase implementations.
pub trait PolynomialEvaluationDefault<BE: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn glwe_eval_baby_step_default<PR, V, P, A, G>(
        &self,
        precision: &PR,
        res: &mut V,
        parity: Parity,
        coeffs: &P,
        power_basis: &G,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        PR: BSGSBabyOps<BE, V, P, A>,
        P: GLWEInfos,
        G: PowerBasisHelper<BE, A>;

    fn glwe_eval_giant_steps_default<PR, R, B, V, A, G, T>(
        &self,
        precision: &PR,
        res: &mut R,
        baby_steps: &mut [B],
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        PR: BSGSGiantOps<BE, V, A, R>,
        B: BabyStep<BE, Value = V>,
        G: PowerBasisHelper<BE, A>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;
}

unsafe impl<BE: Backend> PolynomialEvaluationImpl<BE> for BE
where
    Module<BE>: PolynomialEvaluationDefault<BE>,
{
    fn glwe_eval_baby_step<PR, V, P, A, G>(
        module: &Module<BE>,
        precision: &PR,
        res: &mut V,
        parity: Parity,
        coeffs: &P,
        power_basis: &G,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        PR: BSGSBabyOps<BE, V, P, A>,
        P: GLWEInfos,
        G: PowerBasisHelper<BE, A>,
    {
        module.glwe_eval_baby_step_default::<PR, V, P, A, G>(precision, res, parity, coeffs, power_basis, scratch)
    }

    fn glwe_eval_giant_steps<PR, R, B, V, A, G, T>(
        module: &Module<BE>,
        precision: &PR,
        res: &mut R,
        baby_steps: &mut [B],
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        PR: BSGSGiantOps<BE, V, A, R>,
        B: BabyStep<BE, Value = V>,
        G: PowerBasisHelper<BE, A>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    {
        module.glwe_eval_giant_steps_default::<PR, R, B, V, A, G, T>(precision, res, baby_steps, power_basis, tsk, scratch)
    }
}
