use anyhow::Result;
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{
    BSGSBabyOps, BSGSGiantOps,
    layouts::{BabyStep, GGLWEInfos, GLWEInfos, Parity, PowerBasisHelper, prepared::GLWETensorKeyPreparedToBackendRef},
};

/// Baby-Step / Giant-Step polynomial-evaluation phases.
///
/// All arithmetic is supplied by the scheme through `precision`
/// ([`BSGSBabyOps`] / [`BSGSGiantOps`]); the engine only sequences the schedule.
pub trait GLWEPolynomialEvaluation<BE: Backend> {
    /// Evaluates a single baby step into `res`.
    #[allow(clippy::too_many_arguments)]
    fn glwe_eval_baby_step<PR, V, P, A, G>(
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

    /// Folds the evaluated baby steps into `res` using the giant-step schedule.
    fn glwe_eval_giant_steps<PR, R, B, V, A, G, T>(
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
