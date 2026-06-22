use anyhow::Result;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    BSGSBabyOps, BSGSGiantOps, GLWEPolynomialEvaluation,
    layouts::{BabyStep, GGLWEInfos, GLWEInfos, Parity, PowerBasisHelper, prepared::GLWETensorKeyPreparedToBackendRef},
    oep::PolynomialEvaluationImpl,
};

impl<BE: Backend + PolynomialEvaluationImpl<BE>> GLWEPolynomialEvaluation<BE> for Module<BE> {
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
        G: PowerBasisHelper<BE, A>,
    {
        BE::glwe_eval_baby_step::<PR, V, P, A, G>(self, precision, res, parity, coeffs, power_basis, scratch)
    }

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
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    {
        BE::glwe_eval_giant_steps::<PR, R, B, V, A, G, T>(self, precision, res, baby_steps, power_basis, tsk, scratch)
    }
}
