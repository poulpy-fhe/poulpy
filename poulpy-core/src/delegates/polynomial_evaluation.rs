use anyhow::Result;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    BSGSConstAdd, BSGSPrecision, GLWEAdd, GLWECopy, GLWEMulConst, GLWENormalize, GLWEPolynomialEvaluation, GLWEShift,
    GLWETensoring, GLWEZero, GiantStepTensorBounds,
    layouts::{
        BSGSMeta, BabyStep, GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, Parity, PowerBasisHelper, SetBSGSMeta,
        prepared::GLWETensorKeyPreparedToBackendRef,
    },
    oep::PolynomialEvaluationImpl,
};

impl<BE: Backend + PolynomialEvaluationImpl<BE>> GLWEPolynomialEvaluation<BE> for Module<BE> {
    fn glwe_eval_baby_step<PR, R, C, A, G>(
        &self,
        precision: &PR,
        res: &mut R,
        parity: Parity,
        coeffs: &C,
        power_basis: &G,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEMulConst<BE> + GLWEAdd<BE> + GLWEShift<BE> + GLWENormalize<BE> + GLWEZero<BE> + Sized,
        PR: BSGSPrecision<BE> + BSGSConstAdd<BE, R, C>,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
        A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
        G: PowerBasisHelper<BE, A>,
    {
        BE::glwe_eval_baby_step::<PR, R, C, A, G>(self, precision, res, parity, coeffs, power_basis, scratch)
    }

    fn glwe_eval_giant_steps<PR, R, B, A, G, T>(
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
        BE::glwe_eval_giant_steps::<PR, R, B, A, G, T>(self, precision, res, baby_steps, power_basis, tsk, scratch)
    }
}
