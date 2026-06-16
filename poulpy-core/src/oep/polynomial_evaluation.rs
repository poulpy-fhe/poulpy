use anyhow::Result;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    BSGSConstAdd, BSGSPrecision, GLWEAdd, GLWECopy, GLWEMulConst, GLWENormalize, GLWEShift, GLWETensoring, GLWEZero,
    GiantStepTensorBounds,
    layouts::{
        BSGSMeta, BabyStep, GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, Parity, PowerBasisHelper, SetBSGSMeta,
        prepared::GLWETensorKeyPreparedToBackendRef,
    },
};

/// Backend-provided Baby-Step / Giant-Step polynomial-evaluation phases.
///
/// # Safety
/// Implementations must preserve the BSGS schedule semantics and the precision
/// metadata contract expected by the scheme-supplied `precision` provider.
pub unsafe trait PolynomialEvaluationImpl<BE: Backend>: Backend {
    #[allow(clippy::too_many_arguments)]
    fn glwe_eval_baby_step<PR, R, C, A, G>(
        module: &Module<BE>,
        precision: &PR,
        res: &mut R,
        parity: Parity,
        coeffs: &C,
        power_basis: &G,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEMulConst<BE> + GLWEAdd<BE> + GLWEShift<BE> + GLWENormalize<BE> + GLWEZero<BE>,
        PR: BSGSPrecision<BE> + BSGSConstAdd<BE, R, C>,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
        A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
        G: PowerBasisHelper<BE, A>;

    #[allow(clippy::too_many_arguments)]
    fn glwe_eval_giant_steps<PR, R, B, A, G, T>(
        module: &Module<BE>,
        precision: &PR,
        res: &mut R,
        baby_steps: &mut [B],
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>:
            GiantStepTensorBounds<BE> + GLWEAdd<BE> + GLWEShift<BE> + GLWETensoring<BE> + GLWENormalize<BE> + GLWECopy<BE>,
        PR: BSGSPrecision<BE>,
        R: GLWEToBackendMut<BE> + GLWEInfos + SetBSGSMeta,
        B: BabyStep<BE>,
        A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;
}

/// Override surface carrying the reference BSGS phase implementations.
pub trait PolynomialEvaluationDefault<BE: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn glwe_eval_baby_step_default<PR, R, C, A, G>(
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
        G: PowerBasisHelper<BE, A>;

    fn glwe_eval_giant_steps_default<PR, R, B, A, G, T>(
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
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;
}

unsafe impl<BE: Backend> PolynomialEvaluationImpl<BE> for BE
where
    Module<BE>: PolynomialEvaluationDefault<BE>,
{
    fn glwe_eval_baby_step<PR, R, C, A, G>(
        module: &Module<BE>,
        precision: &PR,
        res: &mut R,
        parity: Parity,
        coeffs: &C,
        power_basis: &G,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEMulConst<BE> + GLWEAdd<BE> + GLWEShift<BE> + GLWENormalize<BE> + GLWEZero<BE>,
        PR: BSGSPrecision<BE> + BSGSConstAdd<BE, R, C>,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
        A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
        G: PowerBasisHelper<BE, A>,
    {
        module.glwe_eval_baby_step_default::<PR, R, C, A, G>(precision, res, parity, coeffs, power_basis, scratch)
    }

    fn glwe_eval_giant_steps<PR, R, B, A, G, T>(
        module: &Module<BE>,
        precision: &PR,
        res: &mut R,
        baby_steps: &mut [B],
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>:
            GiantStepTensorBounds<BE> + GLWEAdd<BE> + GLWEShift<BE> + GLWETensoring<BE> + GLWENormalize<BE> + GLWECopy<BE>,
        PR: BSGSPrecision<BE>,
        R: GLWEToBackendMut<BE> + GLWEInfos + SetBSGSMeta,
        B: BabyStep<BE>,
        A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    {
        module.glwe_eval_giant_steps_default::<PR, R, B, A, G, T>(precision, res, baby_steps, power_basis, tsk, scratch)
    }
}
