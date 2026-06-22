use anyhow::Result;
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{
    BSGSCoeffOps, BSGSPrecision, GLWEAdd, GLWECopy, GLWENormalize, GLWEShift, GLWETensoring, GLWEZero, GiantStepTensorBounds,
    layouts::{
        BSGSMeta, BabyStep, GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, Parity, PowerBasisHelper, SetBSGSMeta,
        prepared::GLWETensorKeyPreparedToBackendRef,
    },
};

/// Baby-Step / Giant-Step polynomial-evaluation phases.
///
/// Per-operation precision integers and the plaintext-coefficient addition are
/// supplied by the scheme through `precision`.
pub trait GLWEPolynomialEvaluation<BE: Backend> {
    /// Evaluates a single baby step into `res`.
    #[allow(clippy::too_many_arguments)]
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
        Self: GLWEZero<BE> + Sized,
        PR: BSGSCoeffOps<BE, R, C, A>,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
        A: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta,
        G: PowerBasisHelper<BE, A>;

    /// Folds the evaluated baby steps into `res` using the giant-step schedule.
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
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;
}
