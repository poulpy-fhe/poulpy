use crate::layouts::GLWERelinearizationKeyHelper;
use anyhow::Result;
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{
    BSGSOps,
    layouts::{
        BabyStep, GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, Parity, PowerBasisHelper,
        prepared::{GGLWEPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
    },
};

/// Baby-Step / Giant-Step polynomial-evaluation phases.
///
/// All arithmetic is supplied by the scheme through `ops`; the engine only
/// sequences the GLWE BSGS schedule.
pub trait GLWEPolynomialEvaluation<BE: Backend> {
    /// Evaluates a single baby step into `res`.
    #[allow(clippy::too_many_arguments)]
    fn glwe_eval_baby_step<Ops, V, P, A, G>(
        &self,
        ops: &Ops,
        res: &mut V,
        parity: Parity,
        coeffs: &P,
        power_basis: &G,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Ops: BSGSOps<BE, V, P, A, V>,
        V: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
        P: GLWEToBackendRef<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE>,
        G: PowerBasisHelper<BE, A>;

    /// Folds the evaluated baby steps into `res` using the giant-step schedule.
    fn glwe_eval_giant_steps<Ops, R, B, V, P, A, G, H>(
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
        H: GLWERelinearizationKeyHelper,
        H::Key: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>;
}
