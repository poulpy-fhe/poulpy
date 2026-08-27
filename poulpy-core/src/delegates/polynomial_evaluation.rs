use crate::layouts::GLWERelinearizationKeyHelper;
use anyhow::Result;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    BSGSOps, GLWEPolynomialEvaluation,
    layouts::{
        BabyStep, GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, Parity, PowerBasisHelper,
        prepared::{GGLWEPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
    },
    oep::PolynomialEvaluationImpl,
};

impl<BE: Backend + PolynomialEvaluationImpl<BE>> GLWEPolynomialEvaluation<BE> for Module<BE> {
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
        G: PowerBasisHelper<BE, A>,
    {
        BE::glwe_eval_baby_step::<Ops, V, P, A, G>(self, ops, res, parity, coeffs, power_basis, scratch)
    }

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
        H::Key: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        BE::glwe_eval_giant_steps::<Ops, R, B, V, P, A, G, H>(self, ops, res, baby_steps, power_basis, tsk, scratch)
    }
}
