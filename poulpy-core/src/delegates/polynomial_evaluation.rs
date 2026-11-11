use crate::layouts::GetTensorKey;
use anyhow::Result;
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena};

use crate::{
    GLWEBSGSOps, GLWEPolynomialEvaluation,
    layouts::{BabyStep, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, Parity, PowerBasisHelper},
    oep::GLWEPolynomialEvaluationImpl,
};

impl<BE: Backend + GLWEPolynomialEvaluationImpl<Rg>, Rg: Ring> GLWEPolynomialEvaluation<BE, Rg> for Module<BE> {
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
        Ops: GLWEBSGSOps<BE, V, P, A, V, Rg>,
        V: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg>,
        P: GLWEToBackendRef<BE, Rg> + GLWEInfos,
        A: GLWEToBackendRef<BE, Rg>,
        G: PowerBasisHelper<BE, A, Rg>,
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
        Ops: GLWEBSGSOps<BE, V, P, A, R, Rg>,
        R: GLWEToBackendMut<BE, Rg>,
        B: BabyStep<BE, Rg, Value = V>,
        V: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg>,
        P: GLWEToBackendRef<BE, Rg>,
        A: GLWEToBackendRef<BE, Rg>,
        G: PowerBasisHelper<BE, A, Rg>,
        H: GetTensorKey<BE, Rg>,
    {
        BE::glwe_eval_giant_steps::<Ops, R, B, V, P, A, G, H>(self, ops, res, baby_steps, power_basis, tsk, scratch)
    }
}
