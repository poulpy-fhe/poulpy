use anyhow::Result;
use poulpy_core::{
    ScratchArenaTakeCore,
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, ModuleCoreAlloc, prepared::GLWETensorKeyPreparedToBackendRef},
};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Module, ScratchArena},
};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{BSGSPolynomialInfos, BabyStep, Parity, PolynomialEvaluation, PowerBasisHelper},
    oep::CKKSPolynomialEvaluationImpl,
};

impl<BE: Backend + CKKSPolynomialEvaluationImpl<BE>> PolynomialEvaluation<BE> for Module<BE>
where
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_eval_baby_step<R, C, A, G>(
        &self,
        res: &mut R,
        coeffs: &C,
        parity: Parity,
        power_basis: &G,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        G: PowerBasisHelper<BE, A>,
    {
        BE::ckks_eval_baby_step::<R, C, A, G>(self, res, coeffs, parity, power_basis, scratch)
    }

    fn ckks_eval_giant_steps<R, B, A, G, T>(
        &self,
        res: &mut R,
        baby_steps: &mut [B],
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        B: BabyStep<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        G: PowerBasisHelper<BE, A>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    {
        BE::ckks_eval_giant_steps::<R, B, A, G, T>(self, res, baby_steps, power_basis, tsk, scratch)
    }

    fn ckks_eval_poly_real_const_coeffs_from_power_basis<R, B, A, G, T>(
        &self,
        res: &mut R,
        poly: &B,
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        B: BSGSPolynomialInfos<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        G: PowerBasisHelper<BE, A>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    {
        BE::ckks_eval_poly_real_const_coeffs_from_power_basis::<R, B, A, G, T>(self, res, poly, power_basis, tsk, scratch)
    }
}
