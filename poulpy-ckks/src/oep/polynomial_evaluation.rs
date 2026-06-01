use anyhow::Result;
use poulpy_core::layouts::{
    BSGSMeta, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta, prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEMulConst, GLWENormalize, GLWEShift, GLWETensoring, GLWEZero, GiantStepTensorBounds,
    ScratchArenaTakeCore,
};
use poulpy_hal::api::ScratchAvailable;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{BSGSPolynomialInfos, BabyStep, CKKSAddOps, CKKSMulAddOps, Parity, PowerBasisHelper},
    default::polynomial_evaluation::PolynomialEvaluationDefault,
    layouts::CKKSModuleAlloc,
};

/// # Safety
///
/// Implementations must satisfy the contracts of the polynomial-evaluation
/// API, including the invariants of the underlying add/mul/copy kernels.
pub unsafe trait CKKSPolynomialEvaluationImpl<BE: Backend>: Backend {
    fn ckks_eval_baby_step<R, C, A, G>(
        module: &Module<BE>,
        res: &mut R,
        coeffs: &C,
        parity: Parity,
        power_basis: &G,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
        A: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
        G: PowerBasisHelper<BE, A>;

    fn ckks_eval_giant_steps<R, B, A, G, T>(
        module: &Module<BE>,
        res: &mut R,
        baby_steps: &mut [B],
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        B: BabyStep<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos;

    fn ckks_eval_poly_real_const_coeffs_from_power_basis<R, B, A, G, T>(
        module: &Module<BE>,
        res: &mut R,
        poly: &B,
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds,
        A: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos;
}

unsafe impl<BE: Backend> CKKSPolynomialEvaluationImpl<BE> for BE
where
    Module<BE>: GiantStepTensorBounds<BE>
        + CKKSAddOps<BE>
        + CKKSMulAddOps<BE>
        + GLWEMulConst<BE>
        + GLWEAdd<BE>
        + GLWEShift<BE>
        + GLWETensoring<BE>
        + GLWECopy<BE>
        + GLWENormalize<BE>
        + GLWEZero<BE>
        + CKKSModuleAlloc<BE>
        + PolynomialEvaluationDefault<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_eval_baby_step<R, C, A, G>(
        module: &Module<BE>,
        res: &mut R,
        coeffs: &C,
        parity: Parity,
        power_basis: &G,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
        A: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
        G: PowerBasisHelper<BE, A>,
    {
        module.ckks_eval_baby_step_default::<R, C, A, G>(res, coeffs, parity, power_basis, scratch)
    }

    fn ckks_eval_giant_steps<R, B, A, G, T>(
        module: &Module<BE>,
        res: &mut R,
        baby_steps: &mut [B],
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        B: BabyStep<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        module.ckks_eval_giant_steps_default::<R, B, A, G, T>(res, baby_steps, power_basis, tsk, scratch)
    }

    fn ckks_eval_poly_real_const_coeffs_from_power_basis<R, B, A, G, T>(
        module: &Module<BE>,
        res: &mut R,
        poly: &B,
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds,
        A: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        module.ckks_eval_poly_real_const_coeffs_from_power_basis_default::<R, B, A, G, T>(res, poly, power_basis, tsk, scratch)
    }
}
