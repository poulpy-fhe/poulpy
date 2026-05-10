use anyhow::Result;
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GLWETensorKeyPreparedToBackendRef};
use poulpy_core::{GLWENormalize, ScratchArenaTakeCore};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::api::CKKSMulAddOps;
use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{BSGSPolynomialInfos, CKKSAddOps, CKKSAddOpsUnnormalized, CKKSCopyOps, CKKSMulOps, PowerBasisHelper},
    default::polynomial_evaluation::PolynomialEvaluationDefault,
    layouts::CKKSModuleAlloc,
};

/// # Safety
///
/// Implementations must satisfy the contracts of the polynomial-evaluation
/// API, including the invariants of the underlying add/mul/copy kernels.
pub unsafe trait CKKSPolynomialEvaluationImpl<BE: Backend>: Backend {
    fn ckks_eval_poly_const_coeffs<R, B, A, G, T>(
        module: &Module<BE>,
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
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos;
}

unsafe impl<BE: Backend> CKKSPolynomialEvaluationImpl<BE> for BE
where
    Module<BE>: CKKSAddOps<BE>
        + CKKSAddOpsUnnormalized<BE>
        + CKKSMulAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSMulOps<BE>
        + GLWENormalize<BE>
        + CKKSModuleAlloc<BE>
        + PolynomialEvaluationDefault<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_eval_poly_const_coeffs<R, B, A, G, T>(
        module: &Module<BE>,
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
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        module.ckks_eval_poly_const_coeffs_default::<R, B, A, G, T>(res, poly, power_basis, tsk, scratch)
    }
}
