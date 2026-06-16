use anyhow::Result;
use poulpy_core::layouts::{
    GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta,
    prepared::{GLWETensorKeyPrepared, GLWETensorKeyPreparedToBackendRef},
};
use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEMulConst, GLWENormalize, GLWEPolynomialEvaluation, GLWEShift, GLWETensoring, GLWEZero,
    GiantStepTensorBounds,
};

use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{BSGSPolynomialInfos, CKKSAddOps, CKKSCopyOps, CKKSImagOps, CKKSMulAddOps, CKKSMulOps, CKKSSubOps, PowerBasisHelper},
    default::polynomial_evaluation::PolynomialEvaluationDefault,
    layouts::{CKKSCiphertext, CKKSModuleAlloc},
    polynomial::ComplexBSGSPolynomial,
    power_basis::{PowerBasis, PowerBasisGen},
};

/// # Safety
///
/// Implementations must satisfy the contracts of the polynomial-evaluation
/// API, including the invariants of the underlying add/mul/copy kernels.
pub unsafe trait CKKSPolynomialEvaluationImpl<BE: Backend>: Backend {
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
        A: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos;

    fn ckks_eval_poly_complex_const_coeffs_from_power_basis<R, C, A, G, T>(
        module: &Module<BE>,
        res: &mut R,
        poly: &ComplexBSGSPolynomial<C>,
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + GLWEInfos + poulpy_core::layouts::BSGSMeta + CKKSCtBounds,
        A: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos;

    fn ckks_eval_poly_real_const_coeffs<R, S, C, B>(
        module: &Module<BE>,
        dst: &mut R,
        src: &S,
        bsgs: &B,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<BE> + CKKSCtBounds,
        C: GLWEToBackendRef<BE> + GLWEInfos + poulpy_core::layouts::BSGSMeta + CKKSCtBounds,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos;

    fn ckks_eval_poly_complex_const_coeffs<R, S, C>(
        module: &Module<BE>,
        dst: &mut R,
        src: &S,
        poly: &ComplexBSGSPolynomial<C>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<BE> + CKKSCtBounds,
        C: GLWEToBackendRef<BE> + GLWEInfos + poulpy_core::layouts::BSGSMeta + CKKSCtBounds,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos;
}

unsafe impl<BE: Backend> CKKSPolynomialEvaluationImpl<BE> for BE
where
    Module<BE>: GiantStepTensorBounds<BE>
        + CKKSAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSImagOps<BE>
        + CKKSMulOps<BE>
        + CKKSSubOps<BE>
        + CKKSMulAddOps<BE>
        + GLWEMulConst<BE>
        + GLWEAdd<BE>
        + GLWEShift<BE>
        + GLWETensoring<BE>
        + GLWECopy<BE>
        + GLWENormalize<BE>
        + GLWEZero<BE>
        + GLWEPolynomialEvaluation<BE>
        + CKKSModuleAlloc<BE>
        + PolynomialEvaluationDefault<BE>,
{
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
        A: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        module.ckks_eval_poly_real_const_coeffs_from_power_basis_default::<R, B, A, G, T>(res, poly, power_basis, tsk, scratch)
    }

    fn ckks_eval_poly_complex_const_coeffs_from_power_basis<R, C, A, G, T>(
        module: &Module<BE>,
        res: &mut R,
        poly: &ComplexBSGSPolynomial<C>,
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + GLWEInfos + poulpy_core::layouts::BSGSMeta + CKKSCtBounds,
        A: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        module.ckks_eval_poly_complex_const_coeffs_from_power_basis_default::<R, C, A, G, T>(res, poly, power_basis, tsk, scratch)
    }

    fn ckks_eval_poly_real_const_coeffs<R, S, C, B>(
        module: &Module<BE>,
        dst: &mut R,
        src: &S,
        bsgs: &B,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<BE> + CKKSCtBounds,
        C: GLWEToBackendRef<BE> + GLWEInfos + poulpy_core::layouts::BSGSMeta + CKKSCtBounds,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        let mut x1 = module.ckks_ciphertext_alloc_from_infos(src);
        module.ckks_copy(&mut x1, src, scratch)?;
        let mut power_basis = PowerBasis::new(bsgs.basis(), x1);
        power_basis.populate(bsgs.degree(), bsgs.log_split(), bsgs.parity(), module, tsk, scratch)?;
        module.ckks_eval_poly_real_const_coeffs_from_power_basis_default(dst, bsgs, &power_basis, tsk, scratch)
    }

    fn ckks_eval_poly_complex_const_coeffs<R, S, C>(
        module: &Module<BE>,
        dst: &mut R,
        src: &S,
        poly: &ComplexBSGSPolynomial<C>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<BE> + CKKSCtBounds,
        C: GLWEToBackendRef<BE> + GLWEInfos + poulpy_core::layouts::BSGSMeta + CKKSCtBounds,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        let mut x1 = module.ckks_ciphertext_alloc_from_infos(src);
        module.ckks_copy(&mut x1, src, scratch)?;
        let mut power_basis = PowerBasis::new(poly.re.basis(), x1);
        power_basis.populate(poly.re.degree(), poly.re.log_split(), poly.re.parity(), module, tsk, scratch)?;
        module.ckks_eval_poly_complex_const_coeffs_from_power_basis_default(dst, poly, &power_basis, tsk, scratch)
    }
}
