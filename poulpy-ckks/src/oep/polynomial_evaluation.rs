use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::layouts::IntPolyInfos;
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
    api::{
        BSGSPolynomialInfos, CKKSAddOps, CKKSCopyOps, CKKSImagOps, CKKSMulAddOps, CKKSMulOps, CKKSPow2Ops, CKKSSubOps,
        PolynomialInputTransform, PowerBasisHelper,
    },
    default::polynomial_evaluation::PolynomialEvaluationDefault,
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc},
    polynomial::ComplexBSGSPolynomial,
    power_basis::{PowerBasis, PowerBasisGen},
};

/// Builds the folded input (`x`, `x²`, or `T₂(x)`) for one-shot evaluation.
/// It lives in the OEP because the default engine only consumes a prepared
/// power basis.
fn polynomial_input<BE, S>(
    module: &Module<BE>,
    src: &S,
    transform: PolynomialInputTransform,
    tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<CKKSCiphertextOwned<BE>>
where
    BE: Backend,
    Module<BE>: CKKSCopyOps<BE> + CKKSMulOps<BE> + CKKSPow2Ops<BE> + CKKSSubOps<BE> + CKKSModuleAlloc<BE>,
    S: GLWEToBackendRef<BE> + CKKSCtBounds,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    let mut input = module.ckks_ciphertext_alloc_from_infos(src);
    module.ckks_copy(&mut input, src, scratch)?;

    match transform {
        PolynomialInputTransform::Identity => Ok(input),
        PolynomialInputTransform::Square | PolynomialInputTransform::SquareTimesInput => {
            let mut input_basis = PowerBasis::new(crate::api::Basis::Monomial, input);
            input_basis.gen_power(2, module, tsk, scratch)?;
            Ok(input_basis
                .take_power(2)
                .expect("generating x² must store the degree-two power"))
        }
        PolynomialInputTransform::ChebyshevT2 | PolynomialInputTransform::ChebyshevT2TimesInput => {
            let mut input_basis = PowerBasis::new(crate::api::Basis::Chebyshev, input);
            input_basis.gen_power_chebyshev(2, module, tsk, scratch)?;
            Ok(input_basis
                .take_power(2)
                .expect("generating T₂ must store the degree-two power"))
        }
    }
}

/// # Safety
///
/// Implementations must satisfy the contracts of the polynomial-evaluation
/// API, including the invariants of the underlying add/mul/copy kernels.
pub unsafe trait CKKSPolynomialEvaluationImpl<BE: Backend>: Backend {
    fn ckks_eval_poly_real_const_coeffs_from_power_basis_impl<R, B, A, G, T>(
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

    fn ckks_eval_poly_complex_const_coeffs_from_power_basis_impl<R, C, A, G, T>(
        module: &Module<BE>,
        res: &mut R,
        poly: &ComplexBSGSPolynomial<C>,
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + GLWEInfos + poulpy_core::layouts::BSGSMeta + CKKSCtBounds + IntPolyInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos;

    fn ckks_eval_poly_real_const_coeffs_impl<R, S, B>(
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
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos;

    fn ckks_eval_poly_complex_const_coeffs_impl<R, S, C>(
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
        C: GLWEToBackendRef<BE> + GLWEInfos + poulpy_core::layouts::BSGSMeta + CKKSCtBounds + IntPolyInfos,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos;
}

unsafe impl<BE: Backend> CKKSPolynomialEvaluationImpl<BE> for BE
where
    Module<BE>: GiantStepTensorBounds<BE>
        + CKKSAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSImagOps<BE>
        + CKKSMulOps<BE>
        + CKKSPow2Ops<BE>
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
    fn ckks_eval_poly_real_const_coeffs_from_power_basis_impl<R, B, A, G, T>(
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

    fn ckks_eval_poly_complex_const_coeffs_from_power_basis_impl<R, C, A, G, T>(
        module: &Module<BE>,
        res: &mut R,
        poly: &ComplexBSGSPolynomial<C>,
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + GLWEInfos + poulpy_core::layouts::BSGSMeta + CKKSCtBounds + IntPolyInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        module.ckks_eval_poly_complex_const_coeffs_from_power_basis_default::<R, C, A, G, T>(res, poly, power_basis, tsk, scratch)
    }

    fn ckks_eval_poly_real_const_coeffs_impl<R, S, B>(
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
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        let transform = bsgs.input_transform();
        let x1 = polynomial_input(module, src, transform, tsk, scratch)?;
        let mut power_basis = PowerBasis::new(bsgs.basis(), x1);
        power_basis.populate(bsgs.degree(), bsgs.log_split(), bsgs.parity(), module, tsk, scratch)?;
        module.ckks_eval_poly_real_const_coeffs_from_power_basis_default(dst, bsgs, &power_basis, tsk, scratch)?;
        if matches!(
            transform,
            PolynomialInputTransform::SquareTimesInput | PolynomialInputTransform::ChebyshevT2TimesInput
        ) {
            module.ckks_mul_assign(dst, src, tsk, scratch)?;
        }
        Ok(())
    }

    fn ckks_eval_poly_complex_const_coeffs_impl<R, S, C>(
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
        C: GLWEToBackendRef<BE> + GLWEInfos + poulpy_core::layouts::BSGSMeta + CKKSCtBounds + IntPolyInfos,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        let transform = poly.re.input_transform();
        ckks_ensure!(
            transform == poly.im.input_transform(),
            "ckks_eval_poly_complex_const_coeffs: real and imaginary input transforms differ"
        );
        let x1 = polynomial_input(module, src, transform, tsk, scratch)?;
        let mut power_basis = PowerBasis::new(poly.re.basis(), x1);
        power_basis.populate(poly.re.degree(), poly.re.log_split(), poly.re.parity(), module, tsk, scratch)?;
        module.ckks_eval_poly_complex_const_coeffs_from_power_basis_default(dst, poly, &power_basis, tsk, scratch)?;
        if matches!(
            transform,
            PolynomialInputTransform::SquareTimesInput | PolynomialInputTransform::ChebyshevT2TimesInput
        ) {
            module.ckks_mul_assign(dst, src, tsk, scratch)?;
        }
        Ok(())
    }
}
