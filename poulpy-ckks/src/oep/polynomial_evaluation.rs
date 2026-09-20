use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta};
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
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc},
    polynomial::ComplexBSGSPolynomial,
    power_basis::{PowerBasis, PowerBasisGen},
    reference::polynomial_evaluation::PolynomialEvaluationReference,
};

/// Builds the folded input (`x`, `x²`, or `T₂(x)`) for one-shot evaluation.
/// It lives in the OEP because the default engine only consumes a prepared
/// power basis.
fn polynomial_input<BE, S, H>(
    module: &Module<BE>,
    src: &S,
    transform: PolynomialInputTransform,
    tsk: &crate::layouts::CKKSKey<H>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<CKKSCiphertextOwned<BE>>
where
    BE: Backend,
    Module<BE>: CKKSCopyOps<BE> + CKKSMulOps<BE> + CKKSPow2Ops<BE> + CKKSSubOps<BE> + CKKSModuleAlloc<BE>,
    S: GLWEToBackendRef<BE> + CKKSCtBounds,
    H: GetTensorKey<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    match transform {
        PolynomialInputTransform::Identity => {
            let mut input = module.ckks_ciphertext_alloc_from_infos(src);
            module.ckks_copy(&mut input, src, scratch)?;
            Ok(input)
        }
        PolynomialInputTransform::Square | PolynomialInputTransform::SquareTimesInput => {
            let k = crate::power_basis::square_ct_k(src)?;
            let mut squared = module.ckks_ciphertext_alloc(src.base2k(), k.into());
            module.ckks_square_into(&mut squared, src, tsk, scratch)?;
            Ok(squared)
        }
        PolynomialInputTransform::ChebyshevT2 | PolynomialInputTransform::ChebyshevT2TimesInput => {
            let k = crate::power_basis::square_ct_k(src)?;
            let mut doubled = module.ckks_ciphertext_alloc(src.base2k(), k.into());
            module.ckks_square_into(&mut doubled, src, tsk, scratch)?;
            module.ckks_mul_pow2_assign(&mut doubled, 1, scratch)?;
            let one = crate::reference::carry_verb::ckks_one_pt(module, src.base2k())?;
            module.ckks_sub_pt_const_assign(&mut doubled, 0, &one, 0, scratch)?;
            Ok(doubled)
        }
    }
}

/// # Safety
///
/// Implementations must satisfy the contracts of the polynomial-evaluation
/// API, including the invariants of the underlying add/mul/copy kernels.
pub unsafe trait CKKSPolynomialEvaluationImpl: Backend {
    fn ckks_eval_poly_real_const_coeffs_from_power_basis_impl<R, B, A, G, H>(
        module: &Module<Self>,
        res: &mut R,
        poly: &B,
        power_basis: &G,
        tsk: &crate::layouts::CKKSKey<H>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        B: BSGSPolynomialInfos<Self>,
        B::Coeffs: CKKSCtBounds,
        A: GLWEToBackendRef<Self> + CKKSCtBounds + poulpy_core::layouts::BSGSMeta,
        G: PowerBasisHelper<Self, A>,
        H: GetTensorKey<Self>;

    fn ckks_eval_poly_complex_const_coeffs_from_power_basis_impl<R, C, A, G, H>(
        module: &Module<Self>,
        res: &mut R,
        poly: &ComplexBSGSPolynomial<C>,
        power_basis: &G,
        tsk: &crate::layouts::CKKSKey<H>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<Self> + GLWEInfos + poulpy_core::layouts::BSGSMeta + CKKSCtBounds + IntPolyInfos,
        A: GLWEToBackendRef<Self> + CKKSCtBounds + poulpy_core::layouts::BSGSMeta,
        G: PowerBasisHelper<Self, A>,
        H: GetTensorKey<Self>;

    fn ckks_eval_poly_real_const_coeffs_impl<R, S, B, H>(
        module: &Module<Self>,
        dst: &mut R,
        src: &S,
        bsgs: &B,
        tsk: &crate::layouts::CKKSKey<H>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<Self> + CKKSCtBounds,
        B: BSGSPolynomialInfos<Self>,
        B::Coeffs: CKKSCtBounds,
        H: GetTensorKey<Self>,
        CKKSCiphertextOwned<Self>: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos;

    fn ckks_eval_poly_complex_const_coeffs_impl<R, S, C, H>(
        module: &Module<Self>,
        dst: &mut R,
        src: &S,
        poly: &ComplexBSGSPolynomial<C>,
        tsk: &crate::layouts::CKKSKey<H>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<Self> + CKKSCtBounds,
        C: GLWEToBackendRef<Self> + GLWEInfos + poulpy_core::layouts::BSGSMeta + CKKSCtBounds + IntPolyInfos,
        H: GetTensorKey<Self>,
        CKKSCiphertextOwned<Self>: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos;
}

unsafe impl<BE: Backend> CKKSPolynomialEvaluationImpl for BE
where
    Module<BE>: GiantStepTensorBounds<BE>
        + poulpy_hal::api::VecZnxRsh<BE>
        + poulpy_hal::api::VecZnxRshTmpBytes
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
        + PolynomialEvaluationReference<BE>,
{
    fn ckks_eval_poly_real_const_coeffs_from_power_basis_impl<R, B, A, G, H>(
        module: &Module<BE>,
        res: &mut R,
        poly: &B,
        power_basis: &G,
        tsk: &crate::layouts::CKKSKey<H>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds,
        A: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        H: GetTensorKey<BE>,
    {
        module.ckks_eval_poly_real_const_coeffs_from_power_basis_reference::<R, B, A, G, H>(res, poly, power_basis, tsk, scratch)
    }

    fn ckks_eval_poly_complex_const_coeffs_from_power_basis_impl<R, C, A, G, H>(
        module: &Module<BE>,
        res: &mut R,
        poly: &ComplexBSGSPolynomial<C>,
        power_basis: &G,
        tsk: &crate::layouts::CKKSKey<H>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + GLWEInfos + poulpy_core::layouts::BSGSMeta + CKKSCtBounds + IntPolyInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        H: GetTensorKey<BE>,
    {
        module.ckks_eval_poly_complex_const_coeffs_from_power_basis_reference::<R, C, A, G, H>(
            res,
            poly,
            power_basis,
            tsk,
            scratch,
        )
    }

    fn ckks_eval_poly_real_const_coeffs_impl<R, S, B, H>(
        module: &Module<BE>,
        dst: &mut R,
        src: &S,
        bsgs: &B,
        tsk: &crate::layouts::CKKSKey<H>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds,
        H: GetTensorKey<BE>,
        CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        let transform = bsgs.input_transform();
        let x1 = polynomial_input(module, src, transform, tsk, scratch)?;
        let mut power_basis = PowerBasis::new(bsgs.basis(), x1);
        power_basis.populate(bsgs.degree(), bsgs.log_split(), bsgs.parity(), module, tsk, scratch)?;
        module.ckks_eval_poly_real_const_coeffs_from_power_basis_reference(dst, bsgs, &power_basis, tsk, scratch)?;
        if matches!(
            transform,
            PolynomialInputTransform::SquareTimesInput | PolynomialInputTransform::ChebyshevT2TimesInput
        ) {
            module.ckks_mul_assign(dst, src, tsk, scratch)?;
        }
        Ok(())
    }

    fn ckks_eval_poly_complex_const_coeffs_impl<R, S, C, H>(
        module: &Module<BE>,
        dst: &mut R,
        src: &S,
        poly: &ComplexBSGSPolynomial<C>,
        tsk: &crate::layouts::CKKSKey<H>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<BE> + CKKSCtBounds,
        C: GLWEToBackendRef<BE> + GLWEInfos + poulpy_core::layouts::BSGSMeta + CKKSCtBounds + IntPolyInfos,
        H: GetTensorKey<BE>,
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
        module.ckks_eval_poly_complex_const_coeffs_from_power_basis_reference(dst, poly, &power_basis, tsk, scratch)?;
        if matches!(
            transform,
            PolynomialInputTransform::SquareTimesInput | PolynomialInputTransform::ChebyshevT2TimesInput
        ) {
            module.ckks_mul_assign(dst, src, tsk, scratch)?;
        }
        Ok(())
    }
}
