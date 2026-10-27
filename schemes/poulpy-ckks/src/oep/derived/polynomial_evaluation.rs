use crate::CKKSResult as Result;
use crate::api::{CKKSComplexPolynomialEvaluationOps, CKKSPolynomialEvaluationOps};
use poulpy_core::GLWENormalize;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta};
use poulpy_core::oep::GLWEMulConstImpl;

use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{BSGSPolynomialInfos, CKKSCopyOps, CKKSMulOps, CKKSPow2Ops, CKKSSubOps, PolynomialInputTransform},
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc},
    polynomial::ComplexBSGSPolynomial,
    power_basis::{PowerBasis, PowerBasisGen},
};

/// Scratch upper bound for [`BSGSOps::eval_baby_linear_combination`]: the constant combination
/// for the widest term (`ct_size`) and constant (`pt_size`) in limbs.
pub(crate) fn eval_baby_linear_combination_tmp_bytes<BE>(module: &Module<BE>, ct_size: usize, pt_size: usize) -> usize
where
    BE: GLWEMulConstImpl,
{
    BE::glwe_const_combination_tmp_bytes(module, ct_size + pt_size, ct_size, pt_size)
}

/// Builds the folded input (`x`, `x²`, or `T₂(x)`) for one-shot evaluation.
/// The prepared-basis kernels are selected through the backend contract.
pub(crate) fn polynomial_input<BE, S, H>(
    module: &Module<BE>,
    src: &S,
    transform: PolynomialInputTransform,
    tsk: &H,
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
            let mut squared = module.ckks_ciphertext_alloc(src.base2k(), k.into());
            module.ckks_square_into(&mut squared, src, tsk, scratch)?;
            let mut doubled = module.ckks_ciphertext_alloc(src.base2k(), k.into());
            module.ckks_double_into(&mut doubled, &squared, scratch)?;
            module.ckks_sub_one_assign(&mut doubled, scratch)?;
            Ok(doubled)
        }
    }
}

pub(crate) fn ckks_eval_poly_real_const_coeffs_derived<BE: Backend, R, S, B, H>(
    module: &Module<BE>,
    dst: &mut R,
    src: &S,
    bsgs: &B,
    tsk: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    Module<BE>: crate::api::CKKSPolynomialEvaluationOps<BE>
        + CKKSCopyOps<BE>
        + CKKSMulOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSSubOps<BE>
        + CKKSModuleAlloc<BE>
        + GLWENormalize<BE>,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    S: GLWEToBackendRef<BE> + CKKSCtBounds,
    B: BSGSPolynomialInfos<BE>,
    B::Coeffs: CKKSCtBounds,
    H: GetTensorKey<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    crate::reference::polynomial_evaluation::ensure_baby_steps("ckks_eval_poly_real_const_coeffs", bsgs)?;
    let transform = bsgs.input_transform();
    let x1 = polynomial_input(module, src, transform, tsk, scratch)?;
    let mut power_basis = PowerBasis::new(bsgs.basis(), x1);
    power_basis.populate(bsgs.degree(), bsgs.log_split(), bsgs.parity(), module, tsk, scratch)?;
    module.ckks_eval_poly_real_const_coeffs_from_power_basis(dst, &power_basis, bsgs, tsk, scratch)?;
    if matches!(
        transform,
        PolynomialInputTransform::SquareTimesInput | PolynomialInputTransform::ChebyshevT2TimesInput
    ) {
        module.ckks_mul_assign(dst, src, tsk, scratch)?;
    }
    Ok(())
}

pub(crate) fn ckks_eval_poly_complex_const_coeffs_derived<BE: Backend, R, S, C, H>(
    module: &Module<BE>,
    dst: &mut R,
    src: &S,
    poly: &ComplexBSGSPolynomial<C>,
    tsk: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    Module<BE>: CKKSComplexPolynomialEvaluationOps<BE>
        + CKKSCopyOps<BE>
        + CKKSMulOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSSubOps<BE>
        + CKKSModuleAlloc<BE>
        + GLWENormalize<BE>,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    S: GLWEToBackendRef<BE> + CKKSCtBounds,
    C: GLWEToBackendRef<BE> + GLWEInfos + poulpy_core::layouts::BSGSMeta + CKKSCtBounds + IntPolyInfos,
    H: GetTensorKey<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    const OP: &str = "ckks_eval_poly_complex_const_coeffs";
    crate::reference::polynomial_evaluation::ensure_complex_schedules::<BE, C>(OP, poly)?;
    let transform = poly.re.input_transform();
    crate::ckks_ensure_operand!(
        OP,
        transform == poly.im.input_transform(),
        "real and imaginary input transforms differ"
    );
    let x1 = polynomial_input(module, src, transform, tsk, scratch)?;
    let mut power_basis = PowerBasis::new(poly.re.basis(), x1);
    power_basis.populate(poly.re.degree(), poly.re.log_split(), poly.re.parity(), module, tsk, scratch)?;
    module.ckks_eval_poly_complex_const_coeffs_from_power_basis(dst, &power_basis, poly, tsk, scratch)?;
    if matches!(
        transform,
        PolynomialInputTransform::SquareTimesInput | PolynomialInputTransform::ChebyshevT2TimesInput
    ) {
        module.ckks_mul_assign(dst, src, tsk, scratch)?;
    }
    Ok(())
}
