use crate::CKKSResult as Result;
use crate::api::{CKKSComplexPolynomialEvaluationOps, CKKSPolynomialEvaluationOps};
use poulpy_core::GLWENormalize;
use poulpy_core::layouts::{
    BSGSMeta, GLWEInfos, GLWELayout, GLWEPlaintextMeta, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, IntPolyInfos, Rank,
    SetBSGSMeta,
};
use poulpy_core::oep::GLWEMulConstImpl;

use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{BSGSPolynomialInfos, CKKSCopyOps, CKKSMulOps, CKKSPow2Ops, CKKSSubOps, PolynomialInputTransform},
    layouts::{CKKSCiphertextAlloc, CKKSCiphertextOwned, ScratchArenaTakeCKKSCiphertext},
    polynomial::ComplexBSGSPolynomial,
    power_basis::{CKKSPowerBasisGen, PowerBasis, PowerStorage},
};

/// Scratch upper bound for [`GLWEBSGSOps::eval_baby_linear_combination`]: the constant combination
/// for the widest term (`ct_size`) and constant (`pt_size`) in limbs.
pub(crate) fn eval_baby_linear_combination_tmp_bytes<BE, Rg: Ring>(module: &Module<BE>, ct_size: usize, pt_size: usize) -> usize
where
    BE: GLWEMulConstImpl<Rg>,
{
    BE::glwe_const_combination_tmp_bytes(module, ct_size + pt_size, ct_size, pt_size)
}

/// Builds the folded input (`x`, `x²`, or `T₂(x)`) for one-shot evaluation.
/// The prepared-basis kernels are selected through the backend contract.
pub(crate) fn polynomial_input<BE, S, H, Rg: Ring>(
    module: &Module<BE>,
    src: &S,
    transform: PolynomialInputTransform,
    tsk: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<CKKSCiphertextOwned<BE, Rg>>
where
    BE: Backend,
    Module<BE>: CKKSCopyOps<BE, Rg> + CKKSMulOps<BE, Rg> + CKKSPow2Ops<BE, Rg> + CKKSSubOps<BE, Rg> + CKKSCiphertextAlloc<BE>,
    S: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    H: GetTensorKey<BE, Rg>,
    CKKSCiphertextOwned<BE, Rg>: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
{
    polynomial_input_in(module, src, transform, tsk, scratch.borrow()).map(|(input, _)| input)
}

/// [`polynomial_input`] in the storage of `A`, with what is left of `scratch`.
pub(crate) fn polynomial_input_in<'a, BE, A, S, H, Rg: Ring>(
    module: &Module<BE>,
    src: &S,
    transform: PolynomialInputTransform,
    tsk: &H,
    scratch: ScratchArena<'a, BE>,
) -> Result<(A, ScratchArena<'a, BE>)>
where
    BE: Backend,
    A: PowerStorage<'a, BE, Rg>,
    Module<BE>: CKKSCopyOps<BE, Rg> + CKKSMulOps<BE, Rg> + CKKSPow2Ops<BE, Rg> + CKKSSubOps<BE, Rg>,
    S: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    H: GetTensorKey<BE, Rg>,
{
    let layout = |k: usize| GLWELayout {
        n: src.n(),
        base2k: src.base2k(),
        k: k.into(),
        rank: Rank(1),
    };
    match transform {
        PolynomialInputTransform::Identity => {
            let (mut input, mut scratch) = A::carve(module, &layout(src.k().as_usize()), scratch);
            input.set_meta(src.meta());
            module.ckks_copy(&mut input, src, &mut scratch)?;
            Ok((input, scratch))
        }
        PolynomialInputTransform::Square | PolynomialInputTransform::SquareTimesInput => {
            let k = crate::power_basis::square_ct_k(src)?;
            let (mut squared, mut scratch) = A::carve(module, &layout(k), scratch);
            module.ckks_square_into(&mut squared, src, tsk, &mut scratch)?;
            Ok((squared, scratch))
        }
        PolynomialInputTransform::ChebyshevT2 | PolynomialInputTransform::ChebyshevT2TimesInput => {
            let k = crate::power_basis::square_ct_k(src)?;
            let (mut doubled, mut scratch) = A::carve(module, &layout(k), scratch);
            scratch.scope(|scratch| -> Result<()> {
                let (mut squared, mut scratch) = scratch.take_ckks_ciphertext_scratch(&layout(k), GLWEPlaintextMeta::default());
                module.ckks_square_into(&mut squared, src, tsk, &mut scratch)?;
                module.ckks_double_into(&mut doubled, &squared, &mut scratch)?;
                module.ckks_sub_one_assign(&mut doubled, &mut scratch)
            })?;
            Ok((doubled, scratch))
        }
    }
}

pub(crate) fn ckks_eval_poly_real_const_coeffs_derived<BE: Backend, R, S, B, H, Rg: Ring>(
    module: &Module<BE>,
    dst: &mut R,
    src: &S,
    bsgs: &B,
    tsk: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    Module<BE>: crate::api::CKKSPolynomialEvaluationOps<BE, Rg>
        + CKKSCopyOps<BE, Rg>
        + CKKSMulOps<BE, Rg>
        + CKKSPow2Ops<BE, Rg>
        + CKKSSubOps<BE, Rg>
        + GLWENormalize<BE, Rg>,
    R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    S: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    B: BSGSPolynomialInfos<BE, Rg>,
    B::Coeffs: CKKSCtBounds,
    H: GetTensorKey<BE, Rg>,
    CKKSCiphertextOwned<BE, Rg>: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
{
    crate::ensure_operand_degrees("ckks_eval_poly_real_const_coeffs", module, [dst.n(), src.n()])?;
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
        + GLWENormalize<BE>,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    S: GLWEToBackendRef<BE> + CKKSCtBounds,
    C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds + IntPolyInfos,
    H: GetTensorKey<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    const OP: &str = "ckks_eval_poly_complex_const_coeffs";
    crate::ensure_operand_degrees(OP, module, [dst.n(), src.n()])?;
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
