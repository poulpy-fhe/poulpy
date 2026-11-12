use crate::CKKSResult as Result;
use crate::api::{CKKSComplexPolynomialEvaluationOps, CKKSPolynomialEvaluationOps};
use poulpy_core::GLWENormalize;
use poulpy_core::layouts::{
    BSGSMeta, GLWEInfos, GLWEPlaintextMeta, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, IntPolyInfos, SetBSGSMeta,
};
use poulpy_core::oep::GLWEMulConstImpl;

use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{BSGSPolynomialInfos, CKKSCopyOps, CKKSMulOps, CKKSPow2Ops, CKKSSubOps, PolynomialInputTransform},
    layouts::{CKKSCiphertextOwned, ScratchArenaTakeCKKSCiphertext},
    polynomial::ComplexBSGSPolynomial,
    power_basis::{CKKSPowerBasisGen, PowerBasis, PowerStorage, populate_in, power_layout, square_ct_k},
    reference::polynomial_evaluation::{ensure_baby_steps, ensure_complex_schedules},
};

/// Scratch upper bound for [`GLWEBSGSOps::eval_baby_linear_combination`]: the constant combination
/// for the widest term (`ct_size`) and constant (`pt_size`) in limbs.
pub(crate) fn eval_baby_linear_combination_tmp_bytes<BE, Rg: Ring>(module: &Module<BE>, ct_size: usize, pt_size: usize) -> usize
where
    BE: GLWEMulConstImpl<Rg>,
{
    BE::glwe_const_combination_tmp_bytes(module, ct_size + pt_size, ct_size, pt_size)
}

/// The folded input (`x`, `x²`, or `T₂(x)`) of `src` for one-shot evaluation, in the storage of `A`, with what is left
/// of `scratch`. The prepared-basis kernels are selected through the backend contract.
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
    match transform {
        PolynomialInputTransform::Identity => {
            let (mut input, mut scratch) = A::carve(module, &power_layout(src, src.k().as_usize()), scratch);
            input.set_meta(src.meta());
            module.ckks_copy(&mut input, src, &mut scratch)?;
            Ok((input, scratch))
        }
        PolynomialInputTransform::Square | PolynomialInputTransform::SquareTimesInput => {
            let k = square_ct_k(src)?;
            let (mut squared, mut scratch) = A::carve(module, &power_layout(src, k), scratch);
            module.ckks_square_into(&mut squared, src, tsk, &mut scratch)?;
            Ok((squared, scratch))
        }
        PolynomialInputTransform::ChebyshevT2 | PolynomialInputTransform::ChebyshevT2TimesInput => {
            let layout = power_layout(src, square_ct_k(src)?);
            let (mut doubled, mut scratch) = A::carve(module, &layout, scratch);
            scratch.scope(|scratch| -> Result<()> {
                let (mut squared, mut scratch) = scratch.take_ckks_ciphertext_scratch(&layout, GLWEPlaintextMeta::default());
                module.ckks_square_into(&mut squared, src, tsk, &mut scratch)?;
                module.ckks_double_into(&mut doubled, &squared, &mut scratch)?;
                module.ckks_sub_one_assign(&mut doubled, &mut scratch)
            })?;
            Ok((doubled, scratch))
        }
    }
}

/// `bsgs` on the power basis of `x1`, its first power, into `dst`: the powers in the storage of `A`, then the
/// evaluation from them. Returns what is left of `scratch`.
pub(crate) fn eval_poly_real_on_power<'a, BE, A, R, B, H, Rg: Ring>(
    module: &Module<BE>,
    dst: &mut R,
    x1: A,
    bsgs: &B,
    tsk: &H,
    scratch: ScratchArena<'a, BE>,
) -> Result<ScratchArena<'a, BE>>
where
    BE: Backend,
    A: PowerStorage<'a, BE, Rg> + BSGSMeta,
    Module<BE>: CKKSPolynomialEvaluationOps<BE, Rg>
        + CKKSMulOps<BE, Rg>
        + CKKSPow2Ops<BE, Rg>
        + CKKSSubOps<BE, Rg>
        + GLWENormalize<BE, Rg>,
    R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    B: BSGSPolynomialInfos<BE, Rg>,
    B::Coeffs: CKKSCtBounds,
    H: GetTensorKey<BE, Rg>,
{
    let mut power_basis = PowerBasis::new(bsgs.basis(), x1);
    let mut scratch = populate_in(
        &mut power_basis,
        bsgs.degree(),
        bsgs.log_split(),
        bsgs.parity(),
        module,
        tsk,
        scratch,
    )?;
    module.ckks_eval_poly_real_const_coeffs_from_power_basis(dst, &power_basis, bsgs, tsk, &mut scratch)?;
    Ok(scratch)
}

/// The real polynomial `bsgs` of `src` into `dst`, its powers in the storage of `A`: the input transform of `bsgs`, its
/// power basis, the evaluation from it, and the product by `src` of a transform times the input. `op` names the
/// caller in the refusals.
pub(crate) fn eval_poly_real_in<'a, BE, A, R, S, B, H, Rg: Ring>(
    op: &'static str,
    module: &Module<BE>,
    dst: &mut R,
    src: &S,
    bsgs: &B,
    tsk: &H,
    scratch: ScratchArena<'a, BE>,
) -> Result<()>
where
    BE: Backend,
    A: PowerStorage<'a, BE, Rg> + BSGSMeta,
    Module<BE>: CKKSPolynomialEvaluationOps<BE, Rg>
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
{
    ensure_baby_steps(op, bsgs)?;
    let transform = bsgs.input_transform();
    let (x1, scratch) = polynomial_input_in::<_, A, _, _, Rg>(module, src, transform, tsk, scratch)?;
    let mut scratch = eval_poly_real_on_power(module, dst, x1, bsgs, tsk, scratch)?;
    if matches!(
        transform,
        PolynomialInputTransform::SquareTimesInput | PolynomialInputTransform::ChebyshevT2TimesInput
    ) {
        module.ckks_mul_assign(dst, src, tsk, &mut scratch)?;
    }
    Ok(())
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
    Module<BE>: CKKSPolynomialEvaluationOps<BE, Rg>
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
    const OP: &str = "ckks_eval_poly_real_const_coeffs";
    crate::ensure_operand_degrees(OP, module, [dst.n(), src.n()])?;
    eval_poly_real_in::<_, CKKSCiphertextOwned<BE, Rg>, _, _, _, _, Rg>(OP, module, dst, src, bsgs, tsk, scratch.borrow())
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
    ensure_complex_schedules::<BE, C>(OP, poly)?;
    let transform = poly.re.input_transform();
    crate::ckks_ensure_operand!(
        OP,
        transform == poly.im.input_transform(),
        "real and imaginary input transforms differ"
    );
    let (x1, _) = polynomial_input_in::<_, CKKSCiphertextOwned<BE>, _, _, _>(module, src, transform, tsk, scratch.borrow())?;
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
