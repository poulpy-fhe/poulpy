//! Approximations compose the interval map with the selected polynomial evaluation.
use poulpy_core::{
    GLWEBytesOf,
    layouts::{
        BSGSMeta, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, IntPolyInfos, LWEInfos, SetBSGSMeta,
        SetGLWEPlaintextInfos,
    },
};
use poulpy_hal::layouts::{Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSResult as Result, SetCKKSInfos, SlotsKind,
    api::{CKKSAddOps, CKKSAffineOps, CKKSAllOpsTmpBytes, CKKSCopyOps, CKKSPolynomialEvaluationOps, CKKSPow2Ops},
    checked_log_budget_sub, ckks_ensure,
    layouts::{PolynomialApproximation, ScratchArenaTakeCKKSCiphertext},
    oep::CKKSApproximationImpl,
};

pub(crate) fn ckks_polynomial_approximation_eval_tmp_bytes_derived<BE, R, I, T, P>(
    module: &Module<BE>,
    res: &R,
    input: &I,
    approximation: &PolynomialApproximation<P>,
    tsk: &T,
) -> usize
where
    BE: CKKSApproximationImpl,
    Module<BE>: CKKSAllOpsTmpBytes<BE>,
    R: CKKSCtBounds,
    I: CKKSCtBounds,
    T: GGLWEInfos,
    P: CKKSInfos + LWEInfos,
{
    let coeffs = approximation.poly.baby_step(0);
    let eval = module
        .ckks_all_ops_tmp_bytes(res, coeffs, tsk)
        .max(module.ckks_all_ops_tmp_bytes(input, coeffs, tsk));
    match &approximation.affine {
        Some(affine) => {
            let ct = BE::scratch_aligned(module.glwe_bytes_of_from_infos(input));
            let map = if approximation.scale_pow2.is_some() {
                eval
            } else {
                eval.max(module.ckks_affine_pt_const_tmp_bytes(input, input, affine))
            };
            ct + map
        }
        None => eval,
    }
}

/// The input-shaped temporary of an affine map plus the larger of its map step's selected queries, sized without a
/// tensor key: the polynomial evaluation that follows checks its constituents' queries itself.
fn map_tmp_bytes<BE, I, P>(module: &Module<BE>, input: &I, approximation: &PolynomialApproximation<P>, affine: &P) -> usize
where
    BE: CKKSApproximationImpl,
    I: CKKSCtBounds,
    P: CKKSCtBounds,
{
    let map = if approximation.scale_pow2.is_some() {
        module
            .ckks_mul_pow2_tmp_bytes(input.size())
            .max(module.ckks_copy_tmp_bytes(input, input))
            .max(module.ckks_add_pt_const_tmp_bytes(input.size()))
    } else {
        module.ckks_affine_pt_const_tmp_bytes(input, input, affine)
    };
    BE::scratch_aligned(module.glwe_bytes_of_from_infos(input)) + map
}

pub(crate) fn ckks_polynomial_approximation_eval_derived<BE, R, I, P, H>(
    module: &Module<BE>,
    res: &mut R,
    input: &I,
    approximation: &PolynomialApproximation<P>,
    tsk: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSApproximationImpl,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    I: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos,
    H: GetTensorKey<BE>,
{
    const OP: &str = "ckks_polynomial_approximation_eval";
    checked_log_budget_sub(OP, input.log_budget(), approximation.consumed_bits(input.log_delta()))?;
    if let Some(affine) = &approximation.affine {
        ckks_ensure!(
            scratch.available() >= map_tmp_bytes(module, input, approximation, affine),
            "{OP}: scratch below its query"
        );
        scratch.scope(|scratch_local| {
            let (mut normalized, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(input);
            if let Some(exponent) = approximation.scale_pow2 {
                if exponent > 0 {
                    module.ckks_mul_pow2_into(exponent as usize, &mut normalized, input, &mut scratch_local)?;
                } else {
                    module.ckks_copy(&mut normalized, input, &mut scratch_local)?;
                    normalized.set_slots(input.slots().join(SlotsKind::Real));
                    module.ckks_div_pow2_assign(exponent.unsigned_abs() as usize, &mut normalized)?;
                }
                module.ckks_add_pt_const_assign(&mut normalized, 0, affine, 0, &mut scratch_local)?;
            } else {
                module.ckks_affine_pt_const_into(&mut normalized, input, affine, 0, 1, &mut scratch_local)?;
            }
            module.ckks_eval_poly_real_const_coeffs(res, &normalized, &approximation.poly, tsk, &mut scratch_local)
        })
    } else {
        module.ckks_eval_poly_real_const_coeffs(res, input, &approximation.poly, tsk, scratch)
    }
}
