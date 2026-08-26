//! Composite evaluation of prepared polynomial approximations.

use poulpy_core::layouts::{
    BSGSMeta, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, IntPolyInfos, LWEInfos, PowerBasisHelper, SetBSGSMeta,
    prepared::{GGLWEPreparedToBackendRef, GLWETensorKeyPrepared, GLWETensorKeyPreparedToBackendRef},
};
use poulpy_core::{GLWEBytesOf, GLWEShift};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, CKKSResult as Result, SetCKKSInfos,
    api::{
        Basis, CKKSAddOps, CKKSAffineOps, CKKSAllOpsTmpBytes, CKKSApproximationOps, CKKSCopyOps, CKKSMulOps,
        CKKSPolynomialEvaluationOps, CKKSPow2Ops, CKKSSubOps, PolynomialInputTransform,
    },
    ckks_ensure,
    layouts::{
        AdaptivePolynomialApproximation, AdaptivePolynomialEvaluationMode, CKKSCiphertextOwned, CKKSModuleAlloc,
        PolynomialApproximation, ScratchArenaTakeCKKS,
    },
    power_basis::{PowerBasis, PowerBasisGen, PowerBasisInsert},
};

impl<BE: Backend> CKKSApproximationOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE>
        + CKKSAffineOps<BE>
        + CKKSAllOpsTmpBytes<BE>
        + CKKSCopyOps<BE>
        + CKKSModuleAlloc<BE>
        + CKKSMulOps<BE>
        + CKKSPolynomialEvaluationOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSSubOps<BE>
        + GLWEShift<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
{
    fn ckks_approximation_tmp_bytes<R, I, T, P>(
        &self,
        res: &R,
        input: &I,
        tsk: &T,
        approximation: &PolynomialApproximation<P>,
    ) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + LWEInfos,
    {
        let coeffs = approximation.poly.baby_step(0);
        let eval = self
            .ckks_all_ops_tmp_bytes(res, tsk, coeffs)
            .max(self.ckks_all_ops_tmp_bytes(input, tsk, coeffs));
        match &approximation.affine {
            Some(affine) => {
                let ct = self.glwe_bytes_of_from_infos(input);
                let map = if approximation.scale_pow2.is_some() {
                    eval
                } else {
                    eval.max(self.ckks_affine_pt_const_tmp_bytes(input, input, affine))
                };
                ct + map
            }
            None => eval,
        }
    }

    fn ckks_eval_approximation<R, I, P>(
        &self,
        res: &mut R,
        input: &I,
        approximation: &PolynomialApproximation<P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos,
    {
        let required = approximation.consumed_bits(input.log_delta());
        ckks_ensure!(
            input.log_budget() >= required,
            "ckks_eval_approximation: log_budget {} < {required} bits required at log_delta {}",
            input.log_budget(),
            input.log_delta()
        );
        if let Some(affine) = &approximation.affine {
            scratch.scope(|scratch_local| {
                let (mut normalized, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(input);
                if let Some(exponent) = approximation.scale_pow2 {
                    self.ckks_copy(&mut normalized, input, &mut scratch_local)?;
                    if exponent < 0 {
                        self.ckks_div_pow2_assign(&mut normalized, exponent.unsigned_abs() as usize)?;
                    } else if exponent > 0 {
                        self.ckks_mul_pow2_assign(&mut normalized, exponent as usize, &mut scratch_local)?;
                    }
                    self.ckks_add_pt_const_assign(&mut normalized, 0, affine, 0, &mut scratch_local)?;
                } else {
                    self.ckks_affine_pt_const_into(&mut normalized, input, affine, 0, 1, &mut scratch_local)?;
                }
                self.ckks_eval_poly_real_const_coeffs(res, &normalized, &approximation.poly, tsk, &mut scratch_local)
            })
        } else {
            self.ckks_eval_poly_real_const_coeffs(res, input, &approximation.poly, tsk, scratch)
        }
    }

    fn ckks_adaptive_approximation_tmp_bytes<R, I, T, P>(
        &self,
        res: &R,
        input: &I,
        tsk: &T,
        approximation: &AdaptivePolynomialApproximation<P>,
    ) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + LWEInfos,
    {
        let low_coeffs = approximation.low.baby_step(0);
        let high_coeffs = approximation.high.baby_step(0);
        let eval = self
            .ckks_all_ops_tmp_bytes(res, tsk, low_coeffs)
            .max(self.ckks_all_ops_tmp_bytes(input, tsk, low_coeffs))
            .max(self.ckks_all_ops_tmp_bytes(res, tsk, high_coeffs))
            .max(self.ckks_all_ops_tmp_bytes(input, tsk, high_coeffs));
        match &approximation.affine {
            Some(affine) => {
                let ct = self.glwe_bytes_of_from_infos(input);
                let map = if approximation.scale_pow2.is_some() {
                    eval
                } else {
                    eval.max(self.ckks_affine_pt_const_tmp_bytes(input, input, affine))
                };
                ct + map
            }
            None => eval,
        }
    }

    fn ckks_eval_adaptive_approximation<R, I, P>(
        &self,
        res: &mut R,
        input: &I,
        approximation: &AdaptivePolynomialApproximation<P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos,
    {
        let eval_log_delta = approximation.output_log_delta(input.log_delta());
        if eval_log_delta < approximation.scale.power_drop_bits {
            return Err(CKKSCompositionError::InsufficientScalePrecision {
                op: "adaptive_approximation",
                available_log_delta: eval_log_delta,
                required_bits: approximation.scale.power_drop_bits,
            }
            .into());
        }
        let required = approximation.consumed_bits(input.log_delta());
        if input.log_budget() < required {
            return Err(CKKSCompositionError::InsufficientHomomorphicCapacity {
                op: "adaptive_approximation",
                available_log_budget: input.log_budget(),
                required_bits: required,
            }
            .into());
        }

        if let Some(affine) = &approximation.affine {
            scratch.scope(|scratch_local| {
                let (mut normalized, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(input);
                if let Some(exponent) = approximation.scale_pow2 {
                    self.ckks_copy(&mut normalized, input, &mut scratch_local)?;
                    if exponent < 0 {
                        self.ckks_div_pow2_assign(&mut normalized, exponent.unsigned_abs() as usize)?;
                    } else if exponent > 0 {
                        self.ckks_mul_pow2_assign(&mut normalized, exponent as usize, &mut scratch_local)?;
                    }
                    self.ckks_add_pt_const_assign(&mut normalized, 0, affine, 0, &mut scratch_local)?;
                } else {
                    self.ckks_affine_pt_const_into(&mut normalized, input, affine, 0, 1, &mut scratch_local)?;
                }
                eval_adaptive_from_normalized(self, res, &normalized, approximation, tsk, &mut scratch_local)
            })
        } else {
            eval_adaptive_from_normalized(self, res, input, approximation, tsk, scratch)
        }
    }
}

fn polynomial_input<BE, I>(
    module: &Module<BE>,
    input: &I,
    transform: PolynomialInputTransform,
    tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<CKKSCiphertextOwned<BE>>
where
    BE: Backend,
    Module<BE>: CKKSCopyOps<BE> + CKKSMulOps<BE> + CKKSPow2Ops<BE> + CKKSSubOps<BE> + CKKSModuleAlloc<BE>,
    I: GLWEToBackendRef<BE> + CKKSCtBounds,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
{
    let mut x = module.ckks_ciphertext_alloc_from_infos(input);
    module.ckks_copy(&mut x, input, scratch)?;
    match transform {
        PolynomialInputTransform::Identity => Ok(x),
        PolynomialInputTransform::Square | PolynomialInputTransform::SquareTimesInput => {
            let mut basis = PowerBasis::new(Basis::Monomial, x);
            basis.gen_power(2, module, tsk, scratch)?;
            Ok(basis.take_power(2).expect("generating x² must store the degree-two power"))
        }
        PolynomialInputTransform::ChebyshevT2 | PolynomialInputTransform::ChebyshevT2TimesInput => {
            let mut basis = PowerBasis::new(Basis::Chebyshev, x);
            basis.gen_power_chebyshev(2, module, tsk, scratch)?;
            Ok(basis.take_power(2).expect("generating T₂ must store the degree-two power"))
        }
    }
}

fn scale_down_assign<BE, C>(module: &Module<BE>, ct: &mut C, bits: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
where
    BE: Backend,
    Module<BE>: GLWEShift<BE>,
    C: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
{
    if bits == 0 {
        return Ok(());
    }
    if ct.log_delta() < bits {
        return Err(CKKSCompositionError::InsufficientScalePrecision {
            op: "adaptive_scale_down",
            available_log_delta: ct.log_delta(),
            required_bits: bits,
        }
        .into());
    }
    module.glwe_rsh(bits, ct, scratch);
    let mut meta = ct.meta();
    meta.log_delta -= bits;
    ct.set_meta(meta);
    Ok(())
}

// The physical shift clears the dirty top bits introduced by scale-down.
fn scale_up_assign<BE, C>(module: &Module<BE>, ct: &mut C, bits: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
where
    BE: Backend,
    Module<BE>: GLWEShift<BE>,
    C: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
{
    if bits == 0 {
        return Ok(());
    }
    if ct.log_budget() < bits {
        return Err(CKKSCompositionError::InsufficientHomomorphicCapacity {
            op: "adaptive_scale_up",
            available_log_budget: ct.log_budget(),
            required_bits: bits,
        }
        .into());
    }
    module.glwe_lsh_assign(ct, bits, scratch);
    let mut meta = ct.meta();
    meta.log_delta += bits;
    ct.set_meta(meta);
    Ok(())
}

fn eval_adaptive_from_normalized<BE, R, I, P>(
    module: &Module<BE>,
    res: &mut R,
    input: &I,
    approximation: &AdaptivePolynomialApproximation<P>,
    tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSMulOps<BE>
        + CKKSModuleAlloc<BE>
        + CKKSPolynomialEvaluationOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSSubOps<BE>
        + GLWEShift<BE>,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    I: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
{
    let x = polynomial_input(module, input, approximation.input_transform, tsk, scratch)?;
    let power_drop_bits = approximation.scale.power_drop_bits;
    let reuse_baby_powers = approximation.mode == AdaptivePolynomialEvaluationMode::ReuseFullScaleBabyPowers;

    let mut x_high = module.ckks_ciphertext_alloc_from_infos(&x);
    module.ckks_copy(&mut x_high, &x, scratch)?;
    let mut low_basis = PowerBasis::new(approximation.low.basis(), x);
    if reuse_baby_powers {
        low_basis.populate(
            approximation.high.base() - 1,
            approximation.high.log_split(),
            approximation.high.parity(),
            module,
            tsk,
            scratch,
        )?;
    } else {
        low_basis.populate(
            approximation.low.degree(),
            approximation.low.log_split(),
            approximation.low.parity(),
            module,
            tsk,
            scratch,
        )?;
    }

    scale_down_assign(module, &mut x_high, power_drop_bits, scratch)?;
    let mut high_basis = PowerBasis::new(approximation.high.basis(), x_high);
    if reuse_baby_powers {
        for power in 2..approximation.high.base() {
            if !low_basis.contains_power(power) {
                continue;
            }
            let source = low_basis.get(power)?;
            let mut reduced = module.ckks_ciphertext_alloc_from_infos(source);
            module.ckks_copy(&mut reduced, source, scratch)?;
            scale_down_assign(module, &mut reduced, power_drop_bits, scratch)?;
            high_basis.insert(power, reduced).map_err(crate::CKKSError::Internal)?;
        }
    }
    high_basis.populate(
        approximation.high.degree(),
        approximation.high.log_split(),
        approximation.high.parity(),
        module,
        tsk,
        scratch,
    )?;

    let mut high = module.ckks_ciphertext_alloc_from_infos(high_basis.get(1)?);
    module.ckks_eval_poly_real_const_coeffs_from_power_basis(&mut high, &approximation.high, &high_basis, tsk, scratch)?;
    scale_up_assign(module, &mut high, power_drop_bits, scratch)?;

    let mut low = module.ckks_ciphertext_alloc_from_infos(low_basis.get(1)?);
    module.ckks_eval_poly_real_const_coeffs_from_power_basis(&mut low, &approximation.low, &low_basis, tsk, scratch)?;
    if matches!(
        approximation.input_transform,
        PolynomialInputTransform::SquareTimesInput | PolynomialInputTransform::ChebyshevT2TimesInput
    ) {
        let mut combined = module.ckks_ciphertext_alloc_from_infos(&low);
        module.ckks_add_into(&mut combined, &high, &low, scratch)?;
        module.ckks_mul_into(res, &combined, input, tsk, scratch)?;
    } else {
        module.ckks_add_into(res, &high, &low, scratch)?;
    }
    Ok(())
}
