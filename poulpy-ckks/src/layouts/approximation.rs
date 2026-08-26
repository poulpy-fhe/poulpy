//! Prepared interval-mapped polynomial approximations.

use std::{collections::HashMap, fmt::Debug};

use anyhow::{Result, anyhow, ensure};
use num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};
use poulpy_core::layouts::Base2K;
use poulpy_hal::layouts::{HostBytesBackend, Module};

use crate::{
    CoeffsMeta, SetCKKSInfos,
    layouts::{CKKSModuleAlloc, CKKSPlaintextOwned, CKKSPlaintextVecHostCodec, CKKSScalar},
    polynomial::{BSGSPolynomial, Basis, EncodeBSGS, Polynomial, PolynomialInputTransform, SplitStrategy, split_degree},
};

/// Power-basis construction strategy for an adaptive approximation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AdaptivePolynomialEvaluationMode {
    /// Reuse full-scale baby powers in the reduced-scale high branch.
    #[default]
    ReuseFullScaleBabyPowers,
    /// Recompute every high-branch power at reduced scale.
    RecomputeReducedScalePowers,
}

/// Independent precision reductions for the adaptive high branch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaptiveScalePolicy {
    /// Bits removed from the high branch's ciphertext-power scale.
    pub power_drop_bits: usize,
    /// Bits removed from the high branch's coefficient scale.
    pub coefficient_drop_bits: usize,
}

impl AdaptiveScalePolicy {
    /// Creates an independently tuned scale policy.
    pub const fn new(power_drop_bits: usize, coefficient_drop_bits: usize) -> Self {
        Self {
            power_drop_bits,
            coefficient_drop_bits,
        }
    }

    /// Applies the same reduction to powers and coefficients.
    pub const fn uniform(drop_bits: usize) -> Self {
        Self::new(drop_bits, drop_bits)
    }
}

/// A BSGS polynomial prepared for homomorphic evaluation, together with the
/// optional affine map from its input interval to its coefficient variable.
pub struct PolynomialApproximation<P> {
    /// BSGS polynomial in its coefficient variable.
    pub poly: BSGSPolynomial<P>,
    /// Packed `[offset, scale]` interval map, if needed.
    pub affine: Option<P>,
    /// Exact power-of-two scale used instead of plaintext multiplication.
    pub scale_pow2: Option<i32>,
    /// Scale of the prepared plaintexts.
    pub coeff_log_delta: usize,
}

impl PolynomialApproximation<CKKSPlaintextOwned<HostBytesBackend>> {
    /// Prepares `poly` and its interval map on the host.
    pub fn from_polynomial<F, M>(
        poly: &Polynomial<F>,
        base2k: Base2K,
        coeff_meta: M,
        strategy: SplitStrategy,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        M: Into<CoeffsMeta>,
        CKKSPlaintextOwned<HostBytesBackend>: CKKSPlaintextVecHostCodec<F>,
    {
        let coeff_meta = coeff_meta.into();
        let bsgs = poly
            .encode_bsgs_with(module, base2k, coeff_meta, strategy)
            .map_err(|e| anyhow!("polynomial approximation: {e}"))?;
        let (scale, offset) = poly.change_of_basis();
        let affine = if scale == F::one() && offset == F::zero() {
            None
        } else {
            let mut pt = module.ckks_pt_coeffs_alloc(2, base2k, coeff_meta.k);
            pt.set_meta(coeff_meta.meta);
            pt.encode_host_floats(&[offset, scale])
                .map_err(|e| anyhow!("polynomial approximation: affine encoding failed: {e}"))?;
            Some(pt)
        };
        let scale_pow2 = affine.as_ref().and_then(|_| exact_power_of_two_exponent(scale));
        Ok(Self {
            poly: bsgs,
            affine,
            scale_pow2,
            coeff_log_delta: coeff_meta.log_delta(),
        })
    }

    /// Prepares an even or odd polynomial through its quadratic parity fold.
    pub fn from_polynomial_folded<F, M>(
        poly: &Polynomial<F>,
        base2k: Base2K,
        coeff_meta: M,
        strategy: SplitStrategy,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        M: Into<CoeffsMeta>,
        CKKSPlaintextOwned<HostBytesBackend>: CKKSPlaintextVecHostCodec<F>,
    {
        let coeff_meta = coeff_meta.into();
        let bsgs = poly
            .encode_bsgs_folded_with(module, base2k, coeff_meta, strategy)
            .map_err(|e| anyhow!("folded polynomial approximation: {e}"))?;
        let (affine, scale_pow2) = encode_affine(poly, base2k, coeff_meta, module)?;
        Ok(Self {
            poly: bsgs,
            affine,
            scale_pow2,
            coeff_log_delta: coeff_meta.log_delta(),
        })
    }
}

/// Adaptive BSGS with a full-scale low branch and a tuned high branch.
pub struct AdaptivePolynomialApproximation<P> {
    /// Terms below the high branch's BSGS baby-step base, encoded at full scale.
    pub low: BSGSPolynomial<P>,
    /// Remaining terms, encoded at the selected coefficient scale.
    pub high: BSGSPolynomial<P>,
    /// Packed `[offset, scale]` interval map, if needed.
    pub affine: Option<P>,
    /// Exact power-of-two scale used instead of plaintext multiplication.
    pub scale_pow2: Option<i32>,
    /// Full scale of the prepared coefficients.
    pub coeff_log_delta: usize,
    /// Precision policy for the high branch.
    pub scale: AdaptiveScalePolicy,
    /// Power-basis construction strategy used during evaluation.
    pub mode: AdaptivePolynomialEvaluationMode,
    /// Optional quadratic parity fold applied before both branches.
    pub input_transform: PolynomialInputTransform,
    source_degree: usize,
    source_interval: (f64, f64),
}

impl AdaptivePolynomialApproximation<CKKSPlaintextOwned<HostBytesBackend>> {
    /// Prepares a direct adaptive evaluation.
    #[allow(clippy::too_many_arguments)]
    pub fn from_polynomial<F, M>(
        poly: &Polynomial<F>,
        base2k: Base2K,
        coeff_meta: M,
        strategy: SplitStrategy,
        scale: AdaptiveScalePolicy,
        mode: AdaptivePolynomialEvaluationMode,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        M: Into<CoeffsMeta>,
        CKKSPlaintextOwned<HostBytesBackend>: CKKSPlaintextVecHostCodec<F>,
    {
        let eval_poly = Polynomial::new_with_parity(poly.basis, poly.coeffs.clone(), poly.parity);
        Self::prepare(
            poly,
            eval_poly,
            PolynomialInputTransform::Identity,
            base2k,
            coeff_meta.into(),
            strategy,
            scale,
            mode,
            module,
        )
    }

    /// Prepares an adaptive evaluation with a quadratic parity fold.
    #[allow(clippy::too_many_arguments)]
    pub fn from_polynomial_folded<F, M>(
        poly: &Polynomial<F>,
        base2k: Base2K,
        coeff_meta: M,
        strategy: SplitStrategy,
        scale: AdaptiveScalePolicy,
        mode: AdaptivePolynomialEvaluationMode,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        M: Into<CoeffsMeta>,
        CKKSPlaintextOwned<HostBytesBackend>: CKKSPlaintextVecHostCodec<F>,
    {
        let (eval_poly, input_transform) = poly.fold_parity()?;
        Self::prepare(
            poly,
            eval_poly,
            input_transform,
            base2k,
            coeff_meta.into(),
            strategy,
            scale,
            mode,
            module,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare<F>(
        source: &Polynomial<F>,
        eval_poly: Polynomial<F>,
        input_transform: PolynomialInputTransform,
        base2k: Base2K,
        coeff_meta: CoeffsMeta,
        strategy: SplitStrategy,
        scale: AdaptiveScalePolicy,
        mode: AdaptivePolynomialEvaluationMode,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        CKKSPlaintextOwned<HostBytesBackend>: CKKSPlaintextVecHostCodec<F>,
    {
        ensure!(
            scale.power_drop_bits > 0 || scale.coefficient_drop_bits > 0,
            "adaptive polynomial approximation: scale policy must reduce powers or coefficients"
        );
        ensure!(
            scale.coefficient_drop_bits < coeff_meta.log_delta(),
            "adaptive polynomial approximation: coefficient_drop_bits {} must be smaller than coefficient log_delta {}",
            scale.coefficient_drop_bits,
            coeff_meta.log_delta()
        );
        ensure!(
            scale.coefficient_drop_bits < coeff_meta.k.as_usize(),
            "adaptive polynomial approximation: coefficient_drop_bits {} must be smaller than coefficient k {}",
            scale.coefficient_drop_bits,
            coeff_meta.k.as_usize()
        );
        let degree = eval_poly.degree();
        let log_split = eval_poly.bsgs_log_split(strategy)?;
        let split = 1usize << log_split;
        ensure!(
            split <= degree,
            "adaptive polynomial approximation: BSGS split {split} exceeds evaluation degree {degree}"
        );

        let mut low_coeffs = eval_poly.coeffs.clone();
        low_coeffs.truncate(split);
        let mut high_coeffs = eval_poly.coeffs.clone();
        for coefficient in high_coeffs.iter_mut().take(split) {
            *coefficient = F::zero();
        }

        let low_poly = Polynomial::new_with_parity(eval_poly.basis, low_coeffs, eval_poly.parity);
        let high_poly = Polynomial::new_with_parity(eval_poly.basis, high_coeffs, eval_poly.parity);
        let low = low_poly
            .encode_bsgs_with(module, base2k, coeff_meta, strategy)
            .map_err(|e| anyhow!("adaptive polynomial approximation low branch: {e}"))?;
        let mut high_meta = coeff_meta;
        high_meta.k = (coeff_meta.k.as_usize() - scale.coefficient_drop_bits).into();
        high_meta.meta.log_delta -= scale.coefficient_drop_bits;
        let high = high_poly
            .encode_bsgs_with(module, base2k, high_meta, strategy)
            .map_err(|e| anyhow!("adaptive polynomial approximation high branch: {e}"))?;
        let (affine, scale_pow2) = encode_affine(source, base2k, coeff_meta, module)?;

        Ok(Self {
            low,
            high,
            affine,
            scale_pow2,
            coeff_log_delta: coeff_meta.log_delta(),
            scale,
            mode,
            input_transform,
            source_degree: source.degree(),
            source_interval: (
                source.a.to_f64().expect("interval lower bound must convert to f64"),
                source.b.to_f64().expect("interval upper bound must convert to f64"),
            ),
        })
    }
}

fn encode_affine<F>(
    poly: &Polynomial<F>,
    base2k: Base2K,
    coeff_meta: CoeffsMeta,
    module: &Module<HostBytesBackend>,
) -> Result<(Option<CKKSPlaintextOwned<HostBytesBackend>>, Option<i32>)>
where
    F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
    CKKSPlaintextOwned<HostBytesBackend>: CKKSPlaintextVecHostCodec<F>,
{
    let (scale, offset) = poly.change_of_basis();
    let affine = if scale == F::one() && offset == F::zero() {
        None
    } else {
        let mut pt = module.ckks_pt_coeffs_alloc(2, base2k, coeff_meta.k);
        pt.set_meta(coeff_meta.meta);
        pt.encode_host_floats(&[offset, scale])
            .map_err(|e| anyhow!("polynomial approximation: affine encoding failed: {e}"))?;
        Some(pt)
    };
    let scale_pow2 = affine.as_ref().and_then(|_| exact_power_of_two_exponent(scale));
    Ok((affine, scale_pow2))
}

/// Returns `e` exactly when `value == 2^e` in `F`.
fn exact_power_of_two_exponent<F>(value: F) -> Option<i32>
where
    F: Float + FromPrimitive + ToPrimitive,
{
    if !value.is_finite() || value <= F::zero() {
        return None;
    }
    let exponent = value.log2().round().to_i32()?;
    let candidate = F::from_i32(exponent)?.exp2();
    (candidate == value).then_some(exponent)
}

impl<P> PolynomialApproximation<P> {
    /// Consumed modulus bits.
    pub fn consumed_bits(&self, input_log_delta: usize) -> usize {
        let affine = match self.scale_pow2 {
            Some(exponent) if exponent < 0 => exponent.unsigned_abs() as usize,
            Some(_) => 0,
            None => usize::from(self.affine.is_some()) * self.coeff_log_delta,
        };
        affine
            + self
                .poly
                .consumed_bits(self.output_log_delta(input_log_delta), self.coeff_log_delta)
    }

    /// Output scale after evaluation.
    ///
    /// Exact division by a power of two is represented by increasing the
    /// ciphertext scale. Other interval maps preserve the input scale.
    pub fn output_log_delta(&self, input_log_delta: usize) -> usize {
        match self.scale_pow2 {
            Some(exponent) if exponent < 0 => input_log_delta + exponent.unsigned_abs() as usize,
            _ => input_log_delta,
        }
    }

    /// Multiplicative depth.
    pub fn depth(&self) -> usize {
        usize::from(self.affine.is_some() && self.scale_pow2.is_none()) + self.poly.eval_depth()
    }

    /// Input interval.
    pub fn interval(&self) -> (f64, f64) {
        self.poly.interval()
    }

    /// Source polynomial degree, before an optional parity fold.
    pub fn degree(&self) -> usize {
        self.poly.source_degree()
    }

    /// Rebuilds the approximation by mapping each borrowed prepared plaintext.
    pub fn map_plaintexts<Q>(self, mut f: impl FnMut(&P) -> Q) -> PolynomialApproximation<Q> {
        PolynomialApproximation {
            poly: self.poly.map_baby_steps_ref(&mut f),
            affine: self.affine.as_ref().map(f),
            scale_pow2: self.scale_pow2,
            coeff_log_delta: self.coeff_log_delta,
        }
    }
}

impl<P> AdaptivePolynomialApproximation<P> {
    /// Consumed modulus bits, including interval mapping and parity folding.
    pub fn consumed_bits(&self, input_log_delta: usize) -> usize
    where
        P: poulpy_core::layouts::LWEInfos,
    {
        let affine = affine_consumed_bits(&self.affine, self.scale_pow2, self.coeff_log_delta);
        let eval_log_delta = self.output_log_delta(input_log_delta);
        let polynomial = self.polynomial_consumed_bits(eval_log_delta);
        affine + polynomial + input_transform_consumed_bits(self.input_transform, eval_log_delta)
    }

    /// Output scale after evaluation.
    pub fn output_log_delta(&self, input_log_delta: usize) -> usize {
        match self.scale_pow2 {
            Some(exponent) if exponent < 0 => input_log_delta + exponent.unsigned_abs() as usize,
            _ => input_log_delta,
        }
    }

    /// Multiplicative depth, including interval mapping and parity folding.
    pub fn depth(&self) -> usize {
        usize::from(self.affine.is_some() && self.scale_pow2.is_none())
            + self.high.eval_depth()
            + input_transform_depth(self.input_transform)
    }

    /// Source polynomial degree, before an optional parity fold.
    pub fn degree(&self) -> usize {
        self.source_degree
    }

    /// Input approximation interval.
    pub fn interval(&self) -> (f64, f64) {
        self.source_interval
    }

    /// Rebuilds the plan by mapping each borrowed prepared plaintext.
    pub fn map_plaintexts<Q>(self, mut f: impl FnMut(&P) -> Q) -> AdaptivePolynomialApproximation<Q> {
        AdaptivePolynomialApproximation {
            low: self.low.map_baby_steps_ref(&mut f),
            high: self.high.map_baby_steps_ref(&mut f),
            affine: self.affine.as_ref().map(f),
            scale_pow2: self.scale_pow2,
            coeff_log_delta: self.coeff_log_delta,
            scale: self.scale,
            mode: self.mode,
            input_transform: self.input_transform,
            source_degree: self.source_degree,
            source_interval: self.source_interval,
        }
    }

    fn polynomial_consumed_bits(&self, input_log_delta: usize) -> usize
    where
        P: poulpy_core::layouts::LWEInfos,
    {
        let low = self
            .low
            .consumed_bits_with_power_cost(input_log_delta, self.coeff_log_delta, |degree| {
                power_depth(degree) * input_log_delta
            });
        let reduced_delta = input_log_delta.saturating_sub(self.scale.power_drop_bits);
        let reduced_coeff_delta = self.coeff_log_delta - self.scale.coefficient_drop_bits;
        let reuse_baby_powers = self.mode == AdaptivePolynomialEvaluationMode::ReuseFullScaleBabyPowers;
        let high = if reuse_baby_powers {
            let mut power_costs = HashMap::new();
            self.high
                .consumed_bits_with_power_cost(reduced_delta, reduced_coeff_delta, |degree| {
                    reuse_power_cost(
                        degree,
                        self.high.base(),
                        self.high.basis(),
                        input_log_delta,
                        reduced_delta,
                        &mut power_costs,
                    )
                })
        } else {
            self.high
                .consumed_bits_with_power_cost(reduced_delta, reduced_coeff_delta, |degree| {
                    power_depth(degree) * reduced_delta
                })
        };
        low.max(high)
    }
}

fn reuse_power_cost(
    degree: usize,
    base: usize,
    basis: Basis,
    input_log_delta: usize,
    reduced_log_delta: usize,
    cached: &mut HashMap<usize, usize>,
) -> usize {
    if let Some(&cost) = cached.get(&degree) {
        return cost;
    }
    let cost = if degree <= 1 {
        0
    } else if degree < base {
        power_depth(degree) * input_log_delta
    } else {
        let (a, b) = split_degree(degree);
        let product = reuse_power_cost(a, base, basis, input_log_delta, reduced_log_delta, cached).max(reuse_power_cost(
            b,
            base,
            basis,
            input_log_delta,
            reduced_log_delta,
            cached,
        )) + reduced_log_delta;
        if basis == Basis::Chebyshev {
            let difference = a.abs_diff(b);
            if difference > 0 {
                product.max(reuse_power_cost(
                    difference,
                    base,
                    basis,
                    input_log_delta,
                    reduced_log_delta,
                    cached,
                ))
            } else {
                product
            }
        } else {
            product
        }
    };
    cached.insert(degree, cost);
    cost
}

fn power_depth(degree: usize) -> usize {
    if degree <= 1 {
        0
    } else {
        let (a, b) = split_degree(degree);
        power_depth(a).max(power_depth(b)) + 1
    }
}

fn affine_consumed_bits<P>(affine: &Option<P>, scale_pow2: Option<i32>, coeff_log_delta: usize) -> usize {
    match scale_pow2 {
        Some(exponent) if exponent < 0 => exponent.unsigned_abs() as usize,
        Some(_) => 0,
        None => usize::from(affine.is_some()) * coeff_log_delta,
    }
}

fn input_transform_depth(transform: PolynomialInputTransform) -> usize {
    match transform {
        PolynomialInputTransform::Identity => 0,
        PolynomialInputTransform::Square | PolynomialInputTransform::ChebyshevT2 => 1,
        PolynomialInputTransform::SquareTimesInput | PolynomialInputTransform::ChebyshevT2TimesInput => 2,
    }
}

fn input_transform_consumed_bits(transform: PolynomialInputTransform, input_log_delta: usize) -> usize {
    input_transform_depth(transform) * input_log_delta
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CKKSInfos, Quad, polynomial::Parity};

    fn prepare(a: f64, b: f64) -> PolynomialApproximation<CKKSPlaintextOwned<HostBytesBackend>> {
        let module = Module::<HostBytesBackend>::new(256);
        let poly = Polynomial::chebyshev_interpolate(4, a, b, |x| x * x).unwrap();
        PolynomialApproximation::from_polynomial(
            &poly,
            Base2K(19),
            CoeffsMeta::from_delta_budget(30, 19),
            SplitStrategy::MinDepth,
            &module,
        )
        .unwrap()
    }

    #[test]
    fn negative_power_of_two_map_updates_scale_and_cost() {
        let plan = prepare(-4.0, 4.0);
        assert_eq!(plan.scale_pow2, Some(-2));
        assert_eq!(plan.output_log_delta(30), 32);
        assert_eq!(plan.consumed_bits(30), 2 + plan.poly.consumed_bits(32, 30));
    }

    #[test]
    fn plaintext_affine_map_preserves_scale_and_charges_coefficients() {
        let plan = prepare(0.0, 3.0);
        assert_eq!(plan.scale_pow2, None);
        assert_eq!(plan.output_log_delta(30), 30);
        assert_eq!(plan.consumed_bits(30), 30 + plan.poly.consumed_bits(30, 30));
    }

    #[test]
    fn quad_near_power_of_two_is_not_exact() {
        let near_one = Quad::new(1.0f128 + 2.0f128.powi(-60));
        assert_eq!(exact_power_of_two_exponent(near_one), None);
        assert_eq!(exact_power_of_two_exponent(Quad::new(0.25)), Some(-2));
    }

    #[test]
    fn adaptive_validates_scale_policy_and_high_branch() {
        let module = Module::<HostBytesBackend>::new(256);
        let metadata = CoeffsMeta::from_delta_budget(8, 10);
        let polynomial = Polynomial::new(Basis::Chebyshev, vec![1.0f64; 8]);
        for (scale, message) in [
            (AdaptiveScalePolicy::new(0, 0), "scale policy"),
            (AdaptiveScalePolicy::new(0, 8), "coefficient_drop_bits"),
        ] {
            let error = AdaptivePolynomialApproximation::from_polynomial(
                &polynomial,
                Base2K(19),
                metadata,
                SplitStrategy::MinDepth,
                scale,
                AdaptivePolynomialEvaluationMode::ReuseFullScaleBabyPowers,
                &module,
            )
            .err()
            .expect("invalid scale policy must fail");
            assert!(error.to_string().contains(message));
        }

        for scale in [AdaptiveScalePolicy::new(1, 0), AdaptiveScalePolicy::new(0, 1)] {
            let approximation = AdaptivePolynomialApproximation::from_polynomial(
                &polynomial,
                Base2K(19),
                metadata,
                SplitStrategy::MinDepth,
                scale,
                AdaptivePolynomialEvaluationMode::ReuseFullScaleBabyPowers,
                &module,
            )
            .expect("either scale component may be zero");
            assert_eq!(approximation.scale, scale);
            assert_eq!(approximation.high.baby_step(0).log_delta(), 8 - scale.coefficient_drop_bits);
        }

        let linear = Polynomial::new(Basis::Chebyshev, vec![1.0f64, 1.0]);
        let error = AdaptivePolynomialApproximation::from_polynomial(
            &linear,
            Base2K(19),
            metadata,
            SplitStrategy::MinDepth,
            AdaptiveScalePolicy::uniform(1),
            AdaptivePolynomialEvaluationMode::ReuseFullScaleBabyPowers,
            &module,
        )
        .err()
        .expect("a polynomial without a high BSGS branch must fail");
        assert!(error.to_string().contains("exceeds evaluation degree"));

        let constant = Polynomial::new(Basis::Chebyshev, vec![1.0f64]);
        let error = AdaptivePolynomialApproximation::from_polynomial(
            &constant,
            Base2K(19),
            metadata,
            SplitStrategy::MinDepth,
            AdaptiveScalePolicy::uniform(1),
            AdaptivePolynomialEvaluationMode::ReuseFullScaleBabyPowers,
            &module,
        )
        .err()
        .expect("a constant polynomial must fail without panicking");
        assert!(error.to_string().contains("degree ≥ 1"));
    }

    #[test]
    fn adaptive_fold_keeps_transform_outside_both_branches() {
        let module = Module::<HostBytesBackend>::new(256);
        let coefficients: Vec<f64> = (0usize..=30)
            .map(|degree| {
                if degree.is_multiple_of(2) {
                    1.0 / (degree + 1) as f64
                } else {
                    0.0
                }
            })
            .collect();
        let polynomial = Polynomial::new_with_parity(Basis::Chebyshev, coefficients, Parity::Even);
        let standard = PolynomialApproximation::from_polynomial_folded(
            &polynomial,
            Base2K(19),
            CoeffsMeta::from_delta_budget(20, 10),
            SplitStrategy::MinDepth,
            &module,
        )
        .unwrap();
        assert_eq!(standard.degree(), 30);
        assert_eq!(standard.poly.degree(), 15);

        let approximation = AdaptivePolynomialApproximation::from_polynomial_folded(
            &polynomial,
            Base2K(19),
            CoeffsMeta::from_delta_budget(20, 10),
            SplitStrategy::MinDepth,
            AdaptiveScalePolicy::uniform(3),
            AdaptivePolynomialEvaluationMode::RecomputeReducedScalePowers,
            &module,
        )
        .unwrap();

        assert_eq!(approximation.input_transform, PolynomialInputTransform::ChebyshevT2);
        assert_eq!(approximation.low.input_transform(), PolynomialInputTransform::Identity);
        assert_eq!(approximation.high.input_transform(), PolynomialInputTransform::Identity);
        assert_eq!(approximation.low.parity(), Parity::Full);
        assert_eq!(approximation.high.parity(), Parity::Full);
        assert_eq!(approximation.degree(), 30);
    }
}
