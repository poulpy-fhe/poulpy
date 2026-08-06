//! Prepared interval-mapped polynomial approximations.

use std::fmt::Debug;

use anyhow::{Result, anyhow};
use num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};
use poulpy_core::layouts::Base2K;
use poulpy_hal::layouts::{HostBytesBackend, Module};

use crate::{
    CoeffsMeta, SetCKKSInfos,
    layouts::{CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar},
    polynomial::{BSGSPolynomial, EncodeBSGS, Polynomial, SplitStrategy},
};

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

impl PolynomialApproximation<CKKSPlaintext<Vec<u8>>> {
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
        CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<F>,
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
        let scale_pow2 = affine.as_ref().and_then(|_| {
            let scale = scale.to_f64()?;
            let exponent = scale.log2().round();
            if exponent >= i32::MIN as f64 && exponent <= i32::MAX as f64 && 2f64.powi(exponent as i32) == scale {
                Some(exponent as i32)
            } else {
                None
            }
        });
        Ok(Self {
            poly: bsgs,
            affine,
            scale_pow2,
            coeff_log_delta: coeff_meta.log_delta(),
        })
    }
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

    /// Polynomial degree.
    pub fn degree(&self) -> usize {
        self.poly.degree()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn prepare(a: f64, b: f64) -> PolynomialApproximation<CKKSPlaintext<Vec<u8>>> {
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
}
