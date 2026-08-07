//! The minimal complex scalar of the PaCo cleartext model.

use anyhow::{Context, Result};
use num_traits::{Float, ToPrimitive};

/// A complex scalar over any float `F` (defaulting to `f64` for the reference
/// oracle and the tests). Minimal on purpose: only what the PaCo cleartext
/// model needs (the crate-wide diagonal maps are
/// [`ComplexDiagonals`](crate::layouts::ComplexDiagonals); `Cpx` only makes
/// the contiguous-vector slot arithmetic readable).
///
/// The scalar is generic so the ψ phases, the σ_t/β_t packings, and the
/// plaintext encodes stay at the working precision `F` end to end (an `f64`
/// waypoint would silently cap an `f128` pipeline at 53 bits).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Cpx<F = f64> {
    /// Real part.
    pub re: F,
    /// Imaginary part.
    pub im: F,
}

#[cfg_attr(not(feature = "test-utils"), allow(dead_code))]
impl Cpx<f64> {
    /// The additive identity.
    pub const ZERO: Cpx = Cpx { re: 0.0, im: 0.0 };
}

impl<F: Float> Cpx<F> {
    /// The additive identity.
    pub(crate) fn zero() -> Self {
        Self::new(F::zero(), F::zero())
    }

    /// The multiplicative identity.
    pub(crate) fn one() -> Self {
        Self::new(F::one(), F::zero())
    }

    /// Builds `re + i·im`.
    pub(crate) fn new(re: F, im: F) -> Self {
        Self { re, im }
    }

    /// Complex conjugate.
    #[cfg_attr(not(feature = "test-utils"), allow(dead_code))]
    pub(crate) fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// Squared modulus.
    #[cfg_attr(not(feature = "test-utils"), allow(dead_code))]
    pub(crate) fn norm_sqr(self) -> F {
        self.re * self.re + self.im * self.im
    }

    /// Modulus.
    #[cfg_attr(not(feature = "test-utils"), allow(dead_code))]
    pub(crate) fn abs(self) -> F {
        self.norm_sqr().sqrt()
    }
}

impl<F: Float + ToPrimitive> Cpx<F> {
    /// Truncates to `f64` parts (the reference oracle's precision).
    #[cfg_attr(not(feature = "test-utils"), allow(dead_code))]
    pub(crate) fn to_f64(self) -> Result<Cpx<f64>> {
        let re = self.re.to_f64().context("complex real part cannot be represented as f64")?;
        let im = self
            .im
            .to_f64()
            .context("complex imaginary part cannot be represented as f64")?;
        Ok(Cpx::new(re, im))
    }
}

impl<F: Float> std::ops::Add for Cpx<F> {
    type Output = Cpx<F>;
    fn add(self, o: Cpx<F>) -> Cpx<F> {
        Cpx::new(self.re + o.re, self.im + o.im)
    }
}

impl<F: Float> std::ops::Sub for Cpx<F> {
    type Output = Cpx<F>;
    fn sub(self, o: Cpx<F>) -> Cpx<F> {
        Cpx::new(self.re - o.re, self.im - o.im)
    }
}

impl<F: Float> std::ops::Mul for Cpx<F> {
    type Output = Cpx<F>;
    fn mul(self, o: Cpx<F>) -> Cpx<F> {
        Cpx::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
}

impl<F: Float> std::ops::Neg for Cpx<F> {
    type Output = Cpx<F>;
    fn neg(self) -> Cpx<F> {
        Cpx::new(-self.re, -self.im)
    }
}
