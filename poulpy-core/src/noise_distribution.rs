use dashu_int::{UBig, ops::BitTest};

/// Distribution of integer error added at the destination precision.
///
/// Gaussian parameters are exact dyadic rationals encoded by `f64`. Every
/// Gaussian is discrete, conditioned on `|z| <= floor(6 sigma)`
/// ([`Noise::CUTOFF_FACTOR`]). Uniform noise covers
/// `[-2^(bits - 1), 2^(bits - 1) - 1]` exactly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Noise {
    /// Discrete Gaussian conditioned on `|z| <= floor(6 sigma)`, with `sigma >= 1`.
    Gaussian {
        sigma: f64,
    },
    Uniform {
        bits: usize,
    },
}

impl Noise {
    /// Truncation of every Gaussian, in multiples of `sigma`.
    pub const CUTOFF_FACTOR: usize = 6;

    /// The discrete Gaussian used by every encryption operation. With
    /// `sigma = 3.2`, integer samples satisfy `|z| <= 19`.
    pub const ENCRYPTION: Self = Self::Gaussian { sigma: 3.2 };

    /// Rejects invalid distribution parameters before drawing randomness.
    /// `sigma >= 1` keeps every Gaussian away from a degenerate zero draw.
    pub fn validate(self) {
        match self {
            Self::Gaussian { sigma } => assert!(
                sigma.is_finite() && sigma >= 1.0,
                "invalid noise: Gaussian sigma must be finite and at least 1"
            ),
            Self::Uniform { bits } => assert!(bits > 0, "invalid noise: uniform width must be positive"),
        }
    }

    /// `sigma` as the exact fraction `numerator / denominator` in lowest terms,
    /// and the support bound `floor(6 sigma)`, as `(numerator, denominator, bound)`.
    /// `sigma` must be positive and finite.
    #[doc(hidden)]
    pub fn gaussian_parts(sigma: f64) -> (UBig, UBig, UBig) {
        let bits = sigma.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i32;
        let mantissa = bits & ((1u64 << 52) - 1);
        let (mantissa, exponent) = if exponent == 0 {
            (mantissa, -1074)
        } else {
            (mantissa | (1u64 << 52), exponent - 1075)
        };
        let trailing = mantissa.trailing_zeros() as i32;
        let numerator = UBig::from(mantissa >> trailing);
        let exponent = exponent + trailing;
        let (numerator, denominator) = if exponent >= 0 {
            (numerator << exponent as usize, UBig::ONE)
        } else {
            (numerator, UBig::ONE << (-exponent) as usize)
        };
        let bound = &numerator * Self::CUTOFF_FACTOR / &denominator;
        (numerator, denominator, bound)
    }

    /// Checks that a flood fits strictly inside the signed `k`-bit window.
    ///
    /// Flooding protocols call this before mutation. It leaves coefficient
    /// headroom for adding a balanced noise polynomial, but does not certify
    /// an application's statistical budget or decoding margin. The Gaussian
    /// bound is computed from the exact dyadic value of `sigma`.
    pub fn assert_valid_for(self, base2k: usize, k: usize) {
        self.validate();
        assert!(
            (1..=62).contains(&base2k),
            "invalid noise: radix outside the coefficient headroom"
        );
        match self {
            Self::Gaussian { sigma } => {
                let (_, _, bound) = Self::gaussian_parts(sigma);
                assert!(bound.bit_len() < k, "invalid noise: Gaussian bound outside the precision");
            }
            Self::Uniform { bits } => {
                assert!(bits < k, "invalid noise: uniform width outside the precision");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Noise, UBig};

    #[test]
    fn flood_bounds_use_exact_dyadic_values() {
        assert_eq!(
            Noise::gaussian_parts(3.2),
            (UBig::from(3602879701896397u64), UBig::ONE << 50, UBig::from(19u8))
        );
        assert_eq!(
            Noise::gaussian_parts(f64::from_bits(1)),
            (UBig::ONE, UBig::ONE << 1074, UBig::ZERO)
        );
        Noise::Gaussian { sigma: 2f64.powi(130) }.assert_valid_for(17, 134);
        Noise::Gaussian { sigma: 1.0 }.assert_valid_for(17, 4);
        // 6 * (4/3 as f64) rounds to 8.0 in binary64, but its exact floor is 7.
        Noise::Gaussian { sigma: 4.0 / 3.0 }.assert_valid_for(17, 4);
        for (noise, base2k, k) in [
            (Noise::Gaussian { sigma: 1.5 }, 17, 4),
            (Noise::Gaussian { sigma: 2f64.powi(130) }, 17, 133),
            (Noise::Gaussian { sigma: 0.1 }, 17, 160),
            (
                Noise::Gaussian {
                    sigma: 1.0 - f64::EPSILON,
                },
                17,
                160,
            ),
            (Noise::Gaussian { sigma: f64::NAN }, 17, 160),
            (Noise::Gaussian { sigma: f64::INFINITY }, 17, 160),
            (Noise::Gaussian { sigma: f64::MAX }, 17, 160),
            (Noise::Uniform { bits: 0 }, 17, 160),
            (Noise::Uniform { bits: 160 }, 17, 160),
            (Noise::Uniform { bits: 80 }, 63, 160),
        ] {
            assert!(std::panic::catch_unwind(|| noise.assert_valid_for(base2k, k)).is_err());
        }
    }
}
