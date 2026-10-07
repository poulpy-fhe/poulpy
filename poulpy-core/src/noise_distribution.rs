/// Distribution of integer error added at the destination precision.
///
/// Gaussian parameters are exact dyadic rationals encoded by `f64`. The
/// Gaussian is discrete, conditioned on `|z| <= floor(cutoff_factor * sigma)`.
/// Uniform noise covers `[-2^(bits - 1), 2^(bits - 1) - 1]` exactly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Noise {
    /// Discrete Gaussian conditioned on `|z| <= floor(cutoff_factor * sigma)`.
    /// `cutoff_factor` is a multiplier of `sigma`: `cutoff_factor: 6` means a six-sigma bound.
    Gaussian {
        sigma: f64,
        cutoff_factor: usize,
    },
    Uniform {
        bits: usize,
    },
}

impl Noise {
    /// The discrete Gaussian used by every encryption operation, truncated at
    /// six sigma. With `sigma = 3.2`, integer samples satisfy `|z| <= 19`.
    pub const ENCRYPTION: Self = Self::Gaussian {
        sigma: 3.2,
        cutoff_factor: 6,
    };

    /// Rejects invalid distribution parameters before drawing randomness.
    pub fn validate(self) {
        match self {
            Self::Gaussian { sigma, .. } => {
                assert!(sigma.is_finite() && sigma > 0.0, "Gaussian sigma must be positive and finite")
            }
            Self::Uniform { bits } => assert!(bits > 0, "uniform noise must have at least one bit"),
        }
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
            Self::Gaussian { sigma, cutoff_factor } => {
                assert!(cutoff_factor > 0, "invalid noise: Gaussian cutoff factor must be positive");
                let bits = sigma.to_bits();
                let exponent = ((bits >> 52) & 0x7ff) as i32;
                let fraction = bits & ((1u64 << 52) - 1);
                let (mantissa, shift) = if exponent == 0 {
                    (fraction, -1074)
                } else {
                    (fraction | (1u64 << 52), exponent - 1075)
                };
                let product = u128::from(mantissa) * cutoff_factor as u128;
                let bound_bits = (128 - product.leading_zeros() as i32 + shift).max(0) as usize;
                assert!(bound_bits < k, "invalid noise: Gaussian bound outside the precision");
            }
            Self::Uniform { bits } => {
                assert!(bits < k, "invalid noise: uniform width outside the precision");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Noise;

    #[test]
    fn flood_bounds_use_exact_dyadic_values() {
        Noise::Gaussian {
            sigma: 2f64.powi(130),
            cutoff_factor: 16,
        }
        .assert_valid_for(17, 136);
        Noise::Gaussian {
            sigma: f64::from_bits(1),
            cutoff_factor: usize::MAX,
        }
        .assert_valid_for(17, 1);
        Noise::Gaussian {
            sigma: f64::from_bits(8f64.to_bits() - 1),
            cutoff_factor: 1,
        }
        .assert_valid_for(17, 4);
        for (noise, base2k, k) in [
            (
                Noise::Gaussian {
                    sigma: 8.0,
                    cutoff_factor: 1,
                },
                17,
                4,
            ),
            (
                Noise::Gaussian {
                    sigma: 1.0,
                    cutoff_factor: 0,
                },
                17,
                160,
            ),
            (
                Noise::Gaussian {
                    sigma: f64::MAX,
                    cutoff_factor: usize::MAX,
                },
                17,
                160,
            ),
            (Noise::Uniform { bits: 160 }, 17, 160),
            (Noise::Uniform { bits: 80 }, 63, 160),
        ] {
            assert!(std::panic::catch_unwind(|| noise.assert_valid_for(base2k, k)).is_err());
        }
    }
}
