//! Full-precision noise for statistically hiding partial decryptions.
//!
//! These parameters describe integer noise at the lattice `2^-k Z`, independently
//! of the radix used to store it. They do not infer an input error bound, a
//! transcript length, or a decoding margin. Ordinary encryption noise continues
//! to use [`crate::NoiseInfos`].

/// The integer distribution of a smudging coefficient before scaling by `2^-k`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmudgingDistribution {
    /// A discrete Gaussian with mass proportional to `exp(-z^2 / (2 sigma^2))`,
    /// `sigma = 2^log_sigma`, conditioned on `|z| <= cutoff * sigma`.
    ///
    /// The omitted tail is at most `2 exp(-cutoff^2 / 2)` per coefficient.
    /// Choose the cutoff for the complete transcript's statistical budget;
    /// a small fixed cutoff is only suitable for functional tests. The CPU
    /// implementation uses exact integer rejection sampling with variable runtime.
    Gaussian { log_sigma: usize, cutoff: usize },
    /// Exactly uniform on the consecutive integers `[-2^(bits-1), 2^(bits-1)-1]`.
    /// Its mean is `-1/2`. Every bit down to the unit integer bit is sampled.
    Uniform { bits: usize },
}

/// A bounded integer smudging distribution at precision `2^-k`.
///
/// For full-precision smudging, set `k` to the sampling destination's precision.
/// A smaller `k` samples on a coarser lattice and can expose fine input-noise
/// bits. Gaussian magnitude and cutoff, or uniform width, must leave room for
/// input error, all parties' floods, and any fresh encryption error inside the
/// application's decoding margin. Increasing `k` alone does not supply that
/// margin for an unchanged integer encoding.
///
/// Validation ensures that one sample fits strictly inside a signed `k`-bit
/// window. It does not certify a statistical security level or the sum of
/// multiple samples. Fields are public for use by parameter planners; samplers
/// revalidate them before consuming randomness or modifying outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmudgingNoise {
    pub k: usize,
    pub distribution: SmudgingDistribution,
}

impl SmudgingNoise {
    /// Constructs a bounded exact discrete Gaussian at the full integer grid.
    pub fn gaussian(k: usize, log_sigma: usize, cutoff: usize) -> Self {
        let noise = Self {
            k,
            distribution: SmudgingDistribution::Gaussian { log_sigma, cutoff },
        };
        noise.assert_valid();
        noise
    }

    /// Constructs an exact uniform distribution over signed `bits`-bit integers.
    pub fn uniform(k: usize, bits: usize) -> Self {
        let noise = Self {
            k,
            distribution: SmudgingDistribution::Uniform { bits },
        };
        noise.assert_valid();
        noise
    }

    /// Checks the precision and distribution bounds, with static panic messages.
    pub fn assert_valid(self) {
        assert!(self.k > 0, "invalid smudging: precision must be positive");
        match self.distribution {
            SmudgingDistribution::Gaussian { log_sigma, cutoff } => {
                assert!(cutoff > 0, "invalid smudging: Gaussian cutoff must be positive");
                let cutoff_bits = (usize::BITS - cutoff.leading_zeros()) as usize;
                assert!(
                    log_sigma.checked_add(cutoff_bits).is_some_and(|bits| bits < self.k),
                    "invalid smudging: Gaussian bound outside the precision"
                );
            }
            SmudgingDistribution::Uniform { bits } => {
                assert!(
                    bits > 0 && bits < self.k,
                    "invalid smudging: uniform width outside the precision"
                );
            }
        }
    }

    /// Checks parameter and radix bounds against a sampling destination.
    ///
    /// `k` is its logical precision; a bare polynomial may use its physical
    /// capacity here. Scheme callers must additionally check the logical one.
    pub fn assert_valid_for(self, base2k: usize, k: usize) {
        assert!(
            self.k > 0 && self.k <= k,
            "invalid smudging: precision outside the destination"
        );
        assert!(
            base2k > 0 && base2k <= 62,
            "invalid smudging: radix outside the coefficient headroom"
        );
        self.assert_valid();
    }
}

/// Supplies dedicated smudging parameters independently of encryption noise.
pub trait SmudgingInfos {
    fn smudging_infos(&self) -> SmudgingNoise;
}

impl SmudgingInfos for SmudgingNoise {
    fn smudging_infos(&self) -> SmudgingNoise {
        *self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smudging_parameter_bounds() {
        SmudgingNoise::gaussian(160, 130, 16).assert_valid_for(17, 160);
        SmudgingNoise::uniform(160, 132).assert_valid_for(17, 160);
        for noise in [
            SmudgingNoise {
                k: 0,
                distribution: SmudgingDistribution::Uniform { bits: 1 },
            },
            SmudgingNoise {
                k: 160,
                distribution: SmudgingDistribution::Uniform { bits: 0 },
            },
            SmudgingNoise {
                k: 160,
                distribution: SmudgingDistribution::Uniform { bits: 160 },
            },
            SmudgingNoise {
                k: 160,
                distribution: SmudgingDistribution::Gaussian {
                    log_sigma: 130,
                    cutoff: 0,
                },
            },
            SmudgingNoise {
                k: 160,
                distribution: SmudgingDistribution::Gaussian {
                    log_sigma: usize::MAX,
                    cutoff: 16,
                },
            },
            SmudgingNoise {
                k: 135,
                distribution: SmudgingDistribution::Gaussian {
                    log_sigma: 130,
                    cutoff: 16,
                },
            },
        ] {
            assert!(std::panic::catch_unwind(|| noise.assert_valid()).is_err());
        }
    }
}
