//! Full-precision noise for statistically hiding partial decryptions.
//!
//! These parameters describe integer noise on the destination's grid `2^-k`,
//! independently of the radix used to store it. They do not infer an input
//! error bound, a transcript length, or a decoding margin. Ordinary encryption
//! noise continues to use [`crate::NoiseInfos`].

/// A bounded integer smudging distribution, sampled on the grid `2^-k` of the
/// destination's precision `k`.
///
/// The flood must reach the bottom bit of the value it hides. On a coarser
/// grid `2^-(k-d) Z`, the low `d` bits of a key-switching share `<a, s>` stay
/// exact, and reducing them modulo `2^d` gives noiseless linear equations in
/// the secret. The sampling operations therefore take `k` from the
/// destination, never from this descriptor.
///
/// Magnitudes count in units of `2^-k`. They must leave room for input error,
/// all parties' floods, and any fresh encryption error inside the
/// application's decoding margin. [`Self::assert_valid_for`] ensures that one
/// sample fits strictly inside a signed `k`-bit window. It does not certify a
/// statistical security level or the sum of multiple samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmudgingNoise {
    /// A discrete Gaussian with mass proportional to `exp(-z^2 / (2 sigma^2))`,
    /// `sigma = 2^log_sigma`, conditioned on `|z| <= cutoff * sigma`.
    ///
    /// The omitted tail is at most `2 exp(-cutoff^2 / 2)` per coefficient.
    /// Choose the cutoff for the complete transcript's statistical budget;
    /// a small fixed cutoff is only suitable for functional tests.
    Gaussian { log_sigma: usize, cutoff: usize },
    /// Exactly uniform on the consecutive integers `[-2^(bits-1), 2^(bits-1)-1]`.
    /// Its mean is `-1/2`. Every bit down to the unit integer bit is sampled.
    Uniform { bits: usize },
}

impl SmudgingNoise {
    /// Checks the distribution and radix bounds against a destination of
    /// precision `k`, with static panic messages.
    pub fn assert_valid_for(self, base2k: usize, k: usize) {
        assert!(
            base2k > 0 && base2k <= 62,
            "invalid smudging: radix outside the coefficient headroom"
        );
        match self {
            Self::Gaussian { log_sigma, cutoff } => {
                assert!(cutoff > 0, "invalid smudging: Gaussian cutoff must be positive");
                let cutoff_bits = (usize::BITS - cutoff.leading_zeros()) as usize;
                assert!(
                    log_sigma.checked_add(cutoff_bits).is_some_and(|bits| bits < k),
                    "invalid smudging: Gaussian bound outside the precision"
                );
            }
            Self::Uniform { bits } => {
                assert!(bits > 0 && bits < k, "invalid smudging: uniform width outside the precision");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smudging_parameter_bounds() {
        SmudgingNoise::Gaussian {
            log_sigma: 130,
            cutoff: 16,
        }
        .assert_valid_for(17, 160);
        SmudgingNoise::Uniform { bits: 132 }.assert_valid_for(17, 160);
        for (noise, base2k, k) in [
            (SmudgingNoise::Uniform { bits: 1 }, 17, 0),
            (SmudgingNoise::Uniform { bits: 0 }, 17, 160),
            (SmudgingNoise::Uniform { bits: 160 }, 17, 160),
            (SmudgingNoise::Uniform { bits: 80 }, 0, 160),
            (SmudgingNoise::Uniform { bits: 80 }, 63, 160),
            (
                SmudgingNoise::Gaussian {
                    log_sigma: 130,
                    cutoff: 0,
                },
                17,
                160,
            ),
            (
                SmudgingNoise::Gaussian {
                    log_sigma: usize::MAX,
                    cutoff: 16,
                },
                17,
                160,
            ),
            (
                SmudgingNoise::Gaussian {
                    log_sigma: 130,
                    cutoff: 16,
                },
                17,
                135,
            ),
        ] {
            assert!(std::panic::catch_unwind(|| noise.assert_valid_for(base2k, k)).is_err());
        }
    }
}
