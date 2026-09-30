//! Exact integer smudging on the full torus coefficient grid.
//!
//! The discrete Gaussian uses the rejection construction of Canonne, Kamath
//! and Steinke, *The Discrete Gaussian for Differential Privacy*, Section 5,
//! <https://arxiv.org/abs/2004.00010>. Its discrete Laplace proposal has constant
//! expected rejection work even for a large sigma. All sampling arithmetic is
//! integer arithmetic; the running time is variable and has no fixed cap.
//! Gaussian outputs are conditioned on the descriptor's symmetric bound.

use dashu_int::UBig;
use poulpy_core::SmudgingNoise;
use poulpy_hal::{
    layouts::{Backend, HostDataMut, VecZnxBackendMut, ZnxView, ZnxViewMut},
    source::Source,
};
use rand_core::Rng;

fn uniform_bits(bits: usize, source: &mut Source) -> UBig {
    let mut bytes = vec![0; bits.div_ceil(8)];
    source.fill_bytes(&mut bytes);
    if !bits.is_multiple_of(8) {
        *bytes.last_mut().unwrap() &= (1 << (bits % 8)) - 1;
    }
    UBig::from_le_bytes(&bytes)
}

/// Unbiased rejection sampling, including non-power-of-two denominators.
fn uniform_below(limit: &UBig, source: &mut Source) -> UBig {
    if limit == &UBig::ONE {
        return UBig::ZERO;
    }
    let mut bytes = (limit - UBig::ONE).to_le_bytes();
    let high_mask = u8::MAX >> bytes.last().unwrap().leading_zeros();
    loop {
        source.fill_bytes(&mut bytes);
        *bytes.last_mut().unwrap() &= high_mask;
        let candidate = UBig::from_le_bytes(&bytes);
        if &candidate < limit {
            return candidate;
        }
    }
}

/// Bernoulli(exp(-a/b)), for 0 <= a <= b, without evaluating an exponential.
fn exp_small(a: &UBig, b: &UBig, source: &mut Source) -> bool {
    let mut denominator = b.clone();
    let mut odd = true;
    while uniform_below(&denominator, source) < *a {
        denominator += b;
        odd = !odd;
    }
    odd
}

fn exp_coin(mut a: UBig, b: &UBig, source: &mut Source) -> bool {
    // Early rejection is essential: evaluating every factor would be linear
    // in a/b. The number of factors inspected is constant in expectation.
    while &a > b {
        if !exp_small(&UBig::ONE, &UBig::ONE, source) {
            return false;
        }
        a -= b;
    }
    exp_small(&a, b, source)
}

/// A signed magnitude, with zero always assigned a positive sign.
struct Sample {
    negative: bool,
    magnitude: UBig,
}

struct Gaussian {
    scale: UBig,
    variance: UBig,
    denominator: UBig,
    bound: UBig,
}

impl Gaussian {
    fn new(log_sigma: usize, cutoff: usize) -> Self {
        let sigma = UBig::ONE << log_sigma;
        let scale = &sigma + UBig::ONE;
        let variance = &sigma * &sigma;
        let denominator = ((&variance * &scale) * &scale) << 1;
        let bound = sigma * UBig::from(cutoff);
        Self {
            scale,
            variance,
            denominator,
            bound,
        }
    }

    fn sample(&self, source: &mut Source) -> Sample {
        loop {
            let u = loop {
                let u = uniform_below(&self.scale, source);
                if exp_small(&u, &self.scale, source) {
                    break u;
                }
            };
            let mut v = UBig::ZERO;
            while exp_small(&UBig::ONE, &UBig::ONE, source) {
                v += UBig::ONE;
            }
            let magnitude = u + &self.scale * v;
            let negative = source.next_u32() & 1 != 0;
            // Reject one representation of zero, then impose the symmetric
            // support bound before the Gaussian acceptance test.
            if (negative && magnitude == UBig::ZERO) || magnitude > self.bound {
                continue;
            }
            let scaled = &magnitude * &self.scale;
            let delta = if scaled >= self.variance {
                scaled - &self.variance
            } else {
                &self.variance - scaled
            };
            if exp_coin(&delta * &delta, &self.denominator, source) {
                return Sample { negative, magnitude };
            }
        }
    }
}

enum Sampler {
    Gaussian(Gaussian),
    Uniform { bits: usize, half: UBig },
}

impl Sampler {
    fn new(noise: SmudgingNoise) -> Self {
        match noise {
            SmudgingNoise::Gaussian { log_sigma, cutoff } => Self::Gaussian(Gaussian::new(log_sigma, cutoff)),
            SmudgingNoise::Uniform { bits } => Self::Uniform {
                bits,
                half: UBig::ONE << (bits - 1),
            },
        }
    }

    fn sample(&self, source: &mut Source) -> Sample {
        match self {
            Self::Gaussian(gaussian) => gaussian.sample(source),
            Self::Uniform { bits, half } => {
                let value = uniform_bits(*bits, source);
                let negative = &value < half;
                let magnitude = if negative { half - value } else { value - half };
                Sample { negative, magnitude }
            }
        }
    }
}

struct Encoding {
    base2k: usize,
    size: usize,
    padding: usize,
    modulus: UBig,
    mask: UBig,
    digit_mask: UBig,
    half: i64,
}

impl Encoding {
    fn new(base2k: usize, k: usize) -> Self {
        let size = k.div_ceil(base2k);
        let physical = size.checked_mul(base2k).expect("invalid smudging: precision overflow");
        let modulus = UBig::ONE << physical;
        Self {
            base2k,
            size,
            padding: physical - k,
            mask: &modulus - UBig::ONE,
            modulus,
            digit_mask: UBig::from((1u64 << base2k) - 1),
            half: 1i64 << (base2k - 1),
        }
    }

    /// Balanced digits, most significant first. Encoding the signed integer
    /// modulo the physical precision also handles negative carries correctly.
    fn digits(&self, sample: Sample, digits: &mut [i64]) {
        let mut value = (sample.magnitude << self.padding) & &self.mask;
        if sample.negative && value != UBig::ZERO {
            value = &self.modulus - value;
        }
        for digit in digits.iter_mut().rev() {
            *digit = u64::try_from(&value & &self.digit_mask).unwrap() as i64;
            value >>= self.base2k;
            if *digit >= self.half {
                *digit -= 1i64 << self.base2k;
                value += UBig::ONE;
            }
        }
    }
}

/// Adds full-grid noise to a canonical column, leaving the sum unnormalized.
/// Other columns and limbs below the sampling precision are untouched. The
/// caller must normalize before another call on the same destination column.
pub fn vec_znx_add_smudging_portable<'r, BE>(
    base2k: usize,
    k: usize,
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    noise: SmudgingNoise,
    source: &mut Source,
) where
    BE: Backend<ZnxWord = i64>,
    BE::BufMut<'r>: HostDataMut,
{
    let capacity = res.size().checked_mul(base2k).expect("invalid smudging: precision overflow");
    assert!(k <= capacity, "invalid smudging: precision outside the destination");
    noise.assert_valid_for(base2k, k);
    assert!(res.n() > 0, "invalid smudging: empty degree");
    assert!(res_col < res.cols(), "invalid smudging: column outside allocation");
    let half = 1i64 << (base2k - 1);
    for limb in 0..res.size() {
        assert!(
            res.at(res_col, limb).iter().all(|&digit| (-half..half).contains(&digit)),
            "invalid smudging: destination column is not canonical"
        );
    }

    let sampler = Sampler::new(noise);
    let encoding = Encoding::new(base2k, k);
    let mut digits = vec![0; encoding.size];
    for coefficient in 0..res.n() {
        encoding.digits(sampler.sample(source), &mut digits);
        for (limb, &digit) in digits.iter().enumerate() {
            res.at_mut(res_col, limb)[coefficient] += digit;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashu_int::{IBig, Sign};

    fn signed(sample: Sample) -> IBig {
        IBig::from_parts(
            if sample.negative { Sign::Negative } else { Sign::Positive },
            sample.magnitude,
        )
    }

    fn decode(digits: impl Iterator<Item = i64>, base2k: usize) -> IBig {
        digits.fold(IBig::ZERO, |value, digit| (value << base2k) + IBig::from(digit))
    }

    #[test]
    fn smudging_small_distribution_pmf() {
        const COUNT: usize = 24_000;
        for (log_sigma, cutoff) in [(0, 1), (1, 3)] {
            let sampler = Gaussian::new(log_sigma, cutoff);
            let sigma = (1usize << log_sigma) as f64;
            let bound = (cutoff << log_sigma) as i64;
            let mut counts = vec![0; (2 * bound + 1) as usize];
            let mut source = Source::new([cutoff as u8; 32]);
            for _ in 0..COUNT {
                let value = i64::try_from(signed(sampler.sample(&mut source))).unwrap();
                assert!((-bound..=bound).contains(&value));
                counts[(value + bound) as usize] += 1;
            }
            // Compare every atom to the independently evaluated conditional
            // PMF; this detects rounded continuous Gaussians and zero bias.
            let weights: Vec<_> = (-bound..=bound)
                .map(|x| (-(x as f64).powi(2) / (2.0 * sigma * sigma)).exp())
                .collect();
            let total: f64 = weights.iter().sum();
            for (&observed, weight) in counts.iter().zip(weights) {
                let p = weight / total;
                let mean = COUNT as f64 * p;
                let tolerance = 7.0 * (mean * (1.0 - p)).sqrt() + 2.0;
                assert!((observed as f64 - mean).abs() <= tolerance);
            }
        }

        let sampler = Sampler::new(SmudgingNoise::Uniform { bits: 3 });
        let mut source = Source::new([91; 32]);
        let mut counts = [0; 8];
        for _ in 0..COUNT {
            let value = i64::try_from(signed(sampler.sample(&mut source))).unwrap();
            assert!((-4..4).contains(&value));
            counts[(value + 4) as usize] += 1;
        }
        let mean = COUNT as f64 / 8.0;
        for count in counts {
            assert!((count as f64 - mean).abs() < 7.0 * (mean * 7.0 / 8.0).sqrt());
        }
    }

    #[test]
    fn smudging_signed_bigint_encoding() {
        for base2k in [1, 17, 62] {
            let encoding = Encoding::new(base2k, 197);
            let mut digits = vec![0; encoding.size];
            for magnitude in [
                UBig::ZERO,
                UBig::ONE,
                UBig::ONE << (base2k - 1),
                (UBig::ONE << 150) + (UBig::ONE << 65) + UBig::from(31u8),
            ] {
                for negative in [false, true] {
                    encoding.digits(
                        Sample {
                            negative,
                            magnitude: magnitude.clone(),
                        },
                        &mut digits,
                    );
                    assert!(digits.iter().all(|&digit| (-encoding.half..encoding.half).contains(&digit)));
                    assert_eq!(digits.last().unwrap() & ((1i64 << encoding.padding) - 1), 0);
                    let decoded = decode(digits.iter().copied(), base2k);
                    let expected = signed(Sample {
                        negative,
                        magnitude: magnitude.clone(),
                    }) << encoding.padding;
                    let modulus = IBig::from(encoding.modulus.clone());
                    assert_eq!(((decoded - expected) % &modulus + &modulus) % &modulus, IBig::ZERO);
                }
            }
        }
    }

    #[test]
    fn smudging_wide_gaussian_has_unit_grid_randomness() {
        let sampler = Gaussian::new(160, 16);
        let mut source = Source::new([17; 32]);
        let mut residues = [0; 8];
        let mut negative = 0;
        let mut wide = 0;
        for _ in 0..512 {
            let sample = sampler.sample(&mut source);
            assert!(sample.magnitude <= sampler.bound);
            negative += usize::from(sample.negative);
            wide += usize::from(sample.magnitude > (UBig::ONE << 128));
            let residue = usize::try_from(&sample.magnitude & UBig::from(7u8)).unwrap();
            residues[residue] += 1;
        }
        assert!((180..332).contains(&negative));
        assert!(wide > 500);
        for count in residues {
            assert!((count as f64 - 64.0).abs() < 7.0 * 56.0_f64.sqrt());
        }
    }

    #[cfg(feature = "enable-core")]
    fn check_backend<BE>(module: &poulpy_hal::layouts::Module<BE>)
    where
        BE: poulpy_core::oep::SmudgingSamplingImpl + Backend<ZnxWord = i64, OwnedBuf = poulpy_hal::AlignedBuf>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        use poulpy_core::VecZnxAddSmudging;
        use poulpy_hal::layouts::{VecZnx, vec_znx_backend_mut};

        const K: usize = 173;
        let n = module.n();
        for (base2k, initial) in [(17, 7), (62, -(1i64 << 61))] {
            let target_size = K.div_ceil(base2k);
            let size = target_size + 2;
            for noise in [
                SmudgingNoise::Gaussian {
                    log_sigma: 128,
                    cutoff: 16,
                },
                SmudgingNoise::Uniform { bits: 132 },
            ] {
                let make = || {
                    let mut value = VecZnx::from_data(
                        BE::alloc_zeroed_bytes(VecZnx::<poulpy_hal::AlignedBuf, i64>::bytes_of(n, 2, size)),
                        n,
                        2,
                        size,
                    );
                    for limb in 0..size {
                        value.at_mut(0, limb).fill(7);
                        value.at_mut(1, limb).fill(initial);
                    }
                    value
                };
                let mut result = make();
                let mut repeated = make();
                BE::vec_znx_add_smudging(
                    module,
                    base2k,
                    K,
                    &mut vec_znx_backend_mut::<BE>(&mut result),
                    1,
                    noise,
                    [23; 32],
                );
                BE::vec_znx_add_smudging(
                    module,
                    base2k,
                    K,
                    &mut vec_znx_backend_mut::<BE>(&mut repeated),
                    1,
                    noise,
                    [23; 32],
                );
                for limb in 0..size {
                    assert!(result.at(0, limb).iter().all(|&v| v == 7));
                    assert_eq!(result.at(1, limb), repeated.at(1, limb));
                    if limb >= target_size {
                        assert!(result.at(1, limb).iter().all(|&v| v == initial));
                    }
                }
                let sampler = Sampler::new(noise);
                let mut source = Source::new([23; 32]);
                let padding = target_size * base2k - K;
                let modulus = IBig::ONE << (target_size * base2k);
                for coefficient in 0..n {
                    let value = decode((0..target_size).map(|limb| result.at(1, limb)[coefficient] - initial), base2k);
                    let expected = signed(sampler.sample(&mut source)) << padding;
                    assert_eq!(((value - expected) % &modulus + &modulus) % &modulus, IBig::ZERO);
                    assert_eq!(
                        (result.at(1, target_size - 1)[coefficient] - initial) & ((1 << padding) - 1),
                        0
                    );
                }

                // The public delegate must advance its private source once per
                // call and give the OEP exactly that child seed.
                let mut private_source = Source::new([37; 32]);
                let mut expected_source = Source::new([37; 32]);
                let mut previous = None;
                for _ in 0..2 {
                    let mut actual = make();
                    let mut expected = make();
                    module.vec_znx_add_smudging(
                        base2k,
                        K,
                        &mut vec_znx_backend_mut::<BE>(&mut actual),
                        1,
                        noise,
                        &mut private_source,
                    );
                    BE::vec_znx_add_smudging(
                        module,
                        base2k,
                        K,
                        &mut vec_znx_backend_mut::<BE>(&mut expected),
                        1,
                        noise,
                        expected_source.new_seed(),
                    );
                    let mut sampled = Vec::new();
                    for limb in 0..size {
                        assert_eq!(actual.at(0, limb), expected.at(0, limb));
                        assert_eq!(actual.at(1, limb), expected.at(1, limb));
                        sampled.extend_from_slice(actual.at(1, limb));
                    }
                    if let Some(previous) = previous {
                        assert_ne!(sampled, previous);
                    }
                    previous = Some(sampled);
                }
                assert_eq!(private_source.new_seed(), expected_source.new_seed());
            }
        }
    }

    #[cfg(feature = "enable-core")]
    #[test]
    fn smudging_fft64_seeded_addition() {
        check_backend(&poulpy_hal::layouts::Module::<crate::FFT64Portable>::new(128));
    }

    #[cfg(feature = "enable-core")]
    #[test]
    fn smudging_ntt4x30_seeded_addition() {
        check_backend(&poulpy_hal::layouts::Module::<crate::NTT4x30Portable>::new(128));
    }

    #[test]
    fn smudging_rejects_before_sampling_or_mutation() {
        use poulpy_hal::layouts::{VecZnx, vec_znx_backend_mut};
        use std::panic::{AssertUnwindSafe, catch_unwind};

        let noise = SmudgingNoise::Uniform { bits: 80 };
        for (base2k, k, col, noncanonical, message) in [
            (17, 100, 2, false, "invalid smudging: column outside allocation"),
            (0, 100, 1, false, "invalid smudging: precision outside the destination"),
            (63, 100, 1, false, "invalid smudging: radix outside the coefficient headroom"),
            (17, 120, 1, false, "invalid smudging: precision outside the destination"),
            (17, 100, 1, true, "invalid smudging: destination column is not canonical"),
        ] {
            let mut result = VecZnx::from_data(
                <crate::FFT64Portable as Backend>::alloc_zeroed_bytes(VecZnx::<poulpy_hal::AlignedBuf, i64>::bytes_of(8, 2, 6)),
                8,
                2,
                6,
            );
            for limb in 0..6 {
                result.at_mut(1, limb).fill(7);
            }
            if noncanonical {
                result.at_mut(1, 5)[7] = 1 << 16;
            }
            let before: Vec<_> = (0..6).flat_map(|limb| result.at(1, limb).iter().copied()).collect();
            let mut source = Source::new([99; 32]);
            let mut pristine = Source::new([99; 32]);
            let error = catch_unwind(AssertUnwindSafe(|| {
                vec_znx_add_smudging_portable::<crate::FFT64Portable>(
                    base2k,
                    k,
                    &mut vec_znx_backend_mut::<crate::FFT64Portable>(&mut result),
                    col,
                    noise,
                    &mut source,
                );
            }))
            .unwrap_err();
            let actual = error
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| error.downcast_ref::<String>().map(String::as_str));
            assert_eq!(actual, Some(message));
            assert_eq!(source.new_seed(), pristine.new_seed());
            let after: Vec<_> = (0..6).flat_map(|limb| result.at(1, limb).iter().copied()).collect();
            assert_eq!(after, before);
            assert!((0..6).all(|limb| result.at(0, limb).iter().all(|&x| x == 0)));
        }
    }
}
