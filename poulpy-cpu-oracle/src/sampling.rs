//! The `poulpy-core` sampling extension point, on host buffers.

use dashu_int::{IBig, UBig, ops::BitTest};
use poulpy_core::{
    Distribution, Noise,
    oep::SamplingImpl,
    test_suite::parity::controlled_sampling::{add_noise_samples, scalar_samples},
};
use poulpy_hal::{
    layouts::{Module, ScalarZnxBackendMut, VecZnxBackendMut, VecZnxBigBackendMut, ZnxViewMut},
    source::Source,
};
use rand::Rng;

use crate::{
    ScalarZnxFill,
    backend::Oracle,
    family::{DFTFamily, Int},
    fft::Fft64,
    ntt::Ntt4x30,
    ring::OracleRing,
    test_suite::ControlledSampling,
};

// The oracle deliberately uses uniform rejection over the whole bounded
// interval, independently of the production CDT and Laplace-proposal sampler.
// Bernoulli(exp(-x)) uses the alternating exponential series and exact ratios.
fn below(limit: &UBig, source: &mut Source) -> UBig {
    let bits = (limit - 1u8).bit_len();
    if bits == 0 {
        return UBig::ZERO;
    }
    let mut bytes = vec![0; bits.div_ceil(8)];
    loop {
        source.fill_bytes(&mut bytes);
        let last = bytes.len() - 1;
        bytes[last] &= u8::MAX >> ((8 - bits % 8) % 8);
        let draw = UBig::from_le_bytes(&bytes);
        if &draw < limit {
            return draw;
        }
    }
}

fn exp_unit(numerator: &UBig, denominator: &UBig, source: &mut Source) -> bool {
    let mut term = UBig::ONE;
    let mut even = true;
    while below(&(denominator * &term), source) < *numerator {
        even = !even;
        term += 1u8;
    }
    even
}

fn exp_coin(numerator: &UBig, denominator: &UBig, source: &mut Source) -> bool {
    let mut whole = numerator / denominator;
    while whole != UBig::ZERO {
        if !exp_unit(&UBig::ONE, &UBig::ONE, source) {
            return false;
        }
        whole -= 1u8;
    }
    exp_unit(&(numerator % denominator), denominator, source)
}

struct Gaussian {
    bound: UBig,
    numerator_squared: UBig,
    denominator_squared: UBig,
}

impl Gaussian {
    fn new(sigma: f64) -> Self {
        let (numerator, denominator, bound) = Noise::gaussian_parts(sigma);
        Self {
            bound,
            numerator_squared: &numerator * &numerator,
            denominator_squared: &denominator * &denominator,
        }
    }

    fn sample(&self, source: &mut Source) -> IBig {
        let count = &self.bound * 2u8 + 1u8;
        loop {
            let sample = IBig::from(below(&count, source)) - IBig::from(self.bound.clone());
            let magnitude = sample.clone().into_parts().1;
            let numerator = &magnitude * &magnitude * &self.denominator_squared;
            if exp_coin(&numerator, &(&self.numerator_squared * 2u8), source) {
                return sample;
            }
        }
    }
}

fn add_noise<R: ZnxViewMut>(base2k: usize, k: usize, res: &mut R, col: usize, noise: Noise, seed: [u8; 32])
where
    R::Scalar: Int,
{
    noise.validate();
    assert!((1..=63).contains(&base2k), "noise radix must be in 1..=63");
    assert!(
        k > 0 && k.div_ceil(base2k) <= res.size(),
        "noise precision exceeds destination allocation"
    );
    assert!(col < res.cols(), "noise column exceeds destination allocation");
    let active = k.div_ceil(base2k);
    let padding = (base2k - k % base2k) % base2k;
    let radix = IBig::ONE << base2k;
    let half = &radix >> 1;
    let gaussian = match noise {
        Noise::Gaussian { sigma } => Some(Gaussian::new(sigma)),
        Noise::Uniform { .. } => None,
    };
    let mut source = Source::new(seed);
    for coefficient in 0..res.n() {
        let sample = match noise {
            Noise::Gaussian { .. } => gaussian.as_ref().unwrap().sample(&mut source),
            Noise::Uniform { bits } => {
                // Only low k bits affect the destination if the requested
                // integer is wider than its torus representation.
                let width = bits.min(k);
                IBig::from(below(&(UBig::ONE << width), &mut source)) - (IBig::ONE << (width - 1))
            }
        };
        let mut value = sample << padding;
        for limb in (0..active).rev() {
            let carry = (&value + &half) >> base2k;
            let digit = i64::try_from(&value - &carry * &radix).unwrap();
            let dst = &mut res.at_mut(col, limb)[coefficient];
            *dst = dst.add(R::Scalar::from(digit));
            value = carry;
        }
    }
}

fn assert_degree<F: DFTFamily, R: OracleRing>(module: &Module<Oracle<F, R>>, n: usize) {
    assert!(n.is_power_of_two() && n <= module.n(), "noise degree outside module");
}

macro_rules! impl_independent_sampling {
    ($($family:ty),+) => {$(
        unsafe impl<R: OracleRing> SamplingImpl for Oracle<$family, R> {
            fn scalar_znx_fill_distribution(
                _module: &Module<Self>,
                res: &mut ScalarZnxBackendMut<'_, Self>,
                res_col: usize,
                dist: Distribution,
                seed: [u8; 32],
            ) {
                let mut source = Source::new(seed);
                match dist {
                    Distribution::TernaryFixed(hw) => res.fill_ternary_hw(res_col, hw, &mut source),
                    Distribution::TernaryProb(prob) => res.fill_ternary_prob(res_col, prob, &mut source),
                    Distribution::BinaryFixed(hw) => res.fill_binary_hw(res_col, hw, &mut source),
                    Distribution::BinaryProb(prob) => res.fill_binary_prob(res_col, prob, &mut source),
                    Distribution::BinaryBlock(block_size) => res.fill_binary_block(res_col, block_size, &mut source),
                    Distribution::ZERO => res.at_mut(res_col, 0).fill(0),
                    Distribution::NONE | Distribution::ENCAPSULATED(_) => {
                        panic!("scalar_znx_fill_distribution: {dist:?} is not a sampleable distribution")
                    }
                }
            }

            fn vec_znx_add_noise(
                module: &Module<Self>,
                base2k: usize,
                k: usize,
                res: &mut VecZnxBackendMut<'_, Self>,
                res_col: usize,
                noise: Noise,
                seed: [u8; 32],
            ) {
                assert_degree(module, res.n());
                add_noise(base2k, k, res, res_col, noise, seed);
            }

            fn vec_znx_big_add_noise(
                module: &Module<Self>,
                base2k: usize,
                k: usize,
                res: &mut VecZnxBigBackendMut<'_, Self>,
                res_col: usize,
                noise: Noise,
                seed: [u8; 32],
            ) {
                assert_degree(module, res.n());
                add_noise(base2k, k, res, res_col, noise, seed);
            }
        }
    )+};
}

impl_independent_sampling!(Fft64, Ntt4x30);

// Safety: copies distribution-correct draws of the backend under test and
// mutates only the selected column and precision, after the same checks.
unsafe impl<F: DFTFamily, R: OracleRing> SamplingImpl for Oracle<ControlledSampling<F>, R> {
    fn scalar_znx_fill_distribution(
        _module: &Module<Self>,
        res: &mut ScalarZnxBackendMut<'_, Self>,
        res_col: usize,
        dist: Distribution,
        seed: [u8; 32],
    ) {
        let samples = scalar_samples(res.n(), dist, seed);
        res.at_mut(res_col, 0).copy_from_slice(&samples);
    }

    fn vec_znx_add_noise(
        module: &Module<Self>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        noise: Noise,
        seed: [u8; 32],
    ) {
        assert_degree(module, res.n());
        add_noise_samples(res, base2k, k, res_col, noise, seed, false, |dst, digit| {
            *dst = dst.add(digit)
        });
    }

    fn vec_znx_big_add_noise(
        module: &Module<Self>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        noise: Noise,
        seed: [u8; 32],
    ) {
        assert_degree(module, res.n());
        add_noise_samples(res, base2k, k, res_col, noise, seed, true, |dst, digit| {
            *dst = dst.add(digit.into())
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FFT64Oracle, NTT4x30Oracle};

    #[test]
    fn independent_gaussian_matches_discrete_pmf() {
        const COUNT: usize = 32_000;
        for (sigma, bound) in [(1.0, 6i64), (3.2, 19)] {
            let sampler = Gaussian::new(sigma);
            assert_eq!(sampler.bound, UBig::from(bound as u64));
            let mut histogram = vec![0usize; (2 * bound + 1) as usize];
            let mut source = Source::new([38; 32]);
            for _ in 0..COUNT {
                let sample = i64::try_from(sampler.sample(&mut source)).unwrap();
                histogram[(sample + bound) as usize] += 1;
            }
            let weights: Vec<_> = (-bound..=bound)
                .map(|z| (-(z as f64).powi(2) / (2.0 * sigma * sigma)).exp())
                .collect();
            let total: f64 = weights.iter().sum();
            for (count, weight) in histogram.into_iter().zip(weights) {
                let probability = weight / total;
                let mean = COUNT as f64 * probability;
                assert!((count as f64 - mean).abs() <= 7.0 * (mean * (1.0 - probability)).sqrt() + 2.0);
            }
        }
    }

    #[test]
    fn fft_full_width_noise_reaches_unit_grid() {
        poulpy_core::test_suite::sampling::test_full_width_noise(&Module::<FFT64Oracle>::new(256));
    }

    #[test]
    fn ntt_full_width_noise_reaches_unit_grid() {
        poulpy_core::test_suite::sampling::test_full_width_noise(&Module::<NTT4x30Oracle>::new(256));
    }
}
