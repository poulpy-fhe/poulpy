//! The `poulpy-core` sampling extension point, on host buffers.

use dashu_int::{IBig, UBig, ops::BitTest};
use poulpy_core::{Distribution, Noise, oep::SamplingImpl};
use poulpy_hal::{
    layouts::{Module, ScalarZnxBackendMut, VecZnxBackendMut, VecZnxBigBackendMut, ZnxViewMut},
    source::Source,
};
use rand::Rng;

use crate::{
    ScalarZnxFill,
    backend::Oracle,
    family::{DFTFamily, Int},
    ring::OracleRing,
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
    fn new(sigma: f64, cutoff_factor: usize) -> Self {
        let raw = sigma.to_bits();
        let encoded_exponent = ((raw >> 52) & 2047) as i32;
        let significand = (raw & ((1u64 << 52) - 1)) | (u64::from(encoded_exponent != 0) << 52);
        let exponent = if encoded_exponent == 0 {
            -1074
        } else {
            encoded_exponent - 1075
        };
        let mut numerator = UBig::from(significand);
        let mut denominator = UBig::ONE;
        if exponent < 0 {
            denominator <<= (-exponent) as usize;
        } else {
            numerator <<= exponent as usize;
        }
        Self {
            bound: &numerator * cutoff_factor / &denominator,
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
        Noise::Gaussian { sigma, cutoff_factor } => Some(Gaussian::new(sigma, cutoff_factor)),
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

fn uses_controlled_sampling<F: DFTFamily, R: OracleRing>(module: &Module<Oracle<F, R>>) -> bool {
    // This flag is fixed before the module is exposed to callers or workers.
    unsafe { (*module.ptr()).controlled_sampling }
}

fn add_controlled_noise<R: ZnxViewMut>(base2k: usize, k: usize, res: &mut R, col: usize, noise: Noise, seed: [u8; 32], big: bool)
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
    let samples = poulpy_core::test_suite::parity::controlled_sampling::noise_samples(res.n(), base2k, k, noise, seed, big);
    for (limb, digits) in samples.chunks(res.n()).enumerate() {
        for (dst, &digit) in res.at_mut(col, limb).iter_mut().zip(digits) {
            *dst = dst.add(R::Scalar::from(digit));
        }
    }
}

unsafe impl<F: DFTFamily, R: OracleRing> SamplingImpl for Oracle<F, R> {
    fn scalar_znx_fill_distribution(
        module: &Module<Self>,
        res: &mut ScalarZnxBackendMut<'_, Self>,
        res_col: usize,
        dist: Distribution,
        seed: [u8; 32],
    ) {
        if uses_controlled_sampling(module) {
            let samples = poulpy_core::test_suite::parity::controlled_sampling::scalar_samples(res.n(), dist, seed);
            res.at_mut(res_col, 0).copy_from_slice(&samples);
            return;
        }
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
        assert!(
            res.n().is_power_of_two() && res.n() <= module.n(),
            "noise degree outside module"
        );
        if uses_controlled_sampling(module) {
            add_controlled_noise(base2k, k, res, res_col, noise, seed, false);
        } else {
            add_noise(base2k, k, res, res_col, noise, seed);
        }
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
        assert!(
            res.n().is_power_of_two() && res.n() <= module.n(),
            "noise degree outside module"
        );
        if uses_controlled_sampling(module) {
            add_controlled_noise(base2k, k, res, res_col, noise, seed, true);
        } else {
            add_noise(base2k, k, res, res_col, noise, seed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FFT64Oracle, NTT4x30Oracle};
    use poulpy_hal::layouts::*;

    #[test]
    fn independent_gaussian_matches_discrete_pmf() {
        const COUNT: usize = 32_000;
        for sigma in [0.75, 3.2] {
            let sampler = Gaussian::new(sigma, 6);
            let bound = i64::try_from(&sampler.bound).unwrap();
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

    fn full_width<BE>()
    where
        BE: SamplingImpl + poulpy_hal::oep::HalModuleImpl + Backend<ZnxWord = i64, OwnedBuf = poulpy_hal::AlignedBuf>,
        BE::BigWord: Int,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        const N: usize = 256;
        const K: usize = 173;
        const BASE: usize = 17;
        let module = Module::<BE>::new(N as u64);
        for noise in [
            Noise::Gaussian {
                sigma: 2f64.powi(132),
                cutoff_factor: 6,
            },
            Noise::Uniform { bits: 135 },
        ] {
            let active = K.div_ceil(BASE);
            let padding = active * BASE - K;
            let mut result = VecZnx::from_data(
                BE::alloc_zeroed_bytes(VecZnx::<poulpy_hal::AlignedBuf, i64>::bytes_of(N, 2, active + 1)),
                N,
                2,
                active + 1,
            );
            for limb in 0..=active {
                result.at_mut(0, limb).fill(91);
            }
            BE::vec_znx_add_noise(
                &module,
                BASE,
                K,
                &mut vec_znx_backend_mut::<BE>(&mut result),
                1,
                noise,
                [64; 32],
            );
            let mut low_bits = [0usize; 8];
            let mut wide = 0;
            let mut negative = 0;
            for i in 0..N {
                let integer = (0..active).fold(IBig::ZERO, |n, limb| (n << BASE) + IBig::from(result.at(1, limb)[i]));
                assert_eq!(&integer % (IBig::ONE << padding), IBig::ZERO);
                let sample = integer >> padding;
                negative += usize::from(sample < IBig::ZERO);
                let (_, magnitude) = sample.into_parts();
                wide += usize::from(magnitude > (UBig::ONE << 128));
                assert!(magnitude <= UBig::ONE << 135);
                low_bits[usize::try_from(&magnitude & UBig::from(7u8)).unwrap()] += 1;
            }
            assert!(wide > 220);
            assert!((80..176).contains(&negative));
            assert!(low_bits.iter().all(|&count| count > 12));
            for limb in 0..=active {
                assert!(result.at(0, limb).iter().all(|&x| x == 91));
            }
            assert!(result.at(1, active).iter().all(|&x| x == 0));
        }
    }

    #[test]
    fn fft_full_width_noise_reaches_unit_grid() {
        full_width::<FFT64Oracle>();
    }

    #[test]
    fn ntt_full_width_noise_reaches_unit_grid() {
        full_width::<NTT4x30Oracle>();
    }
}
