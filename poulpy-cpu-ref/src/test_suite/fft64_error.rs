//! Numerical error of accumulated FFT64 products against an exact NTT oracle.

use poulpy_hal::{
    alloc_aligned,
    api::ModuleNew,
    layouts::{Backend, BackendMaxBase2k, Module, PrimeSet},
};
use rand_chacha::{
    ChaCha8Rng,
    rand_core::{Rng, SeedableRng},
};

use crate::{
    NTT4x30Ref,
    reference::{
        fft64::{
            module::{FFTHandleProvider, FFTModuleHandle},
            reim::{ReimArith, ReimFFTExecute, ReimFFTTable, ReimIFFTTable},
        },
        ntt4x30::{NttDFTExecute, NttFromZnx64, NttModuleHandle, NttToZnx128, primes::Primes30},
    },
};

/// Error before coefficient rounding, across one output polynomial.
#[derive(Clone, Copy, Debug)]
pub struct Fft64ErrorStats {
    pub mean: f64,
    pub rms: f64,
    pub max_abs: f64,
}

/// Measures two forward FFTs per product, sequential Fourier-domain
/// multiply-adds, and one inverse FFT. The exact reference uses the same inputs
/// and an NTT4x30 convolution. Fixed seeds make regressions reproducible.
///
/// The output integers must be exactly representable in `f64`; this is checked
/// before computing their differences from the unrounded FFT output. A finite
/// sample can validate the modeled error scale, but cannot certify a far tail.
pub fn fft64_accumulation_error<BE>(n: usize, base2k: usize, products: usize, seed: u64) -> Fft64ErrorStats
where
    BE: Backend<ZnxWord = i64, DftWord = f64>
        + ReimArith
        + ReimFFTExecute<ReimFFTTable<f64>, f64>
        + ReimFFTExecute<ReimIFFTTable<f64>, f64>,
    BE::Handle: FFTHandleProvider<f64>,
    Module<BE>: ModuleNew<BE>,
{
    assert!((1..=24).contains(&base2k));
    assert!(n.is_power_of_two() && n <= 1 << Primes30::MAX_LOG_N);
    assert!(products > 0);
    let q = Primes30::Q.into_iter().map(u128::from).product::<u128>();
    let hard_bound = (n as u128) * (products as u128) * (1u128 << (2 * base2k - 2));
    assert!(hard_bound < q / 2, "exact NTT oracle must not wrap");

    let module = Module::<BE>::new(n as u64);
    let ntt = Module::<NTT4x30Ref>::new(n as u64);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut a = alloc_aligned::<i64>(n);
    let mut b = alloc_aligned::<i64>(n);
    let mut a_fft = alloc_aligned::<f64>(n);
    let mut b_fft = alloc_aligned::<f64>(n);
    let mut sum_fft = alloc_aligned::<f64>(n);
    let mut a_ntt = vec![0u64; 4 * n];
    let mut b_ntt = vec![0u64; 4 * n];
    let mut sum_ntt = vec![0u64; 4 * n];
    let mask = (1u64 << base2k) - 1;
    let p = 1i64 << (base2k - 1);

    for _ in 0..products {
        for coefficients in [&mut a, &mut b] {
            for coefficient in coefficients.iter_mut() {
                *coefficient = (rng.next_u64() & mask) as i64 - p;
            }
        }
        BE::reim_from_znx(&mut a_fft, &a);
        BE::reim_from_znx(&mut b_fft, &b);
        BE::reim_dft_execute(module.get_fft_table_for(n), &mut a_fft);
        BE::reim_dft_execute(module.get_fft_table_for(n), &mut b_fft);
        BE::reim_addmul(&mut sum_fft, &a_fft, &b_fft);

        NTT4x30Ref::ntt_from_znx64(&mut a_ntt, &a);
        NTT4x30Ref::ntt_from_znx64(&mut b_ntt, &b);
        NTT4x30Ref::ntt_dft_execute(ntt.get_ntt_table_for(n), &mut a_ntt);
        NTT4x30Ref::ntt_dft_execute(ntt.get_ntt_table_for(n), &mut b_ntt);
        for (j, ((sum, &a), &b)) in sum_ntt.iter_mut().zip(&a_ntt).zip(&b_ntt).enumerate() {
            let prime = Primes30::Q[j % 4] as u128;
            let product = (a as u128) * (b as u128) % prime;
            *sum = ((*sum as u128 + product) % prime) as u64;
        }
    }

    BE::reim_dft_execute(module.get_ifft_table_for(n), &mut sum_fft);
    NTT4x30Ref::ntt_dft_execute(ntt.get_intt_table_for(n), &mut sum_ntt);
    let mut exact = vec![0i128; n];
    NTT4x30Ref::ntt_to_znx128(&mut exact, n, &sum_ntt);
    let mut sum = 0.0;
    let mut sum_squares = 0.0;
    let mut max_abs: f64 = 0.0;
    for (&approximate, &integer) in sum_fft.iter().zip(&exact) {
        assert!(
            integer.abs() <= 1i128 << 53,
            "oracle result must be exactly representable in f64"
        );
        // The reim inverse is unnormalized with scale N/2. Division by this
        // power of two is exact and keeps the numerical error observable.
        let error = approximate / (n / 2) as f64 - integer as f64;
        assert!(error.is_finite());
        sum += error;
        sum_squares += error * error;
        max_abs = max_abs.max(error.abs());
    }
    Fft64ErrorStats {
        mean: sum / n as f64,
        rms: (sum_squares / n as f64).sqrt(),
        max_abs,
    }
}

/// Small regression for correct transform scaling, exact reconstruction, and
/// rounding after a chain of Fourier-domain multiply-adds.
pub fn test_fft64_accumulation_error<BE>()
where
    BE: Backend<ZnxWord = i64, DftWord = f64>
        + BackendMaxBase2k
        + ReimArith
        + ReimFFTExecute<ReimFFTTable<f64>, f64>
        + ReimFFTExecute<ReimIFFTTable<f64>, f64>,
    BE::Handle: FFTHandleProvider<f64>,
    Module<BE>: ModuleNew<BE>,
{
    let n = 1024;
    for products in [1, 32, 128] {
        let base2k = Module::<BE>::max_base2k(n, products, 128).unwrap();
        for seed in [0, 1] {
            let stats = fft64_accumulation_error::<BE>(n, base2k, products, seed);
            let sigma = modeled_sigma(n, base2k, products);
            assert!(stats.rms > 0.0);
            assert!(stats.rms <= sigma, "measured {stats:?}, modeled sigma={sigma}");
            assert!(stats.max_abs < 0.5, "{stats:?}");
        }
    }
}

/// Prints larger deterministic measurements for explicit model validation.
/// Kept separate from the small regular test because each row computes an exact
/// NTT oracle. These measurements do not estimate a 128-bit failure probability.
pub fn measure_fft64_accumulation_error<BE>()
where
    BE: Backend<ZnxWord = i64, DftWord = f64>
        + BackendMaxBase2k
        + ReimArith
        + ReimFFTExecute<ReimFFTTable<f64>, f64>
        + ReimFFTExecute<ReimIFFTTable<f64>, f64>,
    BE::Handle: FFTHandleProvider<f64>,
    Module<BE>: ModuleNew<BE>,
{
    for log_n in [15, 16] {
        for products in [1, 32, 128] {
            let selected = Module::<BE>::max_base2k(1 << log_n, products, 128).unwrap();
            let mut radices = vec![19, 20, selected];
            radices.sort_unstable();
            radices.dedup();
            for base2k in radices {
                let stats = fft64_accumulation_error::<BE>(1 << log_n, base2k, products, 0);
                let sigma = modeled_sigma(1 << log_n, base2k, products);
                println!("log_n={log_n}, products={products}, base2k={base2k}, {stats:?}, modeled_sigma={sigma}");
                assert!(stats.rms <= sigma, "measured {stats:?}, modeled sigma={sigma}");
                if base2k == selected {
                    assert!(stats.max_abs < 0.5, "selected radix must round correctly: {stats:?}");
                }
            }
        }
    }
}

// Runtime evaluation separate from the const selector: the statistic above
// comes from actual arithmetic rather than evaluating this expression twice.
fn modeled_sigma(n: usize, base2k: usize, products: usize) -> f64 {
    let d = products as f64;
    let stages = (n as f64).log2() - 1.0;
    let scalar_mac = 2.0 / 3.0 + (d + 1.0) / 6.0 - 1.0 / (3.0 * d);
    let fused_mac = (d + 0.5) / 3.0;
    let relative_variance = 5.0 * stages + scalar_mac.max(fused_mac);
    2.0f64.powi(2 * base2k as i32 - 53) * (n as f64 * d * relative_variance).sqrt() / 12.0
}

#[cfg(test)]
mod tests {
    #[test]
    fn fft64_accumulation_error() {
        super::test_fft64_accumulation_error::<crate::FFT64Ref>();
    }

    #[test]
    #[ignore = "numerical-error model measurements with exact NTT oracles"]
    fn fft64_accumulation_error_measurements() {
        super::measure_fft64_accumulation_error::<crate::FFT64Ref>();
    }
}
