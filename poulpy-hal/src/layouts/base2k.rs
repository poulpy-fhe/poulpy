use std::f64::consts::LN_2;

use super::Backend;

/// Backend-specific radix selection for a requested failure budget.
///
/// Implement this trait on a backend to enable [`Module::max_base2k`](super::Module::max_base2k).
/// The backend chooses the model appropriate for its arithmetic. Backends can
/// reuse [`max_base2k_ntt`] or [`max_base2k_fft64`], or supply a different model
/// with documented input-distribution and numerical-error assumptions.
///
/// Storage or handle delegation alone does not imply that two backends share a model;
/// wrappers that preserve the arithmetic can explicitly forward this trait.
pub trait BackendMaxBase2k: Backend {
    /// Estimates the largest supported limb radix for `products` accumulated
    /// polynomial products of degree `n`, targeting a probability at most
    /// `2^(-failure_bits)` that any coefficient of the output polynomial fails.
    ///
    /// Returns `Some(0)` when no positive radix meets the target, or `None` if
    /// the backend has no applicable model for this workload. An estimate does
    /// not reserve headroom for coefficient-domain additions or normalization.
    ///
    /// [`Module::max_base2k`](super::Module::max_base2k) checks that `n` is a
    /// power of two at least [`Backend::MIN_DEGREE`], and that `products` and
    /// `failure_bits` are positive before forwarding to this method. Direct
    /// callers must supply those valid parameters too.
    fn max_base2k(n: usize, products: usize, failure_bits: usize) -> Option<usize>;
}

/// Selects an NTT radix using a conservative Gaussian failure estimate.
///
/// Model each output coefficient as a sum of `n * products` independent
/// products of centered uniform radix-`k` digits. Its standard deviation is
/// `sigma = 2^(2*k) * sqrt(n * products) / 12`. Centered CRT reconstruction
/// fails beyond `Q / 2`, giving the Gaussian estimate
/// `erfc(Q / (sqrt(8) * sigma))` for one coefficient.
///
/// The envelope `erfc(x) <= exp(-x*x)` and a union bound over all `n`
/// coefficients make that estimate at most `2^(-failure_bits)` when
///
/// ```text
/// k <= (log2(Q) + log2(6) - log2(n * products) / 2
///       - log2(2 * ln(2) * (failure_bits + log2(n))) / 2) / 2.
/// ```
///
/// This is conservative within the Gaussian model; it does not certify the
/// tails of the actual discrete product distribution. The result is capped at
/// the supported signed-digit radix 62, and zero denotes no positive radix.
///
/// # Panics
///
/// Panics if `log2_modulus` is not finite and positive, `n` is not a power of
/// two, or `products` or `failure_bits` is zero.
pub fn max_base2k_ntt(log2_modulus: f64, n: usize, products: usize, failure_bits: usize) -> usize {
    assert!(
        log2_modulus.is_finite() && log2_modulus > 0.0,
        "the modulus logarithm must be finite and positive"
    );
    assert!(n.is_power_of_two(), "n must be a power of two");
    assert!(products > 0, "the number of accumulated products must be positive");
    assert!(failure_bits > 0, "the failure target must be positive");

    let log2_n = n.ilog2() as f64;
    let log2_variance = log2_n + (products as f64).log2();
    let log2_tail = (2.0 * LN_2 * (failure_bits as f64 + log2_n)).log2();
    // Leave a small margin against upward rounding at an integer threshold.
    // Float-to-integer casts saturate negative values to zero.
    let radix = ((log2_modulus + 6.0_f64.log2() - 0.5 * log2_variance - 0.5 * log2_tail) / 2.0 - 1e-12).floor() as usize;
    radix.min(62)
}

/// Selects an FFT64 radix using a first-order stochastic roundoff model.
///
/// With `u = 2^-53`, `L = log2(n) - 1`, and `d = products`, model
///
/// ```text
/// R = 5*L + max(2/3 + (d+1)/6 - 1/(3*d), (d+1/2)/3)
/// sigma_e = 2^(2*k-53) * sqrt(n*d*R) / 12.
/// ```
///
/// Arithmetic roundoff is modeled as independent, centered relative errors
/// with variance `u^2/3`. A complex twiddle has mean squared absolute error
/// at most `2*u^2` in this model. Each FFT stage then contributes `5*u^2/3`
/// relative error variance, hence `5*L` for two forward FFTs and one inverse.
/// Separate complex multiplication contributes `2/3`; sequential addition
/// contributes `sum(j, j=2..d)/(3*d) = (d+1)/6 - 1/(3*d)`.
/// Fused kernels instead update each accumulator twice per product, giving
/// `sum(2*j-1/2, j=1..d)/(3*d) = (d+1/2)/3`. Take the larger MAC variance
/// to cover both implementations. The power-of-two inverse scaling is exact.
///
/// Integer recovery requires error below 1/2. The Gaussian envelope and union
/// bound give `sigma_e <= 1/sqrt(8*ln(2)*(failure_bits+log2(n)))`.
/// These are stochastic assumptions, including decorrelation of reused twiddle
/// errors; this does not certify the far tail of actual floating-point error.
/// The result is capped at radix 62; zero denotes no positive radix.
///
/// # Panics
///
/// Panics if `n` is not a power of two at least 2, or `products` or
/// `failure_bits` is zero.
pub fn max_base2k_fft64(n: usize, products: usize, failure_bits: usize) -> usize {
    assert!(n >= 2 && n.is_power_of_two(), "FFT degree must be a power of two >= 2");
    assert!(products > 0, "the number of accumulated products must be positive");
    assert!(failure_bits > 0, "the failure target must be positive");

    let log2_n = n.ilog2() as f64;
    let d = products as f64;
    // Floating-point count arithmetic avoids usize overflow for large counts.
    let scalar_mac = 2.0 / 3.0 + (d + 1.0) / 6.0 - 1.0 / (3.0 * d);
    let fused_mac = (d + 0.5) / 3.0;
    let mac_variance = scalar_mac.max(fused_mac);
    let variance_factor = 5.0 * (log2_n - 1.0) + mac_variance;
    let log2_variance = log2_n + d.log2() + variance_factor.log2();
    let log2_tail = (8.0 * LN_2 * (failure_bits as f64 + log2_n)).log2();
    let radix = ((53.0 + 12.0_f64.log2() - 0.5 * log2_variance - 0.5 * log2_tail) / 2.0 - 1e-12).floor() as usize;
    radix.min(62)
}

#[cfg(test)]
mod tests {
    use super::{BackendMaxBase2k, max_base2k_fft64, max_base2k_ntt};
    use crate as poulpy_hal;
    use crate::layouts::{HostBytesBackend, Module};

    #[derive(PartialEq, Eq)]
    struct CustomModelBackend;

    // Reuse storage without inheriting the source backend's radix model.
    crate::impl_backend_from!(CustomModelBackend, HostBytesBackend);

    impl BackendMaxBase2k for CustomModelBackend {
        fn max_base2k(n: usize, products: usize, failure_bits: usize) -> Option<usize> {
            // Distinct contributions detect dropped or reordered arguments.
            Some(n.ilog2() as usize + 2 * products + failure_bits / 128)
        }
    }

    #[test]
    fn max_base2k_dispatches_to_the_backend() {
        let radix: Option<usize> = Module::<CustomModelBackend>::max_base2k(16, 3, 256);
        let unsupported: Option<usize> = Module::<HostBytesBackend>::max_base2k(16, 3, 256);
        assert_eq!(radix, Some(12));
        assert_eq!(unsupported, None);
        assert_eq!(Module::<CustomModelBackend>::max_base2k(32, 5, 128), Some(16));
    }

    const LOG2_Q: f64 = 119.886_155_257_481_1;

    #[test]
    fn probability_radix_examples() {
        assert_eq!(max_base2k_ntt(LOG2_Q, 1 << 15, 32, 128), 54);
        assert_eq!(max_base2k_ntt(LOG2_Q, 1 << 16, 32, 128), 54);
        assert_eq!(max_base2k_ntt(LOG2_Q, 1 << 16, 32, 256), 53);
        assert_eq!(max_base2k_ntt(1.0, 1 << 16, 32, 128), 0);
        assert_eq!(max_base2k_ntt(1024.0, 8, 1, 1), 62);
    }

    #[test]
    fn probability_radix_decreases_with_accumulation_and_failure_budget() {
        let counts = [1, 2, 3, 32, 97, 256, 65_535, usize::MAX];
        let targets = [1, 40, 128, 256, 1024, usize::MAX];
        for n in [1 << 15, 1 << 16] {
            for target in targets {
                let mut previous = 62;
                for count in counts {
                    let current = max_base2k_ntt(LOG2_Q, n, count, target);
                    assert!(current <= previous);
                    previous = current;
                }
            }
            for count in counts {
                let mut previous = 62;
                for target in targets {
                    let current = max_base2k_ntt(LOG2_Q, n, count, target);
                    assert!(current <= previous);
                    previous = current;
                }
            }
        }
    }

    #[test]
    fn fft_probability_radix_examples() {
        assert_eq!(max_base2k_fft64(1 << 15, 32, 128), 19);
        assert_eq!(max_base2k_fft64(1 << 16, 32, 128), 19);
        assert_eq!(max_base2k_fft64(1 << 16, 32, 256), 18);
        assert_eq!(max_base2k_fft64(1 << 16, 128, 128), 18);
        assert_eq!(max_base2k_fft64(1usize << (usize::BITS - 1), usize::MAX, usize::MAX), 0);
    }
}
