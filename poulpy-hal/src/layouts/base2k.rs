use std::f64::consts::{LN_2, LOG2_E};

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
pub(super) const fn max_base2k_ntt(log2_modulus: f64, n: usize, products: usize, failure_bits: usize) -> usize {
    assert!(
        log2_modulus.is_finite() && log2_modulus > 0.0,
        "the modulus logarithm must be finite and positive"
    );
    assert!(n.is_power_of_two(), "n must be a power of two");
    assert!(products > 0, "the number of accumulated products must be positive");
    assert!(failure_bits > 0, "the failure target must be positive");

    let log2_n = n.ilog2() as f64;
    let log2_variance = log2_n + log2(products as f64);
    let log2_tail = log2(2.0 * LN_2 * (failure_bits as f64 + log2_n));
    // Leave a small margin against upward rounding at an integer threshold.
    // Float-to-integer casts saturate negative values to zero.
    let radix = ((log2_modulus + 2.584_962_500_721_156 - 0.5 * log2_variance - 0.5 * log2_tail) / 2.0 - 1e-12).floor() as usize;
    if radix > 62 { 62 } else { radix }
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
pub(super) const fn max_base2k_fft64(n: usize, products: usize, failure_bits: usize) -> usize {
    assert!(n >= 2 && n.is_power_of_two(), "FFT degree must be a power of two >= 2");
    assert!(products > 0, "the number of accumulated products must be positive");
    assert!(failure_bits > 0, "the failure target must be positive");

    let log2_n = n.ilog2() as f64;
    let d = products as f64;
    // Floating-point count arithmetic avoids usize overflow for large counts.
    let scalar_mac = 2.0 / 3.0 + (d + 1.0) / 6.0 - 1.0 / (3.0 * d);
    let fused_mac = (d + 0.5) / 3.0;
    let mac_variance = if scalar_mac > fused_mac { scalar_mac } else { fused_mac };
    let variance_factor = 5.0 * (log2_n - 1.0) + mac_variance;
    let log2_variance = log2_n + log2(d) + log2(variance_factor);
    let log2_tail = log2(8.0 * LN_2 * (failure_bits as f64 + log2_n));
    let radix = ((53.0 + 3.584_962_500_721_156 - 0.5 * log2_variance - 0.5 * log2_tail) / 2.0 - 1e-12).floor() as usize;
    if radix > 62 { 62 } else { radix }
}

/// Computes log2 for the positive normal inputs used by the selector.
const fn log2(x: f64) -> f64 {
    assert!(x.is_finite() && x >= f64::MIN_POSITIVE);
    let bits = x.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32 - 1023;
    let mantissa = f64::from_bits((bits & ((1_u64 << 52) - 1)) | (1023_u64 << 52));
    // ln(m) = 2 * atanh((m - 1) / (m + 1)). With 1 <= m < 2,
    // the series argument is below 1/3; 20 terms put truncation below 3e-21.
    let z = (mantissa - 1.0) / (mantissa + 1.0);
    let z_squared = z * z;
    let mut term = z;
    let mut sum = 0.0;
    let mut i = 0;
    while i < 20 {
        sum += term / (2 * i + 1) as f64;
        term *= z_squared;
        i += 1;
    }
    exponent as f64 + (2.0 * LOG2_E) * sum
}

#[cfg(test)]
mod tests {
    use super::{log2, max_base2k_fft64, max_base2k_ntt};

    const LOG2_Q: f64 = 119.886_155_257_481_1;
    const RADIX_N15: usize = max_base2k_ntt(LOG2_Q, 1 << 15, 32, 128);
    const RADIX_N16: usize = max_base2k_ntt(LOG2_Q, 1 << 16, 32, 128);
    const RADIX_N16_256: usize = max_base2k_ntt(LOG2_Q, 1 << 16, 32, 256);

    #[test]
    fn probability_radix_const_examples() {
        assert_eq!(RADIX_N15, 54);
        assert_eq!(RADIX_N16, 54);
        assert_eq!(RADIX_N16_256, 53);
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
    fn fft_probability_radix_const_examples() {
        const N15: usize = max_base2k_fft64(1 << 15, 32, 128);
        const N16: usize = max_base2k_fft64(1 << 16, 32, 128);
        const STRICTER: usize = max_base2k_fft64(1 << 16, 32, 256);
        assert_eq!((N15, N16, STRICTER), (19, 19, 18));
        assert_eq!(max_base2k_fft64(1 << 16, 128, 128), 18);
        assert_eq!(max_base2k_fft64(1usize << (usize::BITS - 1), usize::MAX, usize::MAX), 0);
    }

    #[test]
    fn const_log2_matches_runtime_oracle() {
        for exponent in -1022..=1023 {
            let power = f64::from_bits(((exponent + 1023) as u64) << 52);
            assert_eq!(log2(power), exponent as f64);
        }
        for x in [
            1.000_000_000_000_001,
            1.5,
            1.999_999_999_999_999,
            3.0,
            31.0,
            97.0,
            65_535.0,
            usize::MAX as f64,
        ] {
            assert!((log2(x) - x.log2()).abs() < 2e-14, "x = {x}");
        }
    }
}
