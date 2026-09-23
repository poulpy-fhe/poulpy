use std::f64::consts::LN_2;

use super::Backend;

/// Backend-specific radix selection under an input and numerical-error model.
pub trait BackendMaxBase2k: Backend {
    /// Estimates the largest radix for `products` accumulated degree-`n` products
    /// with a whole-polynomial failure target of `2^(-failure_bits)`.
    /// Returns `Some(0)` if no positive radix fits, or `None` without a model.
    /// See [`Module::max_base2k`](super::Module::max_base2k) for input constraints.
    fn max_base2k(n: usize, products: usize, failure_bits: usize) -> Option<usize>;
}

/// NTT radix under independent, centered uniform inputs and a Gaussian tail model.
/// Uses `sigma = 2^(2*k) * sqrt(n*products) / 12`, reconstruction threshold `Q/2`,
/// and the envelope `erfc(x) <= exp(-x*x)` with a union bound over `n` coefficients.
/// Returns a radix capped at 62; zero means no positive radix fits.
///
/// # Panics
/// Panics unless `log2_modulus` is finite and positive, `n` is a power of two,
/// and `products` and `failure_bits` are positive.
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

/// FFT64 radix under independent centered uniform inputs and Gaussian roundoff.
/// Models two forward FFTs per product, sequential complex accumulation, and one
/// inverse FFT, with independent relative roundoff variance `u^2/3`, `u=2^-53`,
/// and uncorrelated complex twiddle errors of mean square at most `2*u^2`.
/// Uses threshold `1/2` and the same Gaussian envelope and union bound as NTT.
/// Returns a radix capped at 62; zero means no positive radix fits.
///
/// # Panics
/// Panics unless `n` is a power of two at least 2, and `products` and
/// `failure_bits` are positive.
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
