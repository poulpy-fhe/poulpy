use std::f64::consts::{LN_2, PI};

use super::{Backend, ZnxWord};

/// Backend-specific radix selection under an input and numerical-error model.
pub trait MaxBase2k: Backend {
    /// Estimates the largest radix for `products` accumulated degree-`n` products
    /// with a whole-polynomial failure target of `2^(-failure_bits)`.
    /// Caps the radix at `Self::ZnxWord::BITS - 2` for coefficient headroom.
    /// Returns `Some(0)` if no positive radix fits, or `None` without a model.
    /// See [`Module::max_base2k`](super::Module::max_base2k) for input constraints.
    ///
    /// Current NTT and FFT64 models cover independent accumulated terms, each
    /// an independent product or a square of centered uniform coefficients.
    /// Their Gaussian failure estimates are not guarantees; other correlations,
    /// including operand reuse across terms, require a separate model.
    fn max_base2k(n: usize, products: usize, failure_bits: usize) -> Option<usize>;
}

/// NTT radix for independent products or squares of centered uniform inputs.
/// Uses the common Gaussian budget `sigma = 2^(2*k) * sqrt(2*n*products) / 12`,
/// reconstruction threshold `Q/2`, and the Mills-ratio upper bound with a
/// union bound over `n` coefficients.
/// Caps at `B::ZnxWord::BITS - 2` (62 for `i64`, 30 for a 32-bit word); zero means none fits.
///
/// # Panics
/// Panics unless `log2_modulus` is finite and positive, `n` is a power of two
/// at least 2, and `products` and `failure_bits` are positive.
pub fn max_base2k_ntt<B: Backend>(log2_modulus: f64, n: usize, products: usize, failure_bits: usize) -> usize {
    assert!(
        log2_modulus.is_finite() && log2_modulus > 0.0,
        "the modulus logarithm must be finite and positive"
    );
    assert!(n >= 2 && n.is_power_of_two(), "NTT degree must be a power of two >= 2");
    assert!(products > 0, "the number of accumulated products must be positive");
    assert!(failure_bits > 0, "the failure target must be positive");

    let log2_n = n.ilog2() as f64;
    // A square repeats off-diagonal pairs: variance is at most twice a product's.
    let log2_variance = 1.0 + log2_n + (products as f64).log2();
    let log2_x = log2_modulus + 12.0_f64.log2() - 1.5 - 0.5 * log2_variance;
    max_base2k_from_log2_x::<B>(log2_x, failure_bits as f64 + log2_n)
}

/// FFT64 radix for independent products or squares of centered uniform inputs.
/// Models forward FFTs (shared when squaring), sequential complex accumulation,
/// and one inverse FFT. Assumes Gaussian output error, relative roundoff
/// independent of inputs and other roundoff with variance `u^2/3`, `u=2^-53`,
/// and complex twiddle errors of mean square at most `2*u^2`, with uncorrelated
/// propagated contributions, including across accumulated terms.
/// Uses threshold `1/2` and the same Mills-ratio bound and union bound as NTT.
/// Caps at `B::ZnxWord::BITS - 2` (62 for `i64`, 30 for a 32-bit word); zero means none fits.
///
/// # Panics
/// Panics unless `n` is a power of two at least 2, and `products` and
/// `failure_bits` are positive.
pub fn max_base2k_fft64<B: Backend>(n: usize, products: usize, failure_bits: usize) -> usize {
    assert!(n >= 2 && n.is_power_of_two(), "FFT degree must be a power of two >= 2");
    assert!(products > 0, "the number of accumulated products must be positive");
    assert!(failure_bits > 0, "the failure target must be positive");

    let log2_n = n.ilog2() as f64;
    let d = products as f64;
    // Floating-point count arithmetic avoids usize overflow for large counts.
    let scalar_mac = 2.0 / 3.0 + (d + 1.0) / 6.0 - 1.0 / (3.0 * d);
    let fused_mac = (d + 0.5) / 3.0;
    let mac_variance = scalar_mac.max(fused_mac);
    // Relative to the doubled square variance, shared forward error contributes
    // 4*(5/3)*L and inverse error (5/3)*L; this also covers independent products.
    let variance_factor = (25.0 / 3.0) * (log2_n - 1.0) + mac_variance;
    let log2_variance = 1.0 + log2_n + d.log2() + variance_factor.log2();
    let log2_x = 53.0 + 12.0_f64.log2() - 1.5 - 0.5 * log2_variance;
    max_base2k_from_log2_x::<B>(log2_x, failure_bits as f64 + log2_n)
}

// `log2_x` is log2(T / (sqrt(2) * sigma)) at K = 0; sigma grows as 2^(2*K).
// The Mills bound is exp(-x*x - asinh(sqrt(pi)*x/2)); see https://dlmf.nist.gov/7.8.E2.
fn max_base2k_from_log2_x<B: Backend>(log2_x: f64, failure_bits: f64) -> usize {
    let target = LN_2 * failure_bits;
    let (mut lo, mut hi) = (0, B::ZnxWord::BITS.saturating_sub(2));
    while lo < hi {
        let k = lo + (hi - lo).div_ceil(2);
        // Slightly reduce x to leave a numerical margin at the threshold.
        let x = (log2_x - 2.0 * k as f64 - 1e-12).exp2();
        if x * x + (0.5 * PI.sqrt() * x).asinh() >= target {
            lo = k;
        } else {
            hi = k - 1;
        }
    }
    lo
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layouts::HostBytesBackend;

    #[test]
    fn mills_bound_admits_larger_radices_at_the_requested_target() {
        // Including squares, the polynomial bounds are 2^-130.074... and 2^-96.743....
        assert_eq!(
            max_base2k_ntt::<HostBytesBackend>(119.886_155_257_481_1, 1 << 16, 20, 130),
            54
        );
        assert_eq!(
            max_base2k_ntt::<HostBytesBackend>(119.886_155_257_481_1, 1 << 16, 20, 131),
            53
        );
        assert_eq!(max_base2k_fft64::<HostBytesBackend>(8, 1, 96), 24);
        assert_eq!(max_base2k_fft64::<HostBytesBackend>(8, 1, 97), 23);
    }
}
