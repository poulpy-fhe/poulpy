use std::f64::consts::{LN_2, PI};

use super::{Backend, ZnxWord};

/// Backend-specific radix selection under an input and numerical-error model.
pub trait MaxBase2k: Backend {
    /// Estimates the largest radix for `products` accumulated degree-`n` products
    /// with a whole-polynomial failure target of `2^(-failure_bits)`.
    /// Set `squaring` if any term is a square; `products` still counts actual
    /// terms (one for a single square). Otherwise all operands are independent.
    /// Caps the radix at `Self::ZnxWord::BITS - 2` for coefficient headroom.
    /// Returns `Some(0)` if no positive radix fits, or `None` without a model.
    /// See [`Module::max_base2k`](super::Module::max_base2k) for input constraints.
    ///
    /// Current NTT and FFT64 models assume independent centered uniform
    /// coefficients within each input and independent inputs across terms.
    /// `squaring` allows equal operands within a term.
    /// Their Gaussian failure estimates are not guarantees; other correlations,
    /// including operand reuse across terms, require a separate model.
    fn max_base2k(n: usize, products: usize, failure_bits: usize, squaring: bool) -> Option<usize>;
}

/// NTT radix for independent products or squares of centered uniform inputs.
/// Uses `sigma = 2^(2*k) * sqrt(n*products) / 12`, multiplied by `sqrt(2)`
/// when `squaring` is true, reconstruction threshold `Q/2`, and the Mills-ratio
/// upper bound with a union bound over `n` coefficients. Count each square once.
/// Caps at `B::ZnxWord::BITS - 2` (62 for `i64`, 30 for a 32-bit word); zero means none fits.
///
/// # Panics
/// Panics unless `log2_modulus` is finite and positive, `n` is a power of two
/// at least 2, and `products` and `failure_bits` are positive.
pub fn max_base2k_ntt<B: Backend>(log2_modulus: f64, n: usize, products: usize, failure_bits: usize, squaring: bool) -> usize {
    assert!(
        log2_modulus.is_finite() && log2_modulus > 0.0,
        "the modulus logarithm must be finite and positive"
    );
    assert!(n >= 2 && n.is_power_of_two(), "NTT degree must be a power of two >= 2");
    assert!(products > 0, "the number of accumulated products must be positive");
    assert!(failure_bits > 0, "the failure target must be positive");

    let log2_n = n.ilog2() as f64;
    // A square repeats off-diagonal pairs: variance is at most twice a product's.
    let log2_variance = log2_n + (products as f64).log2() + if squaring { 1.0 } else { 0.0 };
    let log2_x = log2_modulus + 12.0_f64.log2() - 1.5 - 0.5 * log2_variance;
    max_base2k_from_log2_x::<B>(log2_x, failure_bits as f64 + log2_n)
}

/// FFT64 radix for independent products or squares of centered uniform inputs.
/// Models forward FFTs (shared when squaring), sequential complex accumulation,
/// and one inverse FFT. Assumes Gaussian output error, relative roundoff
/// independent of inputs and other roundoff with variance `u^2/3`, `u=2^-53`,
/// and complex twiddle errors of mean square at most `2*u^2`, with uncorrelated
/// propagated contributions, including across accumulated terms.
/// Set `squaring` if any term is a square, counting each square once in `products`.
/// Uses threshold `1/2` and the same Mills-ratio bound and union bound as NTT.
/// Caps at `B::ZnxWord::BITS - 2` (62 for `i64`, 30 for a 32-bit word); zero means none fits.
///
/// # Panics
/// Panics unless `n` is a power of two at least 2, and `products` and
/// `failure_bits` are positive.
pub fn max_base2k_fft64<B: Backend>(n: usize, products: usize, failure_bits: usize, squaring: bool) -> usize {
    assert!(n >= 2 && n.is_power_of_two(), "FFT degree must be a power of two >= 2");
    assert!(products > 0, "the number of accumulated products must be positive");
    assert!(failure_bits > 0, "the failure target must be positive");

    let log2_n = n.ilog2() as f64;
    let d = products as f64;
    // Floating-point count arithmetic avoids usize overflow for large counts.
    let scalar_mac = 2.0 / 3.0 + (d + 1.0) / 6.0 - 1.0 / (3.0 * d);
    let fused_mac = (d + 0.5) / 3.0;
    let mac_variance = scalar_mac.max(fused_mac);
    // Ordinary products contribute 3*(5/3)*L. Relative to the doubled square
    // variance, shared forward error contributes 4*(5/3)*L and inverse (5/3)*L.
    let transform_variance = if squaring { 25.0 / 3.0 } else { 5.0 };
    let variance_factor = transform_variance * (log2_n - 1.0) + mac_variance;
    let log2_variance = log2_n + d.log2() + variance_factor.log2() + if squaring { 1.0 } else { 0.0 };
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
        // The NTT variance of d squares matches that of 2*d independent products.
        // Both have a whole-polynomial bound of 2^-130.074... at K = 54.
        for (squaring, products) in [(false, 40), (true, 20)] {
            assert_eq!(
                max_base2k_ntt::<HostBytesBackend>(119.886_155_257_481_1, 1 << 16, products, 130, squaring),
                54
            );
            assert_eq!(
                max_base2k_ntt::<HostBytesBackend>(119.886_155_257_481_1, 1 << 16, products, 131, squaring),
                53
            );
        }
        // FFT bounds are 2^-128.125... for products and 2^-96.743... for squares.
        assert_eq!(max_base2k_fft64::<HostBytesBackend>(1 << 16, 35, 128, false), 19);
        assert_eq!(max_base2k_fft64::<HostBytesBackend>(1 << 16, 35, 129, false), 18);
        assert_eq!(max_base2k_fft64::<HostBytesBackend>(8, 1, 96, true), 24);
        assert_eq!(max_base2k_fft64::<HostBytesBackend>(8, 1, 97, true), 23);
    }
}
