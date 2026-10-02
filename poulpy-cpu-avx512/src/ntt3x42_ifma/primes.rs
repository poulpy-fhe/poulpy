//! Prime sets and modular helpers for the 3-prime IFMA representation.

use poulpy_hal::layouts::{LaneElem, PrimeSet};

/// Extension of the unified [`PrimeSet`] for the 3-prime IFMA family,
/// carrying its **Garner** CRT reconstruction constants.
///
/// The product `Q = Q[0]·Q[1]·Q[2]` is approximately 2^126 (fits in `i128`
/// with two bits of margin).
/// All three primes support a primitive `2^19`-th root of unity, so
/// NTT sizes up to `2^18` are supported.
///
/// Unlike the full-CRT constants of
/// [`PrimeSetCrt4`](poulpy_cpu_ref::reference::ntt4x30::primes::PrimeSetCrt4),
/// the constants here follow Garner's algorithm and are semantically
/// specific to this family.
pub trait PrimeSetNtt3x42Ifma: PrimeSet<PrimeElem = u64, Lanes<u64> = [u64; 3]> {
    /// CRT reconstruction constants (Garner's algorithm).
    ///
    /// - `CRT_CST\[0\]` = `inv(Q\[0\], Q\[1\])` — inverse of `Q\[0\]` modulo `Q\[1\]`
    /// - `CRT_CST\[1\]` = `inv(Q\[0\]*Q\[1\], Q\[2\])` — inverse of `Q\[0\]*Q\[1\]` modulo `Q\[2\]`
    /// - `CRT_CST\[2\]` is unused (reserved / zero)
    const CRT_CST: [u64; 3];
}

/// 42-bit NTT-friendly primes with `2·2^18`-th roots of unity.
///
/// - `Q ≈ 2^126` (product fits in `i128` with two bits of margin).
/// - Each prime is of the form `c·2^19 + 1` with `c·2^19 + 1 < 2^42`.
/// - Designed for use with AVX512-IFMA52 instructions (primes < 2^49).
pub struct Primes42;

impl PrimeSet for Primes42 {
    type PrimeElem = u64;
    type Lanes<T: LaneElem> = [T; 3];
    const Q: [u64; 3] = [
        4_398_044_938_241, // 8_388_605 * 2^19 + 1
        4_398_021_869_569, // 8_388_561 * 2^19 + 1
        4_398_021_345_281, // 8_388_560 * 2^19 + 1
    ];
    const OMEGA: [u64; 3] = [2_628_857_985_221, 3_217_638_597_750, 1_217_792_891_299];
    const LOG_Q: u64 = 42;
    // log2 of the exact prime product, precomputed at high precision.
    const LOG_Q_PRODUCT: f64 = 125.999_983_145_654_84;
    const MAX_LOG_N: u32 = 18;
}

impl PrimeSetNtt3x42Ifma for Primes42 {
    const CRT_CST: [u64; 3] = [2_498_875_871_606, 2_541_070_051_698, 0];
}

/// Computes `x^n mod q` using square-and-multiply with 128-bit intermediates.
///
/// Handles negative exponents via `x^(-(|n| mod (q-1)))`.
pub fn modq_pow64(x: u64, n: i64, q: u64) -> u64 {
    let qm1 = (q - 1) as i64;
    // reduce exponent mod (q-1) to positive representative
    let np = ((n % qm1) + qm1) % qm1;
    let mut np = np as u64;
    let mut val_pow = x % q;
    let q128 = q as u128;
    let mut result = 1u64;
    while np > 0 {
        if np & 1 != 0 {
            result = ((result as u128 * val_pow as u128) % q128) as u64;
        }
        val_pow = ((val_pow as u128 * val_pow as u128) % q128) as u64;
        np >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primes42_are_prime() {
        Primes42::validate();
        for &q in &Primes42::Q {
            assert!(is_prime(q), "{q} is not prime");
            assert_eq!(u64::BITS - q.leading_zeros(), Primes42::LOG_Q as u32);
        }
    }

    #[test]
    fn primes42_omega_are_primitive_roots() {
        for k in 0..3 {
            let q = Primes42::Q[k];
            let omega = Primes42::OMEGA[k];
            assert_eq!(
                modq_pow64(omega, 1 << (Primes42::MAX_LOG_N + 1), q),
                1,
                "omega[{k}]^(2^19) != 1"
            );
            assert_eq!(
                modq_pow64(omega, 1 << Primes42::MAX_LOG_N, q),
                q - 1,
                "omega[{k}] is not primitive"
            );
        }
    }

    #[test]
    fn primes42_crt_roundtrip() {
        let q = Primes42::Q;
        let q01 = q[0] as u128 * q[1] as u128;
        let big_q = q01 * q[2] as u128;

        for x in [
            0i128,
            1,
            -1,
            42,
            i64::MAX as i128,
            i64::MIN as i128,
            (big_q / 2) as i128,
            -((big_q / 2) as i128),
        ] {
            let r: [u64; 3] = [
                (x.rem_euclid(q[0] as i128)) as u64,
                (x.rem_euclid(q[1] as i128)) as u64,
                (x.rem_euclid(q[2] as i128)) as u64,
            ];

            // Garner reconstruction
            let inv01 = <Primes42 as PrimeSetNtt3x42Ifma>::CRT_CST[0];
            let inv012 = <Primes42 as PrimeSetNtt3x42Ifma>::CRT_CST[1];

            let v0 = r[0] as u128;
            // Modular subtraction: (r[1] - v0) mod q[1]
            let r1_mod = (r[1] % q[1]) as u128;
            let v0_mod = v0 % q[1] as u128;
            let diff1 = (r1_mod + q[1] as u128 - v0_mod) % q[1] as u128;
            let v1 = ((diff1 * inv01 as u128) % q[1] as u128) as u64;

            // partial = v0 + v1*q0 (NOT reduced mod q2 yet for final sum)
            let partial_full = v0 + v1 as u128 * q[0] as u128;
            let partial_mod_q2 = partial_full % q[2] as u128;
            let r2_mod = (r[2] % q[2]) as u128;
            let diff2 = (r2_mod + q[2] as u128 - partial_mod_q2) % q[2] as u128;
            let v2 = ((diff2 * inv012 as u128) % q[2] as u128) as u64;

            let mut result = partial_full + v2 as u128 * q01;
            if result > big_q / 2 {
                result = result.wrapping_sub(big_q);
            }
            let result = result as i128;

            let expected = x;
            assert_eq!(result, expected, "CRT roundtrip failed for x={x}");
        }
    }

    /// Simple trial-division primality test (for tests only).
    fn is_prime(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n < 4 {
            return true;
        }
        if n.is_multiple_of(2) || n.is_multiple_of(3) {
            return false;
        }
        let mut i = 5u64;
        while i * i <= n {
            if n.is_multiple_of(i) || n.is_multiple_of(i + 2) {
                return false;
            }
            i += 6;
        }
        true
    }
}
