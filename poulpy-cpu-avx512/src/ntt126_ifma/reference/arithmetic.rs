//! Element-wise CRT conversions for the 3-prime IFMA representation.
//!
//! Provides:
//! - `b_ntt126_ifma_from_znx64_ref`: i64 → 3-prime CRT (4 × u64 per coefficient, lane 3 = 0)
//! - `b_ntt126_ifma_to_znx128_ref`: 3-prime CRT → i128 via Garner's algorithm
//! - `c_ntt126_ifma_from_b_ref`: b → prepared c format (Harvey quotient pairs)

use super::super::primes::{PrimeSetNtt126Ifma, Primes42};

// ─────────────────────────────────────────────────────────────────────────────
// Compile-time constants for Primes42
// ─────────────────────────────────────────────────────────────────────────────

const Q: [u64; 3] = <Primes42 as PrimeSetNtt126Ifma>::Q;

/// `oq[k] = Q[k] - (2^63 mod Q[k])`.
///
/// For a negative input `x`, each prime lane receives
/// `(x as u64 & i64::MAX) + oq[k]`, which equals `x mod Q[k]`.
const OQ: [u64; 3] = {
    let mut oq = [0u64; 3];
    let mut k = 0usize;
    while k < 3 {
        oq[k] = Q[k] - (i64::MIN as u64 % Q[k]);
        k += 1;
    }
    oq
};

/// Garner constants for CRT reconstruction.
/// INV01 = inv(Q[0], Q[1])
/// INV012 = inv(Q[0]*Q[1], Q[2])
const INV01: u64 = Primes42::CRT_CST[0];
const INV012: u64 = Primes42::CRT_CST[1];
const Q0: u64 = Q[0];
const Q1: u64 = Q[1];
const Q2: u64 = Q[2];
const Q01: u128 = Q0 as u128 * Q1 as u128;

/// Full modulus Q = Q0 * Q1 * Q2.
const BIG_Q: u128 = Q01 * Q2 as u128;

// ─────────────────────────────────────────────────────────────────────────────
// i64 → 3-prime CRT (b format)
// ─────────────────────────────────────────────────────────────────────────────

/// Encode `n` i64 coefficients into the 3-prime CRT b format.
///
/// Output layout: `res[4*i + k] = a[i] mod Q[k]` for k ∈ {0,1,2},
/// and `res[4*i + 3] = 0` (padding).
pub fn b_ntt126_ifma_from_znx64_ref(n: usize, res: &mut [u64], a: &[i64]) {
    debug_assert!(res.len() >= 4 * n);
    debug_assert!(a.len() >= n);
    for i in 0..n {
        let x = a[i];
        for k in 0..3 {
            res[4 * i + k] = if x >= 0 {
                (x as u64) % Q[k]
            } else {
                // x as u64 = x + 2^64. We want x mod Q[k].
                // x mod Q[k] = (x as u64 & i64::MAX as u64) + OQ[k], reduced mod Q[k]
                let pos = (x as u64) & (i64::MAX as u64);
                (pos + OQ[k]) % Q[k]
            };
        }
        res[4 * i + 3] = 0; // padding
    }
}

/// Equivalent to [`b_ntt126_ifma_from_znx64_ref`] on `a[j] & mask`.
pub fn b_ntt126_ifma_from_znx64_masked_ref(n: usize, res: &mut [u64], a: &[i64], mask: i64) {
    debug_assert!(res.len() >= 4 * n);
    debug_assert!(a.len() >= n);
    for i in 0..n {
        let x = a[i] & mask;
        for k in 0..3 {
            res[4 * i + k] = if x >= 0 {
                (x as u64) % Q[k]
            } else {
                let pos = (x as u64) & (i64::MAX as u64);
                (pos + OQ[k]) % Q[k]
            };
        }
        res[4 * i + 3] = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3-prime CRT → i128 (Garner's algorithm)
// ─────────────────────────────────────────────────────────────────────────────

/// Decode `nn` coefficients from 3-prime CRT b format to i128.
///
/// Uses Garner's algorithm for 3 primes:
/// ```text
/// v0 = r0
/// v1 = ((r1 - v0) * inv(Q0, Q1)) mod Q1
/// v2 = ((r2 - v0 - v1*Q0) * inv(Q0*Q1, Q2)) mod Q2
/// result = v0 + v1*Q0 + v2*Q0*Q1
/// if result > Q/2: result -= Q
/// ```
///
/// `nn` is the number of coefficients (same as `res.len()`).
/// The `n^{-1}` normalization is already included in the inverse NTT
/// twiddle factors, so no extra division is needed.
pub fn b_ntt126_ifma_to_znx128_ref(nn: usize, res: &mut [i128], a: &[u64]) {
    debug_assert!(res.len() >= nn);
    debug_assert!(a.len() >= 4 * nn);

    for i in 0..nn {
        // Read and reduce residues
        let mut r = [0u64; 3];
        for k in 0..3 {
            r[k] = a[4 * i + k] % Q[k];
        }

        // Garner step 1: v0 = r[0]
        let v0 = r[0] as u128;

        // Garner step 2: v1 = ((r[1] - v0) * INV01) mod Q1
        // Use modular subtraction to avoid underflow
        let r1_mod_q1 = (r[1] % Q1) as u128;
        let v0_mod_q1 = v0 % Q1 as u128;
        let diff1 = (r1_mod_q1 + Q1 as u128 - v0_mod_q1) % Q1 as u128;
        let v1 = ((diff1 * INV01 as u128) % Q1 as u128) as u64;

        // Garner step 3: v2 = ((r[2] - v0 - v1*Q0) * INV012) mod Q2
        let partial = (v0 + v1 as u128 * Q0 as u128) % Q2 as u128;
        let r2_mod_q2 = (r[2] % Q2) as u128;
        let diff2 = (r2_mod_q2 + Q2 as u128 - partial) % Q2 as u128;
        let v2 = ((diff2 * INV012 as u128) % Q2 as u128) as u64;

        // Reconstruct: result = v0 + v1*Q0 + v2*Q0*Q1
        let result_u128 = v0 + v1 as u128 * Q0 as u128 + v2 as u128 * Q01;

        // Convert to signed symmetric representation
        res[i] = if result_u128 > BIG_Q / 2 {
            result_u128 as i128 - BIG_Q as i128
        } else {
            result_u128 as i128
        };
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// b → c (Harvey-prepared format)
// ─────────────────────────────────────────────────────────────────────────────

/// Convert `n` coefficients from b format to Harvey-prepared c format.
///
/// The prepared format uses the same `[u64; 4]` layout as the shared prep scalar,
/// but stores **reduced** residues (mod `Q\[k\]`) rather than lazy values.
/// The Harvey quotients are computed on-the-fly during multiplication
/// since they require 52-bit precision that doesn't fit in u32.
///
/// Output: `res[4*j+k] = a[4*j+k] mod Q[k]` for k ∈ {0,1,2}, res[4*j+3] = 0.
///
/// The `res` parameter is typed as `&mut [u32]` for trait compatibility with
/// the HAL's `cast_slice_mut` from `Q120bScalar`, but the data is actually
/// written as u64 values (each u32 pair forms one u64).
pub fn c_ntt126_ifma_from_b_ref(n: usize, res: &mut [u32], a: &[u64]) {
    debug_assert!(res.len() >= 8 * n);
    debug_assert!(a.len() >= 4 * n);

    // Reinterpret as u64 slice
    let res_u64: &mut [u64] = unsafe { std::slice::from_raw_parts_mut(res.as_mut_ptr() as *mut u64, res.len() / 2) };

    for j in 0..n {
        for k in 0..3 {
            // Store reduced residue as u64
            res_u64[4 * j + k] = a[4 * j + k] % Q[k];
        }
        res_u64[4 * j + 3] = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn b_from_znx64_roundtrip() {
        let n = 8;
        let coeffs: Vec<i64> = vec![0, 1, -1, 42, -42, i64::MAX / 2, i64::MIN / 2 + 1, 12345];

        let mut b = vec![0u64; 4 * n];
        b_ntt126_ifma_from_znx64_ref(n, &mut b, &coeffs);

        let mut result = vec![0i128; n];
        b_ntt126_ifma_to_znx128_ref(n, &mut result, &b);

        for i in 0..n {
            assert_eq!(
                result[i], coeffs[i] as i128,
                "roundtrip mismatch at i={i}: got {}, expected {}",
                result[i], coeffs[i]
            );
        }
    }
}
