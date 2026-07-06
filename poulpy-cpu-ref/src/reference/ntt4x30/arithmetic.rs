// ----------------------------------------------------------------------
// DISCLAIMER
//
// This module contains code that has been directly ported from the
// spqlios-arithmetic library
// (https://github.com/tfhe/spqlios-arithmetic), which is licensed
// under the Apache License, Version 2.0.
//
// The porting process from C to Rust was done with minimal changes
// in order to preserve the semantics and performance characteristics
// of the original implementation.
//
// Both Poulpy and spqlios-arithmetic are distributed under the terms
// of the Apache License, Version 2.0. See the LICENSE file for details.
//
// ----------------------------------------------------------------------

//! Simple (element-wise) Q120 arithmetic.
//!
//! These functions are direct Rust ports of `q120_arithmetic_simple.c`
//! in spqlios-arithmetic.  They operate on flat `u64` / `u32` slices
//! in the q120b / q120c layout (4 or 8 scalars per ring element, one
//! per prime).

use crate::reference::ntt4x30::primes::PrimeSet;

/// Converts a vector of `i64` ring-element coefficients into q120b format.
///
/// For each coefficient `x[j]`:
/// - If `x[j] >= 0`: `res[4*j + k] = x[j] as u64` (same bit pattern).
/// - If `x[j] <  0`: `res[4*j + k] = (x[j] as u64 & i64::MAX as u64) + OQ[k]`
///   where `OQ[k] = Q[k] - (2^63 % Q[k])`.
///
/// The result is congruent to `x[j]` modulo each `Q[k]` but is stored
/// as a non-negative `u64` (not necessarily in `[0, Q[k])`).
///
/// # Panics
/// Panics in debug mode if `res.len() < 4 * nn` or `x.len() < nn`.
pub fn b_from_znx64_ref<P: PrimeSet>(nn: usize, res: &mut [u64], x: &[i64]) {
    debug_assert!(res.len() >= 4 * nn);
    debug_assert!(x.len() >= nn);

    // OQ[k] = Q[k] - (2^63 mod Q[k])
    // Using i64::MIN as u64 = 2^63.
    let oq: [u64; 4] = std::array::from_fn(|k| {
        let q = P::Q[k] as u64;
        q - (i64::MIN as u64 % q)
    });

    let mask_lo: u64 = i64::MAX as u64; // 0x7FFF_FFFF_FFFF_FFFF

    for j in 0..nn {
        let xu = x[j] as u64;
        let is_neg = xu > mask_lo;
        let xl = xu & mask_lo;
        for k in 0..4 {
            res[4 * j + k] = xl + if is_neg { oq[k] } else { 0 };
        }
    }
}

/// Converts a vector of `i64` coefficients into q120b format, applying `mask` to each
/// coefficient before conversion. Equivalent to `b_from_znx64_ref` on `x[j] & mask`.
pub fn b_from_znx64_masked_ref<P: PrimeSet>(nn: usize, res: &mut [u64], x: &[i64], mask: i64) {
    debug_assert!(res.len() >= 4 * nn);
    debug_assert!(x.len() >= nn);

    // OQ[k] = Q[k] - (2^63 mod Q[k])
    // Using i64::MIN as u64 = 2^63.
    let oq: [u64; 4] = std::array::from_fn(|k| {
        let q = P::Q[k] as u64;
        q - (i64::MIN as u64 % q)
    });

    let mask_lo: u64 = i64::MAX as u64; // 0x7FFF_FFFF_FFFF_FFFF

    for j in 0..nn {
        let xu = (x[j] & mask) as u64;
        let is_neg = xu > mask_lo;
        let xl = xu & mask_lo;
        for k in 0..4 {
            res[4 * j + k] = xl + if is_neg { oq[k] } else { 0 };
        }
    }
}

/// Converts a vector of `i64` coefficients into q120c format.
///
/// For each coefficient `x[j]` and each prime index `k`:
/// - `r = x[j].rem_euclid(Q[k])` (canonical representative in `[0, Q[k])`)
/// - `res[8*j + 2*k]     = r`
/// - `res[8*j + 2*k + 1] = (r * 2^32) mod Q[k]`
///
/// # Panics
/// Panics in debug mode if `res.len() < 8 * nn` or `x.len() < nn`.
pub fn c_from_znx64_ref<P: PrimeSet>(nn: usize, res: &mut [u32], x: &[i64]) {
    debug_assert!(res.len() >= 8 * nn);
    debug_assert!(x.len() >= nn);

    for j in 0..nn {
        for k in 0..4 {
            let q = P::Q[k] as u64;
            let r = x[j].rem_euclid(P::Q[k] as i64) as u64;
            res[8 * j + 2 * k] = r as u32;
            res[8 * j + 2 * k + 1] = ((r << 32) % q) as u32;
        }
    }
}

/// Reconstructs `i128` ring-element coefficients from q120b format via CRT.
///
/// For each coefficient `j`:
/// 1. Compute `tmp = sum_k ((x[4*j+k] mod Q[k]) * CRT_CST[k] mod Q[k]) * (Q / Q[k])`
/// 2. Reduce `tmp mod Q` (where `Q = Q[0]*Q[1]*Q[2]*Q[3]`).
/// 3. Map to the symmetric representative in `(-Q/2, Q/2]`.
///
/// # Panics
/// Panics in debug mode if `res.len() < nn` or `x.len() < 4 * nn`.
pub fn b_to_znx128_ref<P: PrimeSet>(nn: usize, res: &mut [i128], x: &[u64]) {
    debug_assert!(res.len() >= nn);
    debug_assert!(x.len() >= 4 * nn);

    let q: [i128; 4] = P::Q.map(|qi| qi as i128);
    let total_q: i128 = q[0] * q[1] * q[2] * q[3];
    // qm[k] = Q / Q[k]
    let qm: [i128; 4] = [q[1] * q[2] * q[3], q[0] * q[2] * q[3], q[0] * q[1] * q[3], q[0] * q[1] * q[2]];
    let crt: [i128; 4] = P::CRT_CST.map(|c| c as i128);

    for j in 0..nn {
        let mut tmp: i128 = 0;
        for k in 0..4 {
            let xk = (x[4 * j + k] % P::Q[k] as u64) as i128;
            let t = (xk * crt[k]) % q[k];
            tmp += t * qm[k];
        }
        tmp %= total_q;
        let half = (total_q + 1) / 2;
        res[j] = if tmp >= half { tmp - total_q } else { tmp };
    }
}

/// Adds two q120b vectors element-wise, with a coarse lazy reduction.
///
/// For each limb `i` and prime `k`:
/// ```text
/// res[i] = (x[i] % (Q[k] << 33)) + (y[i] % (Q[k] << 33))
/// ```
/// The result is congruent to `x[i] + y[i]` modulo `Q[k]` and fits
/// in 64 bits provided the inputs satisfy `x[i], y[i] < Q[k] << 33`.
///
/// # Panics
/// Panics in debug mode if slices are shorter than `4 * nn`.
pub fn add_bbb_ref<P: PrimeSet>(nn: usize, res: &mut [u64], x: &[u64], y: &[u64]) {
    debug_assert!(res.len() >= 4 * nn);
    debug_assert!(x.len() >= 4 * nn);
    debug_assert!(y.len() >= 4 * nn);

    let q_shifted: [u64; 4] = P::Q.map(|qi| (qi as u64) << 33);

    for j in 0..nn {
        for (k, &q_s) in q_shifted.iter().enumerate() {
            let idx = 4 * j + k;
            res[idx] = x[idx] % q_s + y[idx] % q_s;
        }
    }
}

/// Adds two q120c vectors element-wise, reducing modulo each prime.
///
/// For each element `j`, prime `k`, and both sub-limbs `s ∈ {0,1}`:
/// ```text
/// res[8*j + 2*k + s] = (x[...] + y[...]) mod Q[k]
/// ```
///
/// # Panics
/// Panics in debug mode if slices are shorter than `8 * nn`.
pub fn add_ccc_ref<P: PrimeSet>(nn: usize, res: &mut [u32], x: &[u32], y: &[u32]) {
    debug_assert!(res.len() >= 8 * nn);
    debug_assert!(x.len() >= 8 * nn);
    debug_assert!(y.len() >= 8 * nn);

    for j in 0..nn {
        for k in 0..4 {
            let q = P::Q[k] as u64;
            for s in 0..2 {
                let idx = 8 * j + 2 * k + s;
                res[idx] = ((x[idx] as u64 + y[idx] as u64) % q) as u32;
            }
        }
    }
}

/// Converts a q120b vector to q120c format.
///
/// For each element `j` and prime `k`:
/// - `r = x[4*j+k] mod Q[k]`
/// - `res[8*j + 2*k]     = r`
/// - `res[8*j + 2*k + 1] = (r * 2^32) mod Q[k]`
///
/// # Panics
/// Panics in debug mode if slices are shorter than `4 * nn` / `8 * nn`.
pub fn c_from_b_ref<P: PrimeSet>(nn: usize, res: &mut [u32], x: &[u64]) {
    debug_assert!(res.len() >= 8 * nn);
    debug_assert!(x.len() >= 4 * nn);

    for j in 0..nn {
        for k in 0..4 {
            let q = P::Q[k] as u64;
            let r = x[4 * j + k] % q;
            res[8 * j + 2 * k] = r as u32;
            res[8 * j + 2 * k + 1] = ((r << 32) % q) as u32;
        }
    }
}
