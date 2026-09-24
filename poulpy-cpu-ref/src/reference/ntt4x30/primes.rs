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

//! The ntt4x30 family's prime sets.
//!
//! The family-neutral CRT vocabulary ([`PrimeSet`], [`LaneElem`],
//! [`LaneArray`]) lives in `poulpy_hal::layouts` and is re-exported here for
//! convenience; this module owns the 4-lane ~30-bit prime sets and their
//! full-CRT reconstruction extension.

pub use poulpy_hal::layouts::{LaneArray, LaneElem, PrimeSet};

/// 4-lane, `u32`-element prime sets with **full-CRT** reconstruction
/// constants, as consumed by the ntt4x30 kernel family.
///
/// `CRT_CST[k] = (Q / Q[k])^{-1} mod Q[k]`, where `Q = Q[0]·Q[1]·Q[2]·Q[3]`.
/// Used by `b_to_znx128` to recover an integer from its four CRT residues.
pub trait PrimeSetCrt4: PrimeSet<PrimeElem = u32, Lanes<u32> = [u32; 4]> {
    /// CRT reconstruction constants.
    const CRT_CST: [u32; 4];
}

/// 29-bit NTT-friendly primes with `2·2^18`-th roots of unity.
///
/// - `Q ≈ 2^116`
/// - Each prime is of the form `(1 << 29) - c·(1 << 19) + 1`.
pub struct Primes29;

impl PrimeSet for Primes29 {
    type PrimeElem = u32;
    type Lanes<T: LaneElem> = [T; 4];
    const Q: [u32; 4] = [
        531_628_033, // 2^29 - 10 * 2^19 + 1
        524_812_289, // 2^29 - 23 * 2^19 + 1
        518_520_833, // 2^29 - 35 * 2^19 + 1
        517_472_257, // 2^29 - 37 * 2^19 + 1
    ];
    const OMEGA: [u32; 4] = [119_931_893, 194_516_551, 403_971_879, 77_050_655];
    const LOG_Q: u64 = 29;
    // log2 of the exact prime product, precomputed at high precision.
    const LOG_Q_PRODUCT: f64 = 115.849_801_192_087_84;
    const MAX_LOG_N: u32 = 18;
}

impl PrimeSetCrt4 for Primes29 {
    const CRT_CST: [u32; 4] = [148_974_663, 415_017_145, 94_081_818, 386_832_361];
}

/// 30-bit NTT-friendly primes with `2·2^18`-th roots of unity.
///
/// This is the default prime set for the NTT4x30 backends.
///
/// - `Q ≈ 2^120`
/// - Each prime is of the form `(1 << 30) - c·(1 << 19) + 1`.
pub struct Primes30;

impl PrimeSet for Primes30 {
    type PrimeElem = u32;
    type Lanes<T: LaneElem> = [T; 4];
    const Q: [u32; 4] = [
        1_056_440_321, // 2^30 - 33 * 2^19 + 1
        1_053_818_881, // 2^30 - 38 * 2^19 + 1
        1_051_721_729, // 2^30 - 42 * 2^19 + 1
        1_049_100_289, // 2^30 - 47 * 2^19 + 1
    ];
    const OMEGA: [u32; 4] = [195_937_198, 50_863_243, 633_648_745, 87_406_124];
    const LOG_Q: u64 = 30;
    // log2 of the exact prime product, precomputed at high precision.
    const LOG_Q_PRODUCT: f64 = 119.886_155_257_481_1;
    const MAX_LOG_N: u32 = 18;
}

impl PrimeSetCrt4 for Primes30 {
    const CRT_CST: [u32; 4] = [222_597_450, 1_008_704_431, 722_621_871, 152_141_147];
}

/// 31-bit NTT-friendly primes with `2·2^18`-th roots of unity.
///
/// - `Q ≈ 2^124`
/// - Each prime is of the form `(1 << 31) - c·(1 << 19) + 1`.
pub struct Primes31;

impl PrimeSet for Primes31 {
    type PrimeElem = u32;
    type Lanes<T: LaneElem> = [T; 4];
    const Q: [u32; 4] = [
        2_146_959_361, // 2^31 - 1 * 2^19 + 1
        2_132_279_297, // 2^31 - 29 * 2^19 + 1
        2_130_706_433, // 2^31 - 32 * 2^19 + 1
        2_121_793_537, // 2^31 - 49 * 2^19 + 1
    ];
    const OMEGA: [u32; 4] = [1_961_488_829, 1_830_410_192, 339_671_193, 245_713_661];
    const LOG_Q: u64 = 31;
    // log2 of the exact prime product, precomputed at high precision.
    const LOG_Q_PRODUCT: f64 = 123.960_718_835_137_7;
    const MAX_LOG_N: u32 = 18;
}

impl PrimeSetCrt4 for Primes31 {
    const CRT_CST: [u32; 4] = [483_199_030, 1_354_824_273, 1_941_357_861, 484_653_143];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::ntt4x30::{arithmetic::b_to_znx128_ref, ntt::modq_pow};

    fn check<P: PrimeSetCrt4>() {
        P::validate();
        let total: i128 = P::Q.iter().map(|&q| q as i128).product();
        for (k, &q) in P::Q.iter().enumerate() {
            assert_eq!(u32::BITS - q.leading_zeros(), P::LOG_Q as u32);
            assert!((2..).take_while(|d| d * d <= q as u64).all(|d| !(q as u64).is_multiple_of(d)));
            assert_eq!(modq_pow(P::OMEGA[k], 1 << P::MAX_LOG_N, q), q - 1);
            assert_eq!(modq_pow(P::OMEGA[k], 1 << (P::MAX_LOG_N + 1), q), 1);
            assert_eq!((total / q as i128 % q as i128) * P::CRT_CST[k] as i128 % q as i128, 1);
        }
        for x in [
            0,
            1,
            -1,
            i64::MIN as i128,
            i64::MAX as i128,
            total / 2,
            -total / 2,
            total / 2 - 1,
            1 - total / 2,
        ] {
            let residues = P::Q.map(|q| x.rem_euclid(q as i128) as u64);
            let mut result = [0];
            b_to_znx128_ref::<P>(1, &mut result, &residues);
            assert_eq!(result[0], x);
        }
    }

    #[test]
    fn prime_sets_and_crt() {
        check::<Primes29>();
        check::<Primes30>();
        check::<Primes31>();
    }
}
