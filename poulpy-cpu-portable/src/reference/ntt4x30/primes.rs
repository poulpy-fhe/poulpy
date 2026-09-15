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

/// 29-bit NTT-friendly primes with `2·2^16`-th roots of unity.
///
/// - `Q ≈ 2^116`
/// - Each prime is of the form `(1 << 29) - c·(1 << 17) + 1`.
pub struct Primes29;

impl PrimeSet for Primes29 {
    type PrimeElem = u32;
    type Lanes<T: LaneElem> = [T; 4];
    const Q: [u32; 4] = [
        (1u32 << 29) - 2 * (1u32 << 17) + 1,  // 536_608_769
        (1u32 << 29) - 5 * (1u32 << 17) + 1,  // 536_215_553
        (1u32 << 29) - 26 * (1u32 << 17) + 1, // 533_463_041
        (1u32 << 29) - 35 * (1u32 << 17) + 1, // 532_283_393
    ];
    const OMEGA: [u32; 4] = [78_289_835, 178_519_192, 483_889_678, 239_808_033];
    const LOG_Q: u64 = 29;
}

impl PrimeSetCrt4 for Primes29 {
    const CRT_CST: [u32; 4] = [301_701_286, 536_020_447, 86_367_873, 147_030_781];
}

/// 30-bit NTT-friendly primes with `2·2^16`-th roots of unity.
///
/// This is the **default** prime set, matching the spqlios-arithmetic
/// library's default (`SPQLIOS_Q120_USE_30_BIT_PRIMES`).
///
/// - `Q ≈ 2^120`
/// - Each prime is of the form `(1 << 30) - c·(1 << 17) + 1`.
pub struct Primes30;

impl PrimeSet for Primes30 {
    type PrimeElem = u32;
    type Lanes<T: LaneElem> = [T; 4];
    const Q: [u32; 4] = [
        (1u32 << 30) - 2 * (1u32 << 17) + 1,  // 1_073_479_681
        (1u32 << 30) - 17 * (1u32 << 17) + 1, // 1_071_513_601
        (1u32 << 30) - 23 * (1u32 << 17) + 1, // 1_070_727_169
        (1u32 << 30) - 42 * (1u32 << 17) + 1, // 1_068_236_801
    ];
    const OMEGA: [u32; 4] = [1_070_907_127, 315_046_632, 309_185_662, 846_468_380];
    const LOG_Q: u64 = 30;
}

impl PrimeSetCrt4 for Primes30 {
    const CRT_CST: [u32; 4] = [43_599_465, 292_938_863, 594_011_630, 140_177_212];
}

/// 31-bit NTT-friendly primes with `2·2^16`-th roots of unity.
///
/// - `Q ≈ 2^124`
/// - Each prime is of the form `(1 << 31) - c·(1 << 17) + 1`.
pub struct Primes31;

impl PrimeSet for Primes31 {
    type PrimeElem = u32;
    type Lanes<T: LaneElem> = [T; 4];
    const Q: [u32; 4] = [
        (1u32 << 31) - (1u32 << 17) + 1,      // 2_147_352_577
        (1u32 << 31) - 4 * (1u32 << 17) + 1,  // 2_146_959_361
        (1u32 << 31) - 11 * (1u32 << 17) + 1, // 2_146_041_857
        (1u32 << 31) - 23 * (1u32 << 17) + 1, // 2_144_468_993
    ];
    const OMEGA: [u32; 4] = [1_615_402_923, 1_137_738_560, 154_880_552, 558_784_885];
    const LOG_Q: u64 = 31;
}

impl PrimeSetCrt4 for Primes31 {
    const CRT_CST: [u32; 4] = [1_811_422_063, 2_093_150_204, 164_149_010, 225_197_446];
}
