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
//! `LaneArray`) lives in `poulpy_hal::layouts` and is re-exported here for
//! convenience; this module owns the 4-lane ~30-bit prime sets and their
//! full-CRT reconstruction extension.

pub use poulpy_hal::layouts::{LaneElem, PrimeSet};

pub trait PrimeSetCrt4: PrimeSet<PrimeElem = u32, Lanes<u32> = [u32; 4]> {}

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

impl PrimeSetCrt4 for Primes30 {}
