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

/// Selects a set of four NTT-friendly primes and their associated
/// constants for the Q120 CRT representation.
///
/// Q120 represents integers modulo `Q = Q[0]·Q[1]·Q[2]·Q[3]`, a
/// product of four primes each of approximately the same bit-size.
/// All four primes support a primitive `2^17`-th root of unity, so
/// NTT sizes up to `2^16` are supported.
///
/// Three concrete implementations are provided:
/// [`Primes29`], [`Primes30`] (the default, matching the spqlios
/// library), and [`Primes31`].
pub trait PrimeSet: Sized + Sync + Send + 'static {
    /// The four NTT-friendly primes `[Q0, Q1, Q2, Q3]`.
    const Q: [u32; 4];

    /// `OMEGA[k]` is a primitive `2^17`-th root of unity modulo `Q[k]`.
    ///
    /// For an NTT of size `n ≤ 2^16`, the actual primitive `2n`-th root
    /// used is `modq_pow(OMEGA[k], 2^16 / n, Q[k])`.
    const OMEGA: [u32; 4];

    /// CRT reconstruction constants.
    ///
    /// `CRT_CST[k] = (Q / Q[k])^{-1} mod Q[k]`, where `Q = Q[0]·Q[1]·Q[2]·Q[3]`.
    /// Used by `b_to_znx128` to recover an integer from its four CRT residues.
    const CRT_CST: [u32; 4];

    /// `ceil(log2(Q[0]))`.
    ///
    /// All four primes have the same bit-size, so this constant applies
    /// to all of them.  Used during NTT precomputation to track the
    /// growth of intermediate bit-widths through the butterfly levels.
    const LOG_Q: u64;
}

/// 29-bit NTT-friendly primes with `2·2^16`-th roots of unity.
///
/// - `Q ≈ 2^116`
/// - Each prime is of the form `(1 << 29) - c·(1 << 17) + 1`.
pub struct Primes29;

impl PrimeSet for Primes29 {
    const Q: [u32; 4] = [
        (1u32 << 29) - 2 * (1u32 << 17) + 1,  // 536_608_769
        (1u32 << 29) - 5 * (1u32 << 17) + 1,  // 536_215_553
        (1u32 << 29) - 26 * (1u32 << 17) + 1, // 533_463_041
        (1u32 << 29) - 35 * (1u32 << 17) + 1, // 532_283_393
    ];
    const OMEGA: [u32; 4] = [78_289_835, 178_519_192, 483_889_678, 239_808_033];
    const CRT_CST: [u32; 4] = [301_701_286, 536_020_447, 86_367_873, 147_030_781];
    const LOG_Q: u64 = 29;
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
    const Q: [u32; 4] = [
        (1u32 << 30) - 2 * (1u32 << 17) + 1,  // 1_073_479_681
        (1u32 << 30) - 17 * (1u32 << 17) + 1, // 1_071_513_601
        (1u32 << 30) - 23 * (1u32 << 17) + 1, // 1_070_727_169
        (1u32 << 30) - 42 * (1u32 << 17) + 1, // 1_068_236_801
    ];
    const OMEGA: [u32; 4] = [1_070_907_127, 315_046_632, 309_185_662, 846_468_380];
    const CRT_CST: [u32; 4] = [43_599_465, 292_938_863, 594_011_630, 140_177_212];
    const LOG_Q: u64 = 30;
}

/// 31-bit NTT-friendly primes with `2·2^16`-th roots of unity.
///
/// - `Q ≈ 2^124`
/// - Each prime is of the form `(1 << 31) - c·(1 << 17) + 1`.
pub struct Primes31;

impl PrimeSet for Primes31 {
    const Q: [u32; 4] = [
        (1u32 << 31) - (1u32 << 17) + 1,      // 2_147_352_577
        (1u32 << 31) - 4 * (1u32 << 17) + 1,  // 2_146_959_361
        (1u32 << 31) - 11 * (1u32 << 17) + 1, // 2_146_041_857
        (1u32 << 31) - 23 * (1u32 << 17) + 1, // 2_144_468_993
    ];
    const OMEGA: [u32; 4] = [1_615_402_923, 1_137_738_560, 154_880_552, 558_784_885];
    const CRT_CST: [u32; 4] = [1_811_422_063, 2_093_150_204, 164_149_010, 225_197_446];
    const LOG_Q: u64 = 31;
}
