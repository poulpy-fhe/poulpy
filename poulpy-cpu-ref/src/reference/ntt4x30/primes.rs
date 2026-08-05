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

use std::fmt::{Debug, Display, LowerHex};

use bytemuck::Pod;

mod sealed {
    pub trait Sealed {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
}

/// Machine element of a CRT lane: the unsigned integer type holding one
/// residue (`u32` for ~30-bit primes, `u64` for wider primes).
pub trait LaneElem: sealed::Sealed + Copy + Debug + Display + LowerHex + PartialEq + Eq + Send + Sync + Pod + 'static {
    const ZERO: Self;
    fn wrapping_add(self, rhs: Self) -> Self;
}

impl LaneElem for u32 {
    const ZERO: Self = 0;
    fn wrapping_add(self, rhs: Self) -> Self {
        u32::wrapping_add(self, rhs)
    }
}

impl LaneElem for u64 {
    const ZERO: Self = 0;
    fn wrapping_add(self, rhs: Self) -> Self {
        u64::wrapping_add(self, rhs)
    }
}

/// Fixed-size lane container (`[T; LANES]`) abstracted so that a
/// [`PrimeSet`] can pick its exact lane count (3, 4, 6, ...).
pub trait LaneArray<T: LaneElem>: Copy + Debug + PartialEq + Eq + Send + Sync + Pod + 'static {
    const LEN: usize;
    fn as_slice(&self) -> &[T];
    fn as_mut_slice(&mut self) -> &mut [T];
    fn lanes_zeroed() -> Self;
    fn lanes_from_fn(f: impl FnMut(usize) -> T) -> Self;
}

// bytemuck has no blanket `Pod for [T; N]` over all `N`, so lane counts are
// enabled per size; extend the list when a new lane count appears.
macro_rules! impl_lane_array {
    ($($n:literal),* $(,)?) => {$(
        impl<T: LaneElem> LaneArray<T> for [T; $n] {
            const LEN: usize = $n;
            fn as_slice(&self) -> &[T] {
                self
            }
            fn as_mut_slice(&mut self) -> &mut [T] {
                self
            }
            fn lanes_zeroed() -> Self {
                [T::ZERO; $n]
            }
            fn lanes_from_fn(f: impl FnMut(usize) -> T) -> Self {
                std::array::from_fn(f)
            }
        }
    )*};
}

impl_lane_array!(2, 3, 4, 6, 8);

/// Selects a set of NTT-friendly primes and their associated constants
/// for a CRT (residue number system) representation.
///
/// A prime set represents integers modulo `Q = Q[0]·...·Q[LANES-1]`, a
/// product of `LANES` primes each of approximately the same bit-size.
/// All primes support a primitive `2^17`-th root of unity, so NTT sizes
/// up to `2^16` are supported.
///
/// The lane count and the storage element of the prime constants are part
/// of the prime set (`Lanes<T> = [T; LANES]`, `PrimeElem` = `u32` for the
/// ~30-bit family, `u64` for the ~42-bit family). CRT *reconstruction*
/// constants are intentionally not part of this trait — their semantics
/// differ per family (full-CRT vs Garner) and live on extension traits
/// such as [`PrimeSetCrt4`].
///
/// Concrete 4-lane implementations provided here: [`Primes29`],
/// [`Primes30`] (the default, matching the spqlios library), and
/// [`Primes31`].
pub trait PrimeSet: Sized + Sync + Send + 'static {
    /// Storage element of the prime constants.
    type PrimeElem: LaneElem;

    /// Number of CRT lanes (primes).
    const LANES: usize;

    /// Lane container shape: `[T; LANES]`.
    type Lanes<T: LaneElem>: LaneArray<T>;

    /// The NTT-friendly primes `[Q0, ..., Q_{LANES-1}]`.
    const Q: Self::Lanes<Self::PrimeElem>;

    /// `OMEGA[k]` is a primitive `2^17`-th root of unity modulo `Q[k]`.
    ///
    /// For an NTT of size `n ≤ 2^16`, the actual primitive `2n`-th root
    /// used is `modq_pow(OMEGA[k], 2^16 / n, Q[k])`.
    const OMEGA: Self::Lanes<Self::PrimeElem>;

    /// `ceil(log2(Q[0]))`.
    ///
    /// All primes have the same bit-size, so this constant applies
    /// to all of them.  Used during NTT precomputation to track the
    /// growth of intermediate bit-widths through the butterfly levels.
    const LOG_Q: u64;
}

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
    const LANES: usize = 4;
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
    const LANES: usize = 4;
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
    const LANES: usize = 4;
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
