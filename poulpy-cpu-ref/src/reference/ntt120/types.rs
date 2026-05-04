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

use bytemuck::{Pod, Zeroable};
use rand_distr::num_traits::Zero;
use std::{fmt, ops::Add};

use super::primes::{PrimeSet, Primes30};

/// Shared 32-byte NTT prep scalar for 4-lane CRT backends.
///
/// Stores four `u64` lanes in a packed `#[repr(C)]` struct so that:
///
/// - A `VecZnxDft` limb stores `n` consecutive `Q120bScalar` values.
/// - The scalar bytes can be reinterpreted as `[u64; 4]` via
///   [`bytemuck::cast_slice`].
/// - The same 32-byte layout can be reinterpreted as `[u32; 8]` for
///   prepared-constant SVP/VMP multiply–accumulate operations.
///
/// The historical `Q120bScalar` name comes from the original 4-prime NTT120
/// backend. The layout itself is shared by every backend that uses a
/// 4-lane `ScalarPrep`, including 3-prime backends that leave lane 3 as
/// padding.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Q120bScalar(pub [u64; 4]);

// SAFETY: Q120bScalar is #[repr(C)] with a single [u64; 4] field.
// All bit patterns are valid; no padding bytes, no uninit.
unsafe impl Zeroable for Q120bScalar {}
unsafe impl Pod for Q120bScalar {}

impl fmt::Display for Q120bScalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:#x}, {:#x}, {:#x}, {:#x}]", self.0[0], self.0[1], self.0[2], self.0[3])
    }
}

impl Add for Q120bScalar {
    type Output = Self;
    /// Element-wise wrapping addition of the four CRT residues.
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0].wrapping_add(rhs.0[0]),
            self.0[1].wrapping_add(rhs.0[1]),
            self.0[2].wrapping_add(rhs.0[2]),
            self.0[3].wrapping_add(rhs.0[3]),
        ])
    }
}

impl Zero for Q120bScalar {
    fn zero() -> Self {
        Self([0u64; 4])
    }

    fn is_zero(&self) -> bool {
        self.0 == [0u64; 4]
    }
}

/// CRT representation of an integer modulo Q120.
///
/// `Q120a[k]` is the residue modulo `Q[k]`, stored as a `u32`
/// (values in `[0, 2^32)`; may be non-canonical, i.e. not fully reduced
/// to `[0, Q[k])`).
///
/// Memory layout: 4 consecutive `u32` values, matching spqlios `q120a`.
pub type Q120a = [u32; 4];

/// CRT representation of an integer modulo Q120.
///
/// `Q120b[k]` is the residue modulo `Q[k]`, stored as a `u64`
/// (values in `[0, 2^64)`; non-canonical).  This is the primary
/// representation used inside the NTT butterflies, where intermediate
/// values accumulate extra bits before an optional lazy reduction step.
///
/// Memory layout: 4 consecutive `u64` values, matching spqlios `q120b`.
/// An NTT vector of length `n` is stored as `n` consecutive `Q120b`
/// values, i.e. `4n` consecutive `u64` values.
pub type Q120b = [u64; 4];

/// Prepared CRT representation of an integer modulo Q120.
///
/// `Q120c[2k]` = residue modulo `Q[k]` and
/// `Q120c[2k+1]` = `(residue * 2^32) mod Q[k]`.
/// Both stored as `u32` in `[0, Q[k])`.
///
/// This layout pre-computes the high-half product needed by the lazy
/// accumulation algorithm in [`super::mat_vec::vec_mat1col_product_bbc_ref`], halving
/// the number of per-element multiplications at the cost of doubling
/// the storage.
///
/// Memory layout: 8 consecutive `u32` values, matching spqlios `q120c`.
pub type Q120c = [u32; 8];

/// Two `Q120b` elements packed contiguously.
///
/// Used in the `x2` variants of the matrix–vector product, which process
/// two output coefficients in a single accumulation loop for better
/// instruction-level parallelism.
///
/// Memory layout: 8 consecutive `u64` values, matching spqlios `q120x2b`.
pub type Q120x2b = [u64; 8];

/// Two `Q120c` elements packed contiguously.
///
/// Memory layout: 16 consecutive `u32` values, matching spqlios `q120x2c`.
pub type Q120x2c = [u32; 16];

/// Lazy-reduction bound used when adding two q120b values pointwise.
///
/// `Q_SHIFTED[k] = Q[k] << 33`.  Any q120b residue produced by
/// `accum_to_q120b` satisfies `x < 2·Q_SHIFTED[k]`, so reducing
/// modulo `Q_SHIFTED[k]` before adding two such values keeps the result
/// below `4·Q_SHIFTED[k]`, which is safe for a subsequent NTT.
///
/// Shared by [`super::vmp`] and [`super::convolution`].
pub const Q_SHIFTED: [u64; 4] = [
    (Primes30::Q[0] as u64) << 33,
    (Primes30::Q[1] as u64) << 33,
    (Primes30::Q[2] as u64) << 33,
    (Primes30::Q[3] as u64) << 33,
];
