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

use super::primes::{LaneArray, LaneElem, PrimeSet, Primes30};

/// One NTT-domain coefficient of a CRT (residue number system) backend:
/// `P::LANES` packed lanes of `T`, one residue per prime of `P`.
///
/// Byte-layout contract (see [`poulpy_hal::layouts::DftWord`]): a
/// `VecZnxDft` limb stores `n` consecutive `CrtWord<P, T>` blocks in the
/// spqlios NTT ordering for the prime set `P`. Two buffers are
/// interchangeable iff their word types are equal — the prime set, lane
/// element and lane count are all part of the type, so distinct
/// conventions cannot be mixed accidentally.
///
/// The lane count is exact (3, 4, 6, ... — whatever `P` declares), and the
/// lane element is explicit: `CrtWord<Primes30, u64>` is a 32-byte 4-lane
/// block of `u64` residues, `CrtWord<Primes30, u32>` would be its 16-byte
/// compact sibling.
#[repr(transparent)]
pub struct CrtWord<P: PrimeSet, T: LaneElem>(pub P::Lanes<T>);

impl<P: PrimeSet, T: LaneElem> Clone for CrtWord<P, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<P: PrimeSet, T: LaneElem> Copy for CrtWord<P, T> {}

impl<P: PrimeSet, T: LaneElem> Default for CrtWord<P, T> {
    fn default() -> Self {
        Self(P::Lanes::<T>::lanes_zeroed())
    }
}

impl<P: PrimeSet, T: LaneElem> PartialEq for CrtWord<P, T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<P: PrimeSet, T: LaneElem> Eq for CrtWord<P, T> {}

impl<P: PrimeSet, T: LaneElem> fmt::Debug for CrtWord<P, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("CrtWord").field(&self.0.as_slice()).finish()
    }
}

impl<P: PrimeSet, T: LaneElem> fmt::Display for CrtWord<P, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, lane) in self.0.as_slice().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{lane:#x}")?;
        }
        write!(f, "]")
    }
}

// SAFETY: CrtWord is #[repr(transparent)] over `P::Lanes<T>`, which is
// `[T; LANES]` with `T: Pod` (LaneArray requires Pod). All bit patterns
// are valid; no padding bytes, no uninit.
unsafe impl<P: PrimeSet, T: LaneElem> Zeroable for CrtWord<P, T> {}
unsafe impl<P: PrimeSet, T: LaneElem> Pod for CrtWord<P, T> {}

/// Byte-layout contract: `n` consecutive `P::LANES`-lane CRT blocks per
/// limb, in the spqlios NTT ordering of `P`. Two buffers are
/// interchangeable iff their word types are equal.
impl<P: PrimeSet, T: LaneElem> poulpy_hal::layouts::DftWord for CrtWord<P, T> {}

impl<P: PrimeSet, T: LaneElem> Add for CrtWord<P, T> {
    type Output = Self;
    /// Element-wise wrapping addition of the CRT residues.
    fn add(self, rhs: Self) -> Self {
        Self(P::Lanes::<T>::lanes_from_fn(|k| {
            self.0.as_slice()[k].wrapping_add(rhs.0.as_slice()[k])
        }))
    }
}

impl<P: PrimeSet, T: LaneElem> Zero for CrtWord<P, T> {
    fn zero() -> Self {
        Self(P::Lanes::<T>::lanes_zeroed())
    }

    fn is_zero(&self) -> bool {
        self.0.as_slice().iter().all(|x| *x == T::ZERO)
    }
}

/// Shared 32-byte NTT prep scalar for 4-lane CRT backends.
///
/// Stores four `u64` lanes in a packed block so that:
///
/// - A `VecZnxDft` limb stores `n` consecutive `Q120bScalar` values.
/// - The scalar bytes can be reinterpreted as `[u64; 4]` via
///   [`bytemuck::cast_slice`].
/// - The same 32-byte layout can be reinterpreted as `[u32; 8]` for
///   prepared-constant SVP/VMP multiply–accumulate operations.
///
/// The historical `Q120bScalar` name comes from the original 4-prime NTT4x30
/// backend; it is now an alias of the prime-set-parameterized [`CrtWord`].
pub type Q120bScalar = CrtWord<Primes30, u64>;

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
