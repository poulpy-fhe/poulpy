//! Family-neutral CRT (residue number system) word machinery.
//!
//! A CRT backend represents one NTT-domain coefficient as a packed block of
//! per-prime residues. This module provides the backend-agnostic vocabulary
//! for such representations: the lane element ([`LaneElem`]), the fixed-size
//! lane container ([`LaneArray`]), the prime-set descriptor ([`PrimeSet`]),
//! and the resulting word type [`CrtWord<P, T>`], a [`DftWord`] for any
//! `(P, T)`.
//!
//! Concrete prime sets live with their kernel families in the backend crates
//! (e.g. `Primes29/30/31` and the full-CRT `PrimeSetCrt4` extension in
//! `poulpy-cpu-ref`'s ntt4x30 module, `Primes42` and the Garner
//! `PrimeSetNtt3x42Ifma` extension in `poulpy-cpu-avx512`), as do the CRT
//! *reconstruction* constants, whose semantics differ per family.

use std::fmt::{self, Debug, Display, LowerHex};
use std::ops::Add;

use bytemuck::{Pod, Zeroable};
use rand_distr::num_traits::Zero;

use crate::layouts::DftWord;

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

/// Fixed-size lane container (`[T; N]`) abstracted so that a
/// [`PrimeSet`] can pick its exact lane count.
///
/// Enabled lane counts: 1-8, 12, 16 (see `impl_lane_array!` below). A new
/// [`PrimeSet`] with a different lane count needs its count added to that
/// invocation — bytemuck has no blanket `Pod for [T; N]`, so each size is a
/// separate impl.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not an enabled `LaneArray` size",
    note = "lane counts are enabled per size; add the new count to the `impl_lane_array!` invocation in poulpy-hal/src/layouts/crt.rs"
)]
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

impl_lane_array!(1, 2, 3, 4, 5, 6, 7, 8, 12, 16);

/// Selects a set of NTT-friendly primes and their associated constants
/// for a CRT (residue number system) representation.
///
/// A prime set represents integers modulo `Q = Q[0]·...·Q[N-1]`, a
/// product of `N` primes each of approximately the same bit-size.
/// All primes support a primitive `2^17`-th root of unity, so NTT sizes
/// up to `2^16` are supported.
///
/// The lane count and the storage element of the prime constants are part
/// of the prime set (`Lanes<T> = [T; N]`, `PrimeElem` = `u32` for the
/// ~30-bit family, `u64` for the ~42-bit family). CRT *reconstruction*
/// constants are intentionally not part of this trait — their semantics
/// differ per family (full-CRT vs Garner) and live on extension traits in
/// the backend crates (e.g. `PrimeSetCrt4` in `poulpy-cpu-ref`,
/// `PrimeSetNtt3x42Ifma` in `poulpy-cpu-avx512`), next to the concrete
/// prime-set implementations.
pub trait PrimeSet: Sized + Sync + Send + 'static {
    /// Storage element of the prime constants.
    type PrimeElem: LaneElem;

    /// Lane container shape: `[T; N]`.
    ///
    /// The count must be one enabled by `impl_lane_array!` (1-8, 12, 16);
    /// see [`LaneArray`] for extending the list.
    type Lanes<T: LaneElem>: LaneArray<T>;

    /// The NTT-friendly primes `[Q0, ..., Q_{N-1}]`.
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

/// One NTT-domain coefficient of a CRT (residue number system) backend:
/// `P::Lanes<T>::LEN` packed lanes of `T`, one residue per prime of `P`.
///
/// Byte-layout contract (see [`DftWord`]): a `VecZnxDft` limb stores `n`
/// consecutive `CrtWord<P, T>` blocks in the NTT ordering for the prime set
/// `P`. The prime set, lane element, and lane count are all part of the type,
/// so distinct CRT conventions cannot unify accidentally. Cross-backend
/// interchange still requires the relevant layout-compatibility marker for
/// the container family.
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
// `[T; N]` with `T: Pod` (LaneArray requires Pod). All bit patterns
// are valid; no padding bytes, no uninit.
unsafe impl<P: PrimeSet, T: LaneElem> Zeroable for CrtWord<P, T> {}
unsafe impl<P: PrimeSet, T: LaneElem> Pod for CrtWord<P, T> {}

/// Byte-layout contract: `n` consecutive `P::Lanes<T>::LEN`-lane CRT blocks per
/// limb, in the NTT ordering of `P`. Cross-backend interchange also
/// requires the relevant layout-compatibility marker.
impl<P: PrimeSet, T: LaneElem> DftWord for CrtWord<P, T> {}

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
