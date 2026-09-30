//! What distinguishes the two oracles: the transform, its word types and the
//! arithmetic on transformed words. Everything else is shared.

use std::{fmt::Debug, hash::Hash};

use poulpy_hal::layouts::{BigWord, DftWord};

/// Integer word of the coefficient and big layouts, with wrapping arithmetic.
pub trait Int: Copy + Default + Eq + Debug + Send + Sync + 'static + From<i64> + Into<i128> {
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn neg(self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn shl(self, shift: u32) -> Self;
}

macro_rules! impl_int {
    ($t:ty) => {
        impl Int for $t {
            fn add(self, other: Self) -> Self {
                self.wrapping_add(other)
            }
            fn sub(self, other: Self) -> Self {
                self.wrapping_sub(other)
            }
            fn neg(self) -> Self {
                self.wrapping_neg()
            }
            fn mul(self, other: Self) -> Self {
                self.wrapping_mul(other)
            }
            fn shl(self, shift: u32) -> Self {
                self.wrapping_shl(shift)
            }
        }
    };
}

impl_int!(i64);
impl_int!(i128);

/// A transform family: FFT over `f64` or NTT over four primes.
pub trait Family: Copy + Eq + Hash + Debug + Send + Sync + 'static {
    /// Word of the transformed layouts, one per coefficient.
    type Dft: DftWord + Copy + Default + Debug + Send + Sync;
    /// Word of the big coefficient layout.
    type Big: BigWord + Int;
    /// Transform tables for one degree.
    type Table: Send + Sync;

    fn table(n: usize) -> Self::Table;

    /// `res = DFT(a)`, `n = a.len()`.
    fn forward(table: &Self::Table, res: &mut [Self::Dft], a: &[i64]);

    /// `res = IDFT(a)`, the exact integer coefficients.
    fn inverse(table: &Self::Table, res: &mut [Self::Big], a: &[Self::Dft]);

    fn dft_add(a: Self::Dft, b: Self::Dft) -> Self::Dft;

    fn dft_neg(a: Self::Dft) -> Self::Dft;

    /// `res += a * b` on transformed polynomials.
    fn mul_acc(res: &mut [Self::Dft], a: &[Self::Dft], b: &[Self::Dft]);

    /// `res *= a` on transformed polynomials.
    fn mul_assign(res: &mut [Self::Dft], a: &[Self::Dft]);

    /// `res = DFT(sigma_p(IDFT(a)))`, a permutation of the evaluations.
    fn dft_automorphism(p: i64, res: &mut [Self::Dft], a: &[Self::Dft]);

    /// The degree-`2n` spectrum, `n = a.len()`, of a conjugate-invariant
    /// element whose first `n` slots are `a`: its values on the other root of
    /// each conjugate pair.
    fn ci_expand(res: &mut [Self::Dft], a: &[Self::Dft]);

    /// `res += a * b` on conjugate-invariant spectra, slot by slot.
    fn ci_mul_acc(res: &mut [Self::Dft], a: &[Self::Dft], b: &[Self::Dft]);

    /// `res *= a` on conjugate-invariant spectra, slot by slot.
    fn ci_mul_assign(res: &mut [Self::Dft], a: &[Self::Dft]);
}

/// `bitrev(i)` over `bits` bits.
pub(crate) fn bitrev(i: usize, bits: u32) -> usize {
    if bits == 0 {
        0
    } else {
        i.reverse_bits() >> (usize::BITS - bits)
    }
}
