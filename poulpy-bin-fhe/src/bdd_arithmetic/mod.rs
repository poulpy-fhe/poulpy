//! Word-level FHE arithmetic on encrypted unsigned integers.
//!
//! This module provides [`FheUint`] — a packed-GLWE ciphertext encoding all bits
//! of an unsigned integer — together with its DFT-prepared counterpart
//! [`FheUintPrepared`], the key material ([`BDDKey`] / [`BDDKeyPrepared`]) needed
//! to operate on it, and a library of arithmetic and logical operations driven by
//! pre-compiled Binary Decision Diagram (BDD) circuits.
//!
//! ## Computation Model
//!
//! Homomorphic arithmetic proceeds in three stages:
//!
//! 1. **Packing**: Each bit of a plaintext `u32` is independently encrypted as a
//!    GLWE ciphertext and then *packed* into a single GLWE polynomial via
//!    [`FheUint::pack`].  The resulting `FheUint` stores all bits interleaved in
//!    the coefficient slots of a single polynomial.
//!
//! 2. **Circuit bootstrapping** (`FheUint::prepare`): Every bit is extracted from
//!    the packed form, converted to LWE, bootstrapped through the circuit
//!    bootstrapping pipeline, and stored as a DFT-domain GGSW ciphertext.  The
//!    output is a [`FheUintPrepared`] value, ready to act as a CMux selector.
//!
//! 3. **BDD evaluation**: A compiled BDD circuit ([`BitCircuit`] / [`Circuit`])
//!    that encodes the desired operation is evaluated bit-by-bit using CMux gates
//!    ([`Cmux`]).  The result is a fresh packed-GLWE [`FheUint`].
//!
//! ## Key Structures
//!
//! | Type | Role |
//! |---|---|
//! | [`FheUint<D,T>`] | Packed ciphertext; one polynomial per integer |
//! | [`FheUintPrepared<D,T,BE>`] | Per-bit GGSW representation; selector for CMux |
//! | [`BDDKey<D,BRA>`] | Raw key bundle (circuit bootstrapping + switching keys) |
//! | [`BDDKeyPrepared<D,BRA,BE>`] | DFT-prepared key bundle |
//!
//! ## Supported Operations (u32)
//!
//! Two-word to one-word operations: [`Add`], [`Sub`], [`Sll`], [`Srl`], [`Sra`],
//! [`Slt`], [`Sltu`], [`Or`], [`And`], [`Xor`].
//!
//! One-word to one-word operations: [`Identity`].
//!
//! ## Threading
//!
//! Both `FheUintPrepared` construction and BDD circuit evaluation expose
//! `_multi_thread` variants that partition independent output bits across OS
//! threads using `std::thread::scope`.  Each thread receives its own scratch
//! slice; the key structures are `Sync`.
//!
//! ## Scratch-Space Allocation
//!
//! All evaluation and preparation routines accept a mutable `ScratchArena<'_, BE>`.
//! No heap allocation occurs on the hot path; callers must size the arena using
//! the corresponding `*_tmp_bytes` query method.
mod bdd_1w_to_1w;
mod bdd_2w_to_1w;
mod blind_retrieval;
mod blind_rotation;
mod blind_selection;
mod ciphertexts;
mod circuits;
mod eval;
mod key;

pub use bdd_1w_to_1w::*;
pub use bdd_2w_to_1w::*;
pub use blind_retrieval::*;
pub use blind_rotation::*;
pub use blind_selection::*;
pub use ciphertexts::*;
pub(crate) use circuits::*;
pub use eval::*;
pub use key::*;

#[cfg(all(test, feature = "enable-bin-fhe"))]
pub mod tests;

/// Marker trait for unsigned integer types whose bits can be encrypted by [`FheUint`].
///
/// Implemented for `u8`, `u16`, `u32`, `u64`, and `u128`.  The associated
/// constants encode the bit width and derived log values used to compute the
/// interleaved coefficient layout inside a packed GLWE polynomial.
pub trait UnsignedInteger: Copy + Sync + Send + 'static {
    /// Total number of bits in this integer type.
    const BITS: u32;
    /// `ceil(log2(BITS))`.
    const LOG_BITS: u32;
    /// `LOG_BITS - 3` (log₂ of the byte count).
    const LOG_BYTES: u32;
    /// `(1 << LOG_BYTES) - 1`; mask for the byte index within the interleaved layout.
    const LOG_BYTES_MASK: usize;

    /// Maps a logical bit index `i` to its coefficient index in the packed GLWE layout.
    ///
    /// The mapping interleaves bits from different bytes so that a byte can be
    /// isolated via a single GLWE trace operation.
    #[inline(always)]
    fn bit_index(i: usize) -> usize {
        ((i & 7) << Self::LOG_BYTES) | (i >> 3)
    }
}

impl UnsignedInteger for u8 {
    const BITS: u32 = u8::BITS;
    const LOG_BITS: u32 = (u32::BITS - (Self::BITS - 1).leading_zeros());
    const LOG_BYTES: u32 = Self::LOG_BITS - 3;
    const LOG_BYTES_MASK: usize = (1 << Self::LOG_BYTES) - 1;
}
impl UnsignedInteger for u16 {
    const BITS: u32 = u16::BITS;
    const LOG_BITS: u32 = (u32::BITS - (Self::BITS - 1).leading_zeros());
    const LOG_BYTES: u32 = Self::LOG_BITS - 3;
    const LOG_BYTES_MASK: usize = (1 << Self::LOG_BYTES) - 1;
}
impl UnsignedInteger for u32 {
    const BITS: u32 = u32::BITS;
    const LOG_BITS: u32 = (u32::BITS - (Self::BITS - 1).leading_zeros());
    const LOG_BYTES: u32 = Self::LOG_BITS - 3;
    const LOG_BYTES_MASK: usize = (1 << Self::LOG_BYTES) - 1;
}
impl UnsignedInteger for u64 {
    const BITS: u32 = u64::BITS;
    const LOG_BITS: u32 = (u32::BITS - (Self::BITS - 1).leading_zeros());
    const LOG_BYTES: u32 = Self::LOG_BITS - 3;
    const LOG_BYTES_MASK: usize = (1 << Self::LOG_BYTES) - 1;
}
impl UnsignedInteger for u128 {
    const BITS: u32 = u128::BITS;
    const LOG_BITS: u32 = (u32::BITS - (Self::BITS - 1).leading_zeros());
    const LOG_BYTES: u32 = Self::LOG_BITS - 3;
    const LOG_BYTES_MASK: usize = (1 << Self::LOG_BYTES) - 1;
}

/// Extracts individual bits from a plaintext unsigned integer.
///
/// Used during encryption ([`FheUint::encrypt_sk`],
/// [`FheUintPrepared::encrypt_sk`]) to encode each bit of a plaintext
/// value into its corresponding GLWE/GGSW ciphertext slot.
pub trait ToBits {
    /// Returns the `i`-th bit of `self` as `0` or `1`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= bit-width of the type`.
    fn bit(&self, i: usize) -> u8;
}

macro_rules! impl_tobits {
    ($($t:ty),*) => {
        $(
            impl ToBits for $t {
                fn bit(&self, i: usize) -> u8 {
                    if i >= (std::mem::size_of::<$t>() * 8) {
                        panic!("bit index {} out of range for {}", i, stringify!($t));
                    }
                    ((self >> i) & 1) as u8
                }
            }
        )*
    };
}

impl_tobits!(u8, u16, u32, u64, u128);

/// Reconstructs a plaintext unsigned integer from a slice of individual bits.
///
/// Used during decryption ([`FheUint::decrypt`], [`FheUintPrepared::decrypt`])
/// to reassemble the plaintext value from the per-bit decoded coefficients.
pub trait FromBits: Sized {
    /// Constructs `Self` from a slice of bit values (each `0` or `1`).
    ///
    /// Bits are consumed in LSB-first order.  If `bits.len()` is shorter than
    /// the type's bit-width the remaining high bits are treated as zero.
    fn from_bits(bits: &[u8]) -> Self;
}

macro_rules! impl_from_bits {
    ($($t:ty),*) => {
        $(
            impl FromBits for $t {
                fn from_bits(bits: &[u8]) -> Self {
                    let mut value: $t = 0;
                    let max_bits = std::mem::size_of::<$t>() * 8;
                    let n = bits.len().min(max_bits);

                    for (i, &bit) in bits.iter().take(n).enumerate() {
                        if bit != 0 {
                            value |= 1 << i;
                        }
                    }
                    value
                }
            }
        )*
    };
}

impl_from_bits!(u8, u16, u32, u64, u128);
