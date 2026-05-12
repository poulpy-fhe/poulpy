use std::io::{Read, Result, Write};

/// Read-only access to the [`Distribution`] associated with a secret key.
pub trait GetDistribution {
    /// Returns a reference to the distribution descriptor.
    fn dist(&self) -> &Distribution;
}

/// Mutable access to the [`Distribution`] associated with a secret key.
pub trait GetDistributionMut {
    /// Returns a mutable reference to the distribution descriptor.
    fn dist_mut(&mut self) -> &mut Distribution;
}

impl<T: GetDistribution + ?Sized> GetDistribution for &T {
    fn dist(&self) -> &Distribution {
        (*self).dist()
    }
}

impl<T: GetDistribution + ?Sized> GetDistribution for &mut T {
    fn dist(&self) -> &Distribution {
        (**self).dist()
    }
}

impl<T: GetDistributionMut + ?Sized> GetDistributionMut for &mut T {
    fn dist_mut(&mut self) -> &mut Distribution {
        (**self).dist_mut()
    }
}

/// Describes the probability distribution used to sample secret-key
/// coefficients.
///
/// Each variant encodes either a fixed Hamming weight or a per-coefficient
/// probability. The enum is serialised as a single little-endian `u64`
/// word via [`write_to`](Self::write_to) / [`read_from`](Self::read_from).
///
/// For probabilistic variants the `f64` payload is stored with a
/// precision loss below 2^-44 (8 least-significant mantissa bits
/// are discarded to fit the tag byte).
#[derive(Clone, Copy, Debug)]
pub enum Distribution {
    /// Ternary in {-1, 0, 1} with exactly `h` non-zero coefficients.
    TernaryFixed(usize),
    /// Ternary in {-1, 0, 1} where each coefficient is non-zero with probability `p`.
    TernaryProb(f64),
    /// Binary in {0, 1} with exactly `h` ones.
    BinaryFixed(usize),
    /// Binary in {0, 1} where each coefficient is 1 with probability `p`.
    BinaryProb(f64),
    /// Binary in {0, 1} split into blocks of size 2^k, with one 1 per block.
    BinaryBlock(usize),
    /// All-zero secret (debug / testing only).
    ZERO,
    /// Uninitialized — no distribution has been set yet.
    NONE,
}

const TAG_TERNARY_FIXED: u8 = 0;
const TAG_TERNARY_PROB: u8 = 1;
const TAG_BINARY_FIXED: u8 = 2;
const TAG_BINARY_PROB: u8 = 3;
const TAG_BINARY_BLOCK: u8 = 4;
const TAG_ZERO: u8 = 5;
const TAG_NONE: u8 = 6;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl Distribution {
    /// Packs a tag (u8) and an f64 into a single u64.
    /// The f64 is shifted right by 8, discarding the 8 least-significant
    /// mantissa bits (precision loss < 2^-44), and the tag is placed
    /// in the freed top byte.
    #[inline]
    fn pack_f64(tag: u8, p: f64) -> u64 {
        (tag as u64) << 56 | (p.to_bits() >> 8)
    }

    /// Unpacks a tag-stripped 56-bit payload back into an f64
    /// by shifting left by 8 (the 8 LSB mantissa bits become zero).
    #[inline]
    fn unpack_f64(payload: u64) -> f64 {
        f64::from_bits(payload << 8)
    }

    /// Serialises this distribution as a single little-endian `u64` word.
    ///
    /// The top byte carries a variant tag; the lower 56 bits carry either
    /// a `usize` payload (for fixed/block variants) or a truncated `f64`
    /// (for probabilistic variants).
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        let word: u64 = match self {
            Distribution::TernaryFixed(v) => (TAG_TERNARY_FIXED as u64) << 56 | (*v as u64),
            Distribution::TernaryProb(p) => Self::pack_f64(TAG_TERNARY_PROB, *p),
            Distribution::BinaryFixed(v) => (TAG_BINARY_FIXED as u64) << 56 | (*v as u64),
            Distribution::BinaryProb(p) => Self::pack_f64(TAG_BINARY_PROB, *p),
            Distribution::BinaryBlock(v) => (TAG_BINARY_BLOCK as u64) << 56 | (*v as u64),
            Distribution::ZERO => (TAG_ZERO as u64) << 56,
            Distribution::NONE => (TAG_NONE as u64) << 56,
        };
        writer.write_u64::<LittleEndian>(word)
    }

    /// Deserialises a [`Distribution`] from a single little-endian `u64` word.
    ///
    /// Returns [`std::io::ErrorKind::InvalidData`] if the tag byte is unrecognised.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        let word = reader.read_u64::<LittleEndian>()?;
        let tag = (word >> 56) as u8;
        let payload = word & 0x00FF_FFFF_FFFF_FFFF;

        let dist = match tag {
            TAG_TERNARY_FIXED => Distribution::TernaryFixed(payload as usize),
            TAG_TERNARY_PROB => Distribution::TernaryProb(Self::unpack_f64(payload)),
            TAG_BINARY_FIXED => Distribution::BinaryFixed(payload as usize),
            TAG_BINARY_PROB => Distribution::BinaryProb(Self::unpack_f64(payload)),
            TAG_BINARY_BLOCK => Distribution::BinaryBlock(payload as usize),
            TAG_ZERO => Distribution::ZERO,
            TAG_NONE => Distribution::NONE,
            _ => {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid tag"));
            }
        };
        Ok(dist)
    }
}

impl PartialEq for Distribution {
    fn eq(&self, other: &Self) -> bool {
        use Distribution::*;
        match (self, other) {
            (TernaryFixed(a), TernaryFixed(b)) => a == b,
            (TernaryProb(a), TernaryProb(b)) => a.to_bits() == b.to_bits(),
            (BinaryFixed(a), BinaryFixed(b)) => a == b,
            (BinaryProb(a), BinaryProb(b)) => a.to_bits() == b.to_bits(),
            (BinaryBlock(a), BinaryBlock(b)) => a == b,
            (ZERO, ZERO) => true,
            (NONE, NONE) => true,
            _ => false,
        }
    }
}

impl Eq for Distribution {}
