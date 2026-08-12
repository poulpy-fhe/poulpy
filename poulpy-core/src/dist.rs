use std::io::{Read, Result, Write};

/// Read-only access to the [`Distribution`] associated with a secret key.
pub trait GetDistribution {
    /// Returns the distribution the *base* secret was sampled from.
    ///
    /// See [`Distribution`] for what this tag does and does not describe;
    /// in particular it is not re-derived for secrets obtained as products
    /// of other secrets.
    fn dist(&self) -> &Distribution;
}

/// Mutable access to the [`Distribution`] associated with a secret key.
pub trait GetDistributionMut {
    /// Returns a mutable reference to the base-secret distribution tag.
    ///
    /// Only sampling routines and the transforms that propagate the tag
    /// should write through this; see [`Distribution`].
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

/// Describes the probability distribution the *base* secret was sampled
/// from.
///
/// Each variant encodes either a fixed Hamming weight or a per-coefficient
/// probability. The enum is serialised as a single little-endian `u64`
/// word via [`write_to`](Self::write_to) / [`read_from`](Self::read_from).
///
/// For probabilistic variants the `f64` payload is stored with a
/// precision loss below 2^-44 (8 least-significant mantissa bits
/// are discarded to fit the tag byte).
///
/// # What this tag means
///
/// It records how the key material was originally sampled, which is what
/// the security estimate and the noise analysis are stated against. It is
/// *not* a claim that a given buffer's coefficients are, right now, an
/// i.i.d. sample from that distribution.
///
/// The tag is set only by the `fill_*` samplers (and by
/// [`Distribution::ZERO`] for the debug all-zero secret). Every other
/// operation on a secret propagates it verbatim.
///
/// # Transforms that preserve it
///
/// A secret keeps its tag under any transform that permutes and/or negates
/// coefficients, or that only changes the representation:
///
/// - the `X -> X^-1` automorphism used by
///   `glwe_secret_from_lwe_secret` / `lwe_secret_from_glwe_secret`, and
///   any other `X -> X^k` automorphism: the multiset of non-zero
///   coefficients, and hence the Hamming weight and the per-coefficient
///   marginals, are unchanged (up to sign, which the ternary and binary
///   families are analysed against anyway);
/// - flattening a rank-`r` GLWE secret into an LWE secret and back: the
///   tag describes each polynomial component of the source key and is not
///   rescaled by the rank;
/// - DFT preparation ([`GLWESecretPrepared`](crate::layouts::GLWESecretPrepared))
///   and transfers between backends: pure changes of representation.
///
/// # Where it deliberately does not describe the coefficients
///
/// [`GLWESecretTensor`](crate::layouts::GLWESecretTensor) holds the products
/// `s_i * s_j` of a base secret `(s_0, ..., s_{r-1})`, e.g.
/// `(1, s_0, s_1)^(x)2 = (s_0^2, s_0*s_1, s_1^2)`. Those coefficients are
/// *not* ternary or binary any more, and no variant of this enum describes
/// them. The tensor key still carries the base secret's tag, on purpose:
/// it is the handle on the underlying secret's parameters, from which the
/// product's own statistics follow.
///
/// Concretely, if the base secret has zero-mean coefficients of variance
/// `s^2` in ring degree `N` (for instance `s^2 = h/N` for
/// [`TernaryFixed(h)`](Self::TernaryFixed)), then for independent
/// components `i != j` each coefficient of `s_i * s_j` mod `X^N + 1` is a
/// sum of `N` independent products and has variance `N * s^4`. The diagonal
/// blocks `s_i^2` carry twice that, `2 * N * s^4`, because each unordered
/// pair `s_a * s_b` contributes to the same coefficient from both orders.
/// Both are measured to hold on the reference backend. The statistics of
/// the tensor therefore stay a closed-form function of the base
/// distribution recorded here; see `var_tensor_key` in the noise module.
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
    /// Encapsulated category, only valid within its ephemeral context: cannot
    /// back a public key and cannot be serialized.
    ENCAPSULATED(&'static str),
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
    ///
    /// [`ENCAPSULATED`](Self::ENCAPSULATED) has no wire form and returns
    /// [`std::io::ErrorKind::InvalidData`].
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        let word: u64 = match self {
            Distribution::TernaryFixed(v) => (TAG_TERNARY_FIXED as u64) << 56 | (*v as u64),
            Distribution::TernaryProb(p) => Self::pack_f64(TAG_TERNARY_PROB, *p),
            Distribution::BinaryFixed(v) => (TAG_BINARY_FIXED as u64) << 56 | (*v as u64),
            Distribution::BinaryProb(p) => Self::pack_f64(TAG_BINARY_PROB, *p),
            Distribution::BinaryBlock(v) => (TAG_BINARY_BLOCK as u64) << 56 | (*v as u64),
            Distribution::ZERO => (TAG_ZERO as u64) << 56,
            Distribution::NONE => (TAG_NONE as u64) << 56,
            Distribution::ENCAPSULATED(name) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Distribution::ENCAPSULATED({name}) is not serializable"),
                ));
            }
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
            (ENCAPSULATED(a), ENCAPSULATED(b)) => a == b,
            (ZERO, ZERO) => true,
            (NONE, NONE) => true,
            _ => false,
        }
    }
}

impl Eq for Distribution {}
