use std::io::{Error, ErrorKind, Read, Result, Write};

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
/// probability. The wire format is a tag byte followed by a lossless
/// little-endian `u64` payload, shared with component-noise provenance.
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
/// The tensor statistics are therefore a closed-form function of the base
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
    /// Binary blocks of size `b`, with an all-zero block of probability `1/(b+1)`.
    /// Otherwise one uniformly selected position is 1. The degree is a multiple of `b`.
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
    pub(crate) fn validate_wire(&self) -> Result<()> {
        match self {
            Self::ENCAPSULATED(_) => Err(Error::new(
                ErrorKind::InvalidData,
                "secret distribution has no wire representation",
            )),
            Self::TernaryProb(p) | Self::BinaryProb(p) if !p.is_finite() || !(0.0..=1.0).contains(p) => {
                Err(Error::new(ErrorKind::InvalidData, "invalid secret distribution"))
            }
            _ => Ok(()),
        }
    }

    /// Writes a tag byte and a full little-endian `u64` payload.
    /// Probabilities preserve all binary64 bits and must be finite and in `[0, 1]`.
    /// `ENCAPSULATED` has no wire form. Invalid values fail before writing.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        self.validate_wire()?;
        let (tag, payload) = match self {
            Self::TernaryFixed(v) => (TAG_TERNARY_FIXED, *v as u64),
            Self::TernaryProb(p) => (TAG_TERNARY_PROB, p.to_bits()),
            Self::BinaryFixed(v) => (TAG_BINARY_FIXED, *v as u64),
            Self::BinaryProb(p) => (TAG_BINARY_PROB, p.to_bits()),
            Self::BinaryBlock(v) => (TAG_BINARY_BLOCK, *v as u64),
            Self::ZERO => (TAG_ZERO, 0),
            Self::NONE => (TAG_NONE, 0),
            Self::ENCAPSULATED(_) => unreachable!(),
        };
        writer.write_u8(tag)?;
        writer.write_u64::<LittleEndian>(payload)
    }

    /// Reads the lossless nine-byte format, rejecting invalid tags or payloads.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        let tag = reader.read_u8()?;
        let payload = reader.read_u64::<LittleEndian>()?;
        let invalid = || Error::new(ErrorKind::InvalidData, "invalid secret distribution");
        let dist = match tag {
            TAG_TERNARY_FIXED => Self::TernaryFixed(usize::try_from(payload).map_err(|_| invalid())?),
            TAG_TERNARY_PROB => Self::TernaryProb(f64::from_bits(payload)),
            TAG_BINARY_FIXED => Self::BinaryFixed(usize::try_from(payload).map_err(|_| invalid())?),
            TAG_BINARY_PROB => Self::BinaryProb(f64::from_bits(payload)),
            TAG_BINARY_BLOCK => Self::BinaryBlock(usize::try_from(payload).map_err(|_| invalid())?),
            TAG_ZERO if payload == 0 => Self::ZERO,
            TAG_NONE if payload == 0 => Self::NONE,
            _ => return Err(invalid()),
        };
        dist.validate_wire()?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_probabilities_are_lossless_and_validated() {
        for p in [0.0, -0.0, 0.1, 0.3, 1.0 / 3.0, 2.0 / 3.0, 0.7, 0.9, 1.0] {
            for dist in [Distribution::TernaryProb(p), Distribution::BinaryProb(p)] {
                let mut bytes = Vec::new();
                dist.write_to(&mut bytes).unwrap();
                assert_eq!(bytes.len(), 9);
                assert_eq!(Distribution::read_from(&mut bytes.as_slice()).unwrap(), dist);
            }
        }
        for p in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
            let mut bytes = vec![TAG_TERNARY_PROB];
            bytes.extend_from_slice(&p.to_bits().to_le_bytes());
            assert_eq!(
                Distribution::read_from(&mut bytes.as_slice()).unwrap_err().kind(),
                ErrorKind::InvalidData
            );
            let mut writer = Vec::new();
            assert!(Distribution::TernaryProb(p).write_to(&mut writer).is_err());
            assert!(writer.is_empty());
        }
    }
}
