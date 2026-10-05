//! Provenance recorded by encryption, independent of an object's current noise.

use std::io::{self, Read, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::Distribution;

/// A sum of independent secrets sampled from the same base distribution.
///
/// This describes the secret used for encryption. It does not describe an
/// ephemeral secret or the coefficients of a tensor product of secrets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SecretDistribution {
    base: Distribution,
    parties: u64,
}

impl SecretDistribution {
    /// Distribution sampled independently by each contributing party.
    pub fn base(&self) -> Distribution {
        self.base
    }

    /// Number of independent summands in the secret.
    pub fn parties(&self) -> u64 {
        self.parties
    }
}

/// Facts derived when a ciphertext or evaluation key is encrypted.
///
/// This is provenance, not an estimate of the noise after homomorphic
/// operations. Fresh single-party encryption records one party and the
/// encrypting secret's distribution. Uninitialized objects have no provenance.
/// Representation changes preserve it; equality and serialization include it.
/// Encapsulated secrets keep their existing ephemeral, nonserializable tag:
/// an object carrying that provenance cannot be serialized either.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncryptionMetadata {
    secret: SecretDistribution,
}

impl EncryptionMetadata {
    /// Records a single contribution encrypted under a secret sampled from `base`.
    /// Backend and multiparty protocol implementations use this to derive provenance.
    pub fn from_secret(base: Distribution) -> Self {
        Self {
            secret: SecretDistribution { base, parties: 1 },
        }
    }

    /// Combines independent contributions using the same base secret distribution.
    /// Panics if the distributions differ or the party count overflows.
    pub fn aggregate(self, other: Self) -> Self {
        assert_eq!(self.secret.base, other.secret.base, "incompatible secret distributions");
        Self {
            secret: SecretDistribution {
                base: self.secret.base,
                parties: self
                    .secret
                    .parties
                    .checked_add(other.secret.parties)
                    .expect("party count overflow"),
            },
        }
    }

    /// Number of parties that contributed to the original encryption.
    pub fn parties(&self) -> u64 {
        self.secret.parties
    }

    /// Distribution of the encrypting secret, including its independent sums.
    pub fn secret_distribution(&self) -> SecretDistribution {
        self.secret
    }

    /// Initial fresh-error variance parameter in integer coefficient units.
    ///
    /// Uses the encryption noise parameter squared, as in the noise model.
    /// This bounds each conditioned discrete Gaussian draw's variance. It
    /// excludes any additional error inherited from a public key. Multiply by
    /// `2^(-2k)` for its original torus variance.
    pub fn initial_noise_variance(&self) -> f64 {
        self.parties() as f64 * crate::DEFAULT_SIGMA_XE.powi(2)
    }

    /// Initial error standard deviation in integer coefficient units.
    pub fn initial_noise_std_dev(&self) -> f64 {
        self.initial_noise_variance().sqrt()
    }

    pub(crate) fn write_optional<W: Write>(metadata: Option<Self>, writer: &mut W) -> io::Result<()> {
        // Unlike the legacy secret tag encoding, provenance retains every bit
        // of a probabilistic distribution, because equality includes it.
        let distribution = match metadata.map(|m| m.secret.base) {
            None => None,
            Some(Distribution::TernaryFixed(h)) => Some((0, h as u64)),
            Some(Distribution::TernaryProb(p)) => Some((1, p.to_bits())),
            Some(Distribution::BinaryFixed(h)) => Some((2, h as u64)),
            Some(Distribution::BinaryProb(p)) => Some((3, p.to_bits())),
            Some(Distribution::BinaryBlock(b)) => Some((4, b as u64)),
            Some(Distribution::ZERO) => Some((5, 0)),
            Some(Distribution::NONE) => Some((6, 0)),
            Some(Distribution::ENCAPSULATED(_)) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "secret distribution has no wire representation",
                ));
            }
        };
        // A versioned marker prevents legacy ciphertext bytes being silently
        // accepted as provenance by the new readers.
        writer.write_all(b"PNM1")?;
        match metadata {
            None => writer.write_u64::<LittleEndian>(0),
            Some(metadata) => {
                writer.write_u64::<LittleEndian>(metadata.parties())?;
                let (tag, payload) = distribution.unwrap();
                writer.write_u8(tag)?;
                writer.write_u64::<LittleEndian>(payload)
            }
        }
    }

    pub(crate) fn read_optional<R: Read>(reader: &mut R) -> io::Result<Option<Self>> {
        let mut marker = [0; 4];
        reader.read_exact(&mut marker)?;
        if marker != *b"PNM1" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid encryption metadata version",
            ));
        }
        let parties = reader.read_u64::<LittleEndian>()?;
        if parties == 0 {
            return Ok(None);
        }
        let tag = reader.read_u8()?;
        let payload = reader.read_u64::<LittleEndian>()?;
        let invalid = || io::Error::new(io::ErrorKind::InvalidData, "invalid secret distribution");
        let base = match tag {
            0 => Distribution::TernaryFixed(usize::try_from(payload).map_err(|_| invalid())?),
            1 | 3 => {
                let p = f64::from_bits(payload);
                if !p.is_finite() || !(0.0..=1.0).contains(&p) {
                    return Err(invalid());
                }
                if tag == 1 {
                    Distribution::TernaryProb(p)
                } else {
                    Distribution::BinaryProb(p)
                }
            }
            2 => Distribution::BinaryFixed(usize::try_from(payload).map_err(|_| invalid())?),
            4 => Distribution::BinaryBlock(usize::try_from(payload).map_err(|_| invalid())?),
            5 if payload == 0 => Distribution::ZERO,
            6 if payload == 0 => Distribution::NONE,
            _ => return Err(invalid()),
        };
        Ok(Some(Self {
            secret: SecretDistribution { base, parties },
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provenance_roundtrip_and_noise_model() {
        let metadata = EncryptionMetadata::from_secret(Distribution::TernaryProb(0.3));
        assert_eq!(metadata.parties(), 1);
        assert_eq!(metadata.secret_distribution().base(), Distribution::TernaryProb(0.3));
        assert_eq!(metadata.initial_noise_std_dev(), 3.2);
        for value in [None, Some(metadata)] {
            let mut bytes = Vec::new();
            EncryptionMetadata::write_optional(value, &mut bytes).unwrap();
            assert_eq!(EncryptionMetadata::read_optional(&mut bytes.as_slice()).unwrap(), value);
        }
        assert!(EncryptionMetadata::read_optional(&mut [0u8; 12].as_slice()).is_err());
    }

    #[test]
    fn unknown_secret_distribution_roundtrips() {
        let metadata = Some(EncryptionMetadata::from_secret(Distribution::NONE));
        let mut bytes = Vec::new();
        EncryptionMetadata::write_optional(metadata, &mut bytes).unwrap();
        assert_eq!(EncryptionMetadata::read_optional(&mut bytes.as_slice()).unwrap(), metadata);
    }

    #[test]
    fn encapsulated_distribution_rejection_does_not_write() {
        let metadata = Some(EncryptionMetadata::from_secret(Distribution::ENCAPSULATED("ephemeral")));
        let mut bytes = vec![0xaa, 0x55];
        let error = EncryptionMetadata::write_optional(metadata, &mut bytes).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(bytes, [0xaa, 0x55]);
    }

    #[test]
    fn ciphertext_wire_format_preserves_provenance_and_equality() {
        use crate::layouts::{Base2K, Degree, GLWE, LWEInfos, Rank, TorusPrecision};
        use poulpy_hal::layouts::{ReaderFrom, WriterTo};

        let mut ciphertext = GLWE::<poulpy_hal::AlignedBuf, i64>::alloc(Degree(64), Base2K(12), TorusPrecision(35), Rank(1));
        let mut restored = ciphertext.clone();
        ciphertext.metadata = Some(EncryptionMetadata::from_secret(Distribution::TernaryProb(0.3)));
        assert_ne!(ciphertext, restored);

        let mut bytes = Vec::new();
        ciphertext.write_to(&mut bytes).unwrap();
        restored.read_from(&mut bytes.as_slice()).unwrap();
        assert_eq!(ciphertext, restored);
        assert_eq!(restored.encryption_metadata(), ciphertext.encryption_metadata());

        bytes[0] ^= 1;
        assert_eq!(
            restored.read_from(&mut bytes.as_slice()).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }
}
