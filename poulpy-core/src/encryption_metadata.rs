//! Provenance recorded by encryption, independent of an object's current noise.

use std::io::{self, Read, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::{Distribution, layouts::TorusPrecision};

/// Effective fresh phase-error variance at the precision where it was generated.
///
/// This summarizes the noise model, rather than asserting that a sum or product
/// of errors is itself Gaussian. It includes error inherited during encryption
/// and key generation. For an object with heterogeneous columns, it records the
/// largest column estimate. It does not track subsequent homomorphic operations.
/// Positive infinity represents an unbounded estimate (including overflow).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct FreshNoiseEstimate {
    variance_bits: u64,
    precision: TorusPrecision,
}

impl std::fmt::Debug for FreshNoiseEstimate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FreshNoiseEstimate")
            .field("variance", &self.variance())
            .field("precision", &self.precision)
            .finish()
    }
}

impl FreshNoiseEstimate {
    /// Records an effective variance in integer coefficient units at `precision`.
    /// Panics for negative values or NaN; positive infinity is permitted.
    pub fn new(variance: f64, precision: TorusPrecision) -> Self {
        assert!(variance >= 0.0, "fresh noise variance must be nonnegative and not NaN");
        Self {
            variance_bits: if variance == 0.0 { 0 } else { variance.to_bits() },
            precision,
        }
    }

    /// Effective variance in integer coefficient units at the creation precision.
    pub fn variance(&self) -> f64 {
        f64::from_bits(self.variance_bits)
    }

    /// Effective standard deviation at the creation precision.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Precision of the coefficient grid used by the stored estimate.
    pub fn precision(&self) -> TorusPrecision {
        self.precision
    }

    /// Expresses the same historical variance in coefficient units at `precision`.
    ///
    /// This only changes units; it does not model new rounding or evaluation error.
    /// A precision of zero gives torus variance. Overflow yields positive infinity.
    pub fn variance_at(&self, precision: TorusPrecision) -> f64 {
        scale_power_of_two(self.variance(), 2 * (i64::from(precision.0) - i64::from(self.precision.0)))
    }

    /// Expresses the same historical standard deviation at `precision`.
    pub fn std_dev_at(&self, precision: TorusPrecision) -> f64 {
        scale_power_of_two(self.std_dev(), i64::from(precision.0) - i64::from(self.precision.0))
    }
}

fn scale_power_of_two(mut value: f64, mut exponent: i64) -> f64 {
    if value == 0.0 || value.is_infinite() {
        return value;
    }
    // Cover the entire f64 range without overflowing an intermediate power of
    // two or iterating over an arbitrarily large precision difference.
    if exponent > 2098 {
        return f64::INFINITY;
    }
    if exponent < -2098 {
        return 0.0;
    }
    while exponent > 1023 {
        value *= 2.0_f64.powi(1023);
        exponent -= 1023;
    }
    while exponent < -1022 {
        value *= 2.0_f64.powi(-1022);
        exponent += 1022;
    }
    value * 2.0_f64.powi(exponent as i32)
}

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
/// operations. Encryption records the encrypting secret's distribution and an
/// effective fresh phase-error estimate, including amplification during key
/// generation. The fresh variance is independent of the secret's party count.
/// Uninitialized objects have no provenance.
/// Representation changes preserve it; equality and serialization include it.
/// Encapsulated secrets keep their existing ephemeral, nonserializable tag:
/// an object carrying that provenance cannot be serialized either.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncryptionMetadata {
    secret: SecretDistribution,
    fresh_noise: FreshNoiseEstimate,
}

impl EncryptionMetadata {
    /// Records a single base-noise contribution on the unit coefficient grid.
    /// Use [`Self::from_secret_at`] when the encryption precision is known.
    pub fn from_secret(base: Distribution) -> Self {
        Self::from_secret_at(base, TorusPrecision(0))
    }

    /// Records ordinary secret-key encryption with the default fresh Gaussian.
    /// Backend and multiparty protocol implementations use this to derive provenance.
    pub fn from_secret_at(base: Distribution, precision: TorusPrecision) -> Self {
        Self {
            secret: SecretDistribution { base, parties: 1 },
            fresh_noise: FreshNoiseEstimate::new(crate::DEFAULT_SIGMA_XE.powi(2), precision),
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
            fresh_noise: FreshNoiseEstimate::new(
                self.fresh_noise.variance() + other.fresh_noise.variance_at(self.fresh_noise.precision()),
                self.fresh_noise.precision(),
            ),
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

    /// Whether two encryptions use the same secret distribution and party count.
    /// This comparison deliberately allows different fresh-noise estimates.
    pub fn same_secret(&self, other: &Self) -> bool {
        self.secret == other.secret
    }

    /// Effective phase-error estimate recorded during encryption or key generation.
    pub fn fresh_noise(&self) -> FreshNoiseEstimate {
        self.fresh_noise
    }

    /// Records a derived fresh-noise estimate while preserving secret provenance.
    pub fn with_fresh_noise(self, fresh_noise: FreshNoiseEstimate) -> Self {
        Self { fresh_noise, ..self }
    }

    /// Initial effective phase-error variance in creation-grid coefficient units.
    ///
    /// Includes modeled error inherited from public keys and fresh-key generation.
    /// Use [`FreshNoiseEstimate::variance_at`] to express it at another precision.
    pub fn initial_noise_variance(&self) -> f64 {
        self.fresh_noise.variance()
    }

    /// Initial error standard deviation in integer coefficient units.
    pub fn initial_noise_std_dev(&self) -> f64 {
        self.fresh_noise.std_dev()
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
        writer.write_all(b"PNM2")?;
        match metadata {
            None => writer.write_u64::<LittleEndian>(0),
            Some(metadata) => {
                writer.write_u64::<LittleEndian>(metadata.parties())?;
                let (tag, payload) = distribution.unwrap();
                writer.write_u8(tag)?;
                writer.write_u64::<LittleEndian>(payload)?;
                writer.write_u32::<LittleEndian>(metadata.fresh_noise.precision().0)?;
                writer.write_u64::<LittleEndian>(metadata.fresh_noise.variance_bits)
            }
        }
    }

    pub(crate) fn read_optional<R: Read>(reader: &mut R) -> io::Result<Option<Self>> {
        let mut marker = [0; 4];
        reader.read_exact(&mut marker)?;
        if marker != *b"PNM2" {
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
        let precision = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        let variance = f64::from_bits(reader.read_u64::<LittleEndian>()?);
        if variance.is_nan() || variance < 0.0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid fresh noise variance"));
        }
        Ok(Some(Self {
            secret: SecretDistribution { base, parties },
            fresh_noise: FreshNoiseEstimate::new(variance, precision),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provenance_roundtrip_and_noise_model() {
        let metadata = EncryptionMetadata::from_secret_at(Distribution::TernaryProb(0.3), TorusPrecision(60));
        assert_eq!(metadata.parties(), 1);
        assert_eq!(metadata.secret_distribution().base(), Distribution::TernaryProb(0.3));
        assert_eq!(metadata.initial_noise_std_dev(), 3.2);
        let amplified = metadata.with_fresh_noise(FreshNoiseEstimate::new(123.5, TorusPrecision(120)));
        assert_ne!(metadata, amplified);
        assert!(metadata.same_secret(&amplified));
        let unbounded = metadata.with_fresh_noise(FreshNoiseEstimate::new(f64::INFINITY, TorusPrecision(60)));
        for value in [None, Some(metadata), Some(amplified), Some(unbounded)] {
            let mut bytes = Vec::new();
            EncryptionMetadata::write_optional(value, &mut bytes).unwrap();
            assert_eq!(EncryptionMetadata::read_optional(&mut bytes.as_slice()).unwrap(), value);
        }
        assert!(EncryptionMetadata::read_optional(&mut [0u8; 12].as_slice()).is_err());
    }

    #[test]
    fn independent_contributions_add_noise_separately_from_secret_parties() {
        let left = EncryptionMetadata::from_secret_at(Distribution::TernaryProb(0.5), TorusPrecision(60))
            .with_fresh_noise(FreshNoiseEstimate::new(16.0, TorusPrecision(60)));
        let right = EncryptionMetadata::from_secret_at(Distribution::TernaryProb(0.5), TorusPrecision(61))
            .with_fresh_noise(FreshNoiseEstimate::new(64.0, TorusPrecision(61)));
        let combined = left.aggregate(right);
        assert_eq!(combined.parties(), 2);
        assert_eq!(combined.fresh_noise(), FreshNoiseEstimate::new(32.0, TorusPrecision(60)));
        assert_eq!(combined.initial_noise_variance(), 32.0);
        assert_eq!(right.aggregate(left).fresh_noise().variance_at(TorusPrecision(60)), 32.0);
    }

    #[test]
    fn noise_estimate_preserves_creation_grid_and_handles_extreme_precisions() {
        let fresh = FreshNoiseEstimate::new(16.0, TorusPrecision(60));
        assert_eq!(fresh.precision(), TorusPrecision(60));
        assert_eq!(fresh.variance_at(TorusPrecision(62)), 256.0);
        assert_eq!(fresh.std_dev_at(TorusPrecision(58)), 1.0);
        assert_eq!(fresh.variance_at(TorusPrecision(0)), 2.0_f64.powi(-116));
        assert_eq!(fresh.variance_at(TorusPrecision(u32::MAX)), f64::INFINITY);
        assert_eq!(
            FreshNoiseEstimate::new(1.0, TorusPrecision(u32::MAX)).variance_at(TorusPrecision(0)),
            0.0
        );
        assert_eq!(
            FreshNoiseEstimate::new(0.0, TorusPrecision(0)).variance_at(TorusPrecision(u32::MAX)),
            0.0
        );
        assert_eq!(
            FreshNoiseEstimate::new(f64::INFINITY, TorusPrecision(u32::MAX)).variance_at(TorusPrecision(0)),
            f64::INFINITY
        );
        // The scale factor overflows f64, but the correctly scaled variance does not.
        assert_eq!(
            FreshNoiseEstimate::new(2.0_f64.powi(-1000), TorusPrecision(0)).variance_at(TorusPrecision(1000)),
            2.0_f64.powi(1000)
        );
        assert_eq!(
            FreshNoiseEstimate::new(-0.0, TorusPrecision(1)),
            FreshNoiseEstimate::new(0.0, TorusPrecision(1))
        );
    }

    #[test]
    fn invalid_variance_and_legacy_metadata_are_rejected() {
        let metadata = EncryptionMetadata::from_secret_at(Distribution::BinaryProb(0.5), TorusPrecision(60));
        let mut bytes = Vec::new();
        EncryptionMetadata::write_optional(Some(metadata), &mut bytes).unwrap();
        let noise_offset = bytes.len() - 8;
        for invalid in [f64::NAN, f64::NEG_INFINITY, -1.0] {
            assert!(std::panic::catch_unwind(|| FreshNoiseEstimate::new(invalid, TorusPrecision(60))).is_err());
            bytes[noise_offset..].copy_from_slice(&invalid.to_bits().to_le_bytes());
            assert_eq!(
                EncryptionMetadata::read_optional(&mut bytes.as_slice()).unwrap_err().kind(),
                io::ErrorKind::InvalidData
            );
        }
        bytes[..4].copy_from_slice(b"PNM1");
        assert_eq!(
            EncryptionMetadata::read_optional(&mut bytes.as_slice()).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
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
        ciphertext.metadata = Some(
            EncryptionMetadata::from_secret_at(Distribution::TernaryProb(0.3), TorusPrecision(35))
                .with_fresh_noise(FreshNoiseEstimate::new(71.0, TorusPrecision(35))),
        );
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
