//! Fresh component-noise estimates, cleared from homomorphic evaluation outputs.

use std::{
    io::{self, Read, Write},
    sync::Arc,
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::{Distribution, layouts::TorusPrecision};

/// Effective fresh error variance at the precision where it was generated.
///
/// This summarizes the noise model, rather than asserting that a sum or product
/// of errors is itself Gaussian. It includes error inherited during encryption
/// and key generation. Component estimates are recorded before multiplication
/// by the secret. It does not track subsequent homomorphic operations.
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

/// Fresh noise in a ciphertext's body and individual mask components.
///
/// Entries are ordered `[body, mask_0, ...]`: a GLWE of rank `r` has `r + 1`
/// terms, and an LWE of dimension `n` has `n + 1` scalar terms. Each term is a
/// coefficient-noise variance before multiplication by the secret, expressed
/// on the same creation-precision grid. Secret-key encryption starts with a
/// body error and zero mask errors; public-key encryption can affect every term.
/// Heterogeneous evaluation-key rows retain a component-wise upper estimate.
///
/// Copies, preparation, compression, and backend transfers preserve these
/// estimates. Homomorphic operations clear them; noise composition after
/// evaluation is not tracked. The shared immutable terms make view creation
/// cheap without limiting ciphertext rank. Equality and serialization include
/// the component estimates and secret provenance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentNoise {
    secret: SecretDistribution,
    components: Arc<[FreshNoiseEstimate]>,
}

impl ComponentNoise {
    /// Records a body-only base Gaussian on the unit coefficient grid.
    pub fn from_secret(base: Distribution, rank: usize) -> Self {
        Self::from_secret_at(base, TorusPrecision(0), rank)
    }

    /// Records fresh secret-key noise: one body term and `rank` zero mask terms.
    /// For scalar LWE, pass the secret dimension as `rank`. Encapsulated laws
    /// are recorded as `NONE`, since their process-local labels have no wire form.
    pub fn from_secret_at(base: Distribution, precision: TorusPrecision, rank: usize) -> Self {
        let base = if matches!(base, Distribution::ENCAPSULATED(_)) {
            Distribution::NONE
        } else {
            base
        };
        let count = rank.checked_add(1).expect("noise component count overflow");
        let mut components = vec![FreshNoiseEstimate::new(0.0, precision); count];
        components[0] = FreshNoiseEstimate::new(crate::DEFAULT_SIGMA_XE.powi(2), precision);
        Self {
            secret: SecretDistribution { base, parties: 1 },
            components: components.into(),
        }
    }

    /// Body first, followed by one estimate per mask component.
    pub fn components(&self) -> &[FreshNoiseEstimate] {
        &self.components
    }

    /// Number of mask components.
    pub fn rank(&self) -> usize {
        self.components.len() - 1
    }

    /// Noise of the body before subtracting the secret-weighted mask.
    pub fn body(&self) -> FreshNoiseEstimate {
        self.components[0]
    }

    /// Noise of each mask component before multiplication by the secret.
    pub fn masks(&self) -> &[FreshNoiseEstimate] {
        &self.components[1..]
    }

    /// Common precision of the coefficient grid used by all terms.
    pub fn precision(&self) -> TorusPrecision {
        self.body().precision()
    }

    /// Modeled phase variance after weighting mask terms by the secret.
    ///
    /// For a negacyclic ring degree `n`, this is `body + n * E[s^2] * sum(masks)`.
    /// For scalar LWE, use [`Self::lwe_phase_noise`]. Terms may be conservative upper estimates;
    /// this coefficient model does not track arbitrary error covariance.
    pub fn phase_noise(&self, n: usize) -> FreshNoiseEstimate {
        assert!(n > 0, "noise model ring degree must be positive");
        self.weighted_phase_noise(n, n)
    }

    /// Modeled scalar LWE phase variance, using the mask count as secret dimension.
    /// Unlike GLWE, each mask term multiplies one secret coefficient without convolution.
    pub fn lwe_phase_noise(&self) -> FreshNoiseEstimate {
        self.lwe_phase_noise_with_block(self.rank())
    }

    /// Scalar LWE phase estimate using the original sampling block dimension.
    /// For a flattened GLWE with fixed-weight secrets, pass its polynomial degree.
    pub fn lwe_phase_noise_with_block(&self, secret_dimension: usize) -> FreshNoiseEstimate {
        self.weighted_phase_noise(secret_dimension, 1)
    }

    /// Phase estimate with an explicit ring product weight.
    /// `secret_dimension` is the original sampling block dimension. Use a product
    /// weight of `n` for negacyclic products or `4*n` as a conservative CI bound.
    pub fn weighted_phase_noise(&self, secret_dimension: usize, convolution_degree: usize) -> FreshNoiseEstimate {
        let mask_variance: f64 = self.masks().iter().map(FreshNoiseEstimate::variance).sum();
        let weighted_masks = if mask_variance == 0.0 {
            0.0
        } else {
            let second = self
                .secret
                .coefficient_second_moment(secret_dimension)
                .unwrap_or(f64::INFINITY);
            if second == 0.0 {
                0.0
            } else {
                convolution_degree as f64 * second * mask_variance
            }
        };
        FreshNoiseEstimate::new(self.body().variance() + weighted_masks, self.precision())
    }

    /// Replaces all component estimates while preserving the secret provenance.
    /// Panics for an empty vector or inconsistent creation precisions.
    pub fn with_components(&self, components: Vec<FreshNoiseEstimate>) -> Self {
        assert!(!components.is_empty(), "noise must include a body term");
        let precision = components[0].precision();
        assert!(
            components.iter().all(|term| term.precision() == precision),
            "noise component precisions differ"
        );
        Self {
            secret: self.secret,
            components: components.into(),
        }
    }

    /// Records a derived body error and zero mask errors at the given precision.
    pub fn with_body_noise(&self, body: FreshNoiseEstimate) -> Self {
        let mut components = vec![FreshNoiseEstimate::new(0.0, body.precision()); self.components.len()];
        components[0] = body;
        self.with_components(components)
    }

    /// Appends zero-noise masks for a layout copy. Panics when reducing rank.
    /// This preserves the recorded estimates and their creation precision.
    pub fn with_rank(&self, rank: usize) -> Self {
        assert!(rank >= self.rank(), "cannot truncate noise mask components");
        if rank == self.rank() {
            return self.clone();
        }
        let count = rank.checked_add(1).expect("noise component count overflow");
        let mut components = self.components.to_vec();
        components.resize(count, FreshNoiseEstimate::new(0.0, self.precision()));
        self.with_components(components)
    }

    /// Combines independent secret shares and their component errors.
    /// Panics for different base laws, ranks, or an overflowing party count.
    pub fn aggregate(&self, other: &Self) -> Self {
        assert!(self.secret.base == other.secret.base, "incompatible secret distributions");
        assert!(self.rank() == other.rank(), "incompatible noise component counts");
        let precision = self.precision();
        let components: Vec<_> = self
            .components
            .iter()
            .zip(other.components.iter())
            .map(|(left, right)| FreshNoiseEstimate::new(left.variance() + right.variance_at(precision), precision))
            .collect();
        Self {
            secret: SecretDistribution {
                base: self.secret.base,
                parties: self
                    .secret
                    .parties
                    .checked_add(other.secret.parties)
                    .expect("party count overflow"),
            },
            components: components.into(),
        }
    }

    /// Number of independent parties in the encrypting secret.
    pub fn parties(&self) -> u64 {
        self.secret.parties
    }

    /// Distribution of the encrypting secret, including its independent sums.
    pub fn secret_distribution(&self) -> SecretDistribution {
        self.secret
    }

    /// Compares secret provenance independently of component error estimates.
    pub fn same_secret(&self, other: &Self) -> bool {
        self.secret == other.secret
    }

    pub(crate) fn validate_components(&self, expected: usize) -> io::Result<()> {
        if self.components.len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "noise component count does not match ciphertext shape",
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_wire(&self) -> io::Result<()> {
        self.secret.base.validate_wire()
    }

    pub(crate) fn write_optional<W: Write>(noise: Option<&Self>, writer: &mut W) -> io::Result<()> {
        if let Some(noise) = noise {
            noise.validate_wire()?;
        }
        writer.write_all(b"PNM3")?;
        match noise {
            None => writer.write_u64::<LittleEndian>(0),
            Some(noise) => {
                writer.write_u64::<LittleEndian>(noise.parties())?;
                noise.secret.base.write_to(writer)?;
                writer.write_u32::<LittleEndian>(noise.precision().0)?;
                writer.write_u64::<LittleEndian>(noise.components.len() as u64)?;
                let stored = noise
                    .components
                    .iter()
                    .rposition(|term| term.variance_bits != 0)
                    .map_or(0, |i| i + 1);
                writer.write_u64::<LittleEndian>(stored as u64)?;
                for term in &noise.components[..stored] {
                    writer.write_u64::<LittleEndian>(term.variance_bits)?;
                }
                Ok(())
            }
        }
    }

    pub(crate) fn read_optional<R: Read>(reader: &mut R, expected: usize) -> io::Result<Option<Self>> {
        let mut marker = [0; 4];
        reader.read_exact(&mut marker)?;
        if marker != *b"PNM3" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid component noise version"));
        }
        let parties = reader.read_u64::<LittleEndian>()?;
        if parties == 0 {
            return Ok(None);
        }
        let base = Distribution::read_from(reader)?;
        let precision = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        let count = reader.read_u64::<LittleEndian>()?;
        if count == 0 || count != expected as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "noise component count does not match ciphertext shape",
            ));
        }
        let stored = reader.read_u64::<LittleEndian>()?;
        if stored > count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "noise stored prefix exceeds component count",
            ));
        }
        let mut components = vec![FreshNoiseEstimate::new(0.0, precision); expected];
        for term in &mut components[..stored as usize] {
            let variance = f64::from_bits(reader.read_u64::<LittleEndian>()?);
            if variance.is_nan() || variance < 0.0 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid noise component variance"));
            }
            *term = FreshNoiseEstimate::new(variance, precision);
        }
        Ok(Some(Self {
            secret: SecretDistribution { base, parties },
            components: components.into(),
        }))
    }
}

impl std::ops::Index<usize> for ComponentNoise {
    type Output = FreshNoiseEstimate;
    fn index(&self, component: usize) -> &Self::Output {
        &self.components[component]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_roundtrip_and_phase_model() {
        let fresh = ComponentNoise::from_secret_at(Distribution::TernaryProb(0.5), TorusPrecision(60), 2);
        assert_eq!(fresh.components().len(), 3);
        assert_eq!(fresh.body().std_dev(), 3.2);
        assert!(fresh.masks().iter().all(|term| term.variance() == 0.0));
        let pk = fresh.with_components(
            [4.0, 9.0, 16.0]
                .map(|v| FreshNoiseEstimate::new(v, TorusPrecision(60)))
                .to_vec(),
        );
        assert_eq!(pk.phase_noise(64).variance(), 804.0);
        assert!(pk.same_secret(&fresh));
        assert!(fresh != pk);
        let infinite = pk.with_body_noise(FreshNoiseEstimate::new(f64::INFINITY, TorusPrecision(60)));
        for value in [None, Some(fresh), Some(pk), Some(infinite)] {
            let mut bytes = Vec::new();
            ComponentNoise::write_optional(value.as_ref(), &mut bytes).unwrap();
            assert!(
                ComponentNoise::read_optional(&mut bytes.as_slice(), value.as_ref().map_or(0, |n| n.components().len())).unwrap()
                    == value
            );
        }
    }

    #[test]
    fn independent_contributions_add_every_component_across_precisions() {
        let left = ComponentNoise::from_secret_at(Distribution::TernaryProb(0.5), TorusPrecision(60), 1)
            .with_components([16.0, 4.0].map(|v| FreshNoiseEstimate::new(v, TorusPrecision(60))).to_vec());
        let right = ComponentNoise::from_secret_at(Distribution::TernaryProb(0.5), TorusPrecision(61), 1)
            .with_components([64.0, 36.0].map(|v| FreshNoiseEstimate::new(v, TorusPrecision(61))).to_vec());
        let sum = left.aggregate(&right);
        assert_eq!(sum.parties(), 2);
        assert_eq!(sum[0].variance(), 32.0);
        assert_eq!(sum[1].variance(), 13.0);
        assert_eq!(sum.phase_noise(64).variance(), 864.0);
        assert_eq!(right.aggregate(&left)[1].variance_at(TorusPrecision(60)), 13.0);
        let wrong_rank = ComponentNoise::from_secret(Distribution::TernaryProb(0.5), 0);
        assert!(std::panic::catch_unwind(|| left.aggregate(&wrong_rank)).is_err());
        assert!(std::panic::catch_unwind(|| left.with_components(Vec::new())).is_err());
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
    fn invalid_component_variances_counts_and_legacy_versions_are_rejected() {
        let noise = ComponentNoise::from_secret_at(Distribution::BinaryProb(0.5), TorusPrecision(60), 1);
        let mut encoded = Vec::new();
        ComponentNoise::write_optional(Some(&noise), &mut encoded).unwrap();
        for variance in [f64::NAN, f64::NEG_INFINITY, -1.0] {
            assert!(std::panic::catch_unwind(|| FreshNoiseEstimate::new(variance, TorusPrecision(60))).is_err());
            let mut bytes = encoded.clone();
            let offset = bytes.len() - 8;
            bytes[offset..].copy_from_slice(&variance.to_bits().to_le_bytes());
            assert_eq!(
                ComponentNoise::read_optional(&mut bytes.as_slice(), noise.components().len())
                    .unwrap_err()
                    .kind(),
                io::ErrorKind::InvalidData
            );
        }
        for marker in [b"PNM1", b"PNM2"] {
            let mut bytes = encoded.clone();
            bytes[..4].copy_from_slice(marker);
            assert!(ComponentNoise::read_optional(&mut bytes.as_slice(), noise.components().len()).is_err());
        }
        for count in [0, u64::MAX] {
            let mut bytes = encoded[..33].to_vec();
            bytes[25..33].copy_from_slice(&count.to_le_bytes());
            assert!(ComponentNoise::read_optional(&mut bytes.as_slice(), noise.components().len()).is_err());
        }
        assert!(noise.validate_components(2).is_ok());
        assert!(noise.validate_components(3).is_err());
    }

    #[test]
    fn rank_padding_preserves_components_and_appends_zero_masks() {
        let noise = ComponentNoise::from_secret_at(Distribution::TernaryProb(0.5), TorusPrecision(35), 2).with_components(
            [4.0, 9.0, 16.0]
                .map(|v| FreshNoiseEstimate::new(v, TorusPrecision(35)))
                .to_vec(),
        );
        assert!(noise.with_rank(2) == noise);
        let expanded = noise.with_rank(4);
        assert!(expanded.components()[..3] == *noise.components());
        assert!(
            expanded.masks()[2..]
                .iter()
                .all(|term| term.variance() == 0.0 && term.precision() == noise.precision())
        );
        assert!(std::panic::catch_unwind(|| expanded.with_rank(1)).is_err());
    }

    #[test]
    fn scalar_lwe_uses_secret_dimension_without_ring_convolution() {
        for (distribution, second) in [(Distribution::TernaryFixed(16), 0.25), (Distribution::BinaryBlock(4), 0.2)] {
            let noise = ComponentNoise::from_secret(distribution, 64).with_components(vec![
                FreshNoiseEstimate::new(
                    1.0,
                    TorusPrecision(0)
                );
                65
            ]);
            assert_eq!(noise.lwe_phase_noise().variance(), 1.0 + 64.0 * second);
        }
    }

    #[test]
    fn unknown_and_rank_zero_secret_noise_remain_well_defined() {
        let noise = ComponentNoise::from_secret(Distribution::NONE, 0);
        assert_eq!(noise.components().len(), 1);
        assert_eq!(noise.phase_noise(1), noise.body());
        let mut bytes = Vec::new();
        ComponentNoise::write_optional(Some(&noise), &mut bytes).unwrap();
        assert!(ComponentNoise::read_optional(&mut bytes.as_slice(), noise.components().len()).unwrap() == Some(noise));
        let zero = ComponentNoise::from_secret(Distribution::ZERO, 1).with_components(vec![
            FreshNoiseEstimate::new(0.0, TorusPrecision(0)),
            FreshNoiseEstimate::new(f64::INFINITY, TorusPrecision(0)),
        ]);
        assert_eq!(zero.phase_noise(64).variance(), 0.0);
    }

    #[test]
    fn encapsulated_distribution_normalizes_to_unknown() {
        let noise = ComponentNoise::from_secret(Distribution::ENCAPSULATED("ephemeral"), 1);
        assert_eq!(noise.secret_distribution().base(), Distribution::NONE);
        let mut bytes = Vec::new();
        ComponentNoise::write_optional(Some(&noise), &mut bytes).unwrap();
        assert_eq!(ComponentNoise::read_optional(&mut bytes.as_slice(), 2).unwrap(), Some(noise));
    }

    #[test]
    fn ciphertext_wire_format_preserves_components_and_rejects_wrong_rank() {
        use crate::layouts::{Base2K, Degree, GLWE, Rank, TorusPrecision};
        use poulpy_hal::layouts::{ReaderFrom, WriterTo};
        let mut ciphertext = GLWE::<poulpy_hal::AlignedBuf, i64>::alloc(Degree(64), Base2K(12), TorusPrecision(35), Rank(1));
        let mut restored = ciphertext.clone();
        ciphertext.noise = Some(
            ComponentNoise::from_secret_at(Distribution::TernaryProb(0.3), TorusPrecision(35), 1)
                .with_components([71.0, 13.0].map(|v| FreshNoiseEstimate::new(v, TorusPrecision(35))).to_vec()),
        );
        assert!(ciphertext != restored);
        let mut bytes = Vec::new();
        ciphertext.write_to(&mut bytes).unwrap();
        restored.read_from(&mut bytes.as_slice()).unwrap();
        assert!(ciphertext == restored);
        // Build a structurally valid noise vector with the wrong rank, then
        // append the original ciphertext payload to exercise reader validation.
        let mut wrong = Vec::new();
        let wrong_noise = ComponentNoise::from_secret(Distribution::TernaryProb(0.3), 0);
        ComponentNoise::write_optional(Some(&wrong_noise), &mut wrong).unwrap();
        wrong.extend_from_slice(&bytes[57..]);
        assert_eq!(
            restored.read_from(&mut wrong.as_slice()).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        ciphertext.noise = Some(wrong_noise);
        let mut output = Vec::new();
        assert_eq!(
            ciphertext.write_to(&mut output).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        assert!(output.is_empty());
    }
    #[test]
    fn fresh_lwe_prefix_is_compact_and_keeps_logical_terms() {
        use crate::layouts::{Base2K, Degree, LWE, LWEInfos};
        use poulpy_hal::{
            AlignedBuf,
            layouts::{ReaderFrom, WriterTo},
        };
        let noise = ComponentNoise::from_secret_at(Distribution::TernaryProb(2.0 / 3.0), TorusPrecision(35), 1024);
        let mut metadata = Vec::new();
        ComponentNoise::write_optional(Some(&noise), &mut metadata).unwrap();
        assert_eq!(metadata.len(), 49);
        let mut ciphertext = LWE::<AlignedBuf, i64>::alloc(Degree(1024), Base2K(12), TorusPrecision(35));
        ciphertext.noise = Some(noise.clone());
        let mut bytes = Vec::new();
        ciphertext.write_to(&mut bytes).unwrap();
        let mut restored = ciphertext.clone();
        restored.noise = None;
        restored.read_from(&mut bytes.as_slice()).unwrap();
        assert!(restored == ciphertext);
        assert_eq!(restored.noise().unwrap().components().len(), 1025);
        metadata[25..33].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(
            ComponentNoise::read_optional(&mut metadata.as_slice(), 1025)
                .unwrap_err()
                .to_string()
                .contains("count")
        );
    }

    #[test]
    fn flattened_fixed_weight_secret_uses_its_original_block_dimension() {
        let noise = ComponentNoise::from_secret(Distribution::TernaryFixed(16), 128).with_components(vec![
            FreshNoiseEstimate::new(
                1.0,
                TorusPrecision(0)
            );
            129
        ]);
        assert_eq!(noise.lwe_phase_noise_with_block(64).variance(), 33.0);
        assert_eq!(noise.lwe_phase_noise().variance(), 17.0);
    }

    #[test]
    fn invalid_key_provenance_fails_before_wrapper_headers() {
        use crate::layouts::{Base2K, Degree, Dnum, Dsize, GLWEAutomorphismKey, GLWESwitchingKey, Rank};
        use poulpy_hal::{AlignedBuf, layouts::WriterTo};
        let mut invalid = ComponentNoise::from_secret(Distribution::NONE, 1);
        invalid.secret.base = Distribution::ENCAPSULATED("hand-built");
        let mut switching = GLWESwitchingKey::<AlignedBuf, i64>::alloc(
            Degree(8),
            Base2K(12),
            Dnum(2),
            Dsize(1),
            TorusPrecision(15),
            Rank(1),
            Rank(1),
        );
        switching.key.noise = Some(invalid.clone());
        let mut automorphism =
            GLWEAutomorphismKey::<AlignedBuf, i64>::alloc(Degree(8), Base2K(12), Dnum(2), Dsize(1), TorusPrecision(15), Rank(1));
        automorphism.key.noise = Some(invalid);
        let mut bytes = Vec::new();
        assert!(switching.write_to(&mut bytes).is_err());
        assert!(bytes.is_empty());
        assert!(automorphism.write_to(&mut bytes).is_err());
        assert!(bytes.is_empty());
    }
}
