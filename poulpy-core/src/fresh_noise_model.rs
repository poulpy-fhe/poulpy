//! Fresh encryption error estimates. These do not track evaluated noise.

use poulpy_hal::layouts::Backend;

use crate::{
    Distribution, EncryptionMetadata, FreshNoiseEstimate, GetDistribution, Noise, SecretDistribution,
    layouts::{GGLWEInfos, GGSWInfos, GLWEInfos, TorusPrecision, prepared::GLWEPublicKeyPreparedToBackendRef},
};

fn base_moments(distribution: Distribution, n: usize) -> Option<(f64, f64)> {
    let probability = |p: f64| (p.is_finite() && (0.0..=1.0).contains(&p)).then_some(p);
    match distribution {
        Distribution::TernaryFixed(h) if n > 0 && h <= n => Some((0.0, h as f64 / n as f64)),
        Distribution::TernaryProb(p) => probability(p).map(|p| (0.0, p)),
        Distribution::BinaryFixed(h) if n > 0 && h <= n => {
            let p = h as f64 / n as f64;
            Some((p, p))
        }
        Distribution::BinaryProb(p) => probability(p).map(|p| (p, p)),
        // The sampler chooses one of b positions or the all-zero block.
        Distribution::BinaryBlock(b) if b > 0 && n > 0 && n.is_multiple_of(b) => {
            let p = 1.0 / (b as f64 + 1.0);
            Some((p, p))
        }
        Distribution::ZERO => Some((0.0, 0.0)),
        _ => None,
    }
}

impl SecretDistribution {
    /// Mean of a coefficient of the independently summed secret.
    /// Returns `None` for an unknown law or an invalid distribution parameter.
    pub fn coefficient_mean(&self, n: usize) -> Option<f64> {
        base_moments(self.base(), n).map(|(mean, _)| self.parties() as f64 * mean)
    }

    /// Second moment of a coefficient of the independently summed secret.
    ///
    /// For `p` parties this is `p * Var(s) + p^2 * E[s]^2`. A zero-mean
    /// independent error multiplied by the secret is weighted by this second
    /// moment, including the nonzero mean of binary secrets.
    pub fn coefficient_second_moment(&self, n: usize) -> Option<f64> {
        base_moments(self.base(), n).map(|(mean, second)| {
            let parties = self.parties() as f64;
            parties * (second - mean * mean) + (parties * mean).powi(2)
        })
    }
}

/// The sampler's centered variance parameter in integer coefficient units.
/// Gaussian sigma squared conservatively models the conditioned distribution.
pub(crate) fn noise_variance(noise: Noise) -> f64 {
    match noise {
        Noise::Gaussian { sigma, .. } => sigma * sigma,
        Noise::Uniform { bits } => {
            // Divide before the last power of two so widths such as 512 bits
            // retain a finite variance whenever the final f64 fits.
            (2.0 * bits as f64 - 4.0).exp2() * (16.0 / 12.0) - 1.0 / 12.0
        }
    }
}

fn scaled_variance(variance: f64, factor: f64) -> f64 {
    if factor == 0.0 || variance == 0.0 {
        0.0
    } else {
        variance * factor
    }
}

fn variance_with_extra_precision(estimate: FreshNoiseEstimate, precision: TorusPrecision, extra_bits: i64) -> f64 {
    let shift = i64::from(precision.as_u32()) + extra_bits - i64::from(estimate.precision().as_u32());
    // Express the combined shift using two representable grids. Combining
    // first avoids underflow before a large gadget digit restores its scale.
    // Shifts beyond u32 already saturate every nonzero finite f64 value.
    let magnitude = TorusPrecision(shift.unsigned_abs().min(u32::MAX as u64) as u32);
    if shift >= 0 {
        FreshNoiseEstimate::new(estimate.variance(), TorusPrecision(0)).variance_at(magnitude)
    } else {
        FreshNoiseEstimate::new(estimate.variance(), magnitude).variance_at(TorusPrecision(0))
    }
}

/// Ordinary body error is sampled with the masks. Intentional flooding remains
/// at the output grid, even when its descriptor equals `Noise::ENCRYPTION`.
#[derive(Clone, Copy)]
pub(crate) enum PublicKeyBodyNoise {
    Sampled,
    Omitted,
    Flood(Noise),
}

pub(crate) struct PublicKeyEncryptionPlan {
    pub sample_precision: TorusPrecision,
    pub work_precision: TorusPrecision,
    pub encryption_metadata: Option<EncryptionMetadata>,
}

fn public_key_work_precision(sample: TorusPrecision, key: TorusPrecision, base2k: usize) -> TorusPrecision {
    let base2k = base2k as u64;
    TorusPrecision((u64::from(sample.as_u32()).div_ceil(base2k) * base2k).min(u64::from(key.as_u32())) as u32)
}

fn public_key_truncation_variance(rank_n: f64, ephemeral_second: f64, secret_second: f64, centered: bool, base2k: usize) -> f64 {
    // Dropping canonical balanced limbs is not exact nearest rounding. Their
    // geometric tail is bounded by this many working-grid ulps.
    let tail = 0.5 / (1.0 - (-(base2k as f64)).exp2());
    if centered {
        scaled_variance(
            scaled_variance(ephemeral_second, rank_n),
            1.0 + scaled_variance(secret_second, rank_n),
        ) * tail.powi(2)
    } else {
        // Nonzero means can align across convolution terms. The L1 RMS bound
        // includes those cross terms instead of assuming independent errors.
        let u_l1 = scaled_variance(ephemeral_second.sqrt(), rank_n);
        let phase_l1 = 1.0 + scaled_variance(secret_second.sqrt(), rank_n);
        scaled_variance(u_l1, phase_l1 * tail).powi(2)
    }
}

fn combine_inherited_errors(inherited: f64, truncation: f64) -> f64 {
    if truncation == 0.0 {
        inherited
    } else if inherited == 0.0 {
        truncation
    } else {
        // Both terms depend on the same public key and ephemeral. Bound their
        // covariance by adding RMS errors, not by assuming independence.
        (inherited.sqrt() + truncation.sqrt()).powi(2)
    }
}

fn public_key_sample_precision(
    output: TorusPrecision,
    key: TorusPrecision,
    inherited: f64,
    fresh: f64,
    rounding: f64,
    truncation: impl Fn(TorusPrecision) -> f64,
) -> TorusPrecision {
    assert!(key >= output, "invalid public key: less precise than the output");
    if key == output || !inherited.is_finite() || !fresh.is_finite() || !rounding.is_finite() {
        return key;
    }
    if inherited > rounding || (inherited == rounding && fresh > 0.0) {
        return key;
    }
    let fits = |precision| {
        let combined = combine_inherited_errors(inherited, truncation(precision));
        if combined > rounding || (combined == rounding && fresh > 0.0) {
            false
        } else {
            FreshNoiseEstimate::new(fresh, precision).variance_at(output) <= rounding - combined
        }
    };
    if !fits(key) {
        return key;
    }
    // The target is monotone in the integer sampling precision. Integer search
    // avoids logarithm rounding at exact powers of four and also supports the
    // entire u32 precision range without constructing an overflowing scale.
    let mut low = output.as_u32();
    let mut high = key.as_u32();
    while low < high {
        let mid = low + (high - low) / 2;
        if fits(TorusPrecision(mid)) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    TorusPrecision(low)
}

fn public_key_phase_plan<R: GLWEInfos>(
    metadata: Option<EncryptionMetadata>,
    res: &R,
    pk_precision: TorusPrecision,
    ephemeral: Distribution,
    body_noise: PublicKeyBodyNoise,
) -> PublicKeyEncryptionPlan {
    assert!(pk_precision >= res.k(), "invalid public key: less precise than the output");
    let Some(metadata) = metadata else {
        return PublicKeyEncryptionPlan {
            sample_precision: pk_precision,
            work_precision: pk_precision,
            encryption_metadata: None,
        };
    };
    if metadata.secret_distribution().base() != Distribution::NONE {
        assert_eq!(
            ephemeral,
            metadata.secret_distribution().base(),
            "invalid public key: ephemeral distribution differs from its secret provenance"
        );
    }
    let n = res.n().as_usize();
    let rank_n = res.rank().as_usize() as f64 * n as f64;
    let ephemeral_second = base_moments(ephemeral, n).map_or(f64::INFINITY, |(_, second)| second);
    let secret_second = metadata
        .secret_distribution()
        .coefficient_second_moment(n)
        .unwrap_or(f64::INFINITY);
    let inherited = scaled_variance(
        metadata.fresh_noise().variance_at(res.k()),
        scaled_variance(ephemeral_second, rank_n),
    );
    let secret_fold = scaled_variance(secret_second, rank_n);
    let fresh_mask = scaled_variance(noise_variance(Noise::ENCRYPTION), secret_fold);
    let fresh_body = if matches!(body_noise, PublicKeyBodyNoise::Sampled) {
        noise_variance(Noise::ENCRYPTION)
    } else {
        0.0
    };
    let fresh = scaled_variance(
        noise_variance(Noise::ENCRYPTION),
        secret_fold
            + if matches!(body_noise, PublicKeyBodyNoise::Sampled) {
                1.0
            } else {
                0.0
            },
    );
    // As in the core noise models, bound each coefficient rounding error by a
    // half ulp and add its modeled variance independently. This is an estimate,
    // not a proof that rounding and the existing phase error are uncorrelated.
    let rounding = (1.0 + secret_fold) / 4.0;
    let truncation_variance = public_key_truncation_variance(
        rank_n,
        ephemeral_second,
        secret_second,
        metadata.secret_distribution().coefficient_mean(n) == Some(0.0),
        res.base2k().as_usize(),
    );
    let truncation = |sample| {
        let work = public_key_work_precision(sample, pk_precision, res.base2k().as_usize());
        if work == pk_precision {
            0.0
        } else {
            FreshNoiseEstimate::new(truncation_variance, work).variance_at(res.k())
        }
    };
    let sample_precision = if ephemeral_second.is_finite()
        && secret_second.is_finite()
        && metadata.fresh_noise().variance().is_finite()
        && truncation_variance.is_finite()
    {
        public_key_sample_precision(res.k(), pk_precision, inherited, fresh, rounding, truncation)
    } else {
        pk_precision
    };
    let work_precision = public_key_work_precision(sample_precision, pk_precision, res.base2k().as_usize());
    let mask_at_output = FreshNoiseEstimate::new(fresh_mask, sample_precision).variance_at(res.k());
    let body_at_output = FreshNoiseEstimate::new(fresh_body, sample_precision).variance_at(res.k());
    let flood = match body_noise {
        PublicKeyBodyNoise::Flood(noise) => noise_variance(noise),
        _ => 0.0,
    };
    // Keep the historical addition order for an equal-precision key, including
    // the exact f64 metadata bits in existing serialized ciphertexts.
    let variance = combine_inherited_errors(inherited, truncation(sample_precision))
        + mask_at_output
        + body_at_output
        + if work_precision > res.k() { rounding } else { 0.0 }
        + flood;
    PublicKeyEncryptionPlan {
        sample_precision,
        work_precision,
        encryption_metadata: Some(metadata.with_fresh_noise(FreshNoiseEstimate::new(variance, res.k()))),
    }
}

/// Select the fresh-error grid and estimate the resulting output phase error.
/// The product consumes only the selected leading whole limbs of the prepared
/// key. Missing provenance selects full key precision and its full width.
pub(crate) fn public_key_encryption_plan<BE, R, K>(res: &R, pk: &K, body_noise: PublicKeyBodyNoise) -> PublicKeyEncryptionPlan
where
    BE: Backend,
    R: GLWEInfos,
    K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
{
    public_key_phase_plan(pk.encryption_metadata(), res, pk.k(), *pk.to_backend_ref().dist(), body_noise)
}

/// The expansion constructs mask columns by multiplying a row's phase by a
/// secret polynomial, then adding a gadget product with the conversion key.
/// Store the largest column estimate because metadata is shared by the GGSW.
pub(crate) fn ggsw_expansion_metadata<R: GGSWInfos, K: GGLWEInfos>(
    res: &R,
    input: Option<EncryptionMetadata>,
    input_precision: TorusPrecision,
    key: &K,
) -> Option<EncryptionMetadata> {
    let input = input?;
    let rank = res.rank().as_usize() as f64;
    let n = res.n().as_usize() as f64;
    let secret_second = input
        .secret_distribution()
        .coefficient_second_moment(res.n().as_usize())
        .unwrap_or(f64::INFINITY);
    let rounding = (1.0 + scaled_variance(secret_second, rank * n)) / 4.0;
    let body = input.fresh_noise().variance_at(res.k()) + if input_precision > res.k() { rounding } else { 0.0 };
    if rank == 0.0 {
        return Some(input.with_fresh_noise(FreshNoiseEstimate::new(body, res.k())));
    }
    let Some(key_metadata) = key.encryption_metadata().filter(|m| input.same_secret(m)) else {
        return Some(input.with_fresh_noise(FreshNoiseEstimate::new(f64::INFINITY, res.k())));
    };
    let digit_bits = key.dsize().as_usize() * key.base2k().as_usize();
    let digits = res.k().as_usize().div_ceil(digit_bits).min(key.dnum().as_usize());
    let key_error = scaled_variance(
        // Var(digit) = 2^(2B)/12. Move B-2 bits into the precision
        // rescaling, leaving 16/12 outside to avoid premature overflow.
        variance_with_extra_precision(key_metadata.fresh_noise(), res.k(), digit_bits as i64 - 2),
        rank * n * digits as f64 * (16.0 / 12.0),
    );
    let cover = digits * digit_bits;
    let residue = if cover < res.k().as_usize() {
        // The centered-secret product model does not account for the squared
        // convolution means of binary secrets. Retain an unbounded estimate
        // when an uncovered gadget tail would multiply those products.
        if input.secret_distribution().coefficient_mean(res.n().as_usize()) != Some(0.0) {
            f64::INFINITY
        } else {
            scaled_variance(
                (2.0 * (res.k().as_usize() - cover) as f64).exp2() / 12.0,
                (rank + 1.0) * n * n * secret_second.powi(2),
            )
        }
    } else {
        0.0
    };
    let rounding = if key.k() > res.k() || !res.k().as_usize().is_multiple_of(res.base2k().as_usize()) {
        rounding
    } else {
        0.0
    };
    let mask = scaled_variance(body, scaled_variance(secret_second, n)) + key_error + residue + rounding;
    Some(input.with_fresh_noise(FreshNoiseEstimate::new(body.max(mask), res.k())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layouts::{Base2K, Degree, Dnum, Dsize, GGLWE, GGLWELayout, GGSWLayout, GLWELayout, Rank};

    #[test]
    fn collective_binary_moments_include_the_mean() {
        let one = EncryptionMetadata::from_secret(Distribution::BinaryProb(0.5));
        let three = one.aggregate(one).aggregate(one).secret_distribution();
        assert_eq!(three.coefficient_mean(64), Some(1.5));
        assert_eq!(three.coefficient_second_moment(64), Some(3.0));
        let ternary = EncryptionMetadata::from_secret(Distribution::TernaryFixed(16));
        assert_eq!(
            ternary.aggregate(ternary).secret_distribution().coefficient_second_moment(64),
            Some(0.5)
        );
        let block = EncryptionMetadata::from_secret(Distribution::BinaryBlock(3));
        assert_eq!(block.secret_distribution().coefficient_second_moment(64), None);
        assert_eq!(block.secret_distribution().coefficient_second_moment(96), Some(0.25));
    }

    #[test]
    fn public_key_noise_rescales_inherited_error_and_preserves_destination_parties() {
        let layout = GLWELayout {
            n: Degree(64),
            base2k: Base2K(8),
            k: TorusPrecision(32),
            rank: Rank(2),
        };
        let base = Distribution::BinaryProb(0.5);
        let one = EncryptionMetadata::from_secret_at(base, TorusPrecision(36));
        let pk = one.aggregate(one).aggregate(one);
        let plan = public_key_phase_plan(Some(pk), &layout, TorusPrecision(36), base, PublicKeyBodyNoise::Sampled);
        assert_eq!(plan.sample_precision, TorusPrecision(35));
        let result = plan.encryption_metadata.unwrap();
        let sigma2 = noise_variance(Noise::ENCRYPTION);
        // Three binary parties have E[S²] = 3, while a new ephemeral has E[u²] = 1/2.
        let expected = 128.0 * 0.5 * (3.0 * sigma2 / 256.0) + (1.0 + 128.0 * 3.0) * sigma2 / 64.0 + (1.0 + 128.0 * 3.0) / 4.0;
        assert_eq!(result.parties(), 3);
        assert_eq!(result.fresh_noise().precision(), layout.k);
        assert!((result.initial_noise_variance() - expected).abs() < 1e-10);
    }

    #[test]
    fn smudging_replaces_only_the_fresh_body_draw() {
        let layout = GLWELayout {
            n: Degree(64),
            base2k: Base2K(8),
            k: TorusPrecision(32),
            rank: Rank(1),
        };
        let base = Distribution::TernaryProb(0.5);
        let pk = EncryptionMetadata::from_secret_at(base, layout.k);
        let without_body = public_key_phase_plan(Some(pk), &layout, layout.k, base, PublicKeyBodyNoise::Omitted)
            .encryption_metadata
            .unwrap();
        let smudged = public_key_phase_plan(
            Some(pk),
            &layout,
            layout.k,
            base,
            PublicKeyBodyNoise::Flood(Noise::Uniform { bits: 4 }),
        )
        .encryption_metadata
        .unwrap();
        assert_eq!(
            smudged.initial_noise_variance() - without_body.initial_noise_variance(),
            21.25
        );
        let unknown = EncryptionMetadata::from_secret_at(Distribution::NONE, layout.k);
        let conservative = public_key_phase_plan(Some(unknown), &layout, layout.k, base, PublicKeyBodyNoise::Sampled)
            .encryption_metadata
            .unwrap();
        assert_eq!(conservative.secret_distribution(), unknown.secret_distribution());
        assert_eq!(conservative.initial_noise_variance(), f64::INFINITY);
        assert!(noise_variance(Noise::Uniform { bits: 512 }).is_finite());
        assert!(noise_variance(Noise::Uniform { bits: 513 }).is_finite());
        assert_eq!(noise_variance(Noise::Uniform { bits: 514 }), f64::INFINITY);
    }

    #[test]
    fn public_key_precision_is_minimal_or_capped() {
        let output = TorusPrecision(32);
        // Compare the integer search against independent repeated quartering.
        for inherited in [0.0, 0.5, 1.0, 2.0, 8.0] {
            for fresh in [0.0, 0.25, 1.0, 4.0, 16.0, 64.0] {
                for rounding in [0.0, 0.5, 1.0, 4.0] {
                    for gap in [0, 1, 2, 3, 8] {
                        let key = TorusPrecision(output.0 + gap);
                        let mut expected = key;
                        let mut remaining = fresh;
                        for delta in 0..=gap {
                            if inherited + remaining <= rounding {
                                expected = TorusPrecision(output.0 + delta);
                                break;
                            }
                            remaining *= 0.25;
                        }
                        assert_eq!(
                            public_key_sample_precision(output, key, inherited, fresh, rounding, |_| 0.0),
                            expected
                        );
                    }
                }
            }
        }
        assert_eq!(
            public_key_sample_precision(output, TorusPrecision(44), 0.0, 10.24, 0.25, |_| 0.0),
            TorusPrecision(35)
        );
        assert_eq!(
            public_key_sample_precision(output, TorusPrecision(33), 0.0, 10.24, 0.25, |_| 0.0),
            TorusPrecision(33)
        );
        // Exact equality is feasible, while one representable step above it
        // needs one more sampling bit.
        assert_eq!(
            public_key_sample_precision(output, TorusPrecision(40), 0.0, 16.0, 1.0, |_| 0.0),
            TorusPrecision(34)
        );
        assert_eq!(
            public_key_sample_precision(
                output,
                TorusPrecision(40),
                0.0,
                f64::from_bits(16.0_f64.to_bits() + 1),
                1.0,
                |_| 0.0
            ),
            TorusPrecision(35)
        );
    }

    #[test]
    fn public_key_prefix_precision_is_minimal_across_limb_boundaries() {
        let sigma2 = noise_variance(Noise::ENCRYPTION);
        for base2k in [1, 2, 8, 12] {
            for k in [1, 7, 8, 9, 21] {
                for gap in [0, 1, 3, 8, 36] {
                    for base in [Distribution::TernaryProb(0.5), Distribution::BinaryProb(0.5)] {
                        let layout = GLWELayout {
                            n: Degree(64),
                            base2k: Base2K(base2k),
                            k: TorusPrecision(k),
                            rank: Rank(1),
                        };
                        let pk_k = k + gap;
                        let metadata = EncryptionMetadata::from_secret_at(base, TorusPrecision(pk_k));
                        let plan = public_key_phase_plan(
                            Some(metadata),
                            &layout,
                            TorusPrecision(pk_k),
                            base,
                            PublicKeyBodyNoise::Sampled,
                        );
                        let inherited = 32.0 * sigma2 * (-2.0 * gap as f64).exp2();
                        let tail = 0.5 / (1.0 - (-(base2k as f64)).exp2());
                        let prefix = if matches!(base, Distribution::TernaryProb(_)) {
                            32.0 * 33.0 * tail.powi(2)
                        } else {
                            (64.0 * 0.5_f64.sqrt() * (1.0 + 64.0 * 0.5_f64.sqrt()) * tail).powi(2)
                        };
                        let mut fresh = sigma2 * 33.0;
                        let mut expected = pk_k;
                        for sample in k..=pk_k {
                            let work = (sample.div_ceil(base2k) * base2k).min(pk_k);
                            let cut = if work == pk_k {
                                0.0
                            } else {
                                prefix * (-2.0 * (work - k) as f64).exp2()
                            };
                            let combined = if cut == 0.0 {
                                inherited
                            } else {
                                (inherited.sqrt() + cut.sqrt()).powi(2)
                            };
                            if combined + fresh <= 33.0 / 4.0 {
                                expected = sample;
                                break;
                            }
                            fresh *= 0.25;
                        }
                        assert_eq!(
                            plan.sample_precision,
                            TorusPrecision(expected),
                            "base2k={base2k}, k={k}, gap={gap}, law={base:?}"
                        );
                        assert_eq!(
                            plan.work_precision,
                            TorusPrecision((expected.div_ceil(base2k) * base2k).min(pk_k))
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn public_key_prefix_tail_and_binary_covariance_are_bounded() {
        // At radix two, the balanced-digit tail can approach a whole ulp.
        assert_eq!(public_key_truncation_variance(1.0, 1.0, 0.0, true, 1), 1.0);
        let centered = public_key_truncation_variance(64.0, 0.5, 3.0, true, 8);
        let binary = public_key_truncation_variance(64.0, 0.5, 3.0, false, 8);
        let tail = 128.0 / 255.0;
        assert_eq!(centered, 32.0 * 193.0 * tail * tail);
        assert!((binary - (64.0 * 0.5_f64.sqrt() * (1.0 + 64.0 * 3.0_f64.sqrt()) * tail).powi(2)).abs() < binary * 1e-15);
        assert!(binary > centered);
        assert_eq!(combine_inherited_errors(4.0, 9.0), 25.0);
        assert_eq!(combine_inherited_errors(4.0, 0.0), 4.0);
        assert_eq!(
            public_key_work_precision(TorusPrecision(u32::MAX - 1), TorusPrecision(u32::MAX), 52),
            TorusPrecision(u32::MAX)
        );

        // For this high-precision key, two working limbs leave too much tail
        // error. The next sample bit includes a third limb and meets the target.
        let layout = GLWELayout {
            n: Degree(64),
            base2k: Base2K(12),
            k: TorusPrecision(21),
            rank: Rank(1),
        };
        let pk = EncryptionMetadata::from_secret_at(Distribution::TernaryProb(0.5), TorusPrecision(57));
        let plan = public_key_phase_plan(
            Some(pk),
            &layout,
            TorusPrecision(57),
            Distribution::TernaryProb(0.5),
            PublicKeyBodyNoise::Sampled,
        );
        assert_eq!(plan.sample_precision, TorusPrecision(25));
        assert_eq!(plan.work_precision, TorusPrecision(36));
        assert!(plan.work_precision < TorusPrecision(57));
    }

    #[test]
    fn equal_precision_metadata_preserves_its_exact_bits() {
        let layout = GLWELayout {
            n: Degree(2048),
            base2k: Base2K(12),
            k: TorusPrecision(49),
            rank: Rank(1),
        };
        let base = Distribution::TernaryProb(0.5);
        let metadata = EncryptionMetadata::from_secret_at(base, layout.k);
        let sigma2 = noise_variance(Noise::ENCRYPTION);
        let inherited = sigma2 * 1024.0;
        let historical = inherited + sigma2 * 1024.0 + sigma2;
        let plan = public_key_phase_plan(Some(metadata), &layout, layout.k, base, PublicKeyBodyNoise::Sampled);
        assert_eq!(
            plan.encryption_metadata.unwrap().initial_noise_variance().to_bits(),
            historical.to_bits()
        );
    }

    #[test]
    fn public_key_precision_handles_unknown_and_extreme_estimates() {
        let layout = GLWELayout {
            n: Degree(64),
            base2k: Base2K(8),
            k: TorusPrecision(32),
            rank: Rank(1),
        };
        let key = TorusPrecision(64);
        let base = Distribution::TernaryProb(0.5);
        let absent = public_key_phase_plan(None, &layout, key, base, PublicKeyBodyNoise::Sampled);
        assert_eq!(absent.sample_precision, key);
        assert!(absent.encryption_metadata.is_none());
        let unknown = EncryptionMetadata::from_secret_at(Distribution::NONE, key);
        let plan = public_key_phase_plan(Some(unknown), &layout, key, base, PublicKeyBodyNoise::Sampled);
        assert_eq!(plan.sample_precision, key);
        assert_eq!(plan.encryption_metadata.unwrap().initial_noise_variance(), f64::INFINITY);
        let infinite =
            EncryptionMetadata::from_secret_at(base, key).with_fresh_noise(FreshNoiseEstimate::new(f64::INFINITY, key));
        assert_eq!(
            public_key_phase_plan(Some(infinite), &layout, key, base, PublicKeyBodyNoise::Sampled).sample_precision,
            key
        );
        assert_eq!(
            public_key_sample_precision(TorusPrecision(0), TorusPrecision(u32::MAX), 0.0, 16.0, 1.0, |_| 0.0),
            TorusPrecision(2)
        );
        assert_eq!(
            public_key_sample_precision(TorusPrecision(u32::MAX - 4), TorusPrecision(u32::MAX), 0.0, 16.0, 1.0, |_| {
                0.0
            }),
            TorusPrecision(u32::MAX - 2)
        );
        // A positive fresh variance cannot meet a zero remaining budget merely
        // because rescaling eventually underflows in floating-point arithmetic.
        assert_eq!(
            public_key_sample_precision(layout.k, key, 1.0, f64::MIN_POSITIVE, 1.0, |_| 0.0),
            key
        );
    }

    #[test]
    fn public_key_flood_stays_at_output_and_does_not_select_the_grid() {
        let layout = GLWELayout {
            n: Degree(64),
            base2k: Base2K(8),
            k: TorusPrecision(32),
            rank: Rank(1),
        };
        let key = TorusPrecision(44);
        let base = Distribution::TernaryProb(0.5);
        let metadata = EncryptionMetadata::from_secret_at(base, key);
        let masks = public_key_phase_plan(Some(metadata), &layout, key, base, PublicKeyBodyNoise::Omitted);
        for noise in [Noise::ENCRYPTION, Noise::Uniform { bits: 40 }] {
            let flood = public_key_phase_plan(Some(metadata), &layout, key, base, PublicKeyBodyNoise::Flood(noise));
            assert_eq!(flood.sample_precision, masks.sample_precision);
            let expected = masks.encryption_metadata.unwrap().initial_noise_variance() + noise_variance(noise);
            assert_eq!(flood.encryption_metadata.unwrap().initial_noise_variance(), expected);
        }
        let ordinary = public_key_phase_plan(Some(metadata), &layout, key, base, PublicKeyBodyNoise::Sampled);
        let masks_variance = masks.encryption_metadata.unwrap().initial_noise_variance();
        assert!(
            (ordinary.encryption_metadata.unwrap().initial_noise_variance()
                - masks_variance
                - noise_variance(Noise::ENCRYPTION) / 64.0)
                .abs()
                < 1e-12
        );
        // Creation precision may differ from the current key grid. Its stored
        // variance must be rescaled using the creation grid, not key.k().
        let older = metadata.with_fresh_noise(FreshNoiseEstimate::new(
            noise_variance(Noise::ENCRYPTION) / 16.0,
            TorusPrecision(42),
        ));
        let equivalent = public_key_phase_plan(Some(older), &layout, key, base, PublicKeyBodyNoise::Sampled);
        assert_eq!(equivalent.sample_precision, ordinary.sample_precision);
        assert_eq!(equivalent.encryption_metadata, ordinary.encryption_metadata);
    }

    #[test]
    fn expansion_handles_uncovered_binary_products_and_rank_zero() {
        let mut layout = GGSWLayout {
            n: Degree(64),
            base2k: Base2K(8),
            dnum: Dnum(3),
            dsize: Dsize(1),
            k_aux: TorusPrecision(8),
            rank: Rank(1),
        };
        let key_layout = GGLWELayout {
            n: Degree(64),
            base2k: Base2K(8),
            dnum: Dnum(2),
            dsize: Dsize(1),
            k_aux: TorusPrecision(16),
            rank_in: Rank(1),
            rank_out: Rank(1),
            stride: 1,
        };
        let mut key = GGLWE::<poulpy_hal::AlignedBuf, i64>::alloc_from_infos(&key_layout);
        let binary = EncryptionMetadata::from_secret_at(Distribution::BinaryProb(0.5), TorusPrecision(32));
        key.encryption_metadata = Some(binary);
        let expanded = ggsw_expansion_metadata(&layout, Some(binary), TorusPrecision(32), &key).unwrap();
        assert!(expanded.same_secret(&binary));
        assert_eq!(expanded.initial_noise_variance(), f64::INFINITY);

        let zero = EncryptionMetadata::from_secret_at(Distribution::ZERO, TorusPrecision(32));
        key.encryption_metadata = Some(zero);
        assert!(
            ggsw_expansion_metadata(&layout, Some(zero), TorusPrecision(32), &key)
                .unwrap()
                .initial_noise_variance()
                .is_finite()
        );

        layout.rank = Rank(0);
        key.encryption_metadata = None;
        assert_eq!(
            ggsw_expansion_metadata(&layout, Some(binary), TorusPrecision(32), &key),
            Some(binary)
        );
    }

    #[test]
    fn expansion_includes_copy_rounding_before_secret_multiplication() {
        let layout = GGSWLayout {
            n: Degree(64),
            base2k: Base2K(8),
            dnum: Dnum(3),
            dsize: Dsize(1),
            k_aux: TorusPrecision(8),
            rank: Rank(1),
        };
        let key_layout = GGLWELayout {
            n: Degree(64),
            base2k: Base2K(8),
            dnum: Dnum(4),
            dsize: Dsize(1),
            k_aux: TorusPrecision(8),
            rank_in: Rank(1),
            rank_out: Rank(1),
            stride: 1,
        };
        let mut key = GGLWE::<poulpy_hal::AlignedBuf, i64>::alloc_from_infos(&key_layout);
        let metadata = EncryptionMetadata::from_secret_at(Distribution::TernaryProb(0.5), TorusPrecision(40));
        key.encryption_metadata = Some(metadata);
        let narrowed = ggsw_expansion_metadata(&layout, Some(metadata), TorusPrecision(40), &key).unwrap();
        let same_grid = ggsw_expansion_metadata(&layout, Some(metadata), TorusPrecision(32), &key).unwrap();
        assert!(narrowed.initial_noise_variance() > same_grid.initial_noise_variance());
        assert_eq!(narrowed.fresh_noise().precision(), TorusPrecision(32));
    }

    #[test]
    fn gadget_digit_scale_is_combined_before_underflow() {
        let estimate = FreshNoiseEstimate::new(10.24, TorusPrecision(1000));
        assert_eq!(estimate.variance_at(TorusPrecision(32)), 0.0);
        let combined = variance_with_extra_precision(estimate, TorusPrecision(32), 498);
        assert!(combined > 0.0 && combined.is_finite());
        let near_limit = FreshNoiseEstimate::new(1.0, TorusPrecision(u32::MAX));
        assert_eq!(variance_with_extra_precision(near_limit, TorusPrecision(u32::MAX), 4), 256.0);
    }
}
