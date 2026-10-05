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

fn public_key_phase_metadata<R: GLWEInfos>(
    metadata: EncryptionMetadata,
    res: &R,
    pk_precision: TorusPrecision,
    ephemeral: Distribution,
    body_noise: Option<Noise>,
) -> EncryptionMetadata {
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
    let fresh_body = body_noise.map_or(0.0, noise_variance);
    // A more precise key needs rounding at the destination grid. As in the
    // core noise models, use at most a half-ulp error per component and add
    // its modeled variance independently; this is an estimate, not a proof
    // that the rounding error and existing phase error are uncorrelated.
    let rounding = if pk_precision > res.k() {
        (1.0 + secret_fold) / 4.0
    } else {
        0.0
    };
    metadata.with_fresh_noise(FreshNoiseEstimate::new(
        inherited + fresh_mask + fresh_body + rounding,
        res.k(),
    ))
}

/// Estimate the phase error of a fresh public-key encryption at the output grid.
pub(crate) fn public_key_encryption_metadata<BE, R, K>(res: &R, pk: &K, body_noise: Option<Noise>) -> Option<EncryptionMetadata>
where
    BE: Backend,
    R: GLWEInfos,
    K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
{
    let metadata = pk.encryption_metadata()?;
    Some(public_key_phase_metadata(
        metadata,
        res,
        pk.k(),
        *pk.to_backend_ref().dist(),
        body_noise,
    ))
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
        let result = public_key_phase_metadata(pk, &layout, TorusPrecision(36), base, Some(Noise::ENCRYPTION));
        let sigma2 = noise_variance(Noise::ENCRYPTION);
        // Three binary parties have E[S²] = 3, while a new ephemeral has E[u²] = 1/2.
        let expected = 128.0 * 0.5 * (3.0 * sigma2 / 256.0) + (1.0 + 128.0 * 3.0) * sigma2 + (1.0 + 128.0 * 3.0) / 4.0;
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
        let without_body = public_key_phase_metadata(pk, &layout, layout.k, base, None);
        let smudged = public_key_phase_metadata(pk, &layout, layout.k, base, Some(Noise::Uniform { bits: 4 }));
        assert_eq!(
            smudged.initial_noise_variance() - without_body.initial_noise_variance(),
            21.25
        );
        let unknown = EncryptionMetadata::from_secret_at(Distribution::NONE, layout.k);
        let conservative = public_key_phase_metadata(unknown, &layout, layout.k, base, Some(Noise::ENCRYPTION));
        assert_eq!(conservative.secret_distribution(), unknown.secret_distribution());
        assert_eq!(conservative.initial_noise_variance(), f64::INFINITY);
        assert!(noise_variance(Noise::Uniform { bits: 512 }).is_finite());
        assert!(noise_variance(Noise::Uniform { bits: 513 }).is_finite());
        assert_eq!(noise_variance(Noise::Uniform { bits: 514 }), f64::INFINITY);
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
        key.metadata = Some(binary);
        let expanded = ggsw_expansion_metadata(&layout, Some(binary), TorusPrecision(32), &key).unwrap();
        assert!(expanded.same_secret(&binary));
        assert_eq!(expanded.initial_noise_variance(), f64::INFINITY);

        let zero = EncryptionMetadata::from_secret_at(Distribution::ZERO, TorusPrecision(32));
        key.metadata = Some(zero);
        assert!(
            ggsw_expansion_metadata(&layout, Some(zero), TorusPrecision(32), &key)
                .unwrap()
                .initial_noise_variance()
                .is_finite()
        );

        layout.rank = Rank(0);
        key.metadata = None;
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
        key.metadata = Some(metadata);
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
