//! Fresh encryption error estimates. These do not track evaluated noise.

use poulpy_hal::layouts::{Backend, Ring};

use crate::{
    ComponentNoise, Distribution, FreshNoiseEstimate, GetDistribution, Noise, SecretDistribution,
    layouts::{GLWEInfos, TorusPrecision, prepared::GLWEPublicKeyPreparedToBackendRef},
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

fn scaled_variance(variance: f64, factor: f64) -> f64 {
    if factor == 0.0 || variance == 0.0 {
        0.0
    } else {
        variance * factor
    }
}

/// Body error of a public-key encryption.
#[derive(Clone, Copy)]
pub enum PublicKeyBodyNoise {
    /// Fresh body error, drawn with the masks.
    Sampled,
    /// No body error.
    Omitted,
    /// This flood in place of the body error.
    Flood(Noise),
}

fn public_key_noise<R: GLWEInfos>(
    metadata: Option<ComponentNoise>,
    res: &R,
    pk_precision: TorusPrecision,
    ephemeral: Distribution,
    product_weight: usize,
    body_noise: PublicKeyBodyNoise,
) -> Option<ComponentNoise> {
    assert!(pk_precision >= res.k(), "invalid public key: less precise than the output");
    let metadata = metadata?;
    if metadata.secret_distribution().base() != Distribution::NONE {
        assert_eq!(
            ephemeral,
            metadata.secret_distribution().base(),
            "invalid public key: ephemeral distribution differs from its secret provenance"
        );
    }
    let n = res.n().as_usize();
    let rank = res.rank().as_usize();
    assert_eq!(
        metadata.rank(),
        rank,
        "invalid public key: noise component count differs from its rank"
    );
    let rank_n = rank as f64 * product_weight as f64;
    let ephemeral_fold = scaled_variance(base_moments(ephemeral, n).map_or(f64::INFINITY, |(_, second)| second), rank_n);
    let secret_fold = scaled_variance(
        metadata
            .secret_distribution()
            .coefficient_second_moment(n)
            .unwrap_or(f64::INFINITY),
        rank_n,
    );
    let secret_mean = metadata.secret_distribution().coefficient_mean(n);
    let correlated_masks = secret_mean != Some(0.0) && metadata.masks().iter().any(|term| term.variance() != 0.0);
    let fresh = Noise::ENCRYPTION.variance();
    let body = match body_noise {
        PublicKeyBodyNoise::Sampled => fresh,
        PublicKeyBodyNoise::Omitted => 0.0,
        PublicKeyBodyNoise::Flood(noise) => noise.variance(),
    };
    // The key product is normalized once from the key's precision to the
    // output's: a half-ulp error per coefficient, added independently as in
    // the core noise models. Ties round up; noncentered secrets can add that
    // bias coherently.
    let rounding = if pk_precision > res.k() {
        let bias = secret_mean.map_or(f64::INFINITY, |mean| {
            if mean == 0.0 {
                0.0
            } else {
                (-(f64::from(pk_precision.0 - res.k().0) + 1.0)).exp2() * (1.0 + rank_n * mean.abs())
            }
        });
        if bias.is_finite() {
            0.25 + bias.powi(2) / (1.0 + secret_fold)
        } else {
            f64::INFINITY
        }
    } else {
        0.0
    };
    let components = metadata
        .components()
        .iter()
        .enumerate()
        .map(|(index, component)| {
            let inherited = if correlated_masks && index == 0 {
                f64::INFINITY
            } else {
                scaled_variance(component.variance_at(res.k()), ephemeral_fold)
            };
            let fresh = if index == 0 { body } else { fresh };
            FreshNoiseEstimate::new(inherited + fresh + rounding, res.k())
        })
        .collect();
    Some(metadata.with_components(components))
}

/// Output metadata of a public-key encryption, which draws its fresh errors at
/// the output's `k` and normalizes the full-precision key product once.
/// Returns `None` for a key without metadata.
pub fn public_key_encryption_noise<BE, R, K>(res: &R, pk: &K, body_noise: PublicKeyBodyNoise) -> Option<ComponentNoise>
where
    BE: Backend,
    R: GLWEInfos,
    K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
{
    let ring_factor = <BE::Ring as Ring>::CYCLOTOMIC_ORDER_FACTOR as usize / 2;
    let product_weight = res.n().as_usize() * ring_factor * ring_factor;
    public_key_noise(
        pk.noise(),
        res,
        pk.k(),
        *pk.to_backend_ref().dist(),
        product_weight,
        body_noise,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layouts::{Base2K, Degree, GLWELayout, Rank};

    fn public_key_noise<R: GLWEInfos>(
        metadata: Option<ComponentNoise>,
        res: &R,
        pk_precision: TorusPrecision,
        ephemeral: Distribution,
        body_noise: PublicKeyBodyNoise,
    ) -> Option<ComponentNoise> {
        super::public_key_noise(metadata, res, pk_precision, ephemeral, res.n().as_usize(), body_noise)
    }

    #[test]
    fn collective_binary_moments_include_the_mean() {
        let one = ComponentNoise::from_secret(Distribution::BinaryProb(0.5), 1);
        let three = one.aggregate(&one).aggregate(&one).secret_distribution();
        assert_eq!(three.coefficient_mean(64), Some(1.5));
        assert_eq!(three.coefficient_second_moment(64), Some(3.0));
        let ternary = ComponentNoise::from_secret(Distribution::TernaryFixed(16), 1);
        assert_eq!(
            ternary
                .aggregate(&ternary)
                .secret_distribution()
                .coefficient_second_moment(64),
            Some(0.5)
        );
        let block = ComponentNoise::from_secret(Distribution::BinaryBlock(3), 1);
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
        let one = ComponentNoise::from_secret_at(base, TorusPrecision(36), layout.rank.as_usize());
        let pk = one.aggregate(&one).aggregate(&one);
        let result = public_key_noise(
            Some(pk.clone()),
            &layout,
            TorusPrecision(36),
            base,
            PublicKeyBodyNoise::Sampled,
        )
        .unwrap();
        let sigma2 = Noise::ENCRYPTION.variance();
        // Three binary parties have E[S²] = 3, while a new ephemeral has E[u²] = 1/2.
        let bias_squared = (193.0_f64 / 32.0).powi(2);
        let component_rounding = 0.25 + bias_squared / 385.0;
        let inherited = 128.0 * 0.5 * (3.0 * sigma2 / 256.0);
        let expected = bias_squared + inherited + 385.0 * sigma2 + 385.0 / 4.0;
        assert_eq!(result.parties(), 3);
        assert_eq!(result.components().len(), layout.rank.as_usize() + 1);
        assert!((result.body().variance() - (inherited + sigma2 + component_rounding)).abs() < 1e-12);
        for component in &result.components()[1..] {
            assert_eq!(component.variance(), sigma2 + component_rounding);
        }
        assert_eq!(result.phase_noise(layout.n.as_usize()).precision(), layout.k);
        assert!((result.phase_noise(layout.n.as_usize()).variance() - expected).abs() < expected * 1e-15);
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
        let pk = ComponentNoise::from_secret_at(base, layout.k, layout.rank.as_usize());
        let without_body = public_key_noise(Some(pk.clone()), &layout, layout.k, base, PublicKeyBodyNoise::Omitted).unwrap();
        let smudged = public_key_noise(
            Some(pk.clone()),
            &layout,
            layout.k,
            base,
            PublicKeyBodyNoise::Flood(Noise::Uniform { bits: 4 }),
        )
        .unwrap();
        assert!(smudged.components()[1..] == without_body.components()[1..]);
        assert_eq!(smudged.body().variance() - without_body.body().variance(), 21.25);
        assert_eq!(
            smudged.phase_noise(layout.n.as_usize()).variance() - without_body.phase_noise(layout.n.as_usize()).variance(),
            21.25
        );
        let unknown = ComponentNoise::from_secret_at(Distribution::NONE, layout.k, layout.rank.as_usize());
        let conservative = public_key_noise(Some(unknown.clone()), &layout, layout.k, base, PublicKeyBodyNoise::Sampled).unwrap();
        assert_eq!(conservative.secret_distribution(), unknown.secret_distribution());
        assert_eq!(conservative.phase_noise(layout.n.as_usize()).variance(), f64::INFINITY);
        assert!(Noise::Uniform { bits: 512 }.variance().is_finite());
        assert!(Noise::Uniform { bits: 513 }.variance().is_finite());
        assert_eq!(Noise::Uniform { bits: 514 }.variance(), f64::INFINITY);
    }

    #[test]
    fn equal_precision_noise_retains_raw_body_and_mask_variances() {
        let layout = GLWELayout {
            n: Degree(2048),
            base2k: Base2K(12),
            k: TorusPrecision(49),
            rank: Rank(1),
        };
        let base = Distribution::TernaryProb(0.5);
        let metadata = ComponentNoise::from_secret_at(base, layout.k, layout.rank.as_usize());
        let sigma2 = Noise::ENCRYPTION.variance();
        let inherited = sigma2 * 1024.0;
        let historical = inherited + sigma2 * 1024.0 + sigma2;
        let result = public_key_noise(Some(metadata.clone()), &layout, layout.k, base, PublicKeyBodyNoise::Sampled).unwrap();
        assert_eq!(result.components().len(), 2);
        assert_eq!(result.body().variance(), inherited + sigma2);
        assert_eq!(result.components()[1].variance(), sigma2);
        assert!((result.phase_noise(layout.n.as_usize()).variance() - historical).abs() < historical * 1e-15);
    }

    #[test]
    fn public_key_inherited_mask_components_keep_their_positions() {
        let layout = GLWELayout {
            n: Degree(64),
            base2k: Base2K(8),
            k: TorusPrecision(32),
            rank: Rank(2),
        };
        let base = Distribution::TernaryProb(0.5);
        let metadata = ComponentNoise::from_secret_at(base, layout.k, layout.rank.as_usize()).with_components(
            [4.0, 9.0, 16.0]
                .map(|variance| FreshNoiseEstimate::new(variance, layout.k))
                .to_vec(),
        );
        let result = public_key_noise(Some(metadata), &layout, layout.k, base, PublicKeyBodyNoise::Omitted).unwrap();
        let sigma2 = Noise::ENCRYPTION.variance();
        assert_eq!(result.body().variance(), 64.0 * 4.0);
        assert_eq!(result.components()[1].variance(), 64.0 * 9.0 + sigma2);
        assert_eq!(result.components()[2].variance(), 64.0 * 16.0 + sigma2);
    }

    #[test]
    fn noncentered_inherited_masks_require_an_unbounded_estimate() {
        let layout = GLWELayout {
            n: Degree(64),
            base2k: Base2K(8),
            k: TorusPrecision(32),
            rank: Rank(1),
        };
        let base = Distribution::BinaryProb(0.5);
        let metadata = ComponentNoise::from_secret_at(base, TorusPrecision(64), 1).with_components(vec![
            FreshNoiseEstimate::new(4.0, TorusPrecision(64)),
            FreshNoiseEstimate::new(1.0, TorusPrecision(64)),
        ]);
        let noise = public_key_noise(Some(metadata), &layout, TorusPrecision(64), base, PublicKeyBodyNoise::Sampled).unwrap();
        assert!(noise.phase_noise(64).variance().is_infinite());
    }

    #[test]
    fn invariant_ring_uses_a_bound_for_coefficient_zero() {
        let layout = GLWELayout {
            n: Degree(256),
            base2k: Base2K(8),
            k: TorusPrecision(32),
            rank: Rank(1),
        };
        let base = Distribution::TernaryProb(0.5);
        let metadata = ComponentNoise::from_secret_at(base, layout.k, 1);
        let noise = super::public_key_noise(Some(metadata), &layout, layout.k, base, 4 * 256, PublicKeyBodyNoise::Sampled);
        let variance = noise.unwrap().weighted_phase_noise(256, 4 * 256).variance();
        assert!((variance - 1025.0 * Noise::ENCRYPTION.variance()).abs() < 1e-9);
    }

    #[test]
    fn public_key_noise_handles_unknown_and_extreme_estimates() {
        let layout = GLWELayout {
            n: Degree(64),
            base2k: Base2K(8),
            k: TorusPrecision(32),
            rank: Rank(1),
        };
        let key = TorusPrecision(64);
        let base = Distribution::TernaryProb(0.5);
        assert!(public_key_noise(None, &layout, key, base, PublicKeyBodyNoise::Sampled).is_none());
        let unknown = ComponentNoise::from_secret_at(Distribution::NONE, key, layout.rank.as_usize());
        let noise = public_key_noise(Some(unknown), &layout, key, base, PublicKeyBodyNoise::Sampled).unwrap();
        assert_eq!(noise.phase_noise(layout.n.as_usize()).variance(), f64::INFINITY);
        let infinite = ComponentNoise::from_secret_at(base, key, layout.rank.as_usize())
            .with_body_noise(FreshNoiseEstimate::new(f64::INFINITY, key));
        let noise = public_key_noise(Some(infinite), &layout, key, base, PublicKeyBodyNoise::Sampled).unwrap();
        assert_eq!(noise.body().variance(), f64::INFINITY);
    }

    #[test]
    fn public_key_flood_adds_to_the_body_at_the_output() {
        let layout = GLWELayout {
            n: Degree(64),
            base2k: Base2K(8),
            k: TorusPrecision(32),
            rank: Rank(1),
        };
        let key = TorusPrecision(44);
        let base = Distribution::TernaryProb(0.5);
        let metadata = ComponentNoise::from_secret_at(base, key, layout.rank.as_usize());
        let masks = public_key_noise(Some(metadata.clone()), &layout, key, base, PublicKeyBodyNoise::Omitted).unwrap();
        for noise in [Noise::ENCRYPTION, Noise::Uniform { bits: 40 }] {
            let flood = public_key_noise(Some(metadata.clone()), &layout, key, base, PublicKeyBodyNoise::Flood(noise)).unwrap();
            assert!(flood.components()[1..] == masks.components()[1..]);
            let expected_body = masks.body().variance() + noise.variance();
            assert!((flood.body().variance() - expected_body).abs() <= expected_body * 1e-15);
            let expected = masks.phase_noise(layout.n.as_usize()).variance() + noise.variance();
            assert!((flood.phase_noise(layout.n.as_usize()).variance() - expected).abs() <= expected * 1e-15);
        }
        let ordinary = public_key_noise(Some(metadata.clone()), &layout, key, base, PublicKeyBodyNoise::Sampled).unwrap();
        let masks_variance = masks.phase_noise(layout.n.as_usize()).variance();
        assert!(
            (ordinary.phase_noise(layout.n.as_usize()).variance() - masks_variance - Noise::ENCRYPTION.variance()).abs() < 1e-9
        );
        // Creation precision may differ from the current key grid. Its stored
        // variance must be rescaled using the creation grid, not key.k().
        let older = metadata.with_body_noise(FreshNoiseEstimate::new(
            Noise::ENCRYPTION.variance() / 16.0,
            TorusPrecision(42),
        ));
        assert_eq!(
            public_key_noise(Some(older), &layout, key, base, PublicKeyBodyNoise::Sampled),
            Some(ordinary)
        );
    }
}
