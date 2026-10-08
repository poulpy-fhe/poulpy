//! Fresh encryption error estimates. These do not track evaluated noise.

use poulpy_hal::layouts::{Backend, ProductMoments, Ring};

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

/// Fresh errors are drawn one limb past the output, at most at the key's
/// precision, and the product uses the key's limbs down to that grid.
fn sample_precision(output: TorusPrecision, key: TorusPrecision, base2k: usize) -> TorusPrecision {
    let base2k = base2k as u64;
    TorusPrecision(((u64::from(output.0).div_ceil(base2k) + 1) * base2k).min(u64::from(key.0)) as u32)
}

/// Second moment, averaged over coefficients, that the key-tail and rounding
/// errors add to the phase beyond `(r*w*q_u*q_t + q_r) * (1 + r*w*q_S)`, the
/// sum of their per-component terms. `u` and `s` are the ephemeral and secret
/// `(mean, second moment)`, `tail` the `(mean, variance)` of a dropped-limb
/// tail and `mu_r` the output rounding's mean. Each error is split into its
/// mean and an independent fluctuation; products of distinct fluctuations are
/// uncorrelated, and `moments` gives each one's average weight.
fn coherent_excess(
    m: &ProductMoments,
    rank: f64,
    (mu_u, q_u): (f64, f64),
    (mu_s, q_s): (f64, f64),
    (tau, v_t): (f64, f64),
    mu_r: f64,
) -> f64 {
    let (r, w) = (rank, m.weight);
    let (v_u, v_s, q_t) = (q_u - mu_u * mu_u, q_s - mu_s * mu_s, v_t + tau * tau);
    // Triple products of fluctuations, against their per-component product
    // `w^2`: zero for centered laws on the negacyclic ring.
    let triple = (m.triple_weight - w * w) * v_u * v_s * q_t + m.triple_weight * v_t * (v_u * mu_s * mu_s + v_s * mu_u * mu_u)
        - w * w * q_t * (v_u * mu_s * mu_s + v_s * mu_u * mu_u + mu_u * mu_u * mu_s * mu_s);
    // A fluctuating `u` or `S` against the means of the other factors.
    let single = mu_s * v_u * tau * tau * (2.0 * m.sum_cross_weight + r * mu_s * m.sum_weight)
        + mu_u * v_s * tau * (r * mu_u * tau * m.sum_weight + 2.0 * mu_r * m.sum_cross_weight)
        + v_t * (mu_u * mu_s).powi(2) * m.sum_weight;
    // The phase mean `-(a (1 1) + b (1 1 1))`, the body rounding's mean included.
    let (a, b) = (r * (mu_u * tau + mu_r * mu_s), r * r * mu_u * mu_s * tau);
    let mean = a * a * m.sum_square
        + 2.0 * a * b * m.sum_triple_sum
        + b * b * m.triple_sum_square
        + 2.0 * mu_r * (a * m.sum + b * m.triple_sum);
    r * r * triple + r * r * single + mean - r * w * (mu_u * mu_u * tau * tau + mu_s * mu_s * mu_r * mu_r)
}

fn public_key_noise<R: GLWEInfos>(
    metadata: Option<ComponentNoise>,
    res: &R,
    pk_precision: TorusPrecision,
    ephemeral: Distribution,
    moments: ProductMoments,
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
    let rank_n = rank as f64 * moments.weight;
    let base2k = res.base2k().as_usize();
    let sample = sample_precision(res.k(), pk_precision, base2k);
    let ephemeral_moments = base_moments(ephemeral, n);
    let ephemeral_fold = scaled_variance(ephemeral_moments.map_or(f64::INFINITY, |(_, second)| second), rank_n);
    let secret = metadata.secret_distribution();
    let secret_fold = scaled_variance(secret.coefficient_second_moment(n).unwrap_or(f64::INFINITY), rank_n);
    let secret_mean = secret.coefficient_mean(n);
    let correlated_masks = secret_mean != Some(0.0) && metadata.masks().iter().any(|term| term.variance() != 0.0);
    let scale = (-2.0 * f64::from(sample.0 - res.k().0)).exp2();
    // Dropped key limbs leave on every key coefficient a balanced-digit tail,
    // uniform over one ulp of the sampling grid, of mean `-2^-(base2k+1) / (1 - 2^-base2k)`.
    let tail = if sample < pk_precision {
        let mean = -0.5 * (-(base2k as f64)).exp2() / (1.0 - (-(base2k as f64)).exp2());
        (mean * scale.sqrt(), scale / 12.0)
    } else {
        (0.0, 0.0)
    };
    // Normalizing to the output drops `d` uniform bits: variance `(1 - 4^-d)/12`
    // and, ties rounding up, mean `-2^-(d+1)`.
    let rounding = if sample > res.k() {
        (-0.5 * scale.sqrt(), (1.0 - scale) / 12.0)
    } else {
        (0.0, 0.0)
    };
    let excess = if tail == (0.0, 0.0) && rounding == (0.0, 0.0) {
        0.0
    } else {
        match (ephemeral_moments, secret_mean, secret.coefficient_second_moment(n)) {
            (Some(u), Some(mean), Some(second)) => {
                coherent_excess(&moments, rank as f64, u, (mean, second), tail, rounding.0) / (1.0 + secret_fold)
            }
            _ => f64::INFINITY,
        }
    };
    let tail_second = tail.0 * tail.0 + tail.1;
    let rounding_second = rounding.0 * rounding.0 + rounding.1 + excess;
    let fresh = FreshNoiseEstimate::new(Noise::ENCRYPTION.variance(), sample).variance_at(res.k());
    let body = match body_noise {
        PublicKeyBodyNoise::Sampled => fresh,
        PublicKeyBodyNoise::Omitted => 0.0,
        PublicKeyBodyNoise::Flood(noise) => noise.variance(),
    };
    let components = metadata
        .components()
        .iter()
        .enumerate()
        .map(|(index, component)| {
            let inherited = if correlated_masks && index == 0 {
                f64::INFINITY
            } else {
                scaled_variance(component.variance_at(res.k()) + tail_second, ephemeral_fold)
            };
            let fresh = if index == 0 { body } else { fresh };
            FreshNoiseEstimate::new(inherited + fresh + rounding_second, res.k())
        })
        .collect();
    Some(metadata.with_components(components))
}

/// Precision at which a public-key encryption into `res` draws its fresh errors:
/// one limb past the output, at most the key's. The product uses the key's
/// leading `ceil(k_sample / base2k)` limbs.
pub fn public_key_sample_precision<BE, R, K>(res: &R, pk: &K) -> TorusPrecision
where
    BE: Backend,
    R: GLWEInfos,
    K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
{
    sample_precision(res.k(), pk.k(), res.base2k().as_usize())
}

/// Output metadata of a public-key encryption drawing its fresh errors at
/// [`public_key_sample_precision`], normalized once to the output's `k`.
/// Returns `None` for a key without metadata.
pub fn public_key_encryption_noise<BE, R, K>(res: &R, pk: &K, body_noise: PublicKeyBodyNoise) -> Option<ComponentNoise>
where
    BE: Backend,
    R: GLWEInfos,
    K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
{
    public_key_noise(
        pk.noise(),
        res,
        pk.k(),
        *pk.to_backend_ref().dist(),
        <BE::Ring as Ring>::product_moments(res.n().as_usize()),
        body_noise,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layouts::{Base2K, Degree, GLWELayout, Rank};
    use poulpy_hal::layouts::{ConjugateInvariant, Standard};

    fn public_key_noise<R: GLWEInfos>(
        metadata: Option<ComponentNoise>,
        res: &R,
        pk_precision: TorusPrecision,
        ephemeral: Distribution,
        body_noise: PublicKeyBodyNoise,
    ) -> Option<ComponentNoise> {
        super::public_key_noise(
            metadata,
            res,
            pk_precision,
            ephemeral,
            Standard::product_moments(res.n().as_usize()),
            body_noise,
        )
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
        // The 36-bit key lies within one limb past the output: no limb is dropped.
        let sigma2 = Noise::ENCRYPTION.variance() / 256.0;
        // Three binary parties have E[S²] = 3, while a new ephemeral has E[u²] = 1/2.
        // Four dropped bits round with variance (1 - 4^-4)/12 and mean -2^-5;
        // E[S] = 3/2 adds the means up along (1 1), of mean 1 and mean square 1366.
        let rounding = (1.0 - 1.0 / 256.0) / 12.0 + 1.0 / 1024.0;
        let excess = (4.0 * 2.25 * 1366.0 + 4.0 * 1.5 - 128.0 * 2.25) / 1024.0;
        let component_rounding = rounding + excess / 385.0;
        let inherited = 128.0 * 0.5 * (3.0 * Noise::ENCRYPTION.variance() / 256.0);
        let expected = excess + inherited + 385.0 * sigma2 + 385.0 * rounding;
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
    fn fresh_errors_are_drawn_one_limb_past_the_output() {
        let ternary = Distribution::TernaryProb(0.5);
        let k = TorusPrecision(30);
        assert_eq!(sample_precision(k, k, 8), k);
        assert_eq!(sample_precision(k, TorusPrecision(35), 8), TorusPrecision(35));
        assert_eq!(sample_precision(k, TorusPrecision(64), 8), TorusPrecision(40));
        let max = TorusPrecision(u32::MAX);
        assert_eq!(sample_precision(TorusPrecision(u32::MAX - 1), max, 52), max);

        // Key limbs past 40 bits are dropped: each component inherits their tail.
        let layout = GLWELayout {
            n: Degree(64),
            base2k: Base2K(8),
            k: TorusPrecision(32),
            rank: Rank(1),
        };
        let key = TorusPrecision(64);
        let metadata = ComponentNoise::from_secret_at(ternary, key, 1);
        let noise = public_key_noise(Some(metadata), &layout, key, ternary, PublicKeyBodyNoise::Sampled).unwrap();
        // The tail is uniform over one ulp of the 40-bit grid, of mean
        // -2^-9 / (1 - 2^-8) there; rounding to 32 bits drops 8 uniform bits.
        let unit = (-16.0_f64).exp2();
        let tail = (0.5 / 256.0 / (1.0 - 1.0 / 256.0_f64)).powi(2) * unit + unit / 12.0;
        let rounding = unit / 4.0 + (1.0 - unit) / 12.0;
        let fresh = Noise::ENCRYPTION.variance() * unit;
        let body = 32.0 * (Noise::ENCRYPTION.variance() * (-64.0_f64).exp2() + tail) + fresh + rounding;
        assert!((noise.body().variance() - body).abs() <= body * 1e-12);
        let mask = 32.0 * tail + fresh + rounding;
        assert!((noise.components()[1].variance() - mask).abs() <= mask * 1e-12);

        // A binary key of equal second moments adds the coherent means, spread
        // evenly: the body still exceeds a mask by its inherited key error alone.
        let binary = Distribution::BinaryProb(0.5);
        let metadata = ComponentNoise::from_secret_at(binary, key, 1);
        let coherent = public_key_noise(Some(metadata), &layout, key, binary, PublicKeyBodyNoise::Sampled).unwrap();
        let inherited = 32.0 * Noise::ENCRYPTION.variance() * (-64.0_f64).exp2();
        let difference = coherent.body().variance() - coherent.components()[1].variance();
        assert!((difference - inherited).abs() <= 1e-12);
        assert!(coherent.phase_noise(64).variance() > noise.phase_noise(64).variance());
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
    fn invariant_ring_uses_its_average_weight() {
        let layout = GLWELayout {
            n: Degree(256),
            base2k: Base2K(8),
            k: TorusPrecision(32),
            rank: Rank(1),
        };
        let base = Distribution::TernaryProb(0.5);
        let metadata = ComponentNoise::from_secret_at(base, layout.k, 1);
        let moments = ConjugateInvariant::product_moments(256);
        let noise = super::public_key_noise(Some(metadata), &layout, layout.k, base, moments, PublicKeyBodyNoise::Sampled);
        let sigma2 = Noise::ENCRYPTION.variance();
        // The average weight is 2n - 1/n; coefficient zero alone weighs about 4n.
        let body = (0.5 * (512.0 - 1.0 / 256.0) + 1.0) * sigma2;
        let noise = noise.unwrap();
        assert!((noise.body().variance() - body).abs() < 1e-9);
        let variance = noise.weighted_phase_noise(256, 512).variance();
        assert!((variance - body - 256.0 * sigma2).abs() < 1e-9);
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
        // Sampled one limb past the output.
        assert!(
            (ordinary.phase_noise(layout.n.as_usize()).variance() - masks_variance - Noise::ENCRYPTION.variance() / 65536.0)
                .abs()
                < 1e-9
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
