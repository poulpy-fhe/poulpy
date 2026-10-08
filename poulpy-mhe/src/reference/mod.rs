//! Portable multiparty implementations composed from `poulpy-core` and
//! `poulpy-hal` operations, the default dispatch target of the
//! `impl_mhe_*_reference!` opt-ins. A backend replacing one family may keep
//! calling these for the others.
pub mod evaluation_key;
pub mod ggsw;
pub mod keyswitch;
pub mod pat;
pub mod public_key;
pub mod sharing;
pub mod tensor_key;
pub use evaluation_key::*;
pub use ggsw::*;
pub use keyswitch::*;
pub use pat::*;
pub use public_key::*;
pub use sharing::*;
pub use tensor_key::*;

/// Combining seeded shares combines independently sampled secret summands.
/// An untagged share leaves the aggregate untagged.
pub(crate) fn aggregate_metadata(
    left: Option<poulpy_core::ComponentNoise>,
    right: Option<poulpy_core::ComponentNoise>,
) -> Option<poulpy_core::ComponentNoise> {
    match (left, right) {
        (Some(left), Some(right)) => {
            assert!(
                left.secret_distribution().base() == right.secret_distribution().base(),
                "invalid aggregation: secret distributions differ"
            );
            Some(left.aggregate(&right))
        }
        _ => None,
    }
}

/// Summing fresh encryptions under one collective public key keeps its secret
/// distribution. Centered independent ephemerals remove the shared-key error
/// covariance. For noncentered ephemerals, the component estimates lack that
/// covariance decomposition, so the triangle bound safely combines spreads.
/// An untagged share leaves the aggregate untagged.
pub(crate) fn aggregate_common_key_metadata(
    left: Option<poulpy_core::ComponentNoise>,
    right: Option<poulpy_core::ComponentNoise>,
    n: usize,
) -> Option<poulpy_core::ComponentNoise> {
    match (left, right) {
        (Some(left), Some(right)) => {
            assert!(left.same_secret(&right), "invalid aggregation: output key provenance differs");
            assert_eq!(left.rank(), right.rank(), "invalid aggregation: component ranks differ");
            let centered = left.secret_distribution().coefficient_mean(n) == Some(0.0);
            let components = left
                .components()
                .iter()
                .zip(right.components().iter())
                .map(|(noise, right)| {
                    let right_variance = right.variance_at(noise.precision());
                    let variance = if centered {
                        noise.variance() + right_variance
                    } else {
                        (noise.std_dev() + right_variance.sqrt()).powi(2)
                    };
                    poulpy_core::FreshNoiseEstimate::new(variance, noise.precision())
                })
                .collect();
            Some(left.with_components(components))
        }
        _ => None,
    }
}

/// Common-key aggregation uses the provenance's base law to model ephemeral
/// correlations, so reject a separately retagged public key before share work.
pub(crate) fn assert_public_key_distribution<BE, K>(pk: &K)
where
    BE: poulpy_hal::layouts::Backend,
    K: poulpy_core::layouts::GLWEPublicKeyPreparedToBackendRef<BE> + poulpy_core::layouts::GLWEInfos,
{
    use poulpy_core::GetDistribution;
    if let Some(metadata) = pk.noise() {
        let base = metadata.secret_distribution().base();
        if base != poulpy_core::Distribution::NONE {
            assert!(
                *pk.to_backend_ref().dist() == base,
                "invalid public key: ephemeral distribution differs from its secret provenance"
            );
        }
    }
}

#[cfg(test)]
mod fresh_noise_tests {
    use super::*;
    use poulpy_core::{ComponentNoise, Distribution, FreshNoiseEstimate, layouts::TorusPrecision};

    #[test]
    fn common_key_sum_tracks_error_independently_from_secret_parties_and_precision() {
        let single = ComponentNoise::from_secret_at(Distribution::TernaryProb(0.5), TorusPrecision(30), 1);
        let collective = single.aggregate(&single).aggregate(&single);
        let left = collective.with_components(vec![
            FreshNoiseEstimate::new(9.0, TorusPrecision(30)),
            FreshNoiseEstimate::new(4.0, TorusPrecision(30)),
        ]);
        let right = collective.with_components(vec![
            FreshNoiseEstimate::new(36.0, TorusPrecision(31)),
            FreshNoiseEstimate::new(16.0, TorusPrecision(31)),
        ]);
        let sum = aggregate_common_key_metadata(Some(left), Some(right), 64).unwrap();
        assert_eq!(sum.parties(), 3);
        assert_eq!(sum.body().precision(), TorusPrecision(30));
        assert_eq!(sum.body().variance(), 18.0);
        assert_eq!(sum.body().variance_at(TorusPrecision(32)), 288.0);
        assert_eq!(sum.components()[1].variance(), 8.0);
        assert_eq!(sum.phase_noise(64).variance(), 18.0 + 64.0 * 1.5 * 8.0);
    }

    #[test]
    fn shared_public_key_binary_means_need_a_covariance_bound() {
        let one = ComponentNoise::from_secret_at(Distribution::BinaryProb(0.5), TorusPrecision(30), 1);
        let collective = one.aggregate(&one).aggregate(&one);
        let left = collective.with_components(vec![
            FreshNoiseEstimate::new(9.0, TorusPrecision(30)),
            FreshNoiseEstimate::new(4.0, TorusPrecision(30)),
        ]);
        let right = collective.with_components(vec![
            FreshNoiseEstimate::new(36.0, TorusPrecision(31)),
            FreshNoiseEstimate::new(16.0, TorusPrecision(31)),
        ]);
        let sum = aggregate_common_key_metadata(Some(left.clone()), Some(right), 64).unwrap();
        assert_eq!(sum.parties(), 3);
        assert_eq!(sum.body().variance(), 36.0);
        assert_eq!(sum.components()[1].variance(), 16.0);
        // Adding another fresh share remains compatible with the aggregate,
        // even though its effective variance now differs from a single share.
        let three = aggregate_common_key_metadata(Some(sum), Some(left), 64).unwrap();
        assert_eq!(three.parties(), 3);
        assert_eq!(three.body().variance(), 81.0);
        assert_eq!(three.components()[1].variance(), 36.0);
    }
}
