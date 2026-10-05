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
pub(crate) fn aggregate_metadata(
    left: Option<poulpy_core::EncryptionMetadata>,
    right: Option<poulpy_core::EncryptionMetadata>,
) -> Option<poulpy_core::EncryptionMetadata> {
    match (left, right) {
        (Some(left), Some(right)) => {
            assert!(
                left.secret_distribution().base() == right.secret_distribution().base(),
                "invalid aggregation: secret distributions differ"
            );
            Some(left.aggregate(right))
        }
        (None, None) => None,
        _ => panic!("invalid aggregation: encryption provenance differs"),
    }
}

/// Summing fresh encryptions under one collective public key keeps its secret
/// distribution. Centered independent ephemerals remove the shared-key error
/// covariance. For noncentered ephemerals, the scalar estimate lacks that
/// covariance decomposition, so the triangle bound safely combines spreads.
pub(crate) fn aggregate_common_key_metadata(
    left: Option<poulpy_core::EncryptionMetadata>,
    right: Option<poulpy_core::EncryptionMetadata>,
    n: usize,
) -> Option<poulpy_core::EncryptionMetadata> {
    match (left, right) {
        (Some(left), Some(right)) => {
            assert!(left.same_secret(&right), "invalid aggregation: output key provenance differs");
            let noise = left.fresh_noise();
            let right_variance = right.fresh_noise().variance_at(noise.precision());
            let variance = if left.secret_distribution().coefficient_mean(n) == Some(0.0) {
                noise.variance() + right_variance
            } else {
                (noise.std_dev() + right_variance.sqrt()).powi(2)
            };
            Some(left.with_fresh_noise(poulpy_core::FreshNoiseEstimate::new(variance, noise.precision())))
        }
        (None, None) => None,
        _ => panic!("invalid aggregation: output key provenance differs"),
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
    if let Some(metadata) = pk.encryption_metadata() {
        let base = metadata.secret_distribution().base();
        if base != poulpy_core::Distribution::NONE {
            assert!(
                *pk.to_backend_ref().dist() == base,
                "invalid public key: ephemeral distribution differs from its secret provenance"
            );
        }
    }
}

/// A flood's effective centered variance in units of its integer sampling grid.
/// Gaussian sigma is its variance parameter, an upper estimate after cutoff;
/// uniform noise has its exact discrete variance and a separate mean of -1/2.
pub(crate) fn flood_variance(noise: poulpy_core::Noise) -> f64 {
    match noise {
        poulpy_core::Noise::Gaussian { sigma, cutoff_factor } => {
            if cutoff_factor == 0 {
                0.0
            } else {
                sigma * sigma
            }
        }
        poulpy_core::Noise::Uniform { bits } => (2.0 * bits as f64 - 4.0).exp2() * (4.0 / 3.0) - 1.0 / 12.0,
    }
}

#[cfg(test)]
mod fresh_noise_tests {
    use super::*;
    use poulpy_core::{Distribution, EncryptionMetadata, FreshNoiseEstimate, Noise, layouts::TorusPrecision};

    #[test]
    fn common_key_sum_tracks_error_independently_from_secret_parties_and_precision() {
        let single = EncryptionMetadata::from_secret_at(Distribution::TernaryProb(0.5), TorusPrecision(30));
        let collective = single.aggregate(single).aggregate(single);
        let left = collective.with_fresh_noise(FreshNoiseEstimate::new(9.0, TorusPrecision(30)));
        let right = collective.with_fresh_noise(FreshNoiseEstimate::new(36.0, TorusPrecision(31)));
        let sum = aggregate_common_key_metadata(Some(left), Some(right), 64).unwrap();
        assert_eq!(sum.parties(), 3);
        assert_eq!(sum.fresh_noise().precision(), TorusPrecision(30));
        assert_eq!(sum.fresh_noise().variance(), 18.0);
        assert_eq!(sum.fresh_noise().variance_at(TorusPrecision(32)), 288.0);
    }

    #[test]
    fn shared_public_key_binary_means_need_a_covariance_bound() {
        let one = EncryptionMetadata::from_secret_at(Distribution::BinaryProb(0.5), TorusPrecision(30));
        let collective = one.aggregate(one).aggregate(one);
        let left = collective.with_fresh_noise(FreshNoiseEstimate::new(9.0, TorusPrecision(30)));
        let right = collective.with_fresh_noise(FreshNoiseEstimate::new(36.0, TorusPrecision(31)));
        let sum = aggregate_common_key_metadata(Some(left), Some(right), 64).unwrap();
        assert_eq!(sum.parties(), 3);
        assert_eq!(sum.fresh_noise().variance(), 36.0);
        // Adding another fresh share remains compatible with the aggregate,
        // even though its effective variance now differs from a single share.
        let three = aggregate_common_key_metadata(Some(sum), Some(left), 64).unwrap();
        assert_eq!(three.parties(), 3);
        assert_eq!(three.fresh_noise().variance(), 81.0);
    }

    #[test]
    fn flood_estimates_use_the_requested_distribution() {
        assert_eq!(flood_variance(Noise::Uniform { bits: 4 }), 21.25);
        assert!(flood_variance(Noise::Uniform { bits: 513 }).is_finite());
        assert_eq!(flood_variance(Noise::Uniform { bits: 514 }), f64::INFINITY);
        assert_eq!(
            flood_variance(Noise::Gaussian {
                sigma: 1024.0,
                cutoff_factor: 6
            }),
            1048576.0
        );
    }
}
