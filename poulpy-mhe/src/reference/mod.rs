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
