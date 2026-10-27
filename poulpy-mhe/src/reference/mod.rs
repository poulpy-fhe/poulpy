//! Portable multiparty implementations composed from `poulpy-core` and
//! `poulpy-hal` operations, the default dispatch target of the
//! `impl_<family>_reference_full!` opt-ins. A backend replacing one family may keep
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
