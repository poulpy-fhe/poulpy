//! Backend extension points for multiparty operations.
//!
//! # Noise metadata
//!
//! Outputs follow the [core rule](poulpy_core::oep#noise-metadata). Shares,
//! aggregates and finalized keys are fresh constructions and record the estimates
//! of the share metadata contract in `docs/mhe-contracts.md`.
//!
//! A backend selects the reference implementation with the
//! `impl_<family>_reference_full!` opt-ins or implements an `*Impl` contract
//! itself. An override must compute the same result as the reference and pass
//! [`mhe_parity_test_suite!`](crate::mhe_parity_test_suite) against a
//! validated backend.
pub(crate) mod derived;
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

/// Selects every reference multiparty implementation for `$be`.
#[macro_export]
macro_rules! impl_mhe_reference_full {
    ($be:ty) => {
        $crate::impl_glwe_pat_compressed_reference_full!($be);
        $crate::impl_gglwe_pat_compressed_reference_full!($be);
        $crate::impl_gglwe_pat_reference_full!($be);
        $crate::impl_glwe_switching_key_mhe_protocol_reference_full!($be);
        $crate::impl_glwe_automorphism_key_mhe_protocol_reference_full!($be);
        $crate::impl_ggsw_mhe_protocol_reference_full!($be);
        $crate::impl_glwe_private_keyswitch_mhe_protocol_reference_full!($be);
        $crate::impl_glwe_public_keyswitch_mhe_protocol_reference_full!($be);
        $crate::impl_glwe_public_key_mhe_protocol_reference_full!($be);
        $crate::impl_glwe_enc_to_share_mhe_protocol_reference_full!($be);
        $crate::impl_glwe_share_to_enc_mhe_protocol_reference_full!($be);
        $crate::impl_glwe_tensor_key_mhe_protocol_reference_full!($be);
    };
}
