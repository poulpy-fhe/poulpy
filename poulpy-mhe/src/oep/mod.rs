//! Backend extension points for multiparty operations.
//!
//! A backend selects the reference implementation with the `impl_mhe_*_reference!`
//! opt-ins or implements an `*Impl` contract itself. An override must compute the
//! same result as the reference and pass parity against a validated backend; the
//! parity suite arrives with the first override.
pub(crate) mod derived;
pub mod evaluation_key;
pub mod keyswitch;
pub mod pat;
pub mod public_key;
pub mod tensor_key;
pub use evaluation_key::*;
pub use keyswitch::*;
pub use pat::*;
pub use public_key::*;
pub use tensor_key::*;

/// Selects every reference multiparty implementation for `$be`.
#[macro_export]
macro_rules! impl_mhe_reference_full {
    ($be:ty) => {
        $crate::impl_mhe_pat_reference!($be);
        $crate::impl_mhe_evaluation_key_reference!($be);
        $crate::impl_mhe_keyswitch_reference!($be);
        $crate::impl_mhe_public_key_reference!($be);
        $crate::impl_mhe_tensor_key_reference!($be);
    };
}
