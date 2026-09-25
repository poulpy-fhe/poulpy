//! Backend extension points for multiparty operations.
//!
//! A backend selects the reference implementation with the `impl_mhe_*_reference!`
//! opt-ins or implements an `*Impl` contract itself. An override must compute the
//! same result as the reference and pass parity against a validated backend.
pub mod pat;
pub use pat::*;

/// Selects every reference multiparty implementation for `$be`.
#[macro_export]
macro_rules! impl_mhe_reference_full {
    ($be:ty) => {
        $crate::impl_mhe_pat_reference!($be);
    };
}
