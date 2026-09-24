//! Explicit backend extension points for binary-FHE operations.
//!
//! Backend implementations select callable lower-layer reference algorithms or
//! inherit crate-private same-layer derived defaults. An override must reproduce
//! the reference circuit and pass parity against a caller-selected validated
//! backend. Execution and scratch sizing must be selected together.
pub mod bdd;
pub mod blind_rotation;
pub mod circuit_bootstrapping;
pub(crate) mod derived;
pub use bdd::*;
pub use blind_rotation::*;
pub use circuit_bootstrapping::*;

/// Selects the supplied binary-FHE circuits for an explicitly opted-in backend.
///
/// Family macros can be selected separately when replacing an operation. The
/// optional parallel schedule is a backend decision; it does not alter the
/// canonical reference circuit or the CGGI algorithm marker.
#[macro_export]
macro_rules! impl_bin_fhe_reference_full {
    ($be:ty) => {
        $crate::impl_bin_fhe_blind_rotation_reference!($be);
        $crate::impl_bin_fhe_reference_full!(@remaining $be);
    };
    ($be:ty, scheduling = parallel) => {
        $crate::impl_bin_fhe_blind_rotation_reference!($be, scheduling = parallel);
        $crate::impl_bin_fhe_reference_full!(@remaining $be);
    };
    (@remaining $be:ty) => {
        $crate::impl_bin_fhe_circuit_bootstrapping_reference!($be, $crate::blind_rotation::CGGI);
        $crate::impl_bin_fhe_bdd_reference!($be, $crate::blind_rotation::CGGI);
    };
}
pub use crate::impl_bin_fhe_reference_full;
