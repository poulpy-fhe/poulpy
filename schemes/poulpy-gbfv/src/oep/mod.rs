//! Backend contracts selected by the public GBFV API through delegates.
//!
//! Delegates only forward. A backend explicitly implements each `*Impl` family;
//! the `impl_gbfv_<family>_reference_full!` macros wire the algorithms of
//! [`crate::reference`]. An override must reproduce the reference circuit.

mod ntt;

pub use ntt::{GBFVNttImpl, impl_gbfv_ntt_reference_full};
