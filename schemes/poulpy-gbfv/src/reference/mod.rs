//! Callable GBFV algorithms.
//!
//! These bodies define the circuit of each operation. A backend opts into the
//! matching `impl_gbfv_<family>_reference_full!` macro or calls a reference
//! body from a custom `*Impl`. Overrides must reproduce the circuit.

pub mod ntt;

pub use ntt::GBFVNttReference;
