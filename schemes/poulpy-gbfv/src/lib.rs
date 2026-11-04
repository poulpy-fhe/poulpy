#![deny(rustdoc::broken_intra_doc_links)]
//! # poulpy-gbfv
//!
//! Backend-agnostic implementation of the GBFV (generalized BFV) homomorphic
//! encryption scheme.
//!
//! GBFV generalizes BFV by taking a plaintext modulus `t(X)` that is a non-zero
//! element of the ring `Z[X]/(X^N + 1)` instead of an integer.
//!
//! Ring elements are specified in the smallest subring that holds them and stand
//! for their embedding `X -> X^(N/n)` into the encryption ring, as degree-`n`
//! operands do in `poulpy-hal`.
//!
//! ## Modules
//!
//! | Module | Role |
//! |--------|------|
//! | [`layouts`] | The plaintext modulus, the batch encoder, and the NTT operands |
//! | [`api`] | Public operations on `Module<BE>`: the negacyclic NTT over `Z/q` |
//! | [`oep`] | Backend contracts and the reference opt-in macros |
//! | [`reference`](mod@crate::reference) | Reference bodies of the backend contracts |

pub mod api;
mod delegates;
pub mod layouts;
pub mod oep;
pub mod reference;

/// Result of the fallible GBFV operations and setup constructors.
pub type GBFVResult<T> = anyhow::Result<T>;
