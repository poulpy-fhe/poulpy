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
//! | [`layouts`] | The plaintext modulus |

pub mod layouts;
