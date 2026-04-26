//! Leveled CKKS arithmetic, encryption, and decryption.
//!
//! This module provides the core leveled evaluation pipeline:
//!
//! - [`api`]: public trait definitions for all CKKS operations.
//! - [`delegates`]: blanket `impl Trait for Module<BE>` forwarding to OEP/default impls.
//! - [`oep`]: backend dispatch traits bridging delegates to `CKKSImpl`.
//! - Operation-family modules (`add`, `sub`, `neg`, `mul`, `pow2`, `rotate`,
//!   `conjugate`, `pt_znx`, `rescale`): default algorithm implementations.
//!
//! All hot-path operations use scratch-based allocation; no heap allocation
//! occurs during leveled arithmetic.

pub mod api;
pub(crate) mod delegates;
pub(crate) mod oep;

pub(crate) mod default;

pub mod tests;

pub use api::*;
