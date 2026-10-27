#![allow(clippy::too_many_arguments)]

//! Independent scalar backends for correctness testing, not production execution.
//!
//! [`FFT64Oracle`] and [`NTT4x30Oracle`], over the standard or the
//! conjugate-invariant ring, implement the required HAL operations directly,
//! with plain loops, and inherit every optional one. They own their
//! transforms and tables and do not depend on any production CPU backend.
//! `enable-core` registers the generic Core compositions; `enable-ckks`,
//! `enable-bin-fhe` and `enable-mhe` each imply it and register the CKKS,
//! binary-FHE and multiparty ones, the multiparty ones on the standard ring only.
//! `enable-blas` implies `enable-ckks` and wires the oracle backends into `poulpy-blas`.

mod backend;
#[cfg(feature = "enable-bin-fhe")]
mod bin_fhe;
#[cfg(feature = "enable-blas")]
mod blas;
#[cfg(feature = "enable-ckks")]
mod ckks;
#[cfg(feature = "enable-ckks")]
mod ckks_fft;
#[cfg(feature = "enable-ckks")]
mod ckks_roots;
#[cfg(feature = "enable-core")]
mod core_impl;
mod embed;
mod family;
mod fft;
mod hal;
mod limbs;
#[cfg(feature = "enable-mhe")]
mod mhe;
mod normalize;
mod ntt;
mod ring;
#[cfg(feature = "enable-core")]
mod sampling;
#[cfg(feature = "enable-core")]
mod scalar_znx_fill;

#[cfg(feature = "enable-core")]
pub mod test_suite;

#[cfg(test)]
mod tests;

pub use backend::{FFT64CIOracle, FFT64Oracle, Handle, NTT4x30CIOracle, NTT4x30Oracle, Oracle};
pub use family::DFTFamily;
pub use fft::{ComplexFft, Fft64};
pub use ntt::{Ntt4x30, Primes30};
pub use ring::OracleRing;

#[cfg(feature = "enable-ckks")]
pub use ckks_fft::EncodingFft;

#[cfg(feature = "enable-core")]
pub(crate) use scalar_znx_fill::ScalarZnxFill;
