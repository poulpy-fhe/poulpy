//! Scalar reference implementations of the 3-prime IFMA NTT primitives.
//!
//! These are pure-Rust kernel-level oracles used to validate the AVX-512-IFMA
//! SIMD implementations in [`super::kernels`] and [`super::mat_vec_ifma`].
//! End-to-end correctness against the FFT64 backend is exercised by the
//! `cross_backend_test_suite!` blocks in [`super::tests`]; the functions here
//! provide the per-coefficient diff that pinpoints lane-level SIMD bugs before
//! CRT reconstruction.
//!
//! # Contents
//!
//! - [`arithmetic`]: scalar i64 ↔ 3-prime CRT conversions and Garner reconstruction.
//! - [`ntt`]: scalar forward / inverse NTT (`ntt3x42_ifma_ref`, `intt3x42_ifma_ref`).
//! - [`mat_vec`]: scalar BBC inner-product kernels and x2-block extract.
//!
//! Production infrastructure (table types, traits, primes, generic glue) lives
//! in sibling modules, not here.
//!
//! The module builds only under `cfg(test)`; `allow(dead_code)` covers the
//! oracles no test calls yet.
#![allow(dead_code)]

pub mod arithmetic;
pub mod mat_vec;
pub mod ntt;
