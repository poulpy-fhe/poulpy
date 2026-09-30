#![deny(rustdoc::broken_intra_doc_links)]
//! # poulpy-mhe
//!
//! Backend-agnostic multiparty homomorphic encryption, built from
//! `poulpy-core` and `poulpy-hal`.
//!
//! Parties exchange public aggregatable transcripts (PATs): shares any party
//! can sum, then finalize into a key or ciphertext under the ideal secret, the
//! sum of every party's secret. [`layouts`] holds one type per transcript
//! shape.
pub mod layouts;

pub use layouts::*;
