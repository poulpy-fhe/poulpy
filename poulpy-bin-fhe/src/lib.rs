//! # poulpy-bin-fhe
//!
//! Backend-agnostic implementation of binary / gate-level FHE, built on top
//! of the low-level primitives provided by `poulpy-core`, `poulpy-hal`, and
//! the available compute backends (`poulpy-cpu-ref`, `poulpy-cpu-avx`,
//! `poulpy-cpu-avx512`).
//!
//! ## Overview
//!
//! This crate provides three layered building blocks for constructing
//! gate-level and word-level FHE computation over encrypted binary data:
//!
//! - **Blind rotation** ([`blind_rotation`]): Evaluation of a
//!   programmable lookup table (LUT) under a GLWE ciphertext, driven by an LWE
//!   ciphertext acting as a rotation index.  This is the fundamental gate
//!   bootstrapping primitive used throughout the rest of the crate.
//!
//! - **Circuit bootstrapping** ([`circuit_bootstrapping`]): Conversion
//!   of a GLWE ciphertext encrypting a small plaintext into a GGSW ciphertext,
//!   enabling subsequent homomorphic selection (CMux) operations on ciphertexts.
//!   Circuit bootstrapping composes blind rotation, a trace/packing step, and a
//!   GGLWE-to-GGSW key-switch.
//!
//! - **BDD arithmetic** ([`bdd_arithmetic`]): Word-level FHE arithmetic
//!   on encrypted unsigned integers ([`bdd_arithmetic::FheUint`]) via
//!   statically compiled Binary Decision Diagram (BDD) circuits.  Operations
//!   such as addition, subtraction, bitwise logic, and shifts over `u32` are
//!   provided out of the box, evaluated bit-by-bit through GGSW-based CMux
//!   gates.
pub mod bdd_arithmetic;
pub mod blind_rotation;
pub mod circuit_bootstrapping;
