//! Encrypted matrix arithmetic on poulpy-ckks.
//!
//! A matrix is a `rows x cols` grid of `d x d` blocks, one CKKS ciphertext per block. A ciphertext with `S` slots
//! holds `S / d^2` blocks, matrix slot `s` holding entry `(r, c)` of its block at slot `s d^2 + r d + c`, so every
//! operation acts on a batch of matrices: up to `S / d^2`, repeated over the slots, which the grid records
//! ([`layouts::BlasCiphertext::from_blocks`]). Intra operations apply one map to every matrix of the batch; inter
//! operations move whole matrices across it. Products follow Jiang, Kim, Lauter and Song (2018), with every block map
//! encoded once at a caller-chosen precision.

pub mod api;
mod delegates;
pub mod layouts;
pub mod oep;
