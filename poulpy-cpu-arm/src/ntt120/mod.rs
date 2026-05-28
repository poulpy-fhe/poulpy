//! NEON-accelerated NTT120 CPU backend (Q120 NTT, CRT over four ~30-bit primes).

mod module;
mod prim;
mod vec_znx_big;
pub(crate) mod vmp;
mod znx;

#[cfg(test)]
mod tests;

/// NEON-accelerated NTT120 CPU backend for Poulpy HAL.
/// `ScalarPrep = Q120bScalar` (4 × u64 CRT residues), `ScalarBig = i128`, prime set `Primes30`.
#[derive(Debug, Clone, Copy)]
pub struct NTT120Neon;
