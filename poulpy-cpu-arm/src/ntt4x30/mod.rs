//! NEON-accelerated NTT4x30 CPU backend (Q120 NTT, CRT over four ~30-bit primes).

mod conjugate_invariant;
pub(crate) mod convolution;
mod module;
mod prim;
#[cfg(feature = "enable-rayon")]
mod rayon;
mod standard;
pub(crate) mod svp;
mod vec_znx_big;
pub(crate) mod vec_znx_dft;
pub(crate) mod vmp;
mod znx;

use std::marker::PhantomData;

use poulpy_hal::layouts::{Ring, Standard};

#[cfg(test)]
mod tests;

/// NEON-accelerated NTT4x30 CPU backend for Poulpy HAL.
/// `DftWord = CrtWord<Primes30, u32>` (four `u32` CRT residues), `BigWord = i128`, prime set `Primes30`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NTT4x30Neon<R: Ring = Standard>(PhantomData<R>);

/// Rayon-scheduled variant of [`NTT4x30Neon`].
#[cfg(feature = "enable-rayon")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NTT4x30NeonRayon<R: Ring = Standard>(PhantomData<R>);
