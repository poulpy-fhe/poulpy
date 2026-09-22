//! NEON-accelerated NTT4x30 CPU backend (Q120 NTT, CRT over four ~30-bit primes).

use poulpy_cpu_ref::ring::CpuRing;

mod module;
mod prim;
#[cfg(feature = "enable-rayon")]
mod rayon;
mod vec_znx_big;
#[cfg(target_arch = "aarch64")]
pub(crate) mod vmp;
mod znx;

#[cfg(test)]
mod tests;

/// NEON-accelerated NTT4x30 CPU backend for Poulpy HAL.
/// `DftWord = Q120bScalar` (4 × u64 CRT residues), `BigWord = i128`, prime set `Primes30`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NTT4x30NeonBackend<R: CpuRing = poulpy_cpu_ref::ring::Standard>(std::marker::PhantomData<R>);

/// Standard negacyclic backend.
pub type NTT4x30Neon = NTT4x30NeonBackend<poulpy_cpu_ref::ring::Standard>;
/// Conjugate-invariant backend.
pub type NTT4x30CINeon = NTT4x30NeonBackend<poulpy_cpu_ref::ring::ConjugateInvariant>;

/// Rayon-scheduled variant of [`NTT4x30Neon`].
#[cfg(feature = "enable-rayon")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NTT4x30NeonRayonBackend<R: CpuRing = poulpy_cpu_ref::ring::Standard>(std::marker::PhantomData<R>);

/// Standard negacyclic Rayon backend.
#[cfg(feature = "enable-rayon")]
pub type NTT4x30NeonRayon = NTT4x30NeonRayonBackend<poulpy_cpu_ref::ring::Standard>;
/// Conjugate-invariant Rayon backend.
#[cfg(feature = "enable-rayon")]
pub type NTT4x30CINeonRayon = NTT4x30NeonRayonBackend<poulpy_cpu_ref::ring::ConjugateInvariant>;
