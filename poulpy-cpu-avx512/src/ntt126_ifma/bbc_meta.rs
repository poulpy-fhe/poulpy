//! Lazy-accumulation BBC (b × c → b) multiply metadata for the 3-prime IFMA backend.

use std::marker::PhantomData;

use super::primes::{PrimeSetNtt126Ifma, Primes42};

/// Metadata for `b × c → b` (BBC) lazy multiply-accumulate (3-prime variant).
///
/// The fields are not read by the AVX-512-IFMA kernels (which inline the same
/// constants via [`super::primes::PrimeSetNtt126Ifma::Q`]) — they exist so the
/// scalar reference oracles can share a single metadata-carrying parameter
/// shape with the SIMD kernels.
#[allow(dead_code)]
pub struct Bbc126IfmaMeta<P: PrimeSetNtt126Ifma> {
    /// Reduction split point for the final accumulator collapse.
    pub h: u64,
    /// `s2l_pow_red[k] = 2^32 mod Q[k]` — low-half reduction weight.
    pub s2l_pow_red: [u64; 4],
    /// `s2h_pow_red[k]` — high-half reduction weight.
    pub s2h_pow_red: [u64; 4],
    _phantom: PhantomData<P>,
}

impl Bbc126IfmaMeta<Primes42> {
    pub fn new() -> Self {
        let q = Primes42::Q;
        let h = 32u64;
        let mut s2l = [0u64; 4];
        let mut s2h = [0u64; 4];
        for k in 0..3 {
            s2l[k] = ((1u128 << 32) % q[k] as u128) as u64;
            s2h[k] = ((1u128 << (32 + h)) % q[k] as u128) as u64;
        }
        Self {
            h,
            s2l_pow_red: s2l,
            s2h_pow_red: s2h,
            _phantom: PhantomData,
        }
    }
}

impl Default for Bbc126IfmaMeta<Primes42> {
    fn default() -> Self {
        Self::new()
    }
}
