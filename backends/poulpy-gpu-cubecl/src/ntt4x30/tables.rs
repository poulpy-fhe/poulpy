//! Transform tables, built on the host and uploaded once per degree.
//!
//! One device buffer per degree `n`, in planes of `n` words per prime. Every
//! word is the Shoup companion of a multiplier (see `primes::companion`).
//!
//! | words       | content                                                        |
//! |-------------|----------------------------------------------------------------|
//! | `0..4n`     | forward twiddles: entry `2^s + i` is `psi^bitrev(2^s + i)`     |
//! | `4n..8n`    | inverse twiddles `psi^-bitrev(.)`; entry 0 is `n^-1`, entry 1 has it folded in |
//! | `8n..12n`   | conjugate-invariant ring: `alpha^-j`                           |
//! | `12n..16n`  | conjugate-invariant ring: `alpha^j / 2`, entry 0 is 1          |
//! | `16n..20n`  | conjugate-invariant ring: `-alpha^-j / 2`, entry 0 is unused   |
//!
//! `psi` is the primitive `2n`-th root `OMEGA^(2^MAX_LOG_N / n)` and `alpha`
//! the primitive `4n`-th root with `alpha^2 = psi`. Stage `s` of a transform
//! has `2^s` blocks and butterfly distance `n / 2^(s+1)`; its block `i` uses
//! twiddle entry `2^s + i`.

use super::primes::{MAX_LOG_N, Q, companion, inv_mod, mul_mod, omega_pow2};
use crate::runtime::{GpuBuf, GpuRuntime};

/// Smallest degree with tables: `Backend::MIN_DEGREE`.
const MIN_LOG_N: u32 = 3;

fn powers(root: u32, n: usize, q: u32) -> Vec<u32> {
    let mut powers = vec![1u32; n];
    for e in 1..n {
        powers[e] = mul_mod(powers[e - 1], root, q);
    }
    powers
}

fn bit_reversed(powers: &[u32], log_n: u32) -> Vec<u32> {
    (0..powers.len())
        .map(|k| powers[k.reverse_bits() >> (usize::BITS - log_n)])
        .collect()
}

fn build(log_n: u32, invariant: bool) -> Vec<u32> {
    let n = 1usize << log_n;
    let mut words = Vec::with_capacity(if invariant { 20 * n } else { 8 * n });
    let push = |words: &mut Vec<u32>, multipliers: &[u32], q: u32| words.extend(multipliers.iter().map(|&w| companion(w, q)));
    for inverse in [false, true] {
        for (k, &q) in Q.iter().enumerate() {
            let psi = omega_pow2(k, MAX_LOG_N - log_n);
            let root = if inverse { inv_mod(psi, q) } else { psi };
            let mut twiddles = bit_reversed(&powers(root, n, q), log_n);
            if inverse {
                let n_inv = inv_mod(n as u32, q);
                twiddles[0] = n_inv;
                twiddles[1] = mul_mod(twiddles[1], n_inv, q);
            }
            push(&mut words, &twiddles, q);
        }
    }
    if invariant {
        let alpha: [u32; 4] = std::array::from_fn(|k| omega_pow2(k, MAX_LOG_N - 1 - log_n));
        for (k, &q) in Q.iter().enumerate() {
            push(&mut words, &powers(inv_mod(alpha[k], q), n, q), q);
        }
        let halves: Vec<Vec<u32>> = (0..4)
            .map(|k| {
                let half = Q[k].div_ceil(2);
                powers(alpha[k], n, Q[k]).iter().map(|&w| mul_mod(w, half, Q[k])).collect()
            })
            .collect();
        for (k, &q) in Q.iter().enumerate() {
            let mut direct = halves[k].clone();
            direct[0] = 1;
            push(&mut words, &direct, q);
        }
        for (k, &q) in Q.iter().enumerate() {
            // `-alpha^-j / 2 = i * alpha^(n-j) / 2` with `i = alpha^n`.
            let i = omega_pow2(k, MAX_LOG_N - 1);
            let mut reflected = vec![1u32; n];
            for j in 1..n {
                reflected[j] = mul_mod(halves[k][n - j], i, q);
            }
            push(&mut words, &reflected, q);
        }
    }
    words
}

/// The tables of every degree a module serves.
pub(crate) struct Tables<RT: GpuRuntime> {
    by_log_n: Vec<GpuBuf<RT>>,
    invariant: bool,
}

impl<RT: GpuRuntime> Tables<RT> {
    /// Tables of the degrees `8..=n` of the standard ring, or of the
    /// conjugate-invariant ring when `invariant`.
    pub(crate) fn new(n: usize, invariant: bool) -> Self {
        let max_log_n = MAX_LOG_N - invariant as u32;
        assert!(
            n.is_power_of_two() && n <= 1 << max_log_n,
            "ring degree must be a power of two at most 2^{max_log_n}, got {n}"
        );
        let by_log_n = (MIN_LOG_N..=n.ilog2())
            .map(|log_n| GpuBuf::from_host(bytemuck::cast_slice(&build(log_n, invariant))))
            .collect();
        Self { by_log_n, invariant }
    }

    /// The table of degree `n`.
    pub(crate) fn of(&self, n: usize) -> &GpuBuf<RT> {
        assert!(n.is_power_of_two(), "ring degree must be a power of two, got {n}");
        n.ilog2()
            .checked_sub(MIN_LOG_N)
            .and_then(|index| self.by_log_n.get(index as usize))
            .unwrap_or_else(|| panic!("degree {n} is not served by the module"))
    }

    /// Whether the module is over the conjugate-invariant ring.
    pub(crate) fn invariant(&self) -> bool {
        self.invariant
    }
}
