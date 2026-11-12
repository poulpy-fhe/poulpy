//! Transform tables, built on the host and uploaded with the module, for
//! every degree it serves on both rings.
//!
//! Every word is the Shoup companion of a multiplier (see `primes::companion`),
//! in planes of `n` words per prime. Both rings share the twiddles of degree
//! `n`, one device buffer of `8n` words:
//!
//! | words      | content                                                                        |
//! |------------|--------------------------------------------------------------------------------|
//! | `0..4n`    | forward twiddles: entry `2^s + i` is `psi^bitrev(2^s + i)`                     |
//! | `4n..8n`   | inverse twiddles `psi^-bitrev(.)`; entry 0 is `n^-1`, entry 1 has it folded in |
//!
//! The basis change of the conjugate-invariant ring at degree `n` reads one
//! device buffer of `4n` words, plane `k` holding `f[j] = alpha^-j` modulo
//! prime `k`; `i = alpha^n`, the same at every degree, is a kernel constant.
//!
//! `psi` is the primitive `2n`-th root `OMEGA^(2^MAX_LOG_N / n)` and `alpha`
//! the primitive `4n`-th root with `alpha^2 = psi`. Stage `s` of a transform
//! has `2^s` blocks and butterfly distance `n / 2^(s+1)`; its block `i` uses
//! twiddle entry `2^s + i`.

use poulpy_hal::layouts::Ring;

use super::primes::{MAX_LOG_N, Q, companion, inv_mod, mul_mod, omega_pow2};
use crate::runtime::{GpuBuf, GpuRuntime};

/// Smallest degree with tables: `Backend::MIN_DEGREE`, which module construction checks.
pub(crate) const MIN_LOG_N: u32 = 3;

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

fn push(words: &mut Vec<u32>, multipliers: &[u32], q: u32) {
    words.extend(multipliers.iter().map(|&w| companion(w, q)));
}

fn twiddle_words(log_n: u32) -> Vec<u32> {
    let n = 1usize << log_n;
    let mut words = Vec::with_capacity(8 * n);
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
    words
}

/// `f[j] = alpha^-j` for `j < n = 2^log_n`, prime after prime.
fn basis_words(log_n: u32) -> Vec<u32> {
    let n = 1usize << log_n;
    let mut words = Vec::with_capacity(4 * n);
    for (k, &q) in Q.iter().enumerate() {
        let alpha = omega_pow2(k, MAX_LOG_N - 1 - log_n);
        push(&mut words, &powers(inv_mod(alpha, q), n, q), q);
    }
    words
}

/// The tables of every degree a module serves, from `2^MIN_LOG_N` on.
pub(crate) struct Tables<RT: GpuRuntime> {
    twiddles: Vec<GpuBuf<RT>>,
    basis: Vec<GpuBuf<RT>>,
}

impl<RT: GpuRuntime> Tables<RT> {
    /// The tables of the degrees `8..=n`, on both rings.
    pub(crate) fn new(n: usize) -> Self {
        // One module serves both rings, and the basis change takes a primitive `4n`-th root.
        assert!(
            n.is_power_of_two() && n <= 1 << (MAX_LOG_N - 1),
            "ring degree must be a power of two within the prime set's transform"
        );
        let upload = |words: Vec<u32>| GpuBuf::from_host(bytemuck::cast_slice(&words));
        let degrees = MIN_LOG_N..=n.ilog2();
        Self {
            twiddles: degrees.clone().map(|log_n| upload(twiddle_words(log_n))).collect(),
            basis: degrees.map(|log_n| upload(basis_words(log_n))).collect(),
        }
    }

    /// The tables of ring `R`.
    pub(crate) fn ring<R: Ring>(&self) -> RingTables<'_, RT> {
        RingTables {
            twiddles: &self.twiddles,
            basis: &self.basis,
            invariant: R::CYCLOTOMIC_ORDER_FACTOR == 4,
        }
    }
}

/// The tables of one ring.
pub(crate) struct RingTables<'a, RT: GpuRuntime> {
    twiddles: &'a [GpuBuf<RT>],
    basis: &'a [GpuBuf<RT>],
    invariant: bool,
}

impl<RT: GpuRuntime> Clone for RingTables<'_, RT> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<RT: GpuRuntime> Copy for RingTables<'_, RT> {}

/// The tables of one degree.
pub(crate) struct Table<'a, RT: GpuRuntime> {
    pub(crate) twiddles: &'a GpuBuf<RT>,
    /// The basis change of the conjugate-invariant ring.
    pub(crate) basis: &'a GpuBuf<RT>,
}

impl<'a, RT: GpuRuntime> RingTables<'a, RT> {
    /// The tables of degree `n`.
    pub(crate) fn of(&self, n: usize) -> Table<'a, RT> {
        assert!(n.is_power_of_two(), "ring degree must be a power of two");
        let index = n
            .ilog2()
            .checked_sub(MIN_LOG_N)
            .map(|index| index as usize)
            .filter(|&index| index < self.twiddles.len())
            .expect("degree not served by the module");
        Table {
            twiddles: &self.twiddles[index],
            basis: &self.basis[index],
        }
    }

    /// Whether the tables are of the conjugate-invariant ring.
    pub(crate) fn invariant(&self) -> bool {
        self.invariant
    }
}
