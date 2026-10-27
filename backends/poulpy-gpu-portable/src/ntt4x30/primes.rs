//! The prime set as the kernels see it: `Primes30` and every constant derived
//! from it, computed once on the host and passed to the kernels as literals.

use std::sync::OnceLock;

use cubecl::prelude::*;
use poulpy_cpu_portable::kernels::ntt4x30::{ntt::modq_pow_portable, primes::Primes30};
use poulpy_hal::layouts::PrimeSet;

pub(crate) const Q: [u32; 4] = <Primes30 as PrimeSet>::Q;

/// log2 of the order of `Primes30::OMEGA` is `MAX_LOG_N + 1`.
pub(crate) const MAX_LOG_N: u32 = <Primes30 as PrimeSet>::MAX_LOG_N;

pub(crate) fn mul_mod(a: u32, b: u32, q: u32) -> u32 {
    ((a as u64 * b as u64) % q as u64) as u32
}

pub(crate) fn inv_mod(a: u32, q: u32) -> u32 {
    modq_pow_portable(a, -1, q)
}

/// `Primes30::OMEGA[k]^(2^log_e)`.
pub(crate) fn omega_pow2(k: usize, log_e: u32) -> u32 {
    modq_pow_portable(<Primes30 as PrimeSet>::OMEGA[k], 1i64 << log_e, Q[k])
}

/// Shoup companion `floor(w * 2^32 / q)` of a multiplier `w < q`.
pub(crate) fn shoup(w: u32, q: u32) -> u32 {
    (((w as u64) << 32) / q as u64) as u32
}

/// Table word of a nonzero multiplier: its Shoup companion alone. The kernels
/// recover `w = mul_hi(w', q) + 1`: `w' * q = w * 2^32 - r` with `0 < r < q`
/// (`q` is odd and does not divide `w`), so the high word of `w' * q` is `w - 1`.
pub(crate) fn companion(w: u32, q: u32) -> u32 {
    assert!(w != 0 && w < q, "table multipliers are nonzero residues");
    let companion = shoup(w, q);
    debug_assert_eq!(((companion as u64 * q as u64) >> 32) as u32 + 1, w);
    companion
}

/// Constants of the kernels, per prime unless stated otherwise.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct Consts {
    pub(crate) q: [u32; 4],
    /// `2 * q`.
    pub(crate) q2: [u32; 4],
    /// `2^32 mod q`.
    pub(crate) r: [u32; 4],
    /// `-2^63 mod q`: undoes the bias that makes the high word of an `i64` unsigned.
    pub(crate) c63: [u32; 4],
    /// `floor(2^61 / q)`.
    pub(crate) mu61: [u32; 4],
    /// `floor(2^60 / q)`.
    pub(crate) mu60: [u32; 4],
    /// The fourth root of unity of the conjugate-invariant basis change, and its Shoup companion.
    pub(crate) i: [u32; 4],
    pub(crate) i_shoup: [u32; 4],
    /// Garner inverses `q_i^-1 mod q_j` for `(i, j)` = (0,1), (0,2), (1,2), (0,3), (1,3), (2,3).
    pub(crate) g: [u32; 6],
    pub(crate) g_shoup: [u32; 6],
    /// Mixed-radix weights `q0`, `q0 q1`, `q0 q1 q2` in 32-bit words.
    pub(crate) w1: u32,
    pub(crate) w2: [u32; 2],
    pub(crate) w3: [u32; 3],
    /// `Q = q0 q1 q2 q3` and `(Q + 1) / 2`, the smallest value mapped to a negative one.
    pub(crate) q_lo: u64,
    pub(crate) q_hi: u64,
    pub(crate) half_lo: u64,
    pub(crate) half_hi: u64,
}

impl Consts {
    fn new() -> Self {
        let q = Q;
        let r = q.map(|q| ((1u64 << 32) % q as u64) as u32);
        for k in 0..4 {
            assert!(q[k] > 1 << 29 && q[k] < 1 << 30, "the lazy ranges need 30-bit primes");
            // The load folds an `i64` to `lo + hb * r + c63`, reduced by a one-word Barrett step.
            assert!(0xFFFF_FFFFu64 * (r[k] as u64 + 1) + (q[k] as u64) < 1 << 60);
            for j in 0..4 {
                // Garner forms `r_j + 2 q_j - x_i` with `x_i < q_i`.
                assert!(k == j || (q[k] != q[j] && q[k] < 2 * q[j]));
            }
        }
        let pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)];
        let g = pairs.map(|(i, j)| inv_mod(q[i] % q[j], q[j]));
        let product: u128 = q.iter().map(|&q| q as u128).product();
        let half = product.div_ceil(2);
        let w2 = q[0] as u64 * q[1] as u64;
        let w3 = w2 as u128 * q[2] as u128;
        let i: [u32; 4] = std::array::from_fn(|k| omega_pow2(k, MAX_LOG_N - 1));
        Self {
            q,
            q2: q.map(|q| 2 * q),
            r,
            c63: q.map(|q| ((q as u64 - (1u64 << 63) % q as u64) % q as u64) as u32),
            mu61: q.map(|q| u32::try_from((1u64 << 61) / q as u64).expect("2^61 / q fits a word")),
            mu60: q.map(|q| ((1u64 << 60) / q as u64) as u32),
            i,
            i_shoup: std::array::from_fn(|k| shoup(i[k], q[k])),
            g,
            g_shoup: std::array::from_fn(|k| shoup(g[k], q[pairs[k].1])),
            w1: q[0],
            w2: [w2 as u32, (w2 >> 32) as u32],
            w3: [w3 as u32, (w3 >> 32) as u32, (w3 >> 64) as u32],
            q_lo: product as u64,
            q_hi: (product >> 64) as u64,
            half_lo: half as u64,
            half_hi: (half >> 64) as u64,
        }
    }

    pub(crate) fn get() -> Self {
        static CONSTS: OnceLock<Consts> = OnceLock::new();
        *CONSTS.get_or_init(Self::new)
    }
}

/// `c[k]` for a run-time `k < 4`: three selects between literals.
#[cube]
pub(crate) fn sel4(k: u32, #[comptime] c: [u32; 4]) -> u32 {
    let lo = select(k == 1u32, c[1].runtime(), c[0].runtime());
    let hi = select(k == 3u32, c[3].runtime(), c[2].runtime());
    select(k >= 2u32, hi, lo)
}

/// Lazy product `x * w mod q` in `[0, 2q)` for any 32-bit `x`, from the
/// companion `wq = floor(w * 2^32 / q)` of a nonzero `w`.
#[cube]
pub(crate) fn mul_lazy(x: u32, wq: u32, q: u32) -> u32 {
    let w = wq.mul_hi(q) + 1u32;
    x * w - x.mul_hi(wq) * q
}

/// Terms a product accumulator takes between two folds: a folded value and
/// 15 products of canonical residues stay below `2^60 + 15 * 2^60 = 2^64`.
pub(crate) const FOLD_TERMS: u32 = 15;

/// Lazy reduction of a 64-bit accumulator: `lo + hi * (2^32 mod q)`,
/// congruent to `v` and below `2^60` (`2^32 mod q < 2^27`).
#[cube]
pub(crate) fn fold64(v: u64, r: u32) -> u64 {
    (v & 0xFFFF_FFFFu64) + (v >> 32u64) * u64::cast_from(r)
}

/// `v mod q` in `[0, q)` for `v < 2^60`, `mu = floor(2^60 / q)`: the quotient
/// estimate is at most two short, so the remainder fits a word.
#[cube]
pub(crate) fn barrett60(v: u64, q: u32, mu: u32) -> u32 {
    let quotient = ((v >> 29u64) * u64::cast_from(mu)) >> 31u64;
    let r = u32::cast_from(v) - u32::cast_from(quotient) * q;
    let r = min(r, r - q);
    min(r, r - q)
}

/// `x * y mod q` in `[0, q)` for canonical `x` and `y`.
#[cube]
pub(crate) fn mul_mod_canonical(x: u32, y: u32, q: u32, mu: u32) -> u32 {
    barrett60(u64::cast_from(x) * u64::cast_from(y), q, mu)
}
