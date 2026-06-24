//! Precomputed twiddle tables and Harvey arithmetic helpers for the 3-prime IFMA NTT.
//!
//! # IFMA-native arithmetic model
//!
//! Lazy Harvey reduction: butterfly values are kept in `[0, 4q)` internally,
//! and normalised to `[0, 2q)` at NTT boundaries.  On the difference path of
//! each butterfly the Harvey multiplier absorbs the wider range directly —
//! inputs up to `2^52` yield outputs in `[0, 2q)` because `q < 2^42` — so a
//! pre-reduction `cond_sub` before the multiply is unnecessary.  Only the sum
//! path keeps one `cond_sub` (of `4q`) per butterfly pair.
//!
//! # Twiddle factor layout
//!
//! Twiddle factors use a **split (SoA) layout** within each NTT level segment.
//! For a segment with `m` entries, the layout is:
//! - `[ω₀, ω₁, ..., ωₘ₋₁]` — all twiddle values (4 u64 each)
//! - `[ωq₀, ωq₁, ..., ωqₘ₋₁]` — all Harvey quotients (4 u64 each)
//!
//! This enables AVX-512 kernels to load 2 consecutive ω or ωq values with a
//! single 512-bit load, instead of deinterleaving from `[ω, ωq]` pairs.
//!
//! # Data layout
//!
//! Each coefficient = 4 × u64 (3 active CRT residues + 1 padding).
//! All residues are kept in `[0, 2q)` throughout the NTT.

use std::marker::PhantomData;

use poulpy_hal::alloc_aligned;

use super::primes::{PrimeSetNtt126Ifma, modq_pow64};

// ──────────────────────────────────────────────────────────────────────────────
// Precomputation data structures
// ──────────────────────────────────────────────────────────────────────────────

/// Precomputed twiddle-factor table for the forward NTT (3-prime IFMA).
///
/// No per-level metadata is needed — the IFMA-native butterfly keeps all
/// values in `[0, 2q)` without explicit reduction levels.
pub struct Ntt126IfmaTable<P: PrimeSetNtt126Ifma> {
    /// NTT size (power of two, ≤ 2^16).
    pub n: usize,
    /// `2q[k]` for each prime (lane 3 = 0).  Used for the final `[0, 4q)` → `[0, 2q)`
    /// normalisation pass and by external consumers that expect `[0, 2q)` input.
    pub q2: [u64; 4],
    /// `4q[k]` for each prime (lane 3 = 0).  Used inside butterflies under the
    /// lazy `[0, 4q)` invariant: sum path subtracts `4q`, diff path adds `4q`
    /// before subtracting `b`.
    pub q4: [u64; 4],
    /// Packed twiddle factors: each entry is 8 u64.
    /// Layout: level-0 (n entries), then butterfly levels (halfnn-1 entries each).
    pub powomega: Vec<u64>,
    /// Scrambled forward roots, prime-major.
    /// Entry for prime `k`, scrambled index `j` is at `[k*n + j]`.
    /// `root[bitrev(i)] = w^i`, where `w` is the primitive `2n`-th root.
    pub root: Vec<u64>,
    /// Harvey/Shoup preconditioned quotients for `root`, same layout.
    /// `root_quot[k*n + j] = harvey_quotient(root[k*n + j], Q[k])`.
    pub root_quot: Vec<u64>,
    /// Vectorized tail roots for the `t = 4, 2, 1` stages, prime-major, stride
    /// `3n/2`. Per prime, three blocks of length `n/2` with each root duplicated
    /// to match the stage operand width: 4× (distance 4), 2× (distance 2), 1×
    /// (distance 1). Empty when `n < 16` (tail vectorization requires `n >= 16`).
    pub tail_root: Vec<u64>,
    /// Harvey/Shoup preconditioned quotients for `tail_root`, same layout.
    pub tail_quot: Vec<u64>,
    _phantom: PhantomData<P>,
}

/// Precomputed twiddle-factor table for the inverse NTT (3-prime IFMA).
pub struct Ntt126IfmaTableInv<P: PrimeSetNtt126Ifma> {
    pub n: usize,
    pub q2: [u64; 4],
    pub q4: [u64; 4],
    /// Packed twiddle factors: butterfly levels (halfnn-1 entries each),
    /// then last-pass (n entries with ω^{-i}/n baked in).
    pub powomega: Vec<u64>,
    /// Reordered inverse roots, prime-major (stride `n`).
    /// Entry for prime `k`, reordered index `j` is at `[k*n + j]`.
    pub inv_root: Vec<u64>,
    /// Harvey/Shoup preconditioned quotients for `inv_root`, same layout.
    pub inv_quot: Vec<u64>,
    _phantom: PhantomData<P>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Returns the primitive `2n`-th roots of unity for each of the 3 primes.
fn fill_omegas_ntt126_ifma<P: PrimeSetNtt126Ifma>(n: usize) -> [u64; 3] {
    debug_assert!((1..=(1 << 16)).contains(&n), "n must be a power of two in [1, 2^16], got {n}");
    std::array::from_fn(|k| modq_pow64(P::OMEGA[k], (1i64 << 16) / n as i64, P::Q[k]))
}

/// Compute Harvey quotient: `floor(omega * 2^52 / q)`.
#[inline(always)]
pub fn harvey_quotient(omega: u64, q: u64) -> u64 {
    ((omega as u128 * (1u128 << 52)) / q as u128) as u64
}

/// Reverse the low `bits` bits of `x`.
fn reverse_bits(x: usize, bits: u32) -> usize {
    let mut r = 0usize;
    for i in 0..bits {
        r |= ((x >> i) & 1) << (bits - 1 - i);
    }
    r
}

/// Build the scrambled forward root tables (prime-major).
///
/// For each prime `k`: `w = OMEGA[k] ^ (2^16 / n)` (primitive `2n`-th root),
/// then `root[bitrev(i)] = w^i` for `i in 0..n` (`root[0] = 1`).
/// Returns `(root, root_quot)`, each of length `3*n`, with entry for
/// prime `k`, scrambled index `j` at `[k*n + j]`.
fn build_root_table<P: PrimeSetNtt126Ifma>(n: usize) -> (Vec<u64>, Vec<u64>) {
    let mut root_tbl = vec![0u64; 3 * n];
    let mut quot_tbl = vec![0u64; 3 * n];
    if n == 0 {
        return (root_tbl, quot_tbl);
    }
    let log_n = n.trailing_zeros();
    for k in 0..3 {
        let q = P::Q[k];
        let w = modq_pow64(P::OMEGA[k], (1i64 << 16) / n as i64, q);
        // root[bitrev(i)] = w^i, built incrementally from the previous scrambled index.
        let mut root = vec![0u64; n];
        root[0] = 1;
        let mut prev_idx = 0usize;
        for i in 1..n {
            let idx = reverse_bits(i, log_n);
            root[idx] = ((root[prev_idx] as u128 * w as u128) % q as u128) as u64;
            prev_idx = idx;
        }
        for j in 0..n {
            root_tbl[k * n + j] = root[j];
            quot_tbl[k * n + j] = harvey_quotient(root[j], q);
        }
    }
    (root_tbl, quot_tbl)
}

/// Modular inverse via Fermat: `x^{q-2} mod q` (`q` prime).
fn modinv(x: u64, q: u64) -> u64 {
    modq_pow64(x, q as i64 - 2, q)
}

/// Build the reordered inverse root tables (prime-major, stride `n`).
///
/// For each prime `k`: build the scrambled forward roots `fwd[bitrev(i)] = w^i`
/// (`w` the primitive `2n`-th root), take `inv_nat[j] = modinv(fwd[j])`, then
/// reorder level-major: `temp[0] = inv_nat[0]`, then for `m in [n/2, n/4, …, 1]`
/// append `inv_nat[m+i]` for `i in 0..m`.
/// Returns `(inv_root, inv_quot)`, each of length `3*n`.
fn build_inv_root_table<P: PrimeSetNtt126Ifma>(n: usize) -> (Vec<u64>, Vec<u64>) {
    let mut inv_root = vec![0u64; 3 * n];
    let mut inv_precon = vec![0u64; 3 * n];
    if n == 0 {
        return (inv_root, inv_precon);
    }
    let log_n = n.trailing_zeros();
    for k in 0..3 {
        let q = P::Q[k];
        let w = modq_pow64(P::OMEGA[k], (1i64 << 16) / n as i64, q);
        // Scrambled forward roots: root[bitrev(i)] = w^i.
        let mut fwd = vec![0u64; n];
        fwd[0] = 1;
        let mut prev_idx = 0usize;
        for i in 1..n {
            let idx = reverse_bits(i, log_n);
            fwd[idx] = ((fwd[prev_idx] as u128 * w as u128) % q as u128) as u64;
            prev_idx = idx;
        }
        // inv_nat[j] = modinv(fwd[j]).
        let inv_nat: Vec<u64> = fwd.iter().map(|&r| modinv(r, q)).collect();
        // Reorder level-major.
        let mut temp = vec![0u64; n];
        temp[0] = inv_nat[0];
        let mut idx = 1usize;
        let mut m = n >> 1;
        while m > 0 {
            for i in 0..m {
                temp[idx] = inv_nat[m + i];
                idx += 1;
            }
            m >>= 1;
        }
        for j in 0..n {
            inv_root[k * n + j] = temp[j];
            inv_precon[k * n + j] = harvey_quotient(temp[j], q);
        }
    }
    (inv_root, inv_precon)
}

/// Build the vectorized tail root tables for the `t = 4, 2, 1` stages
/// (prime-major, stride `3n/2`).
///
/// Derived from each prime's non-duplicated scrambled `root[]`, each root is
/// duplicated to the stage operand width:
/// - distance-4 block (offset 0, len `n/2`): `root[i]` 4× for `i in n/8..n/4`.
/// - distance-2 block (offset `n/2`, len `n/2`): `root[i]` 2× for `i in n/4..n/2`.
/// - distance-1 block (offset `n`, len `n/2`): `root[i]` 1× for `i in n/2..n`.
///
/// Only built for `n >= 16`; returns empty vectors otherwise.
fn build_tail_root_table<P: PrimeSetNtt126Ifma>(n: usize) -> (Vec<u64>, Vec<u64>) {
    if n < 16 {
        return (Vec::new(), Vec::new());
    }
    let stride = 3 * n / 2;
    let mut tail_root = vec![0u64; 3 * stride];
    let mut tail_quot = vec![0u64; 3 * stride];
    let log_n = n.trailing_zeros();
    for k in 0..3 {
        let q = P::Q[k];
        let w = modq_pow64(P::OMEGA[k], (1i64 << 16) / n as i64, q);
        // root[bitrev(i)] = w^i, built incrementally (same as build_root_table).
        let mut root = vec![0u64; n];
        root[0] = 1;
        let mut prev_idx = 0usize;
        for i in 1..n {
            let idx = reverse_bits(i, log_n);
            root[idx] = ((root[prev_idx] as u128 * w as u128) % q as u128) as u64;
            prev_idx = idx;
        }
        let base = k * stride;
        let mut p = base;
        let mut push = |val: u64, p: &mut usize| {
            tail_root[*p] = val;
            tail_quot[*p] = harvey_quotient(val, q);
            *p += 1;
        };
        // distance-4 block: each root 4×.
        for &r in &root[(n / 8)..(n / 4)] {
            for _ in 0..4 {
                push(r, &mut p);
            }
        }
        // distance-2 block: each root 2×.
        for &r in &root[(n / 4)..(n / 2)] {
            for _ in 0..2 {
                push(r, &mut p);
            }
        }
        // distance-1 block: each root 1×.
        for &r in &root[(n / 2)..n] {
            push(r, &mut p);
        }
        debug_assert_eq!(p - base, stride);
    }
    (tail_root, tail_quot)
}

/// Harvey modular multiply (scalar): `a * omega mod q`.
///
/// Input: `a ∈ [0, 2^52)` (in practice up to `4q` or `8q` under lazy),
/// `omega ∈ [0, q)`.  Output: `r ∈ [0, 2q)` with `r ≡ a*omega (mod q)`.
///
/// Because `omega_quot = floor(omega * 2^52 / q)` rounds down, the computed
/// `qhat` is always `≤ floor(a*omega/q)` (never an overestimate), so the raw
/// remainder `r = a*omega - qhat*q` is non-negative.  It lies in `[0, 2q)`
/// whenever `a < 2^52`, which covers all lazy-reduction ranges we use.
#[inline(always)]
pub fn harvey_modmul(a: u64, omega: u64, omega_quot: u64, q: u64) -> u64 {
    let qhat = ((a as u128 * omega_quot as u128) >> 52) as u64;
    let product_lo = (a as u128 * omega as u128) as u64; // low 64 bits (we only need mod 2^64)
    product_lo.wrapping_sub(qhat.wrapping_mul(q))
}

/// Conditional subtract: if `x >= 2q`, return `x - 2q`, else `x`.
/// Keeps values in `[0, 2q)`.
#[inline(always)]
pub(crate) fn cond_sub_2q(x: u64, q2: u64) -> u64 {
    if x >= q2 { x - q2 } else { x }
}

/// Store a twiddle entry into the split powomega array.
///
/// `omega_base`: start of the ω section for this level segment.
/// `quot_base`: start of the ωq section for this level segment.
/// `idx`: index of this entry within the segment.
fn store_twiddle_split<P: PrimeSetNtt126Ifma>(
    powomega: &mut [u64],
    omega_base: usize,
    quot_base: usize,
    idx: usize,
    omega_vals: &[u64; 3],
) {
    let o = omega_base + 4 * idx;
    let q = quot_base + 4 * idx;
    for k in 0..3 {
        powomega[o + k] = omega_vals[k];
        powomega[q + k] = harvey_quotient(omega_vals[k], P::Q[k]);
    }
    powomega[o + 3] = 0;
    powomega[q + 3] = 0;
}

// ──────────────────────────────────────────────────────────────────────────────
// Forward NTT table construction
// ──────────────────────────────────────────────────────────────────────────────

impl<P: PrimeSetNtt126Ifma> Ntt126IfmaTable<P> {
    pub fn new(n: usize) -> Self {
        assert!(
            n.is_power_of_two() && n <= (1 << 16),
            "NTT size must be a power of two ≤ 2^16, got {n}"
        );

        let q2: [u64; 4] = [2 * P::Q[0], 2 * P::Q[1], 2 * P::Q[2], 0];
        let q4: [u64; 4] = [4 * P::Q[0], 4 * P::Q[1], 4 * P::Q[2], 0];

        let (root, root_quot) = build_root_table::<P>(n);
        let (tail_root, tail_quot) = build_tail_root_table::<P>(n);

        let omega_vec = fill_omegas_ntt126_ifma::<P>(n);

        // Allocate powomega: level-0 needs n entries, butterfly levels need sum of (halfnn-1)
        let total_entries = n
            + (0..)
                .scan(n, |nn, _| {
                    if *nn < 2 {
                        return None;
                    }
                    let h = *nn / 2;
                    *nn /= 2;
                    Some(h.saturating_sub(1))
                })
                .sum::<usize>();

        // Split layout: each segment has m entries of ω (4 u64 each) then m entries of ωq (4 u64 each)
        // Total u64 count is same: 8 * total_entries
        let mut powomega: Vec<u64> = alloc_aligned::<u64>(8 * total_entries);
        powomega.resize(8 * total_entries, 0);
        let mut seg_base = 0usize; // base offset (in u64) for current segment

        if n <= 1 {
            return Self {
                n,
                q2,
                q4,
                powomega,
                root,
                root_quot,
                tail_root,
                tail_quot,
                _phantom: PhantomData,
            };
        }

        // ── Level 0: a[i] *= ω^i (n entries) ────────────────────────────
        {
            let omega_base = seg_base;
            let quot_base = seg_base + 4 * n;
            let mut pow_om: [u64; 3] = [1; 3]; // ω^0 = 1
            for i in 0..n {
                store_twiddle_split::<P>(&mut powomega, omega_base, quot_base, i, &pow_om);
                for k in 0..3 {
                    pow_om[k] = ((pow_om[k] as u128 * omega_vec[k] as u128) % P::Q[k] as u128) as u64;
                }
            }
            seg_base += 8 * n;
        }

        // ── Butterfly levels: nn = n, n/2, …, 2 ─────────────────────────
        let mut nn = n;
        while nn >= 2 {
            let halfnn = nn / 2;
            if halfnn > 1 {
                let count = halfnn - 1;
                let omega_base = seg_base;
                let quot_base = seg_base + 4 * count;
                let m = n / halfnn;
                let omega_m: [u64; 3] = std::array::from_fn(|k| modq_pow64(omega_vec[k], m as i64, P::Q[k]));
                let mut pow_om = omega_m;
                for i in 0..count {
                    store_twiddle_split::<P>(&mut powomega, omega_base, quot_base, i, &pow_om);
                    for k in 0..3 {
                        pow_om[k] = ((pow_om[k] as u128 * omega_m[k] as u128) % P::Q[k] as u128) as u64;
                    }
                }
                seg_base += 8 * count;
            }
            nn /= 2;
        }

        Self {
            n,
            q2,
            q4,
            powomega,
            root,
            root_quot,
            tail_root,
            tail_quot,
            _phantom: PhantomData,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Inverse NTT table construction
// ──────────────────────────────────────────────────────────────────────────────

impl<P: PrimeSetNtt126Ifma> Ntt126IfmaTableInv<P> {
    pub fn new(n: usize) -> Self {
        assert!(
            n.is_power_of_two() && n <= (1 << 16),
            "NTT size must be a power of two ≤ 2^16, got {n}"
        );

        let q2: [u64; 4] = [2 * P::Q[0], 2 * P::Q[1], 2 * P::Q[2], 0];
        let q4: [u64; 4] = [4 * P::Q[0], 4 * P::Q[1], 4 * P::Q[2], 0];
        let omega_vec = fill_omegas_ntt126_ifma::<P>(n);

        let (inv_root, inv_quot) = build_inv_root_table::<P>(n);

        // butterfly levels + last pass (n entries)
        let total_entries = n
            + (0..)
                .scan(2usize, |nn, _| {
                    if *nn > n {
                        return None;
                    }
                    let h = *nn / 2;
                    *nn *= 2;
                    Some(h.saturating_sub(1))
                })
                .sum::<usize>();

        let mut powomega: Vec<u64> = alloc_aligned::<u64>(8 * total_entries);
        powomega.resize(8 * total_entries, 0);
        let mut seg_base = 0usize;

        if n <= 1 {
            return Self {
                n,
                q2,
                q4,
                powomega,
                inv_root,
                inv_quot,
                _phantom: PhantomData,
            };
        }

        // ── Butterfly levels: nn = 2, 4, …, n ───────────────────────────
        let mut nn = 2usize;
        while nn <= n {
            let halfnn = nn / 2;
            if halfnn > 1 {
                let count = halfnn - 1;
                let omega_base = seg_base;
                let quot_base = seg_base + 4 * count;
                let m = n / halfnn;
                let omega_neg_m: [u64; 3] = std::array::from_fn(|k| modq_pow64(omega_vec[k], -(m as i64), P::Q[k]));
                let mut pow_om = omega_neg_m;
                for i in 0..count {
                    store_twiddle_split::<P>(&mut powomega, omega_base, quot_base, i, &pow_om);
                    for k in 0..3 {
                        pow_om[k] = ((pow_om[k] as u128 * omega_neg_m[k] as u128) % P::Q[k] as u128) as u64;
                    }
                }
                seg_base += 8 * count;
            }
            nn *= 2;
        }

        // ── Last pass: ω^{-i} / n (n entries) ──────────────────────────
        {
            let omega_base = seg_base;
            let quot_base = seg_base + 4 * n;
            let omega_inv: [u64; 3] = std::array::from_fn(|k| modq_pow64(omega_vec[k], -1, P::Q[k]));
            let n_inv: [u64; 3] = std::array::from_fn(|k| modq_pow64(n as u64, -1, P::Q[k]));
            let mut pow_om = n_inv; // i=0: just n^{-1}
            for i in 0..n {
                store_twiddle_split::<P>(&mut powomega, omega_base, quot_base, i, &pow_om);
                for k in 0..3 {
                    pow_om[k] = ((pow_om[k] as u128 * omega_inv[k] as u128) % P::Q[k] as u128) as u64;
                }
            }
        }

        Self {
            n,
            q2,
            q4,
            powomega,
            inv_root,
            inv_quot,
            _phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::primes::Primes42;
    use super::*;

    #[test]
    fn harvey_modmul_correctness() {
        for &q in &Primes42::Q {
            // Test with inputs in [0, 2q) — the IFMA-native range
            for a in [0u64, 1, q - 1, q, 2 * q - 1, q / 2, 42] {
                if a >= 2 * q {
                    continue;
                }
                for omega in [0u64, 1, q - 1, q / 2, 7] {
                    let omega_quot = harvey_quotient(omega, q);
                    let got = harvey_modmul(a, omega, omega_quot, q);
                    let expected = ((a as u128 * omega as u128) % q as u128) as u64;
                    assert!(
                        got % q == expected,
                        "harvey_modmul({a}, {omega}, q={q}): got {got} (mod q = {}), expected {expected}",
                        got % q
                    );
                    assert!(got < 2 * q, "harvey_modmul output {got} >= 2q={}", 2 * q);
                }
            }
        }
    }
}
