// ----------------------------------------------------------------------
// DISCLAIMER
//
// This module contains code that has been directly ported from the
// spqlios-arithmetic library
// (https://github.com/tfhe/spqlios-arithmetic), which is licensed
// under the Apache License, Version 2.0.
//
// The porting process from C to Rust was done with minimal changes
// in order to preserve the semantics and performance characteristics
// of the original implementation.
//
// Both Poulpy and spqlios-arithmetic are distributed under the terms
// of the Apache License, Version 2.0. See the LICENSE file for details.
//
// ----------------------------------------------------------------------

//! Q120 NTT precomputation and reference execution.
//!
//! This module is a direct Rust port of:
//! - `q120_ntt.c` (precomputation: [`NttTable`], [`NttTableInv`])
//! - `q120_ntt_avx2.c` (algorithm, re-expressed in scalar Rust: [`ntt_ref`], [`intt_ref`])
//!
//! # Data layout
//!
//! An NTT vector of size `n` is stored as a flat `&mut [u64]` of length
//! `4 * n`.  Each group of 4 consecutive `u64` values represents one
//! NTT coefficient in the q120b format — one value per CRT prime.
//!
//! # Algorithm overview
//!
//! The forward NTT consists of:
//! 1. **First pass**: multiply each coefficient `a[i]` by `ω^i` using the
//!    split-precomputed multiplication (`split_precompmul`).
//! 2. **Butterfly levels**: for `nn = n, n/2, …, 2`, apply the Cooley–Tukey
//!    DIT butterfly across all blocks of size `nn`.
//!
//! The inverse NTT reverses this: butterfly levels first (Gentleman–Sande
//! DIF), then a final element-wise multiply by `ω^{-i} / n`.
//!
//! Lazy Barrett reduction (`modq_red`) is applied at specific levels
//! when the accumulated bit-width would otherwise overflow 64 bits.
//!
//! # Correctness
//!
//! The reference implementation is intended as a correctness oracle.
//! For performance, the AVX2 backend in `poulpy-cpu-avx` will implement
//! the same algorithm using SIMD intrinsics that match the spqlios source.

use std::marker::PhantomData;

use crate::reference::ntt4x30::primes::PrimeSet;
use poulpy_hal::alloc_aligned;

// ──────────────────────────────────────────────────────────────────────────────
// Precomputation data structures
// ──────────────────────────────────────────────────────────────────────────────

/// Per-level metadata for a butterfly iteration.
///
/// The `q2bs[k]` constant is used to turn an unsigned subtraction
/// `a - b` into the equivalent `a + q2bs - b`, keeping all values
/// non-negative throughout the lazy arithmetic.
#[derive(Clone, Debug)]
pub struct NttStepMeta {
    /// `q2bs[k] = Q[k] << (bs - LOG_Q)` — the "multiple of Q" added
    /// before subtraction to keep the result non-negative.
    pub q2bs: [u64; 4],
    /// Output bit-size bound after this level.
    pub bs: u64,
    /// `ceil(bs / 2)`, used in `split_precompmul`.
    pub half_bs: u64,
    /// `(1 << half_bs) - 1`, pre-masked for `split_precompmul`.
    pub mask: u64,
    /// Whether a lazy Barrett reduction step is applied at this level.
    pub reduce: bool,
}

/// Precomputed constants for the lazy Barrett modular reduction.
///
/// For a value `x` with up to `bs_start` bits, the reduction computes:
/// ```text
/// x_reduced = (x & mask) + (x >> h) * modulo_red_cst[k]
/// ```
/// which is congruent to `x mod Q[k]` and has strictly fewer bits.
#[derive(Clone, Debug)]
pub struct NttReducMeta {
    /// `(2^h) mod Q[k]` for each prime — the "high-half correction".
    pub modulo_red_cst: [u64; 4],
    /// `(1 << h) - 1`.
    pub mask: u64,
    /// Split point: `x = (x >> h) * 2^h + (x & mask)`.
    pub h: u64,
}

/// Precomputed twiddle-factor table for the forward Q120 NTT.
///
/// Construct with [`NttTable::new`].
pub struct NttTable<P: PrimeSet> {
    /// NTT size (power of two, ≤ 2^16).
    pub n: usize,
    /// Per-level metadata (length = log2(n) + 1).
    pub level_metadata: Vec<NttStepMeta>,
    /// Packed twiddle factors in q120b layout (4 u64 per entry, 32-byte aligned).
    ///
    /// Packing: each u64 stores `(t1 << 32) | t` where `t = ω^i mod Q[k]`
    /// and `t1 = (t << half_bs) mod Q[k]`.  The four consecutive u64
    /// values (one per prime) form one logical entry.
    pub powomega: Vec<u64>,
    /// Reduction metadata (shared across all levels).
    pub reduc_metadata: NttReducMeta,
    /// Input bit-size (64 for q120b inputs).
    pub input_bit_size: u64,
    /// Output bit-size bound.
    pub output_bit_size: u64,
    _phantom: PhantomData<P>,
}

/// Precomputed twiddle-factor table for the inverse Q120 NTT.
///
/// Construct with [`NttTableInv::new`].
pub struct NttTableInv<P: PrimeSet> {
    /// NTT size (power of two, ≤ 2^16).
    pub n: usize,
    /// Per-level metadata (length = log2(n) + 1).
    pub level_metadata: Vec<NttStepMeta>,
    /// Packed inverse twiddle factors (same packing as `NttTable::powomega`).
    pub powomega: Vec<u64>,
    /// Reduction metadata.
    pub reduc_metadata: NttReducMeta,
    /// Input bit-size.
    pub input_bit_size: u64,
    /// Output bit-size bound.
    pub output_bit_size: u64,
    _phantom: PhantomData<P>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Precomputation
// ──────────────────────────────────────────────────────────────────────────────

/// Computes `x^n mod q` using square-and-multiply.
///
/// Negative exponents compute the modular inverse via `x^(-(|n| mod (q-1)))`.
pub fn modq_pow(x: u32, n: i64, q: u32) -> u32 {
    let qm1 = (q - 1) as i64;
    // reduce exponent mod (q-1) to positive representative
    let np = ((n % qm1) + qm1) % qm1;
    let mut np = np as u64;
    let mut val_pow = x as u64;
    let q64 = q as u64;
    let mut res: u64 = 1;
    while np != 0 {
        if np & 1 != 0 {
            res = (res * val_pow) % q64;
        }
        val_pow = (val_pow * val_pow) % q64;
        np >>= 1;
    }
    res as u32
}

/// Returns the primitive `2n`-th roots of unity for each prime.
fn fill_omegas<P: PrimeSet>(n: usize) -> [u32; 4] {
    debug_assert!((1..=(1 << 16)).contains(&n), "n must be a power of two in [1, 2^16], got {n}");
    std::array::from_fn(|k| modq_pow(P::OMEGA[k], (1i64 << 16) / n as i64, P::Q[k]))
}

/// Finds the optimal `h` for the lazy Barrett reduction step.
///
/// Given that inputs have `bs_start` bits, finds `h ∈ [bs_start/2, bs_start)`
/// that minimises the output bit-size after reduction.
///
/// Returns `(NttReducMeta, bs_after_reduc)`.
fn fill_reduction_meta<P: PrimeSet>(bs_start: u64) -> (NttReducMeta, u64) {
    let mut bs_after_reduc = u64::MAX;
    let mut min_h = bs_start / 2;

    for h in bs_start / 2..bs_start {
        let mut t = 0u64;
        for k in 0..4 {
            let q = P::Q[k] as u64;
            // (2^h) mod Q[k]
            let pow_h_mod_q = pow2_mod(h, q);
            let pow_h_bs = if pow_h_mod_q <= 1 { 0u64 } else { ceil_log2_u64(pow_h_mod_q) };
            let t1 = bs_start - h + pow_h_bs;
            let t2 = 1 + t1.max(h);
            if t < t2 {
                t = t2;
            }
        }
        if t < bs_after_reduc {
            min_h = h;
            bs_after_reduc = t;
        }
    }

    let mask = (1u64 << min_h) - 1;
    let modulo_red_cst: [u64; 4] = std::array::from_fn(|k| pow2_mod(min_h, P::Q[k] as u64));

    (
        NttReducMeta {
            modulo_red_cst,
            mask,
            h: min_h,
        },
        bs_after_reduc,
    )
}

/// Pack a twiddle factor into the 64-bit encoding `(t1 << 32) | t`.
#[inline(always)]
fn pack_omega(t: u64, half_bs: u64, q: u64) -> u64 {
    let t1 = (t << half_bs) % q;
    (t1 << 32) | t
}

impl<P: PrimeSet> NttTable<P> {
    /// Builds the forward NTT precomputation table for size `n`.
    ///
    /// `n` must be a power of two with `1 ≤ n ≤ 2^16`.
    pub fn new(n: usize) -> Self {
        assert!(
            n.is_power_of_two() && n <= (1 << 16),
            "NTT size must be a power of two ≤ 2^16, got {n}"
        );

        let omega_vec = fill_omegas::<P>(n);
        let log_q = P::LOG_Q;
        let mut bs = 64u64; // input_bit_size

        let (reduc_metadata, bs_after_reduc) = fill_reduction_meta::<P>(bs);

        let input_bit_size = bs;

        let powomega_capacity = alloc_aligned::<u64>(4 * 2 * n.max(1));
        let mut powomega: Vec<u64> = powomega_capacity;
        let mut po_ptr = 0usize; // index into powomega

        let mut level_metadata: Vec<NttStepMeta> = Vec::new();

        if n == 1 {
            return Self {
                n,
                level_metadata,
                powomega,
                reduc_metadata,
                input_bit_size,
                output_bit_size: bs,
                _phantom: PhantomData,
            };
        }

        // ── Level 0: first pass  a[i] *= ω^i ──────────────────────────────
        {
            let half_bs = bs.div_ceil(2);
            let mask = (1u64 << half_bs) - 1;
            bs = half_bs + log_q + 1;
            level_metadata.push(NttStepMeta {
                q2bs: [0; 4],
                bs,
                half_bs,
                mask,
                reduce: false,
            });
        }

        // Fill powomega for level 0: n entries, one per coefficient.
        // powomega[i] for prime k: pack(ω^i mod Q[k], half_bs, Q[k])
        let half_bs_0 = level_metadata[0].half_bs;
        {
            // Compute ω^i mod Q[k] for all i and k using successive multiplication.
            let mut pow_om: [u64; 4] = [1; 4]; // ω^0 = 1
            for i in 0..n {
                for k in 0..4 {
                    let q = P::Q[k] as u64;
                    powomega[po_ptr + 4 * i + k] = pack_omega(pow_om[k], half_bs_0, q);
                }
                // Advance: pow_om[k] *= omega_vec[k]
                for k in 0..4 {
                    pow_om[k] = (pow_om[k] * omega_vec[k] as u64) % P::Q[k] as u64;
                }
            }
            po_ptr += 4 * n;
        }

        // ── Levels 1..logN: butterfly levels nn = n, n/2, …, 2 ──────────
        let mut nn = n;
        while nn >= 2 {
            let halfnn = nn / 2;

            let do_reduce = bs == 64;
            if do_reduce {
                bs = bs_after_reduc;
            }

            let q2bs: [u64; 4] = std::array::from_fn(|k| (P::Q[k] as u64) << (bs - log_q));

            let (new_bs, half_bs, mask) = if nn >= 4 {
                let bs1 = bs + 1; // bit-size after a+b or a-b
                let half_bs = bs1.div_ceil(2);
                let bs2 = half_bs + log_q + 1; // bit-size after (a-b)*ω
                let new_bs = bs1.max(bs2);
                assert!(new_bs <= 64, "NTT bit-size overflow at level nn={nn}");
                (new_bs, half_bs, (1u64 << half_bs) - 1)
            } else {
                // nn == 2, last level: only a+b / a-b, no twiddle multiply
                let new_bs = bs + 1;
                (new_bs, 0, 0)
            };

            level_metadata.push(NttStepMeta {
                q2bs,
                bs: new_bs,
                half_bs,
                mask,
                reduce: do_reduce,
            });
            bs = new_bs;

            // Fill powomega for this level: (halfnn-1) entries (i=1..halfnn-1)
            // The stride between consecutive omega powers is m = 2n / nn.
            if halfnn > 1 {
                let m = n / halfnn; // = 2n/nn, step in omega table
                // We need ω^{i*m} for i=1..halfnn-1.
                // Compute successively: start at ω^m, multiply by ω^m each step.
                let omega_m: [u64; 4] = std::array::from_fn(|k| modq_pow(omega_vec[k], m as i64, P::Q[k]) as u64);
                let mut pow_om: [u64; 4] = omega_m; // ω^{1*m}
                let half_bs_level = level_metadata.last().unwrap().half_bs;
                for i in 0..halfnn - 1 {
                    for k in 0..4 {
                        let q = P::Q[k] as u64;
                        powomega[po_ptr + 4 * i + k] = pack_omega(pow_om[k], half_bs_level, q);
                    }
                    // Advance pow_om
                    for k in 0..4 {
                        pow_om[k] = (pow_om[k] * omega_m[k]) % P::Q[k] as u64;
                    }
                }
                po_ptr += 4 * (halfnn - 1);
            }

            nn /= 2;
        }

        let output_bit_size = bs;
        powomega.truncate(po_ptr); // trim to used portion

        Self {
            n,
            level_metadata,
            powomega,
            reduc_metadata,
            input_bit_size,
            output_bit_size,
            _phantom: PhantomData,
        }
    }
}

impl<P: PrimeSet> NttTableInv<P> {
    /// Builds the inverse NTT precomputation table for size `n`.
    ///
    /// `n` must be a power of two with `1 ≤ n ≤ 2^16`.
    pub fn new(n: usize) -> Self {
        assert!(
            n.is_power_of_two() && n <= (1 << 16),
            "iNTT size must be a power of two ≤ 2^16, got {n}"
        );

        let omega_vec = fill_omegas::<P>(n);
        let log_q = P::LOG_Q;
        let mut bs = 64u64;

        let (reduc_metadata, bs_after_reduc) = fill_reduction_meta::<P>(bs);

        let input_bit_size = bs;
        let powomega_capacity = alloc_aligned::<u64>(4 * 2 * n.max(1));
        let mut powomega: Vec<u64> = powomega_capacity;
        let mut po_ptr = 0usize;

        let mut level_metadata: Vec<NttStepMeta> = Vec::new();

        if n == 1 {
            return Self {
                n,
                level_metadata,
                powomega,
                reduc_metadata,
                input_bit_size,
                output_bit_size: bs,
                _phantom: PhantomData,
            };
        }

        // ── Level 0: first butterfly level nn=2 (just a+b, a-b, no twiddle) ─
        {
            let do_reduce = bs == 64;
            if do_reduce {
                bs = bs_after_reduc;
            }
            let q2bs: [u64; 4] = std::array::from_fn(|k| (P::Q[k] as u64) << (bs - log_q));
            let new_bs = bs + 1;
            level_metadata.push(NttStepMeta {
                q2bs,
                bs: new_bs,
                half_bs: 0,
                mask: 0,
                reduce: do_reduce,
            });
            bs = new_bs;
        }
        // No omega entries for level 0 (halfnn=1, no twiddle positions).

        // ── Levels 1..logN-1: butterfly levels nn=4,8,..,n ────────────────
        let mut nn = 4usize;
        while nn <= n {
            let halfnn = nn / 2;

            let do_reduce = bs == 64;
            if do_reduce {
                bs = bs_after_reduc;
            }

            let half_bs = bs.div_ceil(2);
            let bs_mult = half_bs + log_q + 1; // bit-size of b*ω^k
            let new_bs = 1 + bs.max(bs_mult); // bit-size of a ± b*ω^k
            assert!(new_bs <= 64, "iNTT bit-size overflow at level nn={nn}");

            let q2bs: [u64; 4] = std::array::from_fn(|k| (P::Q[k] as u64) << (bs_mult - log_q));
            let mask = (1u64 << half_bs) - 1;
            level_metadata.push(NttStepMeta {
                q2bs,
                bs: new_bs,
                half_bs,
                mask,
                reduce: do_reduce,
            });
            bs = new_bs;

            // Fill omega entries for this level: halfnn-1 entries (i=1..halfnn-1)
            // using inverse ω^{-i*m} for m = 2n/nn.
            let m = n / halfnn;
            let omega_inv_m: [u64; 4] = std::array::from_fn(|k| modq_pow(omega_vec[k], -(m as i64), P::Q[k]) as u64);
            let mut pow_om: [u64; 4] = omega_inv_m; // ω^{-1*m}
            let half_bs_level = level_metadata.last().unwrap().half_bs;
            for i in 0..halfnn - 1 {
                for k in 0..4 {
                    let q = P::Q[k] as u64;
                    powomega[po_ptr + 4 * i + k] = pack_omega(pow_om[k], half_bs_level, q);
                }
                for k in 0..4 {
                    pow_om[k] = (pow_om[k] * omega_inv_m[k]) % P::Q[k] as u64;
                }
            }
            po_ptr += 4 * (halfnn - 1);

            nn *= 2;
        }

        // ── Last level: element-wise multiply by ω^{-i} * n^{-1} ──────────
        {
            let do_reduce = bs == 64;
            if do_reduce {
                bs = bs_after_reduc;
            }

            let half_bs = bs.div_ceil(2);
            let new_bs = half_bs + log_q + 1;
            assert!(new_bs <= 64, "iNTT bit-size overflow at last level");

            let q2bs: [u64; 4] = std::array::from_fn(|k| (P::Q[k] as u64) << (new_bs - log_q));
            let mask = (1u64 << half_bs) - 1;
            level_metadata.push(NttStepMeta {
                q2bs,
                bs: new_bs,
                half_bs,
                mask,
                reduce: do_reduce,
            });
            bs = new_bs;

            // Omega entries: n values of ω^{-i} * n^{-1} for i = 0..n-1
            for k in 0..4 {
                let q = P::Q[k] as u64;
                let inv_n = modq_pow(n as u32, -1, P::Q[k]) as u64;
                // ω_inv = ω^{-1}
                let omega_inv = modq_pow(omega_vec[k], -1, P::Q[k]) as u64;
                let mut pow_om = inv_n; // ω^{-0} * n^{-1}
                for i in 0..n {
                    powomega[po_ptr + 4 * i + k] = pack_omega(pow_om, half_bs, q);
                    pow_om = (pow_om * omega_inv) % q;
                }
            }
            po_ptr += 4 * n;
        }

        let output_bit_size = bs;
        powomega.truncate(po_ptr);

        Self {
            n,
            level_metadata,
            powomega,
            reduc_metadata,
            input_bit_size,
            output_bit_size,
            _phantom: PhantomData,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Core arithmetic primitives (scalar, no SIMD)
// ──────────────────────────────────────────────────────────────────────────────

/// Split-precomputed multiplication.
///
/// Computes an approximation of `inp * ω mod Q` where `ω` is encoded in
/// `powomega_packed = (t1 << 32) | t` with `t = ω mod Q` and
/// `t1 = (ω * 2^{half_bs}) mod Q`.
///
/// The result is `inp_low * t + inp_high * t1` which equals `inp * ω mod Q`
/// up to multiples of Q (the exact value may exceed Q but stays < 2^64
/// under the bit-width guarantees tracked by [`NttStepMeta`]).
#[inline(always)]
pub fn split_precompmul(inp: u64, powomega_packed: u64, half_bs: u64, mask: u64) -> u64 {
    let inp_low = inp & mask;
    let t = powomega_packed & 0xFFFF_FFFF; // low 32 bits = ω mod Q (32-bit prime)
    let t1 = powomega_packed >> 32; // high 32 bits = (ω << half_bs) mod Q
    inp_low.wrapping_mul(t).wrapping_add((inp >> half_bs).wrapping_mul(t1))
}

/// Lazy Barrett-style modular reduction.
///
/// Given `x` with up to `bs_start` bits:
/// ```text
/// x_reduced = (x & mask) + (x >> h) * modulo_red_cst
/// ```
/// The result is congruent to `x mod Q` with fewer bits.
#[inline(always)]
pub fn modq_red(x: u64, h: u64, mask: u64, cst: u64) -> u64 {
    (x & mask).wrapping_add((x >> h).wrapping_mul(cst))
}

// ──────────────────────────────────────────────────────────────────────────────
// Forward NTT
// ──────────────────────────────────────────────────────────────────────────────

/// Forward Q120 NTT on a polynomial of `n` coefficients (reference implementation).
///
/// `data` must be a flat `u64` slice of length `4 * n` in q120b layout.
/// After the call, each group of 4 consecutive u64 values holds the NTT
/// evaluation at the corresponding point, in the same q120b layout.
///
/// # Panics
/// Panics in debug mode if `data.len() < 4 * table.n`.
pub fn ntt_ref<P: PrimeSet>(table: &NttTable<P>, data: &mut [u64]) {
    let n = table.n;
    if n == 1 {
        return;
    }

    debug_assert!(data.len() >= 4 * n);

    let mut po_off = 0usize; // current offset into table.powomega
    let mut meta_idx = 0usize;

    // ── Level 0: a[i] *= ω^i (split_precompmul, no butterfly) ───────────
    {
        let meta = &table.level_metadata[meta_idx];
        let h = meta.half_bs;
        let mask = meta.mask;
        for i in 0..n {
            for k in 0..4 {
                let x = data[4 * i + k];
                let po = table.powomega[po_off + 4 * i + k];
                data[4 * i + k] = split_precompmul(x, po, h, mask);
            }
        }
        po_off += 4 * n;
        meta_idx += 1;
    }

    // ── Butterfly levels: nn = n, n/2, …, 2 ──────────────────────────────
    let mut nn = n;
    while nn >= 2 {
        let halfnn = nn / 2;
        let meta = &table.level_metadata[meta_idx];

        // Process all blocks of size nn.
        let mut blk = 0;
        while blk < n {
            ntt_butterfly_block(data, blk, halfnn, meta, &table.reduc_metadata, &table.powomega, po_off);
            blk += nn;
        }

        po_off += 4 * halfnn.saturating_sub(1);
        meta_idx += 1;
        nn /= 2;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Inverse NTT
// ──────────────────────────────────────────────────────────────────────────────

/// Inverse Q120 NTT on a polynomial of `n` coefficients (reference implementation).
///
/// `data` must be a flat `u64` slice of length `4 * n` in q120b layout
/// (the output of [`ntt_ref`]).  After the call, each group of 4 u64
/// values holds the recovered coefficient (in q120b), scaled by 1 (the
/// `n^{-1}` factor is baked into the last-pass twiddle table).
///
/// # Panics
/// Panics in debug mode if `data.len() < 4 * table.n`.
pub fn intt_ref<P: PrimeSet>(table: &NttTableInv<P>, data: &mut [u64]) {
    let n = table.n;
    if n == 1 {
        return;
    }

    debug_assert!(data.len() >= 4 * n);

    let mut po_off = 0usize;
    let mut meta_idx = 0usize;

    // ── Butterfly levels: nn = 2, 4, …, n ────────────────────────────────
    let log_n = n.trailing_zeros() as usize;

    // Level 0 → nn=2 (no twiddles, po_off unchanged; meta.half_bs = 0, meta.mask = 0)
    {
        let meta = &table.level_metadata[meta_idx];
        let mut blk = 0;
        while blk < n {
            intt_butterfly_block(data, blk, 1, meta, &table.reduc_metadata, &table.powomega, po_off);
            blk += 2;
        }
        // po_off unchanged (halfnn-1 = 0 entries)
        meta_idx += 1;
    }

    // Levels 1..logN-1 → nn=4,8,..,n
    let mut nn = 4usize;
    for _ in 1..log_n {
        let halfnn = nn / 2;
        let meta = &table.level_metadata[meta_idx];

        let mut blk = 0;
        while blk < n {
            intt_butterfly_block(data, blk, halfnn, meta, &table.reduc_metadata, &table.powomega, po_off);
            blk += nn;
        }

        po_off += 4 * (halfnn - 1);
        meta_idx += 1;
        nn *= 2;
    }

    // ── Last pass: a[i] *= ω^{-i} * n^{-1} (split_precompmul) ───────────
    {
        let meta = &table.level_metadata[meta_idx];
        let h = meta.half_bs;
        let mask = meta.mask;
        let do_reduce = meta.reduce;

        for i in 0..n {
            for k in 0..4 {
                let x = if do_reduce {
                    modq_red(
                        data[4 * i + k],
                        table.reduc_metadata.h,
                        table.reduc_metadata.mask,
                        table.reduc_metadata.modulo_red_cst[k],
                    )
                } else {
                    data[4 * i + k]
                };
                let po = table.powomega[po_off + 4 * i + k];
                data[4 * i + k] = split_precompmul(x, po, h, mask);
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Shared butterfly helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Forward NTT butterfly for one block of size `2*halfnn` starting at `data[4*blk..]`.
///
/// For i=0: `(a, b) → (a+b, a + q2bs - b)` (no twiddle).
/// For i=1..halfnn-1: `(a, b) → (a+b, split_precompmul(a + q2bs - b, ω^i, …))`.
#[inline(always)]
fn ntt_butterfly_block(
    data: &mut [u64],
    blk: usize,    // block start (coefficient index)
    halfnn: usize, // half the block size
    meta: &NttStepMeta,
    reduc: &NttReducMeta,
    powomega: &[u64],
    po_off: usize, // offset to the first twiddle entry for this level
) {
    let q2bs = meta.q2bs;
    let h = meta.half_bs;
    let mask = meta.mask;
    let do_reduce = meta.reduce;

    // i = 0: no twiddle factor.
    for (k, &q2bs_k) in q2bs.iter().enumerate() {
        let idx_a = 4 * blk + k;
        let idx_b = 4 * (blk + halfnn) + k;
        let a = if do_reduce {
            modq_red(data[idx_a], reduc.h, reduc.mask, reduc.modulo_red_cst[k])
        } else {
            data[idx_a]
        };
        let b = if do_reduce {
            modq_red(data[idx_b], reduc.h, reduc.mask, reduc.modulo_red_cst[k])
        } else {
            data[idx_b]
        };
        data[idx_a] = a.wrapping_add(b);
        data[idx_b] = a.wrapping_add(q2bs_k).wrapping_sub(b);
    }

    // i = 1..halfnn: with twiddle factor (only if halfnn > 1).
    if halfnn > 1 {
        for i in 1..halfnn {
            for k in 0..4 {
                let idx_a = 4 * (blk + i) + k;
                let idx_b = 4 * (blk + halfnn + i) + k;
                let a = if do_reduce {
                    modq_red(data[idx_a], reduc.h, reduc.mask, reduc.modulo_red_cst[k])
                } else {
                    data[idx_a]
                };
                let b = if do_reduce {
                    modq_red(data[idx_b], reduc.h, reduc.mask, reduc.modulo_red_cst[k])
                } else {
                    data[idx_b]
                };
                data[idx_a] = a.wrapping_add(b);
                let b1 = a.wrapping_add(q2bs[k]).wrapping_sub(b);
                data[idx_b] = split_precompmul(b1, powomega[po_off + 4 * (i - 1) + k], h, mask);
            }
        }
    }
}

/// Inverse NTT (GS DIF) butterfly for one block of size `2*halfnn`.
///
/// For i=0: `(a, b) → (a+b, a + q2bs - b)` (no twiddle).
/// For i=1..halfnn-1: twiddle is applied to `b` *before* the butterfly:
///   `bo = split_precompmul(b, ω^{-i}, …)`, then `(a, bo) → (a+bo, a + q2bs - bo)`.
#[inline(always)]
fn intt_butterfly_block(
    data: &mut [u64],
    blk: usize,
    halfnn: usize,
    meta: &NttStepMeta,
    reduc: &NttReducMeta,
    powomega: &[u64],
    po_off: usize,
) {
    let q2bs = meta.q2bs;
    let h = meta.half_bs;
    let mask = meta.mask;
    let do_reduce = meta.reduce;

    // i = 0: no twiddle factor.
    for (k, &q2bs_k) in q2bs.iter().enumerate() {
        let idx_a = 4 * blk + k;
        let idx_b = 4 * (blk + halfnn) + k;
        let a = if do_reduce {
            modq_red(data[idx_a], reduc.h, reduc.mask, reduc.modulo_red_cst[k])
        } else {
            data[idx_a]
        };
        let bo = if do_reduce {
            modq_red(data[idx_b], reduc.h, reduc.mask, reduc.modulo_red_cst[k])
        } else {
            data[idx_b]
        };
        data[idx_a] = a.wrapping_add(bo);
        data[idx_b] = a.wrapping_add(q2bs_k).wrapping_sub(bo);
    }

    // i = 1..halfnn: apply twiddle to b before the butterfly (only if halfnn > 1).
    if halfnn > 1 {
        for i in 1..halfnn {
            for k in 0..4 {
                let idx_a = 4 * (blk + i) + k;
                let idx_b = 4 * (blk + halfnn + i) + k;
                let a = if do_reduce {
                    modq_red(data[idx_a], reduc.h, reduc.mask, reduc.modulo_red_cst[k])
                } else {
                    data[idx_a]
                };
                let b_raw = if do_reduce {
                    modq_red(data[idx_b], reduc.h, reduc.mask, reduc.modulo_red_cst[k])
                } else {
                    data[idx_b]
                };
                let bo = split_precompmul(b_raw, powomega[po_off + 4 * (i - 1) + k], h, mask);
                data[idx_a] = a.wrapping_add(bo);
                data[idx_b] = a.wrapping_add(q2bs[k]).wrapping_sub(bo);
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

use super::pow2_mod;

/// `ceil(log2(x))` for `x ≥ 1`.
fn ceil_log2_u64(x: u64) -> u64 {
    if x <= 1 {
        return 0;
    }
    let floor_log2 = 63 - x.leading_zeros() as u64;
    if x.is_power_of_two() { floor_log2 } else { floor_log2 + 1 }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::ntt4x30::{
        arithmetic::{b_from_znx64_ref, b_to_znx128_ref},
        primes::Primes30,
    };

    /// Verify that NTT followed by iNTT is the identity on a polynomial
    /// with small coefficients (fitting comfortably in q120b).
    #[test]
    fn ntt_intt_identity() {
        for log_n in 1..=8usize {
            let n = 1 << log_n;
            let fwd = NttTable::<Primes30>::new(n);
            let inv = NttTableInv::<Primes30>::new(n);

            // Polynomial with small random-ish coefficients in [-100, 100].
            let coeffs: Vec<i64> = (0..n as i64).map(|i| (i * 7 + 3) % 201 - 100).collect();

            let mut data = vec![0u64; 4 * n];
            b_from_znx64_ref::<Primes30>(n, &mut data, &coeffs);

            // Save a copy for comparison.
            let data_orig = data.clone();

            ntt_ref::<Primes30>(&fwd, &mut data);
            intt_ref::<Primes30>(&inv, &mut data);

            // After round-trip, the q120b representation should match (mod each Q[k]).
            for i in 0..n {
                for k in 0..4 {
                    let orig = data_orig[4 * i + k] % Primes30::Q[k] as u64;
                    let got = data[4 * i + k] % Primes30::Q[k] as u64;
                    assert_eq!(orig, got, "n={n} i={i} k={k}: mismatch after NTT+iNTT round-trip");
                }
            }
        }
    }

    /// Verify the NTT-based convolution: (A*B)(X) mod (X^n + 1) over Z/Q.
    ///
    /// The NTT diagonalises the cyclic-negacyclic convolution, so
    /// `iNTT(NTT(a) ⊙ NTT(b)) = a * b mod (X^n + 1)`.
    #[test]
    fn ntt_convolution() {
        let n = 8usize;
        let fwd = NttTable::<Primes30>::new(n);
        let inv = NttTableInv::<Primes30>::new(n);

        // a = [1, 2, 0, 0, …], b = [3, 4, 0, 0, …]
        // a*b mod (X^8+1) = [3, 10, 8, 0, …]
        let a: Vec<i64> = [1, 2, 0, 0, 0, 0, 0, 0].to_vec();
        let b: Vec<i64> = [3, 4, 0, 0, 0, 0, 0, 0].to_vec();

        let mut da = vec![0u64; 4 * n];
        let mut db = vec![0u64; 4 * n];
        b_from_znx64_ref::<Primes30>(n, &mut da, &a);
        b_from_znx64_ref::<Primes30>(n, &mut db, &b);

        ntt_ref::<Primes30>(&fwd, &mut da);
        ntt_ref::<Primes30>(&fwd, &mut db);

        // Pointwise multiply (mod each Q[k])
        let mut dc = vec![0u64; 4 * n];
        for i in 0..n {
            for k in 0..4 {
                let q = Primes30::Q[k] as u64;
                dc[4 * i + k] = (da[4 * i + k] % q * (db[4 * i + k] % q)) % q;
            }
        }

        intt_ref::<Primes30>(&inv, &mut dc);

        let mut result = vec![0i128; n];
        b_to_znx128_ref::<Primes30>(n, &mut result, &dc);

        let expected: Vec<i128> = [3, 10, 8, 0, 0, 0, 0, 0].to_vec();
        assert_eq!(result, expected, "NTT convolution mismatch");
    }
}
