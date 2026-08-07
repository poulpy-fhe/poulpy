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

//! Lazy-accumulation matrix–vector dot products in Q120 arithmetic.
//!
//! These functions are direct Rust ports of `q120_arithmetic_ref.c`
//! in spqlios-arithmetic.  They implement:
//!
//! - **baa**: inner product of q120a × q120a → q120b (inputs in `[0, 2^32)`)
//! - **bbb**: inner product of q120b × q120b → q120b (inputs in `[0, 2^64)`)
//! - **bbc**: inner product of q120b × q120c → q120b (NTT × prepared-const)
//! - **x2** / **2cols** variants that process two output elements at once.
//! - Block extract/save helpers.
//!
//! The accumulation is designed so that `ell < 10 000` inner products
//! fit without overflow (all intermediate sums stay below 64 bits).

use crate::reference::ntt4x30::primes::PrimeSetCrt4;

// ──────────────────────────────────────────────────────────────────────────────
// Precomputed metadata
// ──────────────────────────────────────────────────────────────────────────────

/// Precomputed metadata for the q120a × q120a → q120b dot product.
///
/// Constructed once (per prime set) and reused for any `ell < 10 000`.
pub struct BaaMeta<P: PrimeSetCrt4> {
    pub h: u64,
    pub h_pow_red: [u64; 4], // (2^h) % Q[k]
    _phantom: std::marker::PhantomData<P>,
}

impl<P: PrimeSetCrt4> BaaMeta<P> {
    /// Computes the optimal split point `h` that minimises the output
    /// bit-width for an accumulation of up to `MAX_ELL = 10 000` terms.
    pub fn new() -> Self {
        const MAX_ELL: f64 = 10_000.0;
        let ell_bs = MAX_ELL.log2();

        let mut min_res_bs = f64::MAX;
        let mut min_h = 0u64;

        for h in 1u64..64 {
            let h_pow2_bs = compute_bit_size_red(h, P::Q);
            // S1 has (h + ell_bs) bits, S2 has (64-h + ell_bs + h_pow2_bs) bits
            let res_bs = log2_sum_two(h as f64 + ell_bs, (64.0 - h as f64) + ell_bs + h_pow2_bs);
            if res_bs < min_res_bs {
                min_res_bs = res_bs;
                min_h = h;
            }
        }

        let h_pow_red: [u64; 4] = std::array::from_fn(|k| {
            let q = P::Q[k] as u64;
            pow2_mod(min_h, q)
        });

        Self {
            h: min_h,
            h_pow_red,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<P: PrimeSetCrt4> Default for BaaMeta<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Precomputed metadata for the q120b × q120b → q120b dot product.
pub struct BbbMeta<P: PrimeSetCrt4> {
    pub h: u64,
    pub s1h_pow_red: u64,      // 2^h (prime-independent)
    pub s2l_pow_red: [u64; 4], // 2^32 mod Q[k]
    pub s2h_pow_red: [u64; 4], // 2^(32+h) mod Q[k]
    pub s3l_pow_red: [u64; 4], // 2^64 mod Q[k]
    pub s3h_pow_red: [u64; 4], // 2^(64+h) mod Q[k]
    pub s4l_pow_red: [u64; 4], // 2^96 mod Q[k]
    pub s4h_pow_red: [u64; 4], // 2^(96+h) mod Q[k]
    _phantom: std::marker::PhantomData<P>,
}

impl<P: PrimeSetCrt4> BbbMeta<P> {
    /// Computes the optimal `h` for the four-term accumulation scheme.
    pub fn new() -> Self {
        const MAX_ELL: f64 = 10_000.0;
        let ell_bs = MAX_ELL.log2();
        let pow2_32_bs = compute_bit_size_red(32, P::Q);

        let s1_bs = 32.0 + ell_bs;
        let s2_bs = 32.0 + ell_bs + 3.0f64.log2(); // +log2(3) from Ah+Bl+Cl
        let s3_bs = 32.0 + ell_bs + 3.0f64.log2();
        let s4_bs = 32.0 + ell_bs;

        let mut min_res_bs = f64::MAX;
        let mut min_h = 16u64;

        for h in 16u64..32 {
            let s1l_bs = h as f64;
            let s1h_bs = (s1_bs - h as f64) + compute_bit_size_red(h, P::Q);
            let s2l_bs = h as f64 + pow2_32_bs;
            let s2h_bs = (s2_bs - h as f64) + compute_bit_size_red(32 + h, P::Q);
            let s3l_bs = h as f64 + compute_bit_size_red(64, P::Q);
            let s3h_bs = (s3_bs - h as f64) + compute_bit_size_red(64 + h, P::Q);
            let s4l_bs = h as f64 + compute_bit_size_red(96, P::Q);
            let s4h_bs = (s4_bs - h as f64) + compute_bit_size_red(96 + h, P::Q);

            let res_bs = log2_sum_n(&[s1l_bs, s1h_bs, s2l_bs, s2h_bs, s3l_bs, s3h_bs, s4l_bs, s4h_bs]);
            if res_bs < min_res_bs {
                min_res_bs = res_bs;
                min_h = h;
            }
        }

        let s1h_pow_red: u64 = 1u64 << min_h; // prime-independent: 2^h is the same for all primes
        let s2l_pow_red: [u64; 4] = std::array::from_fn(|k| pow2_mod(32, P::Q[k] as u64));
        let s2h_pow_red: [u64; 4] = std::array::from_fn(|k| {
            let q = P::Q[k] as u64;
            (s2l_pow_red[k] * s1h_pow_red) % q
        });
        let s3l_pow_red: [u64; 4] = std::array::from_fn(|k| {
            let q = P::Q[k] as u64;
            (s2l_pow_red[k] * s2l_pow_red[k]) % q
        });
        let s3h_pow_red: [u64; 4] = std::array::from_fn(|k| {
            let q = P::Q[k] as u64;
            (s3l_pow_red[k] * s1h_pow_red) % q
        });
        let s4l_pow_red: [u64; 4] = std::array::from_fn(|k| {
            let q = P::Q[k] as u64;
            (s3l_pow_red[k] * s2l_pow_red[k]) % q
        });
        let s4h_pow_red: [u64; 4] = std::array::from_fn(|k| {
            let q = P::Q[k] as u64;
            (s4l_pow_red[k] * s1h_pow_red) % q
        });

        Self {
            h: min_h,
            s1h_pow_red,
            s2l_pow_red,
            s2h_pow_red,
            s3l_pow_red,
            s3h_pow_red,
            s4l_pow_red,
            s4h_pow_red,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<P: PrimeSetCrt4> Default for BbbMeta<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Precomputed metadata for the q120b × q120c → q120b dot product.
pub struct BbcMeta<P: PrimeSetCrt4> {
    pub h: u64,
    pub s2l_pow_red: [u64; 4], // 2^32 mod Q[k]
    pub s2h_pow_red: [u64; 4], // 2^(32+h) mod Q[k]
    _phantom: std::marker::PhantomData<P>,
}

impl<P: PrimeSetCrt4> BbcMeta<P> {
    /// Computes the optimal `h` for the two-term accumulation scheme.
    pub fn new() -> Self {
        const MAX_ELL: f64 = 10_000.0;
        let ell_bs = MAX_ELL.log2();
        let pow2_32_bs = compute_bit_size_red(32, P::Q);
        let s1_bs = 32.0 + ell_bs;

        let mut min_res_bs = f64::MAX;
        let mut min_h = 16u64;

        for h in 16u64..32 {
            let s2l_bs = pow2_32_bs + h as f64;
            let s2h_bs = (s1_bs - h as f64) + compute_bit_size_red(32 + h, P::Q);
            let res_bs = log2_sum_n(&[s1_bs, s2l_bs, s2h_bs]);
            if res_bs < min_res_bs {
                min_res_bs = res_bs;
                min_h = h;
            }
        }

        let s2l_pow_red: [u64; 4] = std::array::from_fn(|k| pow2_mod(32, P::Q[k] as u64));
        let s2h_pow_red: [u64; 4] = std::array::from_fn(|k| pow2_mod(32 + min_h, P::Q[k] as u64));

        Self {
            h: min_h,
            s2l_pow_red,
            s2h_pow_red,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<P: PrimeSetCrt4> Default for BbcMeta<P> {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Dot-product kernels
//
// PERF POLICY: `assert!` guards appear at public function entry where an
// incorrect slice length would silently access out-of-bounds memory in a
// release build.  Pure arithmetic inner loops (no slice indexing beyond the
// entry check) use `debug_assert!` to avoid redundant bounds work per
// iteration.
// ──────────────────────────────────────────────────────────────────────────────

/// Computes `res = sum_{i=0}^{ell-1} x[i] * y[i]` in Q120b format,
/// where `x` and `y` are in q120a (values in `[0, 2^32)`).
///
/// `ell` must be < 10 000.
///
/// Inputs: `x` and `y` as flat `u32` slices with stride 4 (one group
/// of 4 per ring element), `res` as a `u64` slice of length 4.
pub fn vec_mat1col_product_baa_ref<P: PrimeSetCrt4>(meta: &BaaMeta<P>, ell: usize, res: &mut [u64], x: &[u32], y: &[u32]) {
    assert!(res.len() >= 4);
    assert!(x.len() >= 4 * ell);
    assert!(y.len() >= 4 * ell);

    let h = meta.h;
    let mask = (1u64 << h) - 1;

    let mut acc1 = [0u64; 4];
    let mut acc2 = [0u64; 4];

    for i in 0..ell {
        for k in 0..4 {
            let t = x[4 * i + k] as u64 * y[4 * i + k] as u64;
            acc1[k] += t & mask;
            acc2[k] += t >> h;
        }
    }

    for k in 0..4 {
        res[k] = acc1[k] + acc2[k] * meta.h_pow_red[k];
    }
}

/// Computes `res = sum_{i=0}^{ell-1} x[i] * y[i]` in Q120b format,
/// where `x` and `y` are in q120b (values in `[0, 2^64)`).
///
/// `ell` must be < 10 000.
///
/// Both inputs and output are flat `u64` slices with stride 4.
pub fn vec_mat1col_product_bbb_ref<P: PrimeSetCrt4>(meta: &BbbMeta<P>, ell: usize, res: &mut [u64], x: &[u64], y: &[u64]) {
    assert!(res.len() >= 4);
    assert!(x.len() >= 4 * ell);
    assert!(y.len() >= 4 * ell);

    const MASK1: u64 = u32::MAX as u64; // lower 32 bits

    let mut s1 = [0u64; 4];
    let mut s2 = [0u64; 4];
    let mut s3 = [0u64; 4];
    let mut s4 = [0u64; 4];

    for i in 0..ell {
        for k in 0..4 {
            let xv = x[4 * i + k];
            let yv = y[4 * i + k];
            let xl = xv & MASK1;
            let xh = xv >> 32;
            let yl = yv & MASK1;
            let yh = yv >> 32;

            let a = xl * yl;
            let al = a & MASK1;
            let ah = a >> 32;

            let b = xl * yh;
            let bl = b & MASK1;
            let bh = b >> 32;

            let c = xh * yl;
            let cl = c & MASK1;
            let ch = c >> 32;

            let d = xh * yh;
            let dl = d & MASK1;
            let dh = d >> 32;

            s1[k] += al;
            s2[k] += ah + bl + cl;
            s3[k] += bh + ch + dl;
            s4[k] += dh;
        }
    }

    let h2 = meta.h;
    let mask2 = (1u64 << h2) - 1;

    for k in 0..4 {
        let s1l = s1[k] & mask2;
        let s1h = s1[k] >> h2;
        let s2l = s2[k] & mask2;
        let s2h = s2[k] >> h2;
        let s3l = s3[k] & mask2;
        let s3h = s3[k] >> h2;
        let s4l = s4[k] & mask2;
        let s4h = s4[k] >> h2;

        let mut t = s1l;
        t += s1h * meta.s1h_pow_red;
        t += s2l * meta.s2l_pow_red[k];
        t += s2h * meta.s2h_pow_red[k];
        t += s3l * meta.s3l_pow_red[k];
        t += s3h * meta.s3h_pow_red[k];
        t += s4l * meta.s4l_pow_red[k];
        t += s4h * meta.s4h_pow_red[k];

        res[k] = t;
    }
}

/// Inner helper: accumulate one q120b × q120c pair into an 8-wide `u64` sum.
///
/// `s[2*k]` collects the low-32-bit part and `s[2*k+1]` the high-32-bit part
/// of the per-prime product for prime index `k`.
#[inline(always)]
pub(crate) fn accum_mul_q120_bc(s: &mut [u64; 8], x: &[u32; 8], y: &[u32; 8]) {
    const MASK32: u64 = u32::MAX as u64;
    for i in 0..4 {
        let x_lo = x[2 * i] as u64;
        let x_hi = x[2 * i + 1] as u64;
        let y_lo = y[2 * i] as u64;
        let y_hi = y[2 * i + 1] as u64;
        let xy_lo = x_lo * y_lo;
        let xy_hi = x_hi * y_hi;
        s[2 * i] += (xy_lo & MASK32) + (xy_hi & MASK32);
        s[2 * i + 1] += (xy_lo >> 32) + (xy_hi >> 32);
    }
}

/// Collapses the 8-wide accumulator `s` into a 4-wide q120b result.
#[inline(always)]
pub(crate) fn accum_to_q120b<P: PrimeSetCrt4>(res: &mut [u64; 4], s: &[u64; 8], meta: &BbcMeta<P>) {
    let h2 = meta.h;
    let mask2 = (1u64 << h2) - 1;
    for k in 0..4 {
        let s2l = s[2 * k + 1] & mask2;
        let s2h = s[2 * k + 1] >> h2;
        let t = s[2 * k] + s2l * meta.s2l_pow_red[k] + s2h * meta.s2h_pow_red[k];
        res[k] = t;
    }
}

/// Computes `res = sum_{i=0}^{ell-1} x[i] * y[i]` in Q120b format,
/// where `x` is in q120b and `y` is in q120c.
///
/// `ell` must be < 10 000.
pub fn vec_mat1col_product_bbc_ref<P: PrimeSetCrt4>(meta: &BbcMeta<P>, ell: usize, res: &mut [u64], x: &[u32], y: &[u32]) {
    assert!(res.len() >= 4);
    assert!(x.len() >= 8 * ell);
    assert!(y.len() >= 8 * ell);

    let mut s = [0u64; 8];
    for i in 0..ell {
        let xi: &[u32; 8] = x[8 * i..8 * i + 8].try_into().unwrap();
        let yi: &[u32; 8] = y[8 * i..8 * i + 8].try_into().unwrap();
        accum_mul_q120_bc(&mut s, xi, yi);
    }
    let res4: &mut [u64; 4] = (&mut res[..4]).try_into().unwrap();
    accum_to_q120b::<P>(res4, &s, meta);
}

/// Computes two q120b dot products simultaneously (x2 variant).
///
/// `x` contains two interleaved q120b vectors (each of length `ell`),
/// and `y` contains two interleaved q120c vectors.
/// Both output q120b values are written into `res` (8 contiguous u64s).
pub fn vec_mat1col_product_x2_bbc_ref<P: PrimeSetCrt4>(meta: &BbcMeta<P>, ell: usize, res: &mut [u64], x: &[u32], y: &[u32]) {
    assert!(res.len() >= 8);
    assert!(x.len() >= 16 * ell);
    assert!(y.len() >= 16 * ell);

    let mut s = [[0u64; 8]; 2];

    for i in 0..ell {
        // Each element: 2 × q120b (16 u32) in x, 2 × q120c (16 u32) in y
        let x0: &[u32; 8] = x[16 * i..16 * i + 8].try_into().unwrap();
        let x1: &[u32; 8] = x[16 * i + 8..16 * i + 16].try_into().unwrap();
        let y0: &[u32; 8] = y[16 * i..16 * i + 8].try_into().unwrap();
        let y1: &[u32; 8] = y[16 * i + 8..16 * i + 16].try_into().unwrap();
        accum_mul_q120_bc(&mut s[0], x0, y0);
        accum_mul_q120_bc(&mut s[1], x1, y1);
    }

    let (res0, res1) = res.split_at_mut(4);
    let r0: &mut [u64; 4] = res0.try_into().unwrap();
    accum_to_q120b::<P>(r0, &s[0], meta);
    let r1: &mut [u64; 4] = (&mut res1[..4]).try_into().unwrap();
    accum_to_q120b::<P>(r1, &s[1], meta);
}

/// Computes four q120b dot products (two output, two columns).
///
/// Equivalent to calling `vec_mat1col_product_x2_bbc_ref` twice with
/// two different column slices of `y`, accumulating into `res[0..8]`
/// and `res[8..16]` respectively.
pub fn vec_mat2cols_product_x2_bbc_ref<P: PrimeSetCrt4>(meta: &BbcMeta<P>, ell: usize, res: &mut [u64], x: &[u32], y: &[u32]) {
    assert!(res.len() >= 16);
    assert!(x.len() >= 16 * ell);
    assert!(y.len() >= 32 * ell);

    let mut s = [[0u64; 8]; 4];

    for i in 0..ell {
        let x0: &[u32; 8] = x[16 * i..16 * i + 8].try_into().unwrap();
        let x1: &[u32; 8] = x[16 * i + 8..16 * i + 16].try_into().unwrap();
        let y0: &[u32; 8] = y[32 * i..32 * i + 8].try_into().unwrap();
        let y1: &[u32; 8] = y[32 * i + 8..32 * i + 16].try_into().unwrap();
        let y2: &[u32; 8] = y[32 * i + 16..32 * i + 24].try_into().unwrap();
        let y3: &[u32; 8] = y[32 * i + 24..32 * i + 32].try_into().unwrap();
        accum_mul_q120_bc(&mut s[0], x0, y0);
        accum_mul_q120_bc(&mut s[1], x1, y1);
        accum_mul_q120_bc(&mut s[2], x0, y2);
        accum_mul_q120_bc(&mut s[3], x1, y3);
    }

    for (out_idx, si) in s.iter().enumerate() {
        let r: &mut [u64; 4] = (&mut res[4 * out_idx..4 * out_idx + 4]).try_into().unwrap();
        accum_to_q120b::<P>(r, si, meta);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Block extract / save helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Extracts one block of 4 q120b coefficients (= 8 u64 values) from
/// a q120b NTT vector of length `nn`, copying into `dst`.
///
/// A "block" here groups 2 consecutive NTT coefficients (indices
/// `2*blk` and `2*blk+1`), so `blk < nn/2`.
///
/// This is the Rust port of `q120x2_extract_1blk_from_q120b_ref`.
pub fn extract_1blk_from_q120b_ref(nn: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    debug_assert!(blk < nn / 2);
    debug_assert!(dst.len() >= 8);
    debug_assert!(src.len() >= 4 * nn);

    dst[..8].copy_from_slice(&src[8 * blk..8 * blk + 8]);
}

/// Extracts one block from a contiguous array of `nrows` q120b NTT
/// vectors, each of length `nn`.
///
/// `dst` receives `nrows` consecutive blocks of 8 u64 each.
/// `src` is laid out as `[row_0 || row_1 || ... || row_{nrows-1}]`
/// where each row has `4*nn` u64 values.
///
/// Port of `q120x2_extract_1blk_from_contiguous_q120b_ref`.
pub fn extract_1blk_from_contiguous_q120b_ref(nn: usize, nrows: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    debug_assert!(blk < nn / 2);
    debug_assert!(dst.len() >= 8 * nrows);
    debug_assert!(src.len() >= 4 * nn * nrows);

    for row in 0..nrows {
        let src_base = 4 * nn * row;
        let dst_base = 8 * row;
        dst[dst_base..dst_base + 8].copy_from_slice(&src[src_base + 8 * blk..src_base + 8 * blk + 8]);
    }
}

/// Saves one q120x2b block (8 u64 values) into the corresponding
/// position of a q120b NTT vector of length `nn`.
///
/// Port of `q120x2b_save_1blk_to_q120b_ref`.
pub fn save_1blk_to_q120b_ref(nn: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    debug_assert!(blk < nn / 2);
    debug_assert!(src.len() >= 8);
    debug_assert!(dst.len() >= 4 * nn);

    dst[8 * blk..8 * blk + 8].copy_from_slice(&src[..8]);
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

use super::pow2_mod;

/// `ceil(log2(x))` for x ≥ 1 encoded as bit-size estimate.
///
/// Returns the maximum, over all four primes, of `ceil(log2((2^exp) % Q[k]))`.
fn compute_bit_size_red(exp: u64, q: [u32; 4]) -> f64 {
    let mut max_bs = 0.0f64;
    for &qi in &q {
        let val = pow2_mod(exp, qi as u64);
        if val > 1 {
            let bs = (val as f64).log2();
            if bs > max_bs {
                max_bs = bs;
            }
        }
    }
    max_bs
}

/// `log2(2^a + 2^b)`.
fn log2_sum_two(a: f64, b: f64) -> f64 {
    let sum = 2.0f64.powf(a) + 2.0f64.powf(b);
    sum.log2()
}

/// `log2(sum_i 2^{bs[i]})`.
fn log2_sum_n(bs: &[f64]) -> f64 {
    let sum: f64 = bs.iter().map(|&b| 2.0f64.powf(b)).sum();
    sum.log2()
}
