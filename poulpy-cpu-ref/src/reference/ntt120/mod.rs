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

//! Q120 NTT reference implementation.
//!
//! This module is a Rust port of the `q120` component of the
//! [spqlios-arithmetic](https://github.com/tfhe/spqlios-arithmetic) library.
//! It provides a pure-scalar (no SIMD) reference implementation of the
//! number-theoretic transform (NTT) and associated arithmetic over a
//! degree-120 composite modulus `Q = Q₀·Q₁·Q₂·Q₃`.
//!
//! # Representation
//!
//! Ring elements are stored in **CRT form**: each integer is represented
//! as four residues, one per prime factor of `Q`.  Three concrete prime
//! sets (29-, 30-, and 31-bit) are provided via the [`primes::PrimeSet`]
//! trait; the 30-bit variant ([`primes::Primes30`]) is the default and
//! matches the spqlios library default.
//!
//! The concrete storage types are:
//!
//! | Type | Element | Content |
//! |------|---------|---------|
//! | [`types::Q120a`] | `[u32; 4]` | Residues in `[0, 2^32)` |
//! | [`types::Q120b`] | `[u64; 4]` | Residues in `[0, 2^64)` — NTT domain |
//! | [`types::Q120c`] | `[u32; 8]` | `(rᵢ, rᵢ·2^32 mod Qᵢ)` pairs — prepared for lazy multiply |
//!
//! An NTT vector of length `n` is stored as a flat `[u64]` slice of
//! length `4 * n` (i.e., `n` consecutive [`types::Q120b`] values).
//!
//! # Submodules
//!
//! - [`primes`]: [`primes::PrimeSet`] trait and [`primes::Primes29`] /
//!   [`primes::Primes30`] / [`primes::Primes31`] implementations.
//! - [`types`]: CRT type aliases ([`types::Q120a`], [`types::Q120b`], etc.).
//! - [`arithmetic`]: Simple element-wise operations (conversion to/from
//!   `i64` / `i128`, component-wise addition).
//! - [`mat_vec`]: Lazy-accumulation matrix–vector products
//!   ([`mat_vec::BaaMeta`], [`mat_vec::BbbMeta`], [`mat_vec::BbcMeta`]
//!   and the corresponding product functions).
//! - [`ntt`]: NTT precomputation tables ([`ntt::NttTable`],
//!   [`ntt::NttTableInv`]) and reference execution
//!   ([`ntt::ntt_ref`], [`ntt::intt_ref`]).
//!
//! # Trait overview
//!
//! The traits defined at this level mirror the `Reim*` traits in
//! [`crate::reference::fft64::reim`] and provide the NTT-domain
//! operations that a backend implementation must satisfy:
//!
//! | Trait | Description |
//! |-------|-------------|
//! | [`NttDFTExecute`] | Forward or inverse NTT execution |
//! | [`NttFromZnx64`] | Load `i64` coefficients into q120b format |
//! | [`NttToZnx128`] | CRT-reconstruct from q120b to `i128` coefficients |
//! | [`NttAdd`] | Component-wise addition of two q120b vectors |
//! | [`NttAddAssign`] | In-place component-wise addition |
//! | [`NttSub`] | Component-wise subtraction of two q120b vectors |
//! | [`NttSubAssign`] | In-place component-wise subtraction |
//! | [`NttSubNegateAssign`] | In-place swap-subtract: `res = a - res` |
//! | [`NttNegate`] | Component-wise negation |
//! | [`NttNegateAssign`] | In-place component-wise negation |
//! | [`NttZero`] | Zero a q120b vector |
//! | [`NttCopy`] | Copy a q120b vector |
//! | [`NttMulBbb`] | Lazy product: q120b × q120b → q120b |
//! | [`NttMulBbc`] | Pointwise product: q120b × q120c → q120b (overwrite) |
//! | [`NttCFromB`] | Convert q120b → q120c (Montgomery-prepared form) |
//! | [`NttMulBbc1ColX2`] | x2-block 1-column bbc product (VMP inner loop) |
//! | [`NttMulBbc2ColsX2`] | x2-block 2-column bbc product (VMP inner loop) |
//! | [`NttExtract1BlkContiguous`] | Extract one x2-block from a contiguous q120b array |

pub mod arithmetic;
pub mod convolution;
pub mod mat_vec;
pub mod ntt;
pub mod primes;
pub mod svp;
pub mod types;
pub mod vec_znx_big;
pub mod vec_znx_dft;
pub mod vmp;

pub use arithmetic::*;
pub use convolution::*;
pub use mat_vec::*;
pub use ntt::*;
pub use primes::*;
pub use svp::*;
pub use types::*;
pub use vec_znx_big::*;
pub use vec_znx_dft::*;
pub use vmp::*;

// ──────────────────────────────────────────────────────────────────────────────
// Shared internal utilities
// ──────────────────────────────────────────────────────────────────────────────

/// `2^exp mod q` using 128-bit intermediate arithmetic.
///
/// Shared by [`mat_vec`] and [`ntt`] to avoid duplicating this function.
pub(super) fn pow2_mod(exp: u64, q: u64) -> u64 {
    let mut result: u64 = 1;
    let mut base: u64 = 2 % q;
    let mut e = exp;
    while e > 0 {
        if e & 1 != 0 {
            result = ((result as u128 * base as u128) % q as u128) as u64;
        }
        base = ((base as u128 * base as u128) % q as u128) as u64;
        e >>= 1;
    }
    result
}

// ──────────────────────────────────────────────────────────────────────────────
// NTT-domain operation traits
// ──────────────────────────────────────────────────────────────────────────────

/// Execute a forward or inverse NTT using a precomputed table.
///
/// `Table` is either [`NttTable`] or [`NttTableInv`] (both generic over a
/// [`PrimeSet`]).  `data` is a flat `u64` slice of length `4 * n` in
/// q120b layout.
pub trait NttDFTExecute<Table> {
    /// Apply the NTT (or iNTT) described by `table` to `data` in place.
    fn ntt_dft_execute(table: &Table, data: &mut [u64]);
}

/// Load a polynomial from the standard `i64` coefficient representation
/// into the q120b NTT-domain format.
pub trait NttFromZnx64 {
    /// Encode the `a.len()` coefficients of `a` into `res` (q120b layout).
    ///
    /// `res` must have length `4 * a.len()`.
    fn ntt_from_znx64(res: &mut [u64], a: &[i64]);

    /// Encode `a` into `res` (q120b layout), applying `mask` to each coefficient
    /// before conversion. Equivalent to `ntt_from_znx64` on `a[j] & mask`.
    fn ntt_from_znx64_masked(res: &mut [u64], a: &[i64], mask: i64) {
        arithmetic::b_from_znx64_masked_ref::<primes::Primes30>(a.len(), res, a, mask)
    }
}

/// Recover `i128` ring-element coefficients from a q120b NTT vector.
///
/// The `divisor_is_n` parameter specifies the polynomial degree `n`; it
/// is used to apply the `1/n` scaling that the inverse NTT does not
/// include automatically (see [`NttTableInv`]).
pub trait NttToZnx128 {
    /// Decode `a` (q120b layout, length `4 * n`) into `res` (`n` × `i128`).
    fn ntt_to_znx128(res: &mut [i128], divisor_is_n: usize, a: &[u64]);
}

/// Component-wise addition of two q120b vectors.
pub trait NttAdd {
    /// `res[i] = a[i] + b[i]` for each CRT component.
    ///
    /// All three slices must have the same length (a multiple of 4).
    fn ntt_add(res: &mut [u64], a: &[u64], b: &[u64]);
}

/// In-place component-wise addition of a q120b vector.
pub trait NttAddAssign {
    /// `res[i] += a[i]` for each CRT component.
    fn ntt_add_assign(res: &mut [u64], a: &[u64]);
}

/// Zero a q120b vector.
pub trait NttZero {
    /// Set all elements of `res` to zero.
    fn ntt_zero(res: &mut [u64]);
}

/// Copy a q120b vector.
pub trait NttCopy {
    /// Copy all elements from `a` into `res`.
    fn ntt_copy(res: &mut [u64], a: &[u64]);
}

/// Lazy matrix–vector product: q120b × q120b → q120b.
///
/// Multiplies each of the `ell` rows of a column vector `b` (in q120b
/// format) by the corresponding entry in `a` (also q120b), accumulating
/// the results into `res`. `meta` carries the precomputed lazy-reduction
/// constants and should be obtained via [`vec_znx_dft::NttModuleHandle::get_bbb_meta`].
pub trait NttMulBbb {
    /// `res += a[0..ell] ⊙ b[0..ell]` using lazy modular arithmetic.
    fn ntt_mul_bbb(meta: &BbbMeta<Primes30>, ell: usize, res: &mut [u64], a: &[u64], b: &[u64]);
}

/// Pointwise product: q120b × q120c → q120b (overwrite).
///
/// Like [`NttMulBbb`] but the right-hand operand `b` is in the
/// **prepared** q120c format ([`types::Q120c`]: 8 × `u32` per element),
/// which pre-stores `(r, r·2^32 mod Q)` pairs for faster multiply-accumulate.
/// `meta` carries the precomputed lazy-reduction parameters for the prime set.
///
/// **Overwrites** `res` with the result (does not accumulate into `res`).
///
/// <!-- DOCUMENTED EXCEPTION: Primes30 hardcoded for spqlios compatibility.
///   The generalisation path is to add an associated type PrimeSet to
///   NttModuleHandle; until then every bbc/VMP/SVP/convolution trait method
///   is intentionally fixed to Primes30. -->
pub trait NttMulBbc {
    /// `res = sum_{i<ell} ntt_coeff[i] ⊙ prepared[i]` with `prepared` in q120c layout, using `meta`.
    fn ntt_mul_bbc(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], ntt_coeff: &[u32], prepared: &[u32]);
}

// ──────────────────────────────────────────────────────────────────────────────
// Sub / negate variants
// ──────────────────────────────────────────────────────────────────────────────

/// Component-wise subtraction of two q120b vectors.
pub trait NttSub {
    /// `res[i] = a[i] - b[i]` (lazy q120b arithmetic) for each CRT component.
    fn ntt_sub(res: &mut [u64], a: &[u64], b: &[u64]);
}

/// In-place component-wise subtraction of a q120b vector.
pub trait NttSubAssign {
    /// `res[i] -= a[i]` (lazy q120b arithmetic) for each CRT component.
    fn ntt_sub_assign(res: &mut [u64], a: &[u64]);
}

/// In-place swap-subtract: `res = a - res`.
///
/// Equivalent to negating `res` then adding `a`, in lazy q120b arithmetic.
pub trait NttSubNegateAssign {
    /// `res[i] = a[i] - res[i]` (lazy q120b arithmetic).
    fn ntt_sub_negate_assign(res: &mut [u64], a: &[u64]);
}

/// Component-wise negation of a q120b vector.
pub trait NttNegate {
    /// `res[i] = -a[i]` (lazy q120b arithmetic).
    fn ntt_negate(res: &mut [u64], a: &[u64]);
}

/// In-place component-wise negation of a q120b vector.
pub trait NttNegateAssign {
    /// `res[i] = -res[i]` (lazy q120b arithmetic).
    fn ntt_negate_assign(res: &mut [u64]);
}

// ──────────────────────────────────────────────────────────────────────────────
// q120b → q120c conversion
// ──────────────────────────────────────────────────────────────────────────────

/// Convert a q120b vector to q120c (Montgomery-prepared) form.
///
/// For each element `j` and prime `k`:
/// - `r = a[4*j+k] mod Q[k]`
/// - `res[8*j + 2*k]     = r`
/// - `res[8*j + 2*k + 1] = (r * 2^32) mod Q[k]`
pub trait NttCFromB {
    /// Encode `a` (q120b, length `4*n`) into `res` (q120c, length `8*n`).
    fn ntt_c_from_b(n: usize, res: &mut [u32], a: &[u64]);
}

// ──────────────────────────────────────────────────────────────────────────────
// VMP x2-block kernels
// ──────────────────────────────────────────────────────────────────────────────

/// VMP inner loop: x2-block 1-column bbc product.
///
/// Computes the inner product of one x2-block from `a` (q120b, as u32)
/// against one column of the prepared matrix `b` (q120c), producing 8 u64
/// output values (two q120b coefficients).
pub trait NttMulBbc1ColX2 {
    /// `res[0..8] = sum_{i<ell} a_x2[i] ⊙ b_x2[i]`.
    fn ntt_mul_bbc_1col_x2(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], a: &[u32], b: &[u32]);
}

/// VMP inner loop: x2-block 2-column bbc product.
///
/// Like [`NttMulBbc1ColX2`] but computes two output columns simultaneously,
/// writing 16 u64 values: `res[0..8]` for col 0, `res[8..16]` for col 1.
pub trait NttMulBbc2ColsX2 {
    /// `res[0..16] = [sum_i a_x2[i] ⊙ b_col0_x2[i], sum_i a_x2[i] ⊙ b_col1_x2[i]]`.
    fn ntt_mul_bbc_2cols_x2(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], a: &[u32], b: &[u32]);
}

/// Extract one x2-block from a contiguous q120b array.
///
/// Reads block `blk` (8 u64 values: two consecutive coefficients × 4 primes)
/// from `src`, which holds `row_max` q120b polynomials of degree `n` in
/// contiguous layout, and writes the extracted values into `dst`.
pub trait NttExtract1BlkContiguous {
    /// Copy x2-block `blk` from `src` into `dst`.
    fn ntt_extract_1blk_contiguous(n: usize, row_max: usize, blk: usize, dst: &mut [u64], src: &[u64]);
}

/// Pack a row range of q120b x2-blocks into the u32 layout expected by BBC kernels.
///
/// `a` is a column-start q120b slice with row stride `row_stride` (in `u64` units).
/// For each row, block `blk` is reduced to canonical residues and written to `dst`
/// as 16 u32 values in x2 q120b/u32 layout.
pub trait NttPackLeft1BlkX2 {
    /// Pack `row_count` q120b x2-blocks for block `blk`.
    fn ntt_pack_left_1blk_x2(dst: &mut [u32], a: &[u64], row_count: usize, row_stride: usize, blk: usize);
}

/// Pack a row range of q120c x2-blocks in reversed row order.
///
/// `a` is a column-start q120c slice with row stride `row_stride` (in `u32` units).
/// For each row, block `blk` is copied to `dst` in reversed row order so convolution
/// windows can consume contiguous slices directly.
pub trait NttPackRight1BlkX2 {
    /// Pack `row_count` q120c x2-blocks for block `blk` in reversed row order.
    fn ntt_pack_right_1blk_x2(dst: &mut [u32], a: &[u32], row_count: usize, row_stride: usize, blk: usize);
}

/// Pack a row range of pairwise-summed q120b x2-blocks into the u32 layout expected by BBC kernels.
///
/// `a` and `b` are column-start q120b slices with row stride `row_stride` (in `u64` units).
/// For each row, block `blk` (two consecutive coefficients) is reduced to canonical residues,
/// summed mod `Q`, and written to `dst` as 16 u32 values:
/// `[r0, 0, r1, 0, r2, 0, r3, 0, r0', 0, ..., r3', 0]`.
pub trait NttPairwisePackLeft1BlkX2 {
    /// Pack `row_count` pairwise-summed q120b x2-blocks for block `blk`.
    fn ntt_pairwise_pack_left_1blk_x2(dst: &mut [u32], a: &[u64], b: &[u64], row_count: usize, row_stride: usize, blk: usize);
}

/// Pack a row range of pairwise-summed q120c x2-blocks.
///
/// `a` and `b` are column-start q120c slices with row stride `row_stride` (in `u32` units).
/// For each row, block `blk` is written to `dst` in reversed row order so convolution windows
/// can consume contiguous slices directly.
pub trait NttPairwisePackRight1BlkX2 {
    /// Pack `row_count` pairwise-summed q120c x2-blocks for block `blk`.
    fn ntt_pairwise_pack_right_1blk_x2(dst: &mut [u32], a: &[u32], b: &[u32], row_count: usize, row_stride: usize, blk: usize);
}
