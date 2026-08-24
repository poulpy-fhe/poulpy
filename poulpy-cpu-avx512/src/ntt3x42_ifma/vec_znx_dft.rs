//! Packed NTT-domain SIMD helpers for [`NTT3x42Ifma`](crate::NTT3x42Ifma).

use crate::NTT3x42Ifma;
use crate::ntt3x42_ifma::{
    execution::{SendPtr, for_index_exec, for_index_with},
    kernels::{cond_sub_2q_si512, harvey_modmul_si512, ntt_avx512},
    module::handle,
    primes::{PrimeSetNtt3x42Ifma, Primes42},
    tables::Ntt3x42IfmaTableInv,
    traits::{Ntt3x42IfmaDFTExecute, Ntt3x42IfmaFromZnx64, Ntt3x42IfmaToZnx128},
    vmp::{pack_y, unpack_y},
};
use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::x86_64::{
    __m512i, __mmask8, _MM_CMPINT_LT, _mm512_add_epi64, _mm512_and_si512, _mm512_cmp_epu64_mask, _mm512_cmpeq_epi64_mask,
    _mm512_loadu_si512, _mm512_madd52hi_epu64, _mm512_madd52lo_epu64, _mm512_mask_sub_epi64, _mm512_permutex2var_epi64,
    _mm512_set_epi64, _mm512_set1_epi64, _mm512_setzero_si512, _mm512_slli_epi64, _mm512_srli_epi64, _mm512_storeu_si512,
    _mm512_sub_epi64,
};
use poulpy_hal::layouts::PrimeSet;
use poulpy_hal::layouts::{
    DataView, DataViewMut, Module, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut, VecZnxDftBackendRef, ZnxView,
    ZnxViewMut,
};

// 3-prime CRT -> i128 reconstruction helpers.

const Q: [u64; 3] = Primes42::Q;
const INV01: u64 = Primes42::CRT_CST[0];
const INV012: u64 = Primes42::CRT_CST[1];
const Q0: u64 = Q[0];
const Q1: u64 = Q[1];
const Q2: u64 = Q[2];
const Q01: u128 = Q0 as u128 * Q1 as u128;
const BIG_Q: u128 = Q01 * Q2 as u128;
const HALF_BIG_Q: u128 = BIG_Q / 2;
const BIG_Q_LO: u64 = BIG_Q as u64;
const BIG_Q_HI: u64 = (BIG_Q >> 64) as u64;
const HALF_BIG_Q_LO: u64 = HALF_BIG_Q as u64;
const HALF_BIG_Q_HI: u64 = (HALF_BIG_Q >> 64) as u64;
const Q01_LO: u64 = (Q01 & ((1u128 << 52) - 1)) as u64;
const Q01_HI: u64 = (Q01 >> 52) as u64;

// Harvey quotients for the Garner steps.
const INV01_QUOT: u64 = ((INV01 as u128 * (1u128 << 52)) / Q1 as u128) as u64;
const INV012_QUOT: u64 = ((INV012 as u128 * (1u128 << 52)) / Q2 as u128) as u64;
// `Q0 mod Q2` and its Harvey quotient.
const Q0_MOD_Q2: u64 = Q0 % Q2;
const Q0_MOD_Q2_QUOT: u64 = ((Q0_MOD_Q2 as u128 * (1u128 << 52)) / Q2 as u128) as u64;

/// Harvey scalar modular multiply: `(a * omega) mod q`, result in `[0, q)`.
///
/// Input: `a ∈ [0, q)`, `omega ∈ [0, q)`.
/// `omega_quot = floor(omega * 2^52 / q)`.
#[inline(always)]
fn harvey_modmul_scalar(a: u64, omega: u64, omega_quot: u64, q: u64) -> u64 {
    let qhat = ((a as u128 * omega_quot as u128) >> 52) as u64;
    let product_lo = (a as u128 * omega as u128) as u64;
    let qhat_times_q = (qhat as u128 * q as u128) as u64;
    let mut r = product_lo.wrapping_sub(qhat_times_q);
    if (r as i64) < 0 {
        r = r.wrapping_add(q);
    }
    if r >= q { r - q } else { r }
}

/// Conditional subtract: if x >= q, return x - q.
#[inline(always)]
fn cond_sub_scalar(x: u64, q: u64) -> u64 {
    if x >= q { x - q } else { x }
}

/// Scalar Garner CRT reconstruction from 3 reduced residues.
///
/// Input: `r0 ∈ [0, Q0)`, `r1 ∈ [0, Q1)`, `r2 ∈ [0, Q2)`.
/// Output: reconstructed `i128` in symmetric representation `(-Q/2, Q/2]`.
#[inline(always)]
fn garner_from_residues(r0: u64, r1: u64, r2: u64) -> i128 {
    let v0 = r0;

    let v0_mod_q1 = cond_sub_scalar(v0, Q1);
    let diff1 = cond_sub_scalar(r1 + Q1 - v0_mod_q1, Q1);
    let v1 = harvey_modmul_scalar(diff1, INV01, INV01_QUOT, Q1);

    let v0_mod_q2 = cond_sub_scalar(v0, Q2);
    let v1q0_mod_q2 = harvey_modmul_scalar(v1, Q0_MOD_Q2, Q0_MOD_Q2_QUOT, Q2);
    let partial = cond_sub_scalar(v0_mod_q2 + v1q0_mod_q2, Q2);
    let diff2 = cond_sub_scalar(r2 + Q2 - partial, Q2);
    let v2 = harvey_modmul_scalar(diff2, INV012, INV012_QUOT, Q2);

    let result_u128 = v0 as u128 + v1 as u128 * Q0 as u128 + v2 as u128 * Q01;

    if result_u128 > HALF_BIG_Q {
        result_u128 as i128 - BIG_Q as i128
    } else {
        result_u128 as i128
    }
}

/// Convert eight unsigned CRT representatives to the symmetric i128 range and
/// store them in native interleaved `[lo, hi]` memory order.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn store_symmetric_i128x8(dst: *mut i128, lo: __m512i, hi: __m512i) {
    unsafe {
        let half_lo = _mm512_set1_epi64(HALF_BIG_Q_LO as i64);
        let half_hi = _mm512_set1_epi64(HALF_BIG_Q_HI as i64);
        let big_lo = _mm512_set1_epi64(BIG_Q_LO as i64);
        let big_hi = _mm512_set1_epi64(BIG_Q_HI as i64);
        let one = _mm512_set1_epi64(1);

        // Strict unsigned 128-bit comparison: (hi, lo) > HALF_BIG_Q.
        let hi_gt = _mm512_cmp_epu64_mask(half_hi, hi, _MM_CMPINT_LT);
        let hi_eq = _mm512_cmpeq_epi64_mask(hi, half_hi);
        let lo_gt = _mm512_cmp_epu64_mask(half_lo, lo, _MM_CMPINT_LT);
        let subtract_q: __mmask8 = hi_gt | (hi_eq & lo_gt);

        // Masked 128-bit subtraction with a low-to-high borrow.
        let borrow = subtract_q & _mm512_cmp_epu64_mask(lo, big_lo, _MM_CMPINT_LT);
        let lo = _mm512_mask_sub_epi64(lo, subtract_q, lo, big_lo);
        let hi = _mm512_mask_sub_epi64(hi, subtract_q, hi, big_hi);
        let hi = _mm512_mask_sub_epi64(hi, borrow, hi, one);

        // [lo0..lo7] + [hi0..hi7] -> two native i128 store vectors.
        let interleave_lo = _mm512_set_epi64(11, 3, 10, 2, 9, 1, 8, 0);
        let interleave_hi = _mm512_set_epi64(15, 7, 14, 6, 13, 5, 12, 4);
        let out0 = _mm512_permutex2var_epi64(lo, interleave_lo, hi);
        let out1 = _mm512_permutex2var_epi64(lo, interleave_hi, hi);
        let dst = dst as *mut __m512i;
        _mm512_storeu_si512(dst, out0);
        _mm512_storeu_si512(dst.add(1), out1);
    }
}

/// CRT reconstruction: planar 3-prime IFMA b-format to i128.
///
/// Input residues must be in `[0, 2q)` (b-format after iNTT).
///
/// # Safety
///
/// - `a` must contain at least `3 * nn` u64 values.
/// - `res` must have room for at least `nn` i128 values.
/// - Caller must ensure AVX512-IFMA and AVX512-VL support.
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn simd_b_ntt3x42_ifma_to_znx128(nn: usize, res: &mut [i128], a: &[u64]) {
    debug_assert!(res.len() >= nn);
    debug_assert!(a.len() >= 3 * nn);

    unsafe {
        let q0 = _mm512_set1_epi64(Q0 as i64);
        let q1 = _mm512_set1_epi64(Q1 as i64);
        let q2 = _mm512_set1_epi64(Q2 as i64);
        let inv01 = _mm512_set1_epi64(INV01 as i64);
        let inv01_quot = _mm512_set1_epi64(INV01_QUOT as i64);
        let inv012 = _mm512_set1_epi64(INV012 as i64);
        let inv012_quot = _mm512_set1_epi64(INV012_QUOT as i64);
        let q0_mod_q2 = _mm512_set1_epi64(Q0_MOD_Q2 as i64);
        let q0_mod_q2_quot = _mm512_set1_epi64(Q0_MOD_Q2_QUOT as i64);
        let q01_lo = _mm512_set1_epi64(Q01_LO as i64);
        let q01_hi = _mm512_set1_epi64(Q01_HI as i64);
        let mask52 = _mm512_set1_epi64(((1u64 << 52) - 1) as i64);
        let mask12 = _mm512_set1_epi64(0xFFF);
        let zero = _mm512_setzero_si512();
        let mut c = 0usize;
        while c + 8 <= nn {
            let r0 = cond_sub_2q_si512(_mm512_loadu_si512(a.as_ptr().add(c) as *const __m512i), q0);
            let r1 = cond_sub_2q_si512(_mm512_loadu_si512(a.as_ptr().add(nn + c) as *const __m512i), q1);
            let r2 = cond_sub_2q_si512(_mm512_loadu_si512(a.as_ptr().add(2 * nn + c) as *const __m512i), q2);

            let v0_mod_q1 = cond_sub_2q_si512(r0, q1);
            let diff1 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(r1, q1), v0_mod_q1), q1);
            let v1 = harvey_modmul_si512(diff1, inv01, inv01_quot, q1);

            let v0_mod_q2 = cond_sub_2q_si512(r0, q2);
            let v1q0_mod_q2 = harvey_modmul_si512(v1, q0_mod_q2, q0_mod_q2_quot, q2);
            let partial = cond_sub_2q_si512(_mm512_add_epi64(v0_mod_q2, v1q0_mod_q2), q2);
            let diff2 = cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(r2, q2), partial), q2);
            let v2 = harvey_modmul_si512(diff2, inv012, inv012_quot, q2);

            // result = v0 + v1*Q0 + v2*Q01, accumulated in base-2^52 limbs.
            let p10 = _mm512_madd52lo_epu64(zero, v1, q0);
            let p1h = _mm512_madd52hi_epu64(zero, v1, q0);
            let p2ll = _mm512_madd52lo_epu64(zero, v2, q01_lo);
            let p2lh = _mm512_madd52hi_epu64(zero, v2, q01_lo);
            let p2hl = _mm512_madd52lo_epu64(zero, v2, q01_hi);
            let p2hh = _mm512_madd52hi_epu64(zero, v2, q01_hi);
            let a0 = _mm512_add_epi64(_mm512_add_epi64(r0, p10), p2ll);
            let a1 = _mm512_add_epi64(_mm512_add_epi64(p1h, p2lh), p2hl);
            let a1 = _mm512_add_epi64(a1, _mm512_srli_epi64::<52>(a0));
            let a0 = _mm512_and_si512(a0, mask52);
            let a2 = _mm512_add_epi64(p2hh, _mm512_srli_epi64::<52>(a1));
            let a1 = _mm512_and_si512(a1, mask52);
            let lo = _mm512_add_epi64(a0, _mm512_slli_epi64::<52>(_mm512_and_si512(a1, mask12)));
            let hi = _mm512_add_epi64(_mm512_srli_epi64::<12>(a1), _mm512_slli_epi64::<40>(a2));
            store_symmetric_i128x8(res.as_mut_ptr().add(c), lo, hi);

            c += 8;
        }

        while c < nn {
            let r0 = cond_sub_scalar(a[c], Q0);
            let r1 = cond_sub_scalar(a[nn + c], Q1);
            let r2 = cond_sub_scalar(a[2 * nn + c], Q2);
            res[c] = garner_from_residues(r0, r1, r2);
            c += 1;
        }
    }
}

/// iNTT (in place on `src`) + Garner CRT-compact (writing i128 to `dst`).
///
/// # Safety
/// - `src_ptr` covers `3 * n * n_blocks` u64; `dst_ptr` covers `n * n_blocks` i128.
/// - If aliased, the dst window must lie in the first half of the src window.
/// - AVX-512-IFMA and AVX-512-VL required at runtime.
#[target_feature(enable = "avx512ifma,avx512vl")]
unsafe fn intt_then_compact_ifma(
    n: usize,
    n_blocks: usize,
    src_ptr: *mut u64,
    dst_ptr: *mut i128,
    table: &Ntt3x42IfmaTableInv<Primes42>,
) {
    unsafe {
        for k in 0..n_blocks {
            let src_off_u64 = 3 * n * k;
            let dst_off_i128 = n * k;

            // Step 1: inverse NTT in-place on `src`.
            {
                let blk = std::slice::from_raw_parts_mut(src_ptr.add(src_off_u64), 3 * n);
                <NTT3x42Ifma as Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTableInv<Primes42>>>::ntt3x42_ifma_dft_execute(table, blk);
            }

            // Step 2: Garner CRT-compact 3n u64s → n i128s, writing to `dst`.
            let src_base = src_ptr.add(src_off_u64);
            let dst_base = dst_ptr.add(dst_off_i128);
            let src = std::slice::from_raw_parts(src_base, 3 * n);
            let dst = std::slice::from_raw_parts_mut(dst_base, n);

            simd_b_ntt3x42_ifma_to_znx128(n, dst, src);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Packed 3x42 layout helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Bit masks of the packed 2-word coefficient encoding.
pub(crate) const MASK42: u64 = (1u64 << 42) - 1;
pub(crate) const MASK22: u64 = (1u64 << 22) - 1;
pub(crate) const MASK20: u64 = (1u64 << 20) - 1;

#[inline(always)]
fn packed_limb(data: &[u64], n: usize, cols: usize, col: usize, j: usize) -> &[u64] {
    let start = 2 * n * (j * cols + col);
    &data[start..start + 2 * n]
}

#[inline(always)]
fn packed_limb_mut(data: &mut [u64], n: usize, cols: usize, col: usize, j: usize) -> &mut [u64] {
    let start = 2 * n * (j * cols + col);
    &mut data[start..start + 2 * n]
}

#[inline(always)]
unsafe fn packed_limb_raw_mut<'a>(data: *mut u64, n: usize, cols: usize, col: usize, j: usize) -> &'a mut [u64] {
    let start = 2 * n * (j * cols + col);
    unsafe { std::slice::from_raw_parts_mut(data.add(start), 2 * n) }
}

/// Load and unpack the x8 group at u64 offset `off` of a packed limb.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn load_group(src: &[u64], off: usize, m42: __m512i, m20: __m512i) -> [__m512i; 3] {
    unsafe {
        let w0 = _mm512_loadu_si512(src.as_ptr().add(off) as *const __m512i);
        let w1 = _mm512_loadu_si512(src.as_ptr().add(off + 8) as *const __m512i);
        unpack_y(w0, w1, m42, m20)
    }
}

/// Pack and store the x8 group at u64 offset `off` of a packed limb.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn store_group(dst: &mut [u64], off: usize, y: [__m512i; 3], m22: __m512i) {
    unsafe {
        let [w0, w1] = pack_y(y, m22);
        _mm512_storeu_si512(dst.as_mut_ptr().add(off) as *mut __m512i, w0);
        _mm512_storeu_si512(dst.as_mut_ptr().add(off + 8) as *mut __m512i, w1);
    }
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn q_vec_512() -> [__m512i; 3] {
    [
        _mm512_set1_epi64(Q0 as i64),
        _mm512_set1_epi64(Q1 as i64),
        _mm512_set1_epi64(Q2 as i64),
    ]
}

/// Reduce a lazy planar NTT limb from `[0, 4q)` while packing it, keeping the
/// final reduction in registers instead of a separate canonicalize pass.
#[target_feature(enable = "avx512f")]
unsafe fn pack_limb_3x42_lazy(n: usize, dst: &mut [u64], src: &[u64]) {
    unsafe {
        let m22 = _mm512_set1_epi64(MASK22 as i64);
        let q = q_vec_512();
        let q2 = [
            _mm512_add_epi64(q[0], q[0]),
            _mm512_add_epi64(q[1], q[1]),
            _mm512_add_epi64(q[2], q[2]),
        ];
        for g in 0..n / 8 {
            let p0 = _mm512_loadu_si512(src.as_ptr().add(8 * g) as *const __m512i);
            let p1 = _mm512_loadu_si512(src.as_ptr().add(n + 8 * g) as *const __m512i);
            let p2 = _mm512_loadu_si512(src.as_ptr().add(2 * n + 8 * g) as *const __m512i);
            let p0 = cond_sub_2q_si512(cond_sub_2q_si512(p0, q2[0]), q[0]);
            let p1 = cond_sub_2q_si512(cond_sub_2q_si512(p1, q2[1]), q[1]);
            let p2 = cond_sub_2q_si512(cond_sub_2q_si512(p2, q2[2]), q[2]);
            store_group(dst, 16 * g, [p0, p1, p2], m22);
        }
    }
}

/// Unpack a packed limb into planar residues.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn unpack_limb_3x42(n: usize, dst: &mut [u64], src: &[u64]) {
    unsafe {
        let m42 = _mm512_set1_epi64(MASK42 as i64);
        let m20 = _mm512_set1_epi64(MASK20 as i64);
        for g in 0..n / 8 {
            let y = load_group(src, 16 * g, m42, m20);
            _mm512_storeu_si512(dst.as_mut_ptr().add(8 * g) as *mut __m512i, y[0]);
            _mm512_storeu_si512(dst.as_mut_ptr().add(n + 8 * g) as *mut __m512i, y[1]);
            _mm512_storeu_si512(dst.as_mut_ptr().add(2 * n + 8 * g) as *mut __m512i, y[2]);
        }
    }
}

/// `dst = a + b` on packed limbs, canonical (`x + y` then conditional subtract `q`).
#[target_feature(enable = "avx512f")]
unsafe fn packed_add(n: usize, dst: &mut [u64], a: &[u64], b: &[u64]) {
    unsafe {
        let m42 = _mm512_set1_epi64(MASK42 as i64);
        let m20 = _mm512_set1_epi64(MASK20 as i64);
        let m22 = _mm512_set1_epi64(MASK22 as i64);
        let q = q_vec_512();
        for g in 0..n / 8 {
            let off = 16 * g;
            let ya = load_group(a, off, m42, m20);
            let yb = load_group(b, off, m42, m20);
            let r = [
                cond_sub_2q_si512(_mm512_add_epi64(ya[0], yb[0]), q[0]),
                cond_sub_2q_si512(_mm512_add_epi64(ya[1], yb[1]), q[1]),
                cond_sub_2q_si512(_mm512_add_epi64(ya[2], yb[2]), q[2]),
            ];
            store_group(dst, off, r, m22);
        }
    }
}

/// `dst += a` on packed limbs, canonical.
#[target_feature(enable = "avx512f")]
unsafe fn packed_add_assign(n: usize, dst: &mut [u64], a: &[u64]) {
    unsafe {
        let m42 = _mm512_set1_epi64(MASK42 as i64);
        let m20 = _mm512_set1_epi64(MASK20 as i64);
        let m22 = _mm512_set1_epi64(MASK22 as i64);
        let q = q_vec_512();
        for g in 0..n / 8 {
            let off = 16 * g;
            let yd = load_group(dst, off, m42, m20);
            let ya = load_group(a, off, m42, m20);
            let r = [
                cond_sub_2q_si512(_mm512_add_epi64(yd[0], ya[0]), q[0]),
                cond_sub_2q_si512(_mm512_add_epi64(yd[1], ya[1]), q[1]),
                cond_sub_2q_si512(_mm512_add_epi64(yd[2], ya[2]), q[2]),
            ];
            store_group(dst, off, r, m22);
        }
    }
}

/// `dst = a - b` on packed limbs, canonical (`x + q - y` then conditional subtract `q`).
#[target_feature(enable = "avx512f")]
unsafe fn packed_sub(n: usize, dst: &mut [u64], a: &[u64], b: &[u64]) {
    unsafe {
        let m42 = _mm512_set1_epi64(MASK42 as i64);
        let m20 = _mm512_set1_epi64(MASK20 as i64);
        let m22 = _mm512_set1_epi64(MASK22 as i64);
        let q = q_vec_512();
        for g in 0..n / 8 {
            let off = 16 * g;
            let ya = load_group(a, off, m42, m20);
            let yb = load_group(b, off, m42, m20);
            let r = [
                cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(ya[0], q[0]), yb[0]), q[0]),
                cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(ya[1], q[1]), yb[1]), q[1]),
                cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(ya[2], q[2]), yb[2]), q[2]),
            ];
            store_group(dst, off, r, m22);
        }
    }
}

/// `dst -= a` on packed limbs, canonical.
#[target_feature(enable = "avx512f")]
unsafe fn packed_sub_assign(n: usize, dst: &mut [u64], a: &[u64]) {
    unsafe {
        let m42 = _mm512_set1_epi64(MASK42 as i64);
        let m20 = _mm512_set1_epi64(MASK20 as i64);
        let m22 = _mm512_set1_epi64(MASK22 as i64);
        let q = q_vec_512();
        for g in 0..n / 8 {
            let off = 16 * g;
            let yd = load_group(dst, off, m42, m20);
            let ya = load_group(a, off, m42, m20);
            let r = [
                cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(yd[0], q[0]), ya[0]), q[0]),
                cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(yd[1], q[1]), ya[1]), q[1]),
                cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(yd[2], q[2]), ya[2]), q[2]),
            ];
            store_group(dst, off, r, m22);
        }
    }
}

/// `dst = a - dst` on packed limbs, canonical.
#[target_feature(enable = "avx512f")]
unsafe fn packed_sub_negate_assign(n: usize, dst: &mut [u64], a: &[u64]) {
    unsafe {
        let m42 = _mm512_set1_epi64(MASK42 as i64);
        let m20 = _mm512_set1_epi64(MASK20 as i64);
        let m22 = _mm512_set1_epi64(MASK22 as i64);
        let q = q_vec_512();
        for g in 0..n / 8 {
            let off = 16 * g;
            let yd = load_group(dst, off, m42, m20);
            let ya = load_group(a, off, m42, m20);
            let r = [
                cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(ya[0], q[0]), yd[0]), q[0]),
                cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(ya[1], q[1]), yd[1]), q[1]),
                cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(ya[2], q[2]), yd[2]), q[2]),
            ];
            store_group(dst, off, r, m22);
        }
    }
}

/// `dst = -a` on packed limbs, canonical (`q - x` then conditional subtract `q`).
#[target_feature(enable = "avx512f")]
unsafe fn packed_negate(n: usize, dst: &mut [u64], a: &[u64]) {
    unsafe {
        let m42 = _mm512_set1_epi64(MASK42 as i64);
        let m20 = _mm512_set1_epi64(MASK20 as i64);
        let m22 = _mm512_set1_epi64(MASK22 as i64);
        let q = q_vec_512();
        for g in 0..n / 8 {
            let off = 16 * g;
            let ya = load_group(a, off, m42, m20);
            let r = [
                cond_sub_2q_si512(_mm512_sub_epi64(q[0], ya[0]), q[0]),
                cond_sub_2q_si512(_mm512_sub_epi64(q[1], ya[1]), q[1]),
                cond_sub_2q_si512(_mm512_sub_epi64(q[2], ya[2]), q[2]),
            ];
            store_group(dst, off, r, m22);
        }
    }
}

/// `dst = -dst` on packed limbs, canonical.
#[target_feature(enable = "avx512f")]
unsafe fn packed_negate_assign(n: usize, dst: &mut [u64]) {
    unsafe {
        let m42 = _mm512_set1_epi64(MASK42 as i64);
        let m20 = _mm512_set1_epi64(MASK20 as i64);
        let m22 = _mm512_set1_epi64(MASK22 as i64);
        let q = q_vec_512();
        for g in 0..n / 8 {
            let off = 16 * g;
            let yd = load_group(dst, off, m42, m20);
            let r = [
                cond_sub_2q_si512(_mm512_sub_epi64(q[0], yd[0]), q[0]),
                cond_sub_2q_si512(_mm512_sub_epi64(q[1], yd[1]), q[1]),
                cond_sub_2q_si512(_mm512_sub_epi64(q[2], yd[2]), q[2]),
            ];
            store_group(dst, off, r, m22);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward / inverse NTT
// ─────────────────────────────────────────────────────────────────────────────

#[inline(always)]
#[cfg(feature = "enable-rayon")]
pub(crate) fn vec_znx_idft_apply_tmpa_limb_ifma(
    module: &Module<NTT3x42Ifma>,
    dst: &mut [i128],
    src: Option<&[u64]>,
    scratch: &mut [u64],
) {
    let n = module.n();
    assert_eq!(dst.len(), n);
    assert!(scratch.len() >= 3 * n);
    let Some(src) = src else {
        dst.fill(0);
        return;
    };
    assert_eq!(src.len(), 2 * n);
    unsafe {
        unpack_limb_3x42(n, scratch, src);
        intt_then_compact_ifma(n, 1, scratch.as_mut_ptr(), dst.as_mut_ptr(), &handle(module).table_intt);
    }
}

/// In-place iNTT of `a[a_col]`'s limbs: each packed limb is replaced by its
/// `i128` compaction, leaving that column of the buffer in `VecZnxBig` layout.
pub(crate) fn idft_compact_in_place_ifma<E: poulpy_hal::execution::TaskExecutor>(
    module: &Module<NTT3x42Ifma>,
    a: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    a_col: usize,
    tmp: &mut [u64],
) {
    let table = &handle(module).table_intt;
    let n = a.n();
    let a_cols = a.cols();
    let a_size = a.size();
    let data: &mut [u64] = cast_slice_mut(a.data_mut());
    let data_ptr = SendPtr(data.as_mut_ptr());
    E::for_each_chunked(a_size, tmp, 3 * n, |scratch, j| {
        let slot = unsafe { packed_limb_raw_mut(data_ptr.get(), n, a_cols, a_col, j) };
        unsafe {
            unpack_limb_3x42(n, scratch, slot);
            intt_then_compact_ifma(n, 1, scratch.as_mut_ptr(), slot.as_mut_ptr() as *mut i128, table);
        }
    });
}

/// `VecZnxIdftApplyTmpA` packed fast path.
pub(crate) fn vec_znx_idft_apply_tmpa_ifma(
    module: &Module<crate::NTT3x42Ifma>,
    res: &mut VecZnxBigBackendMut<'_, crate::NTT3x42Ifma>,
    res_col: usize,
    a: &mut VecZnxDftBackendMut<'_, crate::NTT3x42Ifma>,
    a_col: usize,
) {
    let table = &handle(module).table_intt;
    let n = a.n();
    let min_size = res.size().min(a.size());
    let a_cols = a.cols();
    let res_cols = res.cols();
    let res_size = res.size();

    let src_data: &[u64] = cast_slice(a.data());
    let dst_data = res.raw_mut();

    for_index_with(
        res_size,
        3 * n * res_size,
        || vec![0u64; 3 * n],
        |scratch, j| {
            let start = n * (j * res_cols + res_col);
            let dst = &mut dst_data[start..start + n];
            if j < min_size {
                let src = packed_limb(src_data, n, a_cols, a_col, j);
                unsafe {
                    unpack_limb_3x42(n, scratch, src);
                    intt_then_compact_ifma(n, 1, scratch.as_mut_ptr(), dst.as_mut_ptr(), table);
                }
            } else {
                dst.fill(0i128);
            }
        },
    );
}

#[inline(always)]
pub(crate) fn vec_znx_dft_apply_limb(module: &Module<NTT3x42Ifma>, dst: &mut [u64], src: Option<&[i64]>, scratch: &mut [u64]) {
    let n = module.n();
    assert_eq!(dst.len(), 2 * n);
    assert!(scratch.len() >= 3 * n);
    let Some(src) = src else {
        dst.fill(0);
        return;
    };
    assert_eq!(src.len(), n);
    NTT3x42Ifma::ntt3x42_ifma_from_znx64(scratch, src);
    unsafe {
        ntt_avx512::<Primes42>(&handle(module).table_ntt, scratch, true);
        pack_limb_3x42_lazy(n, dst, scratch);
    }
}

/// Forward NTT into the packed layout.
pub(crate) fn vec_znx_dft_apply(
    module: &Module<NTT3x42Ifma>,
    step: usize,
    offset: usize,
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
) {
    let a_size = a.size();
    let res_size = res.size();
    let n = res.n();
    let cols = res.cols();
    let steps = a_size.div_ceil(step);
    let min_steps = res_size.min(steps);

    let res_data: &mut [u64] = cast_slice_mut(res.data_mut());
    for_index_with(
        res_size,
        3 * n * res_size,
        || vec![0u64; 3 * n],
        |scratch, j| {
            let res_slice = packed_limb_mut(res_data, n, cols, res_col, j);
            let limb = offset + j * step;
            let src = (j < min_steps && limb < a_size).then(|| a.at(a_col, limb));
            vec_znx_dft_apply_limb(module, res_slice, src, scratch);
        },
    );
}

/// Scratch space (in bytes) for [`vec_znx_idft_apply`].
pub(crate) fn vec_znx_idft_apply_tmp_bytes(n: usize) -> usize {
    use std::mem::size_of;
    3 * n * size_of::<u64>()
}

/// Inverse NTT (non-destructive) for the IFMA backend.
#[inline(always)]
#[cfg(feature = "enable-rayon")]
pub(crate) fn vec_znx_idft_apply_limb(module: &Module<NTT3x42Ifma>, dst: &mut [i128], src: Option<&[u64]>, scratch: &mut [u64]) {
    let n = module.n();
    assert_eq!(dst.len(), n);
    assert!(scratch.len() >= 3 * n);
    let Some(src) = src else {
        dst.fill(0);
        return;
    };
    assert_eq!(src.len(), 2 * n);
    unsafe { unpack_limb_3x42(n, scratch, src) };
    <NTT3x42Ifma as Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTableInv<Primes42>>>::ntt3x42_ifma_dft_execute(
        &handle(module).table_intt,
        scratch,
    );
    NTT3x42Ifma::ntt3x42_ifma_to_znx128(dst, n, scratch);
}

pub(crate) fn vec_znx_idft_apply(
    module: &Module<NTT3x42Ifma>,
    res: &mut VecZnxBigBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
    tmp: &mut [u64],
) {
    let n = res.n();
    let res_cols = res.cols();
    let res_size = res.size();
    let min_size = res_size.min(a.size());
    let a_cols = a.cols();
    let table = &handle(module).table_intt;
    let _ = tmp;

    let a_u64: &[u64] = cast_slice(a.data());
    let res_data = res.raw_mut();
    for_index_with(
        res_size,
        3 * n * res_size,
        || vec![0u64; 3 * n],
        |scratch, j| {
            let start = n * (j * res_cols + res_col);
            let dst = &mut res_data[start..start + n];
            if j < min_size {
                let a_slice: &[u64] = &a_u64[2 * n * (j * a_cols + a_col)..][..2 * n];
                unsafe { unpack_limb_3x42(n, scratch, a_slice) };
                <NTT3x42Ifma as Ntt3x42IfmaDFTExecute<Ntt3x42IfmaTableInv<Primes42>>>::ntt3x42_ifma_dft_execute(table, scratch);
                NTT3x42Ifma::ntt3x42_ifma_to_znx128(dst, n, scratch);
            } else {
                dst.fill(0i128);
            }
        },
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// DFT-domain VecZnxDft operations
// ─────────────────────────────────────────────────────────────────────────────

/// DFT-domain add: `res[res_col] = a[a_col] + b[b_col]`.
pub(crate) fn vec_znx_dft_add_into<E: poulpy_hal::execution::TaskExecutor>(
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
    b_col: usize,
) {
    let n = res.n();
    let (rc, ac, bc) = (res.cols(), a.cols(), b.cols());
    let (res_size, a_size, b_size) = (res.size(), a.size(), b.size());
    let (sum_size, cpy_size, cpy_from_b) = if a_size <= b_size {
        (a_size.min(res_size), b_size.min(res_size), true)
    } else {
        (b_size.min(res_size), a_size.min(res_size), false)
    };
    let rp: &mut [u64] = cast_slice_mut(res.data_mut());
    let rp_ptr = SendPtr(rp.as_mut_ptr());
    let ap: &[u64] = cast_slice(a.data());
    let bp: &[u64] = cast_slice(b.data());
    for_index_exec::<E>(res_size, 2 * n * res_size, |j| {
        let dst = unsafe { packed_limb_raw_mut(rp_ptr.get(), n, rc, res_col, j) };
        if j < sum_size {
            let av = packed_limb(ap, n, ac, a_col, j);
            let bv = packed_limb(bp, n, bc, b_col, j);
            unsafe { packed_add(n, dst, av, bv) };
        } else if j < cpy_size {
            let sv = if cpy_from_b {
                packed_limb(bp, n, bc, b_col, j)
            } else {
                packed_limb(ap, n, ac, a_col, j)
            };
            dst.copy_from_slice(sv);
        } else {
            dst.fill(0);
        }
    });
}

/// DFT-domain in-place add: `res[res_col] += a[a_col]`.
pub(crate) fn vec_znx_dft_add_assign<E: poulpy_hal::execution::TaskExecutor>(
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
) {
    let n = res.n();
    let (rc, ac) = (res.cols(), a.cols());
    let sum_size = res.size().min(a.size());
    let rp: &mut [u64] = cast_slice_mut(res.data_mut());
    let rp_ptr = SendPtr(rp.as_mut_ptr());
    let ap: &[u64] = cast_slice(a.data());
    for_index_exec::<E>(sum_size, 2 * n * sum_size, |j| {
        let dst = unsafe { packed_limb_raw_mut(rp_ptr.get(), n, rc, res_col, j) };
        let av = packed_limb(ap, n, ac, a_col, j);
        unsafe { packed_add_assign(n, dst, av) };
    });
}

/// DFT-domain scaled in-place add: `res[res_col] += a[a_col] >> (a_scale * base2k)`.
pub(crate) fn vec_znx_dft_add_scaled_assign<E: poulpy_hal::execution::TaskExecutor>(
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
    a_scale: i64,
) {
    let n = res.n();
    let (rc, ac) = (res.cols(), a.cols());
    let res_size = res.size();
    let a_size = a.size();

    let (res_shift, a_shift, sum_size) = if a_scale > 0 {
        let shift = (a_scale as usize).min(a_size);
        (0, shift, a_size.min(res_size).saturating_sub(shift))
    } else if a_scale < 0 {
        let shift = (a_scale.unsigned_abs() as usize).min(res_size);
        (shift, 0, a_size.min(res_size.saturating_sub(shift)))
    } else {
        (0, 0, a_size.min(res_size))
    };

    let rp: &mut [u64] = cast_slice_mut(res.data_mut());
    let rp_ptr = SendPtr(rp.as_mut_ptr());
    let ap: &[u64] = cast_slice(a.data());
    for_index_exec::<E>(sum_size, 2 * n * sum_size, |j| {
        let dst = unsafe { packed_limb_raw_mut(rp_ptr.get(), n, rc, res_col, j + res_shift) };
        let av = packed_limb(ap, n, ac, a_col, j + a_shift);
        unsafe { packed_add_assign(n, dst, av) };
    });
}

/// DFT-domain sub: `res[res_col] = a[a_col] - b[b_col]`.
pub(crate) fn vec_znx_dft_sub<E: poulpy_hal::execution::TaskExecutor>(
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
    b_col: usize,
) {
    let n = res.n();
    let (rc, ac, bc) = (res.cols(), a.cols(), b.cols());
    let (res_size, a_size, b_size) = (res.size(), a.size(), b.size());
    let (sum_size, cpy_size, negate_tail) = if a_size <= b_size {
        (a_size.min(res_size), b_size.min(res_size), true)
    } else {
        (b_size.min(res_size), a_size.min(res_size), false)
    };
    let rp: &mut [u64] = cast_slice_mut(res.data_mut());
    let rp_ptr = SendPtr(rp.as_mut_ptr());
    let ap: &[u64] = cast_slice(a.data());
    let bp: &[u64] = cast_slice(b.data());
    for_index_exec::<E>(res_size, 2 * n * res_size, |j| {
        let dst = unsafe { packed_limb_raw_mut(rp_ptr.get(), n, rc, res_col, j) };
        if j < sum_size {
            let av = packed_limb(ap, n, ac, a_col, j);
            let bv = packed_limb(bp, n, bc, b_col, j);
            unsafe { packed_sub(n, dst, av, bv) };
        } else if j < cpy_size {
            if negate_tail {
                let bv = packed_limb(bp, n, bc, b_col, j);
                unsafe { packed_negate(n, dst, bv) };
            } else {
                let av = packed_limb(ap, n, ac, a_col, j);
                dst.copy_from_slice(av);
            }
        } else {
            dst.fill(0);
        }
    });
}

/// DFT-domain in-place sub: `res[res_col] -= a[a_col]`.
pub(crate) fn vec_znx_dft_sub_assign<E: poulpy_hal::execution::TaskExecutor>(
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
) {
    let n = res.n();
    let (rc, ac) = (res.cols(), a.cols());
    let sum_size = res.size().min(a.size());
    let rp: &mut [u64] = cast_slice_mut(res.data_mut());
    let rp_ptr = SendPtr(rp.as_mut_ptr());
    let ap: &[u64] = cast_slice(a.data());
    for_index_exec::<E>(sum_size, 2 * n * sum_size, |j| {
        let dst = unsafe { packed_limb_raw_mut(rp_ptr.get(), n, rc, res_col, j) };
        let av = packed_limb(ap, n, ac, a_col, j);
        unsafe { packed_sub_assign(n, dst, av) };
    });
}

/// DFT-domain in-place swap-sub: `res[res_col] = a[a_col] - res[res_col]`.
pub(crate) fn vec_znx_dft_sub_negate_assign<E: poulpy_hal::execution::TaskExecutor>(
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
) {
    let n = res.n();
    let (rc, ac) = (res.cols(), a.cols());
    let res_size = res.size();
    let sum_size = res_size.min(a.size());
    let rp: &mut [u64] = cast_slice_mut(res.data_mut());
    let rp_ptr = SendPtr(rp.as_mut_ptr());
    let ap: &[u64] = cast_slice(a.data());
    for_index_exec::<E>(res_size, 2 * n * res_size, |j| {
        let dst = unsafe { packed_limb_raw_mut(rp_ptr.get(), n, rc, res_col, j) };
        if j < sum_size {
            let av = packed_limb(ap, n, ac, a_col, j);
            unsafe { packed_sub_negate_assign(n, dst, av) };
        } else {
            unsafe { packed_negate_assign(n, dst) };
        }
    });
}

/// DFT-domain copy with stride: `res[res_col][j] = a[a_col][offset + j*step]`.
pub(crate) fn vec_znx_dft_copy<E: poulpy_hal::execution::TaskExecutor>(
    step: usize,
    offset: usize,
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), a.n())
    }

    let n = res.n();
    let (rc, ac) = (res.cols(), a.cols());
    let res_size = res.size();
    let a_size = a.size();
    let steps: usize = a_size.div_ceil(step);
    let min_steps: usize = res_size.min(steps);

    let rp: &mut [u64] = cast_slice_mut(res.data_mut());
    let rp_ptr = SendPtr(rp.as_mut_ptr());
    let ap: &[u64] = cast_slice(a.data());
    for_index_exec::<E>(res_size, 2 * n * res_size, |j| {
        let dst = unsafe { packed_limb_raw_mut(rp_ptr.get(), n, rc, res_col, j) };
        let limb = offset + j * step;
        if j < min_steps && limb < a_size {
            let av = packed_limb(ap, n, ac, a_col, limb);
            dst.copy_from_slice(av);
        } else {
            dst.fill(0);
        }
    });
}

/// Zero all limbs of `res[res_col]`.
pub(crate) fn vec_znx_dft_zero<E: poulpy_hal::execution::TaskExecutor>(
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
) {
    let n = res.n();
    let rc = res.cols();
    let res_size = res.size();
    let rp: &mut [u64] = cast_slice_mut(res.data_mut());
    let rp_ptr = SendPtr(rp.as_mut_ptr());
    for_index_exec::<E>(res_size, 2 * n * res_size, |j| {
        let dst = unsafe { packed_limb_raw_mut(rp_ptr.get(), n, rc, res_col, j) };
        dst.fill(0);
    });
}

/// Packed-layout NTT3x42 automorphism fused with accumulation: `res += automorphism(a)`.
pub(crate) fn vec_znx_dft_automorphism_add<E: poulpy_hal::execution::TaskExecutor>(
    plan: &poulpy_cpu_ref::reference::ntt4x30::vec_znx_dft::NttAutomorphismPlan,
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
        assert_eq!(plan.perm.len(), res.n());
    }

    let n: usize = res.n();
    let (rc, ac) = (res.cols(), a.cols());
    let min_size: usize = res.size().min(a.size());
    let perm: &[u32] = &plan.perm;

    let rp: &mut [u64] = cast_slice_mut(res.data_mut());
    let rp_ptr = SendPtr(rp.as_mut_ptr());
    let ap: &[u64] = cast_slice(a.data());
    for_index_exec::<E>(min_size, 2 * n * min_size, |limb| {
        let res_slice = unsafe { packed_limb_raw_mut(rp_ptr.get(), n, rc, res_col, limb) };
        let a_slice = packed_limb(ap, n, ac, a_col, limb);
        unsafe { automorphism_add_limb(n, perm, res_slice, a_slice) };
    });
}

/// One packed limb of `dst += perm(a)`: scalar gather per 8-group, canonical packed add.
#[target_feature(enable = "avx512f")]
unsafe fn automorphism_add_limb(n: usize, perm: &[u32], dst: &mut [u64], a: &[u64]) {
    unsafe {
        let m42 = _mm512_set1_epi64(MASK42 as i64);
        let m20 = _mm512_set1_epi64(MASK20 as i64);
        let m22 = _mm512_set1_epi64(MASK22 as i64);
        let q = q_vec_512();
        let mut buf = [0u64; 16];
        for g in 0..n / 8 {
            for (l, &p) in perm[8 * g..8 * g + 8].iter().enumerate() {
                let p = p as usize;
                let src_off = 16 * (p >> 3) + (p & 7);
                buf[l] = a[src_off];
                buf[l + 8] = a[src_off + 8];
            }
            let off = 16 * g;
            let ya = load_group(&buf, 0, m42, m20);
            let yd = load_group(dst, off, m42, m20);
            let r = [
                cond_sub_2q_si512(_mm512_add_epi64(yd[0], ya[0]), q[0]),
                cond_sub_2q_si512(_mm512_add_epi64(yd[1], ya[1]), q[1]),
                cond_sub_2q_si512(_mm512_add_epi64(yd[2], ya[2]), q[2]),
            ];
            store_group(dst, off, r, m22);
        }
    }
}

/// Packed-layout NTT3x42 automorphism.
pub(crate) fn vec_znx_dft_automorphism<E: poulpy_hal::execution::TaskExecutor>(
    plan: &poulpy_cpu_ref::reference::ntt4x30::vec_znx_dft::NttAutomorphismPlan,
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
        assert_eq!(plan.perm.len(), res.n());
    }

    let n: usize = res.n();
    let (rc, ac) = (res.cols(), a.cols());
    let res_size: usize = res.size();
    let a_size: usize = a.size();
    let min_size: usize = res_size.min(a_size);
    let perm: &[u32] = &plan.perm;

    let rp: &mut [u64] = cast_slice_mut(res.data_mut());
    let rp_ptr = SendPtr(rp.as_mut_ptr());
    let ap: &[u64] = cast_slice(a.data());
    for_index_exec::<E>(res_size, 2 * n * res_size, |limb| {
        let res_slice = unsafe { packed_limb_raw_mut(rp_ptr.get(), n, rc, res_col, limb) };
        if limb < min_size {
            let a_slice = packed_limb(ap, n, ac, a_col, limb);
            for (i, &p) in perm.iter().enumerate() {
                let p = p as usize;
                let src_off = 16 * (p >> 3) + (p & 7);
                let dst_off = 16 * (i >> 3) + (i & 7);
                res_slice[dst_off] = a_slice[src_off];
                res_slice[dst_off + 8] = a_slice[src_off + 8];
            }
        } else {
            res_slice.fill(0);
        }
    });
}
