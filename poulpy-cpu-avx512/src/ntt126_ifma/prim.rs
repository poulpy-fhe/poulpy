//! Primitive NTT-domain trait implementations for [`NTT126Ifma`](crate::NTT126Ifma).
//!
//! This module connects the IFMA backend type to the low-level reference IFMA traits:
//! NTT execution, b/c domain conversion, BBC multiply-accumulate, and basic
//! transform-domain arithmetic on the shared 4-lane prep scalar.

use crate::ntt126_ifma::{
    bbc_meta::Bbc126IfmaMeta,
    primes::Primes42,
    tables::{Ntt126IfmaTable, Ntt126IfmaTableInv},
    traits::{
        Ntt126IfmaAdd, Ntt126IfmaAddAssign, Ntt126IfmaCFromB, Ntt126IfmaCopy, Ntt126IfmaDFTExecute, Ntt126IfmaFromZnx64,
        Ntt126IfmaMulBbc, Ntt126IfmaNegate, Ntt126IfmaNegateAssign, Ntt126IfmaSub, Ntt126IfmaSubAssign,
        Ntt126IfmaSubNegateAssign, Ntt126IfmaToZnx128, Ntt126IfmaZero,
    },
    types::Q_SHIFTED_NTT126IFMA,
};

use super::mat_vec_ifma::vec_mat1col_product_bbc_ifma;

use super::kernels::{cond_sub_2q_si256, cond_sub_2q_si512, intt_avx512, ntt_avx512};

use core::arch::x86_64::{
    __m256i, __m512i, _mm256_add_epi64, _mm256_and_si256, _mm256_cmpgt_epi64, _mm256_loadu_si256, _mm256_mul_epu32,
    _mm256_set_epi64x, _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_srli_epi64, _mm256_storeu_si256, _mm256_sub_epi64,
    _mm512_add_epi64, _mm512_loadu_si512, _mm512_storeu_si512, _mm512_sub_epi64,
};

use poulpy_cpu_ref::reference::ntt120::{
    NttAdd, NttAddAssign, NttCopy, NttNegate, NttNegateAssign, NttSub, NttSubAssign, NttSubNegateAssign, NttZero,
};

use crate::NTT126Ifma;

/// Q_SHIFTED_NTT126IFMA as 256-bit: `[2*Q[0], 2*Q[1], 2*Q[2], 0]`.
fn q2_vec() -> __m256i {
    unsafe { _mm256_loadu_si256(Q_SHIFTED_NTT126IFMA.as_ptr() as *const __m256i) }
}

/// Q_SHIFTED_NTT126IFMA duplicated for 512-bit: `[2Q0,2Q1,2Q2,0, 2Q0,2Q1,2Q2,0]`.
const Q2_512: [u64; 8] = {
    let q = <Primes42 as crate::ntt126_ifma::primes::PrimeSetNtt126Ifma>::Q;
    [2 * q[0], 2 * q[1], 2 * q[2], 0, 2 * q[0], 2 * q[1], 2 * q[2], 0]
};

/// Generic pattern for DFT-domain operations: 512-bit main loop + 256-bit tail.
/// This macro avoids repeating the 512/256 dispatch for each operation variant.
macro_rules! dft_op_512 {
    // 3-operand: res = f(a, b)
    (res=$res:ident, a=$a:ident, b=$b:ident, op512=$op512:expr, op256=$op256:expr) => {{
        let n = $res.len() / 4;
        let pairs = n / 2;
        let res_512 = $res.as_mut_ptr() as *mut __m512i;
        let a_512 = $a.as_ptr() as *const __m512i;
        let b_512 = $b.as_ptr() as *const __m512i;
        let q2_512v = _mm512_loadu_si512(Q2_512.as_ptr() as *const __m512i);
        for i in 0..pairs {
            let av = _mm512_loadu_si512(a_512.add(i));
            let bv = _mm512_loadu_si512(b_512.add(i));
            _mm512_storeu_si512(res_512.add(i), $op512(av, bv, q2_512v));
        }
        if n % 2 != 0 {
            let idx = n - 1;
            let res_256 = $res.as_mut_ptr() as *mut __m256i;
            let a_256 = $a.as_ptr() as *const __m256i;
            let b_256 = $b.as_ptr() as *const __m256i;
            let q2 = q2_vec();
            let av = _mm256_loadu_si256(a_256.add(idx));
            let bv = _mm256_loadu_si256(b_256.add(idx));
            _mm256_storeu_si256(res_256.add(idx), $op256(av, bv, q2));
        }
    }};
    // 2-operand inplace: res = f(res, a)
    (res_assign=$res:ident, a=$a:ident, op512=$op512:expr, op256=$op256:expr) => {{
        let n = $res.len() / 4;
        let pairs = n / 2;
        let res_512 = $res.as_mut_ptr() as *mut __m512i;
        let a_512 = $a.as_ptr() as *const __m512i;
        let q2_512v = _mm512_loadu_si512(Q2_512.as_ptr() as *const __m512i);
        for i in 0..pairs {
            let rv = _mm512_loadu_si512(res_512.add(i) as *const __m512i);
            let av = _mm512_loadu_si512(a_512.add(i));
            _mm512_storeu_si512(res_512.add(i), $op512(rv, av, q2_512v));
        }
        if n % 2 != 0 {
            let idx = n - 1;
            let res_256 = $res.as_mut_ptr() as *mut __m256i;
            let a_256 = $a.as_ptr() as *const __m256i;
            let q2 = q2_vec();
            let rv = _mm256_loadu_si256(res_256.add(idx) as *const __m256i);
            let av = _mm256_loadu_si256(a_256.add(idx));
            _mm256_storeu_si256(res_256.add(idx), $op256(rv, av, q2));
        }
    }};
    // 1-operand inplace negate: res = f(res)
    (negate_assign=$res:ident, op512=$op512:expr, op256=$op256:expr) => {{
        let n = $res.len() / 4;
        let pairs = n / 2;
        let res_512 = $res.as_mut_ptr() as *mut __m512i;
        let q2_512v = _mm512_loadu_si512(Q2_512.as_ptr() as *const __m512i);
        for i in 0..pairs {
            let rv = _mm512_loadu_si512(res_512.add(i) as *const __m512i);
            _mm512_storeu_si512(res_512.add(i), $op512(rv, q2_512v));
        }
        if n % 2 != 0 {
            let idx = n - 1;
            let res_256 = $res.as_mut_ptr() as *mut __m256i;
            let q2 = q2_vec();
            let rv = _mm256_loadu_si256(res_256.add(idx) as *const __m256i);
            _mm256_storeu_si256(res_256.add(idx), $op256(rv, q2));
        }
    }};
}

#[target_feature(enable = "avx512f")]
unsafe fn simd_add(res: &mut [u64], a: &[u64], b: &[u64]) {
    unsafe {
        dft_op_512!(
            res = res,
            a = a,
            b = b,
            op512 = |av: __m512i, bv: __m512i, q2: __m512i| cond_sub_2q_si512(_mm512_add_epi64(av, bv), q2),
            op256 = |av: __m256i, bv: __m256i, q2: __m256i| cond_sub_2q_si256(_mm256_add_epi64(av, bv), q2)
        );
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn simd_add_assign(res: &mut [u64], a: &[u64]) {
    unsafe {
        dft_op_512!(
            res_assign = res,
            a = a,
            op512 = |rv: __m512i, av: __m512i, q2: __m512i| cond_sub_2q_si512(_mm512_add_epi64(rv, av), q2),
            op256 = |rv: __m256i, av: __m256i, q2: __m256i| cond_sub_2q_si256(_mm256_add_epi64(rv, av), q2)
        );
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn simd_sub(res: &mut [u64], a: &[u64], b: &[u64]) {
    unsafe {
        dft_op_512!(
            res = res,
            a = a,
            b = b,
            op512 = |av: __m512i, bv: __m512i, q2: __m512i| cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(av, q2), bv), q2),
            op256 = |av: __m256i, bv: __m256i, q2: __m256i| cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(av, q2), bv), q2)
        );
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn simd_sub_assign(res: &mut [u64], a: &[u64]) {
    unsafe {
        dft_op_512!(
            res_assign = res,
            a = a,
            op512 = |rv: __m512i, av: __m512i, q2: __m512i| cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(rv, q2), av), q2),
            op256 = |rv: __m256i, av: __m256i, q2: __m256i| cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(rv, q2), av), q2)
        );
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn simd_sub_negate_assign(res: &mut [u64], a: &[u64]) {
    unsafe {
        dft_op_512!(
            res_assign = res,
            a = a,
            op512 = |rv: __m512i, av: __m512i, q2: __m512i| cond_sub_2q_si512(_mm512_sub_epi64(_mm512_add_epi64(av, q2), rv), q2),
            op256 = |rv: __m256i, av: __m256i, q2: __m256i| cond_sub_2q_si256(_mm256_sub_epi64(_mm256_add_epi64(av, q2), rv), q2)
        );
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn simd_negate(res: &mut [u64], a: &[u64]) {
    unsafe {
        dft_op_512!(
            res = res,
            a = a,
            b = a,
            op512 = |av: __m512i, _bv: __m512i, q2: __m512i| cond_sub_2q_si512(_mm512_sub_epi64(q2, av), q2),
            op256 = |av: __m256i, _bv: __m256i, q2: __m256i| cond_sub_2q_si256(_mm256_sub_epi64(q2, av), q2)
        );
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn simd_negate_assign(res: &mut [u64]) {
    unsafe {
        dft_op_512!(
            negate_assign = res,
            op512 = |rv: __m512i, q2: __m512i| cond_sub_2q_si512(_mm512_sub_epi64(q2, rv), q2),
            op256 = |rv: __m256i, q2: __m256i| cond_sub_2q_si256(_mm256_sub_epi64(q2, rv), q2)
        );
    }
}

/// Q vector (not 2Q) for c_from_b reduction: `[Q[0], Q[1], Q[2], 0]`.
fn q_vec() -> __m256i {
    use crate::ntt126_ifma::primes::{PrimeSetNtt126Ifma, Primes42};
    let q = <Primes42 as PrimeSetNtt126Ifma>::Q;
    unsafe { _mm256_set_epi64x(0, q[2] as i64, q[1] as i64, q[0] as i64) }
}

/// `oq[k] = Q[k] - (2^63 mod Q[k])` for negative i64 handling.
const OQ_IFMA: [u64; 4] = {
    let q = <Primes42 as crate::ntt126_ifma::primes::PrimeSetNtt126Ifma>::Q;
    let mut oq = [0u64; 4];
    let mut k = 0;
    while k < 3 {
        oq[k] = q[k] - (i64::MIN as u64 % q[k]);
        k += 1;
    }
    oq
};

/// `2^42 mod Q[k]` — used for two-pass modular reduction of values up to 2^63.
///
/// Since Q[k] ≈ 2^42, `2^42 mod Q[k] = 2^42 - Q[k]` which is small (< 2^23).
const POW42_MOD_Q_IFMA: [u64; 4] = {
    let q = <Primes42 as crate::ntt126_ifma::primes::PrimeSetNtt126Ifma>::Q;
    let pow42 = 1u64 << 42;
    // All three primes are < 2^43, so pow42 mod Q[k] = pow42 - Q[k]
    [pow42 - q[0], pow42 - q[1], pow42 - q[2], 0]
};

/// SIMD: convert n i64 coefficients to 3-prime CRT b format.
///
/// For each i64 x:
/// 1. Strip sign bit, conditionally add oq[k] for negative inputs
/// 2. Two-pass reduction: split at bit 42, multiply high part by (2^42 mod Q),
///    add to low part, repeat. Final conditional subtract gives [0, Q).
///
/// Result: `res[4*i+k] = a[i] mod Q[k]` for k in {0,1,2}, `res[4*i+3] = 0`.
#[target_feature(enable = "avx512vl")]
unsafe fn simd_b_from_znx64(n: usize, res: &mut [u64], a: &[i64]) {
    unsafe { simd_b_from_znx64_impl(n, res, a, !0i64) }
}

/// Same as [`simd_b_from_znx64`] but ANDs each input by `mask` first.
#[target_feature(enable = "avx512vl")]
unsafe fn simd_b_from_znx64_masked(n: usize, res: &mut [u64], a: &[i64], mask: i64) {
    unsafe { simd_b_from_znx64_impl(n, res, a, mask) }
}

#[inline]
#[target_feature(enable = "avx512vl")]
unsafe fn simd_b_from_znx64_impl(n: usize, res: &mut [u64], a: &[i64], mask: i64) {
    unsafe {
        let oq_vec = _mm256_loadu_si256(OQ_IFMA.as_ptr() as *const __m256i);
        let i64_max = _mm256_set1_epi64x(i64::MAX);
        let zero = _mm256_setzero_si256();
        let mask42 = _mm256_set1_epi64x((1i64 << 42) - 1);
        let pow42 = _mm256_loadu_si256(POW42_MOD_Q_IFMA.as_ptr() as *const __m256i);
        let q = q_vec();
        let lane3_zero = _mm256_set_epi64x(0, -1, -1, -1);
        let mask_vec = _mm256_set1_epi64x(mask);
        let mut r_ptr = res.as_mut_ptr() as *mut __m256i;

        for &xval in &a[..n] {
            let xv = _mm256_and_si256(_mm256_set1_epi64x(xval), mask_vec);
            let xl = _mm256_and_si256(xv, i64_max);
            let sign = _mm256_cmpgt_epi64(zero, xv);
            let add = _mm256_and_si256(sign, oq_vec);
            let val = _mm256_add_epi64(xl, add);

            let hi = _mm256_srli_epi64::<42>(val);
            let lo = _mm256_and_si256(val, mask42);
            let y = _mm256_add_epi64(_mm256_mul_epu32(hi, pow42), lo);

            let hi2 = _mm256_srli_epi64::<42>(y);
            let lo2 = _mm256_and_si256(y, mask42);
            let z = _mm256_add_epi64(_mm256_mul_epu32(hi2, pow42), lo2);

            let result = cond_sub_2q_si256(z, q);
            let result = _mm256_and_si256(result, lane3_zero);

            _mm256_storeu_si256(r_ptr, result);
            r_ptr = r_ptr.add(1);
        }
    }
}

/// AVX-512: reduce b-format values in [0, 2q) to c-format values in [0, q).
/// Uses 512-bit main loop (2 coefficients per iteration).
#[target_feature(enable = "avx512f")]
unsafe fn simd_c_from_b(n: usize, res: &mut [u64], a: &[u64]) {
    unsafe {
        let q_512 = {
            let q = <Primes42 as crate::ntt126_ifma::primes::PrimeSetNtt126Ifma>::Q;
            let arr: [u64; 8] = [q[0], q[1], q[2], 0, q[0], q[1], q[2], 0];
            _mm512_loadu_si512(arr.as_ptr() as *const __m512i)
        };
        let res_512 = res.as_mut_ptr() as *mut __m512i;
        let a_512 = a.as_ptr() as *const __m512i;
        let pairs = n / 2;
        for i in 0..pairs {
            let av = _mm512_loadu_si512(a_512.add(i));
            _mm512_storeu_si512(res_512.add(i), cond_sub_2q_si512(av, q_512));
        }
        if !n.is_multiple_of(2) {
            let idx = n - 1;
            let res_256 = res.as_mut_ptr() as *mut __m256i;
            let a_256 = a.as_ptr() as *const __m256i;
            let q = q_vec();
            let av = _mm256_loadu_si256(a_256.add(idx));
            _mm256_storeu_si256(res_256.add(idx), cond_sub_2q_si256(av, q));
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// IFMA NTT execution
// ──────────────────────────────────────────────────────────────────────────────

impl Ntt126IfmaDFTExecute<Ntt126IfmaTable<Primes42>> for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_dft_execute(table: &Ntt126IfmaTable<Primes42>, data: &mut [u64]) {
        unsafe { ntt_avx512::<Primes42>(table, data) }
    }
}

impl Ntt126IfmaDFTExecute<Ntt126IfmaTableInv<Primes42>> for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_dft_execute(table: &Ntt126IfmaTableInv<Primes42>, data: &mut [u64]) {
        unsafe { intt_avx512::<Primes42>(table, data) }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Domain conversion
// ──────────────────────────────────────────────────────────────────────────────

impl Ntt126IfmaFromZnx64 for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_from_znx64(res: &mut [u64], a: &[i64]) {
        unsafe { simd_b_from_znx64(a.len(), res, a) };
    }

    #[inline(always)]
    fn ntt126_ifma_from_znx64_masked(res: &mut [u64], a: &[i64], mask: i64) {
        unsafe { simd_b_from_znx64_masked(a.len(), res, a, mask) };
    }
}

impl Ntt126IfmaToZnx128 for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_to_znx128(res: &mut [i128], divisor_is_n: usize, a: &[u64]) {
        unsafe { super::vec_znx_dft::simd_b_ntt126_ifma_to_znx128(divisor_is_n, res, a) };
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// IFMA-specific addition / subtraction / negation / copy / zero
// ──────────────────────────────────────────────────────────────────────────────

impl Ntt126IfmaAdd for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_add(res: &mut [u64], a: &[u64], b: &[u64]) {
        unsafe { simd_add(res, a, b) };
    }
}

impl Ntt126IfmaAddAssign for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_add_assign(res: &mut [u64], a: &[u64]) {
        unsafe { simd_add_assign(res, a) };
    }
}

impl Ntt126IfmaSub for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_sub(res: &mut [u64], a: &[u64], b: &[u64]) {
        unsafe { simd_sub(res, a, b) };
    }
}

impl Ntt126IfmaSubAssign for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_sub_assign(res: &mut [u64], a: &[u64]) {
        unsafe { simd_sub_assign(res, a) };
    }
}

impl Ntt126IfmaSubNegateAssign for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_sub_negate_assign(res: &mut [u64], a: &[u64]) {
        unsafe { simd_sub_negate_assign(res, a) };
    }
}

impl Ntt126IfmaNegate for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_negate(res: &mut [u64], a: &[u64]) {
        unsafe { simd_negate(res, a) };
    }
}

impl Ntt126IfmaNegateAssign for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_negate_assign(res: &mut [u64]) {
        unsafe { simd_negate_assign(res) };
    }
}

impl Ntt126IfmaZero for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_zero(res: &mut [u64]) {
        res.fill(0);
    }
}

impl Ntt126IfmaCopy for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_copy(res: &mut [u64], a: &[u64]) {
        res.copy_from_slice(a);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// IFMA multiply-accumulate
// ──────────────────────────────────────────────────────────────────────────────

impl Ntt126IfmaMulBbc for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_mul_bbc(meta: &Bbc126IfmaMeta<Primes42>, ell: usize, res: &mut [u64], ntt_coeff: &[u32], prepared: &[u32]) {
        unsafe { vec_mat1col_product_bbc_ifma(meta, ell, res, ntt_coeff, prepared) };
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// b -> c conversion
// ──────────────────────────────────────────────────────────────────────────────

impl Ntt126IfmaCFromB for NTT126Ifma {
    #[inline(always)]
    fn ntt126_ifma_c_from_b(n: usize, res: &mut [u32], a: &[u64]) {
        // c format for IFMA = reduced residues: a[k] mod Q[k].
        // Values in [0, 2q) → cond_sub with q → [0, q).
        // res is typed as &mut [u32] for trait compatibility but is actually u64 data.
        let res_u64: &mut [u64] = unsafe { std::slice::from_raw_parts_mut(res.as_mut_ptr() as *mut u64, res.len() / 2) };
        unsafe { simd_c_from_b(n, res_u64, a) };
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// NTT120 Ntt* traits (DFT-domain arithmetic reuse)
//
// The ntt120 generic functions (ntt120_vec_znx_dft_add, etc.) require these
// traits. Since the 3-prime IFMA layout uses the same 4 x u64 per coefficient
// representation (with lane 3 as padding), the implementation is identical
// to the NTT120 version but uses Q_SHIFTED_NTT126IFMA for the 3 active lanes.
// ──────────────────────────────────────────────────────────────────────────────

impl NttAdd for NTT126Ifma {
    #[inline(always)]
    fn ntt_add(res: &mut [u64], a: &[u64], b: &[u64]) {
        unsafe { simd_add(res, a, b) };
    }
}

impl NttAddAssign for NTT126Ifma {
    #[inline(always)]
    fn ntt_add_assign(res: &mut [u64], a: &[u64]) {
        unsafe { simd_add_assign(res, a) };
    }
}

impl NttSub for NTT126Ifma {
    #[inline(always)]
    fn ntt_sub(res: &mut [u64], a: &[u64], b: &[u64]) {
        unsafe { simd_sub(res, a, b) };
    }
}

impl NttSubAssign for NTT126Ifma {
    #[inline(always)]
    fn ntt_sub_assign(res: &mut [u64], a: &[u64]) {
        unsafe { simd_sub_assign(res, a) };
    }
}

impl NttSubNegateAssign for NTT126Ifma {
    #[inline(always)]
    fn ntt_sub_negate_assign(res: &mut [u64], a: &[u64]) {
        unsafe { simd_sub_negate_assign(res, a) };
    }
}

impl NttNegate for NTT126Ifma {
    #[inline(always)]
    fn ntt_negate(res: &mut [u64], a: &[u64]) {
        unsafe { simd_negate(res, a) };
    }
}

impl NttNegateAssign for NTT126Ifma {
    #[inline(always)]
    fn ntt_negate_assign(res: &mut [u64]) {
        unsafe { simd_negate_assign(res) };
    }
}

impl NttZero for NTT126Ifma {
    #[inline(always)]
    fn ntt_zero(res: &mut [u64]) {
        res.fill(0);
    }
}

impl NttCopy for NTT126Ifma {
    #[inline(always)]
    fn ntt_copy(res: &mut [u64], a: &[u64]) {
        res.copy_from_slice(a);
    }
}
