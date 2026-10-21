//! Large-coefficient (i128) ring element vector support for [`NTT4x30Avx512`](super::NTT4x30Avx512).
//!
//! The shared `poulpy-cpu-portable` NTT4x30 defaults rely on backend-provided `I128BigOps`
//! hooks for vectorized i128 operations. The kernels themselves live in
//! [`crate::vec_znx_big_avx512`] and operate on **4 i128 per `__m512i`**
//! (the 256-bit AVX backend handles 2 i128 per `__m256i`). Mask-form compares
//! (`_mm512_cmpgt_epi64_mask`, `_mm512_cmpeq_epi64_mask`) drive borrow propagation
//! in place of the 256-bit vector-andnot trick.

use super::NTT4x30Avx512;
use crate::vec_znx_big_avx512::{
    vi128_add_assign_avx512, vi128_add_avx512, vi128_add_small_assign_avx512, vi128_add_small_avx512, vi128_from_small_avx512,
    vi128_hadamard_i64_avx512, vi128_neg_from_small_avx512, vi128_negate_assign_avx512, vi128_negate_avx512,
    vi128_sub_assign_avx512, vi128_sub_avx512, vi128_sub_negate_assign_avx512, vi128_sub_small_a_avx512,
    vi128_sub_small_assign_avx512, vi128_sub_small_b_avx512, vi128_sub_small_negate_assign_avx512,
};
use poulpy_cpu_portable::hal_defaults::BigWordHadamardProduct;
use poulpy_cpu_portable::kernels::ntt4x30::I128BigOps;
use poulpy_hal::layouts::Ring;

impl<R: Ring> I128BigOps for NTT4x30Avx512<R> {
    #[inline(always)]
    fn i128_hadamard_product_i64(res: &mut [i128], a: &[i64], b: &[i64]) {
        unsafe { vi128_hadamard_i64_avx512(res.len(), res, a, b) }
    }

    #[inline(always)]
    fn i128_add(res: &mut [i128], a: &[i128], b: &[i128]) {
        // SAFETY: NTT4x30Avx512::new() verifies AVX-512F availability at construction time.
        unsafe { vi128_add_avx512(res.len(), res, a, b) }
    }
    #[inline(always)]
    fn i128_add_assign(res: &mut [i128], a: &[i128]) {
        unsafe { vi128_add_assign_avx512(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_add_small(res: &mut [i128], a: &[i128], b: &[i64]) {
        unsafe { vi128_add_small_avx512(res.len(), res, a, b) }
    }
    #[inline(always)]
    fn i128_add_small_assign(res: &mut [i128], a: &[i64]) {
        unsafe { vi128_add_small_assign_avx512(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_sub(res: &mut [i128], a: &[i128], b: &[i128]) {
        unsafe { vi128_sub_avx512(res.len(), res, a, b) }
    }
    #[inline(always)]
    fn i128_sub_assign(res: &mut [i128], a: &[i128]) {
        unsafe { vi128_sub_assign_avx512(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_sub_negate_assign(res: &mut [i128], a: &[i128]) {
        unsafe { vi128_sub_negate_assign_avx512(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_sub_small_a(res: &mut [i128], a: &[i64], b: &[i128]) {
        unsafe { vi128_sub_small_a_avx512(res.len(), res, a, b) }
    }
    #[inline(always)]
    fn i128_sub_small_b(res: &mut [i128], a: &[i128], b: &[i64]) {
        unsafe { vi128_sub_small_b_avx512(res.len(), res, a, b) }
    }
    #[inline(always)]
    fn i128_sub_small_assign(res: &mut [i128], a: &[i64]) {
        unsafe { vi128_sub_small_assign_avx512(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_sub_small_negate_assign(res: &mut [i128], a: &[i64]) {
        unsafe { vi128_sub_small_negate_assign_avx512(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_negate(res: &mut [i128], a: &[i128]) {
        unsafe { vi128_negate_avx512(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_negate_assign(res: &mut [i128]) {
        unsafe { vi128_negate_assign_avx512(res.len(), res) }
    }
    #[inline(always)]
    fn i128_neg_from_small(res: &mut [i128], a: &[i64]) {
        unsafe { vi128_neg_from_small_avx512(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_from_small(res: &mut [i128], a: &[i64]) {
        unsafe { vi128_from_small_avx512(res.len(), res, a) }
    }
}

impl<R: Ring> BigWordHadamardProduct for NTT4x30Avx512<R> {
    #[inline(always)]
    fn big_word_hadamard_product(res: &mut [i128], a: &[i64], b: &[i64]) {
        Self::i128_hadamard_product_i64(res, a, b)
    }
}
