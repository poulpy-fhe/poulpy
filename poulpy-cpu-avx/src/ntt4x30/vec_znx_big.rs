//! Large-coefficient (i128) ring element vector support for [`NTT4x30Avx`](super::NTT4x30Avx).
//!
//! The shared `poulpy-cpu-portable` NTT4x30 defaults rely on backend-provided `I128BigOps`
//! hooks for vectorized i128 operations.

use super::{
    NTT4x30Avx,
    vec_znx_big_avx::{
        vi128_add_assign_avx2, vi128_add_avx2, vi128_add_small_assign_avx2, vi128_add_small_avx2, vi128_from_small_avx2,
        vi128_hadamard_i64_avx2, vi128_neg_from_small_avx2, vi128_negate_assign_avx2, vi128_negate_avx2, vi128_sub_assign_avx2,
        vi128_sub_avx2, vi128_sub_negate_assign_avx2, vi128_sub_small_a_avx2, vi128_sub_small_assign_avx2,
        vi128_sub_small_b_avx2, vi128_sub_small_negate_assign_avx2,
    },
};
use poulpy_cpu_portable::hal_defaults::BigWordHadamardProduct;
use poulpy_cpu_portable::kernels::ntt4x30::I128BigOps;
use poulpy_hal::layouts::Ring;

impl<R: Ring> I128BigOps for NTT4x30Avx<R> {
    #[inline(always)]
    fn i128_hadamard_product_i64(res: &mut [i128], a: &[i64], b: &[i64]) {
        unsafe { vi128_hadamard_i64_avx2(res.len(), res, a, b) }
    }

    #[inline(always)]
    fn i128_add(res: &mut [i128], a: &[i128], b: &[i128]) {
        // SAFETY: NTT4x30Avx::new() verifies AVX2 availability at construction time.
        unsafe { vi128_add_avx2(res.len(), res, a, b) }
    }
    #[inline(always)]
    fn i128_add_assign(res: &mut [i128], a: &[i128]) {
        unsafe { vi128_add_assign_avx2(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_add_small(res: &mut [i128], a: &[i128], b: &[i64]) {
        unsafe { vi128_add_small_avx2(res.len(), res, a, b) }
    }
    #[inline(always)]
    fn i128_add_small_assign(res: &mut [i128], a: &[i64]) {
        unsafe { vi128_add_small_assign_avx2(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_sub(res: &mut [i128], a: &[i128], b: &[i128]) {
        unsafe { vi128_sub_avx2(res.len(), res, a, b) }
    }
    #[inline(always)]
    fn i128_sub_assign(res: &mut [i128], a: &[i128]) {
        unsafe { vi128_sub_assign_avx2(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_sub_negate_assign(res: &mut [i128], a: &[i128]) {
        unsafe { vi128_sub_negate_assign_avx2(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_sub_small_a(res: &mut [i128], a: &[i64], b: &[i128]) {
        unsafe { vi128_sub_small_a_avx2(res.len(), res, a, b) }
    }
    #[inline(always)]
    fn i128_sub_small_b(res: &mut [i128], a: &[i128], b: &[i64]) {
        unsafe { vi128_sub_small_b_avx2(res.len(), res, a, b) }
    }
    #[inline(always)]
    fn i128_sub_small_assign(res: &mut [i128], a: &[i64]) {
        unsafe { vi128_sub_small_assign_avx2(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_sub_small_negate_assign(res: &mut [i128], a: &[i64]) {
        unsafe { vi128_sub_small_negate_assign_avx2(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_negate(res: &mut [i128], a: &[i128]) {
        unsafe { vi128_negate_avx2(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_negate_assign(res: &mut [i128]) {
        unsafe { vi128_negate_assign_avx2(res.len(), res) }
    }
    #[inline(always)]
    fn i128_neg_from_small(res: &mut [i128], a: &[i64]) {
        unsafe { vi128_neg_from_small_avx2(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_from_small(res: &mut [i128], a: &[i64]) {
        unsafe { vi128_from_small_avx2(res.len(), res, a) }
    }
}

impl<R: Ring> BigWordHadamardProduct for NTT4x30Avx<R> {
    #[inline(always)]
    fn big_word_hadamard_product(res: &mut [i128], a: &[i64], b: &[i64]) {
        Self::i128_hadamard_product_i64(res, a, b)
    }
}
