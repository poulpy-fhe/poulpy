//! Large-coefficient (i128) ring element vector support for [`NTT4x30Avx`](super::NTT4x30Avx).
//!
//! The shared `poulpy-hal` NTT4x30 defaults rely on backend-provided `I128BigOps`
//! and `I128NormalizeOps` hooks for vectorized i128 operations.

use super::{
    NTT4x30Avx,
    vec_znx_big_avx::{
        nfc_final_step_assign_avx2, nfc_final_step_assign_scalar, nfc_middle_step_assign_avx2, nfc_middle_step_assign_scalar,
        nfc_middle_step_avx2, nfc_middle_step_scalar, vi128_add_assign_avx2, vi128_add_avx2, vi128_add_small_assign_avx2,
        vi128_add_small_avx2, vi128_from_small_avx2, vi128_hadamard_i64_avx2, vi128_neg_from_small_avx2,
        vi128_negate_assign_avx2, vi128_negate_avx2, vi128_sub_assign_avx2, vi128_sub_avx2, vi128_sub_negate_assign_avx2,
        vi128_sub_small_a_avx2, vi128_sub_small_assign_avx2, vi128_sub_small_b_avx2, vi128_sub_small_negate_assign_avx2,
    },
};
use poulpy_cpu_ref::hal_defaults::BigWordHadamardProduct;
use poulpy_cpu_ref::reference::ntt4x30::{I128BigOps, I128NormalizeOps};

impl I128BigOps for NTT4x30Avx {
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

impl BigWordHadamardProduct for NTT4x30Avx {
    #[inline(always)]
    fn big_word_hadamard_product(res: &mut [i128], a: &[i64], b: &[i64]) {
        Self::i128_hadamard_product_i64(res, a, b)
    }
}

impl I128NormalizeOps for NTT4x30Avx {
    #[inline(always)]
    fn nfc_middle_step(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
        // SAFETY: NTT4x30Avx::new() verifies AVX2 availability at construction time.
        if base2k <= 64 && res.len() >= 4 {
            unsafe { nfc_middle_step_avx2(base2k as u32, lsh as u32, res.len(), res, a, carry) }
        } else {
            nfc_middle_step_scalar(base2k, lsh, res, a, carry);
        }
    }

    #[inline(always)]
    fn nfc_middle_step_assign(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        // SAFETY: NTT4x30Avx::new() verifies AVX2 availability at construction time.
        if base2k <= 64 && res.len() >= 4 {
            unsafe { nfc_middle_step_assign_avx2(base2k as u32, lsh as u32, res.len(), res, carry) }
        } else {
            nfc_middle_step_assign_scalar(base2k, lsh, res, carry);
        }
    }

    #[inline(always)]
    fn nfc_final_step_assign(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        // SAFETY: NTT4x30Avx::new() verifies AVX2 availability at construction time.
        if base2k <= 64 && res.len() >= 4 {
            unsafe { nfc_final_step_assign_avx2(base2k as u32, lsh as u32, res.len(), res, carry) }
        } else {
            nfc_final_step_assign_scalar(base2k, lsh, res, carry);
        }
    }
}
