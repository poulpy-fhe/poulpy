//! Large-coefficient (i128) ring element vector support for [`NTT3x42Ifma`](super::NTT3x42Ifma).
//!
//! The shared `poulpy-hal` NTT4x30 defaults rely on backend-provided `I128BigOps`
//! and `I128NormalizeOps` hooks for vectorized i128 operations. The kernels themselves
//! live in [`crate::vec_znx_big_avx512`] and are shared with [`NTT4x30Avx512`](crate::NTT4x30Avx512):
//! both backends use an i128 `ScalarBig`, so the AVX-512F mask-form borrow propagation
//! and the fused `±= normalize(a)` middle/final steps apply unchanged.

use super::NTT3x42Ifma;
use crate::vec_znx_big_avx512::{
    nfc_final_step_add_assign_avx512, nfc_final_step_add_assign_scalar, nfc_final_step_assign_avx512,
    nfc_final_step_assign_scalar, nfc_final_step_sub_assign_avx512, nfc_final_step_sub_assign_scalar,
    nfc_middle_step_add_assign_avx512, nfc_middle_step_add_assign_scalar, nfc_middle_step_assign_avx512,
    nfc_middle_step_assign_scalar, nfc_middle_step_avx512, nfc_middle_step_scalar, nfc_middle_step_sub_assign_avx512,
    nfc_middle_step_sub_assign_scalar, vi128_add_assign_avx512, vi128_add_avx512, vi128_add_small_assign_avx512,
    vi128_add_small_avx512, vi128_from_small_avx512, vi128_hadamard_i64_avx512, vi128_neg_from_small_avx512,
    vi128_negate_assign_avx512, vi128_negate_avx512, vi128_sub_assign_avx512, vi128_sub_avx512, vi128_sub_negate_assign_avx512,
    vi128_sub_small_a_avx512, vi128_sub_small_assign_avx512, vi128_sub_small_b_avx512, vi128_sub_small_negate_assign_avx512,
};
use poulpy_cpu_ref::hal_defaults::ScalarBigHadamardProduct;
use poulpy_cpu_ref::reference::ntt4x30::{I128BigOps, I128NormalizeOps, vec_znx_big::AssignOp};

impl I128BigOps for NTT3x42Ifma {
    #[inline(always)]
    fn i128_hadamard_product_i64(res: &mut [i128], a: &[i64], b: &[i64]) {
        unsafe { vi128_hadamard_i64_avx512(res.len(), res, a, b) }
    }

    #[inline(always)]
    fn i128_add(res: &mut [i128], a: &[i128], b: &[i128]) {
        // SAFETY: NTT3x42Ifma::new() verifies AVX-512F availability at construction time.
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

impl ScalarBigHadamardProduct for NTT3x42Ifma {
    #[inline(always)]
    fn scalar_big_hadamard_product(res: &mut [i128], a: &[i64], b: &[i64]) {
        Self::i128_hadamard_product_i64(res, a, b)
    }
}

impl I128NormalizeOps for NTT3x42Ifma {
    #[inline(always)]
    fn nfc_middle_step(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
        if base2k <= 64 && res.len() >= 8 {
            unsafe { nfc_middle_step_avx512(base2k as u32, lsh as u32, res.len(), res, a, carry) }
        } else {
            nfc_middle_step_scalar(base2k, lsh, res, a, carry);
        }
    }

    #[inline(always)]
    fn nfc_middle_step_assign(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        if base2k <= 64 && res.len() >= 8 {
            unsafe { nfc_middle_step_assign_avx512(base2k as u32, lsh as u32, res.len(), res, carry) }
        } else {
            nfc_middle_step_assign_scalar(base2k, lsh, res, carry);
        }
    }

    #[inline(always)]
    fn nfc_middle_step_into<O: AssignOp>(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
        if base2k <= 64 && res.len() >= 8 {
            if O::SUB {
                unsafe { nfc_middle_step_sub_assign_avx512(base2k as u32, lsh as u32, res.len(), res, a, carry) }
            } else {
                unsafe { nfc_middle_step_add_assign_avx512(base2k as u32, lsh as u32, res.len(), res, a, carry) }
            }
        } else if O::SUB {
            nfc_middle_step_sub_assign_scalar(base2k, lsh, res, a, carry);
        } else {
            nfc_middle_step_add_assign_scalar(base2k, lsh, res, a, carry);
        }
    }

    #[inline(always)]
    fn nfc_final_step_assign(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        if base2k <= 64 && res.len() >= 8 {
            unsafe { nfc_final_step_assign_avx512(base2k as u32, lsh as u32, res.len(), res, carry) }
        } else {
            nfc_final_step_assign_scalar(base2k, lsh, res, carry);
        }
    }

    #[inline(always)]
    fn nfc_final_step_into<O: AssignOp>(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        if base2k <= 64 && res.len() >= 8 {
            if O::SUB {
                unsafe { nfc_final_step_sub_assign_avx512(base2k as u32, lsh as u32, res.len(), res, carry) }
            } else {
                unsafe { nfc_final_step_add_assign_avx512(base2k as u32, lsh as u32, res.len(), res, carry) }
            }
        } else if O::SUB {
            nfc_final_step_sub_assign_scalar(base2k, lsh, res, carry);
        } else {
            nfc_final_step_add_assign_scalar(base2k, lsh, res, carry);
        }
    }
}
