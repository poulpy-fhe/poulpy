//! Large-coefficient (i128) ring element vector support for [`NTT4x30Neon`](super::NTT4x30Neon).

use super::NTT4x30Neon;
#[cfg(target_arch = "aarch64")]
use poulpy_cpu_ref::reference::ntt4x30::vec_znx_big::AssignOp;
use poulpy_cpu_ref::reference::ntt4x30::{I128BigOps, I128NormalizeOps};

#[cfg(target_arch = "aarch64")]
use crate::neon::normalize::{
    nfc_final_step_assign_neon, nfc_final_step_into_neon, nfc_middle_step_assign_neon, nfc_middle_step_into_neon,
    nfc_middle_step_neon,
};
#[cfg(target_arch = "aarch64")]
use crate::neon::vec_znx_big::{
    vi128_add_assign_neon, vi128_add_neon, vi128_add_small_assign_neon, vi128_add_small_neon, vi128_from_small_neon,
    vi128_neg_from_small_neon, vi128_negate_assign_neon, vi128_negate_neon, vi128_sub_assign_neon, vi128_sub_negate_assign_neon,
    vi128_sub_neon, vi128_sub_small_a_neon, vi128_sub_small_assign_neon, vi128_sub_small_b_neon,
    vi128_sub_small_negate_assign_neon,
};

#[cfg(target_arch = "aarch64")]
impl I128BigOps for NTT4x30Neon {
    #[inline(always)]
    fn i128_add(res: &mut [i128], a: &[i128], b: &[i128]) {
        vi128_add_neon(res.len(), res, a, b);
    }
    #[inline(always)]
    fn i128_add_assign(res: &mut [i128], a: &[i128]) {
        vi128_add_assign_neon(res.len(), res, a);
    }
    #[inline(always)]
    fn i128_add_small(res: &mut [i128], a: &[i128], b: &[i64]) {
        vi128_add_small_neon(res.len(), res, a, b);
    }
    #[inline(always)]
    fn i128_add_small_assign(res: &mut [i128], a: &[i64]) {
        vi128_add_small_assign_neon(res.len(), res, a);
    }
    #[inline(always)]
    fn i128_sub(res: &mut [i128], a: &[i128], b: &[i128]) {
        vi128_sub_neon(res.len(), res, a, b);
    }
    #[inline(always)]
    fn i128_sub_assign(res: &mut [i128], a: &[i128]) {
        vi128_sub_assign_neon(res.len(), res, a);
    }
    #[inline(always)]
    fn i128_sub_negate_assign(res: &mut [i128], a: &[i128]) {
        vi128_sub_negate_assign_neon(res.len(), res, a);
    }
    #[inline(always)]
    fn i128_sub_small_a(res: &mut [i128], a: &[i64], b: &[i128]) {
        vi128_sub_small_a_neon(res.len(), res, a, b);
    }
    #[inline(always)]
    fn i128_sub_small_b(res: &mut [i128], a: &[i128], b: &[i64]) {
        vi128_sub_small_b_neon(res.len(), res, a, b);
    }
    #[inline(always)]
    fn i128_sub_small_assign(res: &mut [i128], a: &[i64]) {
        vi128_sub_small_assign_neon(res.len(), res, a);
    }
    #[inline(always)]
    fn i128_sub_small_negate_assign(res: &mut [i128], a: &[i64]) {
        vi128_sub_small_negate_assign_neon(res.len(), res, a);
    }
    #[inline(always)]
    fn i128_negate(res: &mut [i128], a: &[i128]) {
        vi128_negate_neon(res.len(), res, a);
    }
    #[inline(always)]
    fn i128_negate_assign(res: &mut [i128]) {
        vi128_negate_assign_neon(res.len(), res);
    }
    #[inline(always)]
    fn i128_neg_from_small(res: &mut [i128], a: &[i64]) {
        vi128_neg_from_small_neon(res.len(), res, a);
    }
    #[inline(always)]
    fn i128_from_small(res: &mut [i128], a: &[i64]) {
        vi128_from_small_neon(res.len(), res, a);
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl I128BigOps for NTT4x30Neon {}

impl poulpy_cpu_ref::hal_defaults::BigWordHadamardProduct for NTT4x30Neon {
    #[inline(always)]
    fn big_word_hadamard_product(res: &mut [i128], a: &[i64], b: &[i64]) {
        <Self as I128BigOps>::i128_hadamard_product_i64(res, a, b)
    }
}

#[cfg(target_arch = "aarch64")]
impl I128NormalizeOps for NTT4x30Neon {
    #[inline(always)]
    fn nfc_middle_step(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
        nfc_middle_step_neon(base2k, lsh, res, a, carry);
    }
    #[inline(always)]
    fn nfc_middle_step_into<O: AssignOp>(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
        nfc_middle_step_into_neon::<O>(base2k, lsh, res, a, carry);
    }
    #[inline(always)]
    fn nfc_middle_step_assign(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        nfc_middle_step_assign_neon(base2k, lsh, res, carry);
    }
    #[inline(always)]
    fn nfc_final_step_assign(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        nfc_final_step_assign_neon(base2k, lsh, res, carry);
    }
    #[inline(always)]
    fn nfc_final_step_into<O: AssignOp>(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        nfc_final_step_into_neon::<O>(base2k, lsh, res, carry);
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl I128NormalizeOps for NTT4x30Neon {}
