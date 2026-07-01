//! Single ring element (`Z[X]/(X^n+1)`) arithmetic for [`NTT4x30Avx`](super::NTT4x30Avx).
//!
//! Implements the `Znx*` traits from `poulpy_cpu_ref::reference::znx`. All implementations
//! delegate to the AVX2-accelerated functions in `crate::znx_avx` (same kernels used
//! by `FFT64Avx`). These operate on plain `&[i64]` slices and are backend-independent.

use poulpy_cpu_ref::reference::znx::{
    ZnxAdd, ZnxAddAssign, ZnxAutomorphism, ZnxAutomorphismRotate, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulAddPowerOfTwo,
    ZnxMulPowerOfTwo, ZnxMulPowerOfTwoAssign, ZnxNegate, ZnxNegateAssign, ZnxNormalizeDigit, ZnxNormalizeFinalStep,
    ZnxNormalizeFinalStepAssign, ZnxNormalizeFinalStepSub, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepAssign,
    ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepAssign, ZnxNormalizeMiddleStepCarryOnly,
    ZnxNormalizeMiddleStepSub, ZnxRotate, ZnxSub, ZnxSubAssign, ZnxSubNegateAssign, ZnxSwitchRing, ZnxZero, znx_copy_ref,
    znx_rotate, znx_zero_ref,
};

use crate::znx_avx::{
    znx_add_assign_avx, znx_add_avx, znx_automorphism_avx, znx_automorphism_rotate_avx, znx_extract_digit_addmul_avx,
    znx_mul_add_power_of_two_avx, znx_mul_power_of_two_assign_avx, znx_mul_power_of_two_avx, znx_negate_assign_avx,
    znx_negate_avx, znx_normalize_digit_avx, znx_normalize_final_step_assign_avx, znx_normalize_final_step_avx,
    znx_normalize_final_step_sub_avx, znx_normalize_first_step_assign_avx, znx_normalize_first_step_avx,
    znx_normalize_first_step_carry_only_avx, znx_normalize_middle_step_assign_avx, znx_normalize_middle_step_avx,
    znx_normalize_middle_step_carry_only_avx, znx_normalize_middle_step_sub_avx, znx_sub_assign_avx, znx_sub_avx,
    znx_sub_negate_assign_avx, znx_switch_ring_avx,
};

use super::NTT4x30Avx;

impl ZnxAdd for NTT4x30Avx {
    #[inline(always)]
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]) {
        unsafe { znx_add_avx(res, a, b) }
    }
}

impl ZnxAddAssign for NTT4x30Avx {
    #[inline(always)]
    fn znx_add_assign(res: &mut [i64], a: &[i64]) {
        unsafe { znx_add_assign_avx(res, a) }
    }
}

impl ZnxSub for NTT4x30Avx {
    #[inline(always)]
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]) {
        unsafe { znx_sub_avx(res, a, b) }
    }
}

impl ZnxSubAssign for NTT4x30Avx {
    #[inline(always)]
    fn znx_sub_assign(res: &mut [i64], a: &[i64]) {
        unsafe { znx_sub_assign_avx(res, a) }
    }
}

impl ZnxSubNegateAssign for NTT4x30Avx {
    #[inline(always)]
    fn znx_sub_negate_assign(res: &mut [i64], a: &[i64]) {
        unsafe { znx_sub_negate_assign_avx(res, a) }
    }
}

impl ZnxMulAddPowerOfTwo for NTT4x30Avx {
    #[inline(always)]
    fn znx_muladd_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        unsafe { znx_mul_add_power_of_two_avx(k, res, a) }
    }
}

impl ZnxMulPowerOfTwo for NTT4x30Avx {
    #[inline(always)]
    fn znx_mul_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        unsafe { znx_mul_power_of_two_avx(k, res, a) }
    }
}

impl ZnxMulPowerOfTwoAssign for NTT4x30Avx {
    #[inline(always)]
    fn znx_mul_power_of_two_assign(k: i64, res: &mut [i64]) {
        unsafe { znx_mul_power_of_two_assign_avx(k, res) }
    }
}

impl ZnxAutomorphism for NTT4x30Avx {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        unsafe { znx_automorphism_avx(p, res, a) }
    }
}

impl ZnxAutomorphismRotate for NTT4x30Avx {
    #[inline(always)]
    fn znx_automorphism_rotate(p: i64, k: i64, res: &mut [i64], a: &[i64]) {
        unsafe { znx_automorphism_rotate_avx(p, k, res, a) }
    }
}

impl ZnxCopy for NTT4x30Avx {
    #[inline(always)]
    fn znx_copy(res: &mut [i64], a: &[i64]) {
        znx_copy_ref(res, a);
    }
}

impl ZnxNegate for NTT4x30Avx {
    #[inline(always)]
    fn znx_negate(res: &mut [i64], src: &[i64]) {
        unsafe { znx_negate_avx(res, src) }
    }
}

impl ZnxNegateAssign for NTT4x30Avx {
    #[inline(always)]
    fn znx_negate_assign(res: &mut [i64]) {
        unsafe { znx_negate_assign_avx(res) }
    }
}

impl ZnxRotate for NTT4x30Avx {
    #[inline(always)]
    fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]) {
        znx_rotate::<Self>(p, res, src);
    }
}

impl ZnxZero for NTT4x30Avx {
    #[inline(always)]
    fn znx_zero(res: &mut [i64]) {
        znx_zero_ref(res);
    }
}

impl ZnxSwitchRing for NTT4x30Avx {
    #[inline(always)]
    fn znx_switch_ring(res: &mut [i64], a: &[i64]) {
        unsafe { znx_switch_ring_avx(res, a) }
    }
}

impl ZnxNormalizeFirstStep for NTT4x30Avx {
    #[inline(always)]
    fn znx_normalize_first_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        unsafe { znx_normalize_first_step_avx::<OVERWRITE>(base2k, lsh, x, a, carry) }
    }
}

impl ZnxNormalizeMiddleStep for NTT4x30Avx {
    #[inline(always)]
    fn znx_normalize_middle_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        unsafe { znx_normalize_middle_step_avx::<OVERWRITE>(base2k, lsh, x, a, carry) }
    }
}

impl ZnxNormalizeFinalStep for NTT4x30Avx {
    #[inline(always)]
    fn znx_normalize_final_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        unsafe { znx_normalize_final_step_avx::<OVERWRITE>(base2k, lsh, x, a, carry) }
    }
}

impl ZnxNormalizeMiddleStepSub for NTT4x30Avx {
    #[inline(always)]
    fn znx_normalize_middle_step_sub(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        unsafe { znx_normalize_middle_step_sub_avx(base2k, lsh, x, a, carry) }
    }
}

impl ZnxNormalizeFinalStepSub for NTT4x30Avx {
    #[inline(always)]
    fn znx_normalize_final_step_sub(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        unsafe { znx_normalize_final_step_sub_avx(base2k, lsh, x, a, carry) }
    }
}

impl ZnxNormalizeFinalStepAssign for NTT4x30Avx {
    #[inline(always)]
    fn znx_normalize_final_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        unsafe { znx_normalize_final_step_assign_avx(base2k, lsh, x, carry) }
    }
}

impl ZnxNormalizeFirstStepCarryOnly for NTT4x30Avx {
    #[inline(always)]
    fn znx_normalize_first_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        unsafe { znx_normalize_first_step_carry_only_avx(base2k, lsh, x, carry) }
    }
}

impl ZnxNormalizeFirstStepAssign for NTT4x30Avx {
    #[inline(always)]
    fn znx_normalize_first_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        unsafe { znx_normalize_first_step_assign_avx(base2k, lsh, x, carry) }
    }
}

impl ZnxNormalizeMiddleStepCarryOnly for NTT4x30Avx {
    #[inline(always)]
    fn znx_normalize_middle_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        unsafe { znx_normalize_middle_step_carry_only_avx(base2k, lsh, x, carry) }
    }
}

impl ZnxNormalizeMiddleStepAssign for NTT4x30Avx {
    #[inline(always)]
    fn znx_normalize_middle_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        unsafe { znx_normalize_middle_step_assign_avx(base2k, lsh, x, carry) }
    }
}

impl ZnxExtractDigitAddMul for NTT4x30Avx {
    #[inline(always)]
    fn znx_extract_digit_addmul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
        unsafe { znx_extract_digit_addmul_avx(base2k, lsh, res, src) }
    }
}

impl ZnxNormalizeDigit for NTT4x30Avx {
    #[inline(always)]
    fn znx_normalize_digit(base2k: usize, res: &mut [i64], src: &mut [i64]) {
        unsafe { znx_normalize_digit_avx(base2k, res, src) }
    }
}
