//! Single ring element (`Z[X]/(X^n+1)`) arithmetic for [`NTT126Ifma`](super::NTT126Ifma).
//!
//! Implements the `Znx*` traits from `poulpy_cpu_ref::reference::znx`. All implementations
//! delegate to the AVX512-accelerated functions in `crate::znx_avx512` (same kernels used
//! by `FFT64Avx512`). These operate on plain `&[i64]` slices and are backend-independent.

use poulpy_cpu_ref::reference::znx::{
    ZnxAdd, ZnxAddAssign, ZnxAutomorphism, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulAddPowerOfTwo, ZnxMulPowerOfTwo,
    ZnxMulPowerOfTwoAssign, ZnxNegate, ZnxNegateAssign, ZnxNormalizeDigit, ZnxNormalizeFinalStep, ZnxNormalizeFinalStepAssign,
    ZnxNormalizeFinalStepSub, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepAssign, ZnxNormalizeFirstStepCarryOnly,
    ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepAssign, ZnxNormalizeMiddleStepCarryOnly, ZnxNormalizeMiddleStepSub, ZnxRotate,
    ZnxSub, ZnxSubAssign, ZnxSubNegateAssign, ZnxSwitchRing, ZnxZero, znx_copy_ref, znx_rotate, znx_zero_ref,
};

use crate::znx_avx512::{
    znx_add_assign_avx512, znx_add_avx512, znx_automorphism_avx512, znx_extract_digit_addmul_avx512,
    znx_mul_add_power_of_two_avx512, znx_mul_power_of_two_assign_avx512, znx_mul_power_of_two_avx512, znx_negate_assign_avx512,
    znx_negate_avx512, znx_normalize_digit_avx512, znx_normalize_final_step_assign_avx512, znx_normalize_final_step_avx512,
    znx_normalize_final_step_sub_avx512, znx_normalize_first_step_assign_avx512, znx_normalize_first_step_avx512,
    znx_normalize_first_step_carry_only_avx512, znx_normalize_middle_step_assign_avx512, znx_normalize_middle_step_avx512,
    znx_normalize_middle_step_carry_only_avx512, znx_normalize_middle_step_sub_avx512, znx_sub_assign_avx512, znx_sub_avx512,
    znx_sub_negate_assign_avx512, znx_switch_ring_avx512,
};

use super::NTT126Ifma;

impl ZnxAdd for NTT126Ifma {
    #[inline(always)]
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]) {
        unsafe { znx_add_avx512(res, a, b) }
    }
}

impl ZnxAddAssign for NTT126Ifma {
    #[inline(always)]
    fn znx_add_assign(res: &mut [i64], a: &[i64]) {
        unsafe { znx_add_assign_avx512(res, a) }
    }
}

impl ZnxSub for NTT126Ifma {
    #[inline(always)]
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]) {
        unsafe { znx_sub_avx512(res, a, b) }
    }
}

impl ZnxSubAssign for NTT126Ifma {
    #[inline(always)]
    fn znx_sub_assign(res: &mut [i64], a: &[i64]) {
        unsafe { znx_sub_assign_avx512(res, a) }
    }
}

impl ZnxSubNegateAssign for NTT126Ifma {
    #[inline(always)]
    fn znx_sub_negate_assign(res: &mut [i64], a: &[i64]) {
        unsafe { znx_sub_negate_assign_avx512(res, a) }
    }
}

impl ZnxMulAddPowerOfTwo for NTT126Ifma {
    #[inline(always)]
    fn znx_muladd_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        unsafe { znx_mul_add_power_of_two_avx512(k, res, a) }
    }
}

impl ZnxMulPowerOfTwo for NTT126Ifma {
    #[inline(always)]
    fn znx_mul_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        unsafe { znx_mul_power_of_two_avx512(k, res, a) }
    }
}

impl ZnxMulPowerOfTwoAssign for NTT126Ifma {
    #[inline(always)]
    fn znx_mul_power_of_two_assign(k: i64, res: &mut [i64]) {
        unsafe { znx_mul_power_of_two_assign_avx512(k, res) }
    }
}

impl ZnxAutomorphism for NTT126Ifma {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        unsafe { znx_automorphism_avx512(p, res, a) }
    }
}

impl ZnxCopy for NTT126Ifma {
    #[inline(always)]
    fn znx_copy(res: &mut [i64], a: &[i64]) {
        znx_copy_ref(res, a);
    }
}

impl ZnxNegate for NTT126Ifma {
    #[inline(always)]
    fn znx_negate(res: &mut [i64], src: &[i64]) {
        unsafe { znx_negate_avx512(res, src) }
    }
}

impl ZnxNegateAssign for NTT126Ifma {
    #[inline(always)]
    fn znx_negate_assign(res: &mut [i64]) {
        unsafe { znx_negate_assign_avx512(res) }
    }
}

impl ZnxRotate for NTT126Ifma {
    #[inline(always)]
    fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]) {
        znx_rotate::<Self>(p, res, src);
    }
}

impl ZnxZero for NTT126Ifma {
    #[inline(always)]
    fn znx_zero(res: &mut [i64]) {
        znx_zero_ref(res);
    }
}

impl ZnxSwitchRing for NTT126Ifma {
    #[inline(always)]
    fn znx_switch_ring(res: &mut [i64], a: &[i64]) {
        unsafe { znx_switch_ring_avx512(res, a) }
    }
}

impl ZnxNormalizeFinalStep for NTT126Ifma {
    #[inline(always)]
    fn znx_normalize_final_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        unsafe { znx_normalize_final_step_avx512::<OVERWRITE>(base2k, lsh, x, a, carry) }
    }
}

impl ZnxNormalizeFinalStepSub for NTT126Ifma {
    #[inline(always)]
    fn znx_normalize_final_step_sub(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        unsafe { znx_normalize_final_step_sub_avx512(base2k, lsh, x, a, carry) }
    }
}

impl ZnxNormalizeFinalStepAssign for NTT126Ifma {
    #[inline(always)]
    fn znx_normalize_final_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        unsafe { znx_normalize_final_step_assign_avx512(base2k, lsh, x, carry) }
    }
}

impl ZnxNormalizeFirstStep for NTT126Ifma {
    #[inline(always)]
    fn znx_normalize_first_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        unsafe { znx_normalize_first_step_avx512::<OVERWRITE>(base2k, lsh, x, a, carry) }
    }
}

impl ZnxNormalizeFirstStepCarryOnly for NTT126Ifma {
    #[inline(always)]
    fn znx_normalize_first_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        unsafe { znx_normalize_first_step_carry_only_avx512(base2k, lsh, x, carry) }
    }
}

impl ZnxNormalizeFirstStepAssign for NTT126Ifma {
    #[inline(always)]
    fn znx_normalize_first_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        unsafe { znx_normalize_first_step_assign_avx512(base2k, lsh, x, carry) }
    }
}

impl ZnxNormalizeMiddleStep for NTT126Ifma {
    #[inline(always)]
    fn znx_normalize_middle_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        unsafe { znx_normalize_middle_step_avx512::<OVERWRITE>(base2k, lsh, x, a, carry) }
    }
}

impl ZnxNormalizeMiddleStepSub for NTT126Ifma {
    #[inline(always)]
    fn znx_normalize_middle_step_sub(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        unsafe { znx_normalize_middle_step_sub_avx512(base2k, lsh, x, a, carry) }
    }
}

impl ZnxNormalizeMiddleStepCarryOnly for NTT126Ifma {
    #[inline(always)]
    fn znx_normalize_middle_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        unsafe { znx_normalize_middle_step_carry_only_avx512(base2k, lsh, x, carry) }
    }
}

impl ZnxNormalizeMiddleStepAssign for NTT126Ifma {
    #[inline(always)]
    fn znx_normalize_middle_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        unsafe { znx_normalize_middle_step_assign_avx512(base2k, lsh, x, carry) }
    }
}

impl ZnxExtractDigitAddMul for NTT126Ifma {
    #[inline(always)]
    fn znx_extract_digit_addmul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
        unsafe { znx_extract_digit_addmul_avx512(base2k, lsh, res, src) }
    }
}

impl ZnxNormalizeDigit for NTT126Ifma {
    #[inline(always)]
    fn znx_normalize_digit(base2k: usize, res: &mut [i64], src: &mut [i64]) {
        unsafe { znx_normalize_digit_avx512(base2k, res, src) }
    }
}
