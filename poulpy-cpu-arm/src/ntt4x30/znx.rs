//! `Znx*` trait impls for [`NTT4x30Neon`](super::NTT4x30Neon).

use poulpy_cpu_ref::reference::znx::{
    ZnxAdd, ZnxAddAssign, ZnxAutomorphism, ZnxAutomorphismRotate, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulAddPowerOfTwo,
    ZnxMulPowerOfTwo, ZnxMulPowerOfTwoAssign, ZnxNegate, ZnxNegateAssign, ZnxNormalizeDigit, ZnxNormalizeFinalStep,
    ZnxNormalizeFinalStepAssign, ZnxNormalizeFinalStepSub, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepAssign,
    ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepAssign, ZnxNormalizeMiddleStepCarryOnly,
    ZnxNormalizeMiddleStepSub, ZnxRotate, ZnxSub, ZnxSubAssign, ZnxSubNegateAssign, ZnxSwitchRing, ZnxZero, znx_copy_ref,
    znx_rotate, znx_zero_ref,
};

use super::NTT4x30Neon;

#[cfg(target_arch = "aarch64")]
use crate::neon::{
    znx::{
        znx_add_assign_neon as kn_add_assign, znx_add_neon as kn_add, znx_automorphism_neon as kn_automorphism,
        znx_automorphism_rotate_neon as kn_automorphism_rotate, znx_mul_add_power_of_two_neon as kn_mul_add_p2,
        znx_mul_power_of_two_assign_neon as kn_mul_p2_assign, znx_mul_power_of_two_neon as kn_mul_p2,
        znx_negate_assign_neon as kn_negate_assign, znx_negate_neon as kn_negate, znx_sub_assign_neon as kn_sub_assign,
        znx_sub_negate_assign_neon as kn_sub_negate_assign, znx_sub_neon as kn_sub, znx_switch_ring_neon as kn_switch_ring,
    },
    znx_normalize::{
        znx_extract_digit_addmul_neon as kn_extract_digit_addmul, znx_normalize_digit_neon as kn_normalize_digit,
        znx_normalize_final_step_assign_neon as kn_normalize_final_step_assign,
        znx_normalize_final_step_neon as kn_normalize_final_step,
        znx_normalize_final_step_sub_neon as kn_normalize_final_step_sub,
        znx_normalize_first_step_assign_neon as kn_normalize_first_step_assign,
        znx_normalize_first_step_carry_only_neon as kn_normalize_first_step_carry_only,
        znx_normalize_first_step_neon as kn_normalize_first_step,
        znx_normalize_middle_step_assign_neon as kn_normalize_middle_step_assign,
        znx_normalize_middle_step_carry_only_neon as kn_normalize_middle_step_carry_only,
        znx_normalize_middle_step_neon as kn_normalize_middle_step,
        znx_normalize_middle_step_sub_neon as kn_normalize_middle_step_sub,
    },
};
#[cfg(not(target_arch = "aarch64"))]
use poulpy_cpu_ref::reference::znx::{
    znx_add_assign_ref as kn_add_assign, znx_add_ref as kn_add, znx_automorphism_ref as kn_automorphism,
    znx_automorphism_rotate_ref as kn_automorphism_rotate, znx_extract_digit_addmul_ref as kn_extract_digit_addmul,
    znx_mul_add_power_of_two_ref as kn_mul_add_p2, znx_mul_power_of_two_assign_ref as kn_mul_p2_assign,
    znx_mul_power_of_two_ref as kn_mul_p2, znx_negate_assign_ref as kn_negate_assign, znx_negate_ref as kn_negate,
    znx_normalize_digit_ref as kn_normalize_digit, znx_normalize_final_step_assign_ref as kn_normalize_final_step_assign,
    znx_normalize_final_step_ref as kn_normalize_final_step, znx_normalize_final_step_sub_ref as kn_normalize_final_step_sub,
    znx_normalize_first_step_assign_ref as kn_normalize_first_step_assign,
    znx_normalize_first_step_carry_only_ref as kn_normalize_first_step_carry_only,
    znx_normalize_first_step_ref as kn_normalize_first_step,
    znx_normalize_middle_step_assign_ref as kn_normalize_middle_step_assign,
    znx_normalize_middle_step_carry_only_ref as kn_normalize_middle_step_carry_only,
    znx_normalize_middle_step_ref as kn_normalize_middle_step, znx_normalize_middle_step_sub_ref as kn_normalize_middle_step_sub,
    znx_sub_assign_ref as kn_sub_assign, znx_sub_negate_assign_ref as kn_sub_negate_assign, znx_sub_ref as kn_sub,
    znx_switch_ring_ref as kn_switch_ring,
};

impl ZnxAdd for NTT4x30Neon {
    #[inline(always)]
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]) {
        kn_add(res, a, b);
    }
}

impl ZnxAddAssign for NTT4x30Neon {
    #[inline(always)]
    fn znx_add_assign(res: &mut [i64], a: &[i64]) {
        kn_add_assign(res, a);
    }
}

impl ZnxSub for NTT4x30Neon {
    #[inline(always)]
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]) {
        kn_sub(res, a, b);
    }
}

impl ZnxSubAssign for NTT4x30Neon {
    #[inline(always)]
    fn znx_sub_assign(res: &mut [i64], a: &[i64]) {
        kn_sub_assign(res, a);
    }
}

impl ZnxSubNegateAssign for NTT4x30Neon {
    #[inline(always)]
    fn znx_sub_negate_assign(res: &mut [i64], a: &[i64]) {
        kn_sub_negate_assign(res, a);
    }
}

impl ZnxMulAddPowerOfTwo for NTT4x30Neon {
    #[inline(always)]
    fn znx_muladd_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        kn_mul_add_p2(k, res, a);
    }
}

impl ZnxMulPowerOfTwo for NTT4x30Neon {
    #[inline(always)]
    fn znx_mul_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        kn_mul_p2(k, res, a);
    }
}

impl ZnxMulPowerOfTwoAssign for NTT4x30Neon {
    #[inline(always)]
    fn znx_mul_power_of_two_assign(k: i64, res: &mut [i64]) {
        kn_mul_p2_assign(k, res);
    }
}

impl ZnxAutomorphism for NTT4x30Neon {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        kn_automorphism(p, res, a);
    }
}

impl ZnxAutomorphismRotate for NTT4x30Neon {
    #[inline(always)]
    fn znx_automorphism_rotate(p: i64, k: i64, res: &mut [i64], a: &[i64]) {
        kn_automorphism_rotate(p, k, res, a);
    }
}

impl ZnxCopy for NTT4x30Neon {
    #[inline(always)]
    fn znx_copy(res: &mut [i64], a: &[i64]) {
        znx_copy_ref(res, a);
    }
}

impl ZnxNegate for NTT4x30Neon {
    #[inline(always)]
    fn znx_negate(res: &mut [i64], src: &[i64]) {
        kn_negate(res, src);
    }
}

impl ZnxNegateAssign for NTT4x30Neon {
    #[inline(always)]
    fn znx_negate_assign(res: &mut [i64]) {
        kn_negate_assign(res);
    }
}

impl ZnxRotate for NTT4x30Neon {
    #[inline(always)]
    fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]) {
        znx_rotate::<Self>(p, res, src);
    }
}

impl ZnxZero for NTT4x30Neon {
    #[inline(always)]
    fn znx_zero(res: &mut [i64]) {
        znx_zero_ref(res);
    }
}

impl ZnxSwitchRing for NTT4x30Neon {
    #[inline(always)]
    fn znx_switch_ring(res: &mut [i64], a: &[i64]) {
        kn_switch_ring(res, a);
    }
}

impl ZnxNormalizeFirstStep for NTT4x30Neon {
    #[inline(always)]
    fn znx_normalize_first_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        kn_normalize_first_step::<OVERWRITE>(base2k, lsh, x, a, carry);
    }
}

impl ZnxNormalizeMiddleStep for NTT4x30Neon {
    #[inline(always)]
    fn znx_normalize_middle_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        kn_normalize_middle_step::<OVERWRITE>(base2k, lsh, x, a, carry);
    }
}

impl ZnxNormalizeFinalStep for NTT4x30Neon {
    #[inline(always)]
    fn znx_normalize_final_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        kn_normalize_final_step::<OVERWRITE>(base2k, lsh, x, a, carry);
    }
}

impl ZnxNormalizeMiddleStepSub for NTT4x30Neon {
    #[inline(always)]
    fn znx_normalize_middle_step_sub(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        kn_normalize_middle_step_sub(base2k, lsh, x, a, carry);
    }
}

impl ZnxNormalizeFinalStepSub for NTT4x30Neon {
    #[inline(always)]
    fn znx_normalize_final_step_sub(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        kn_normalize_final_step_sub(base2k, lsh, x, a, carry);
    }
}

impl ZnxNormalizeFinalStepAssign for NTT4x30Neon {
    #[inline(always)]
    fn znx_normalize_final_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        kn_normalize_final_step_assign(base2k, lsh, x, carry);
    }
}

impl ZnxNormalizeFirstStepCarryOnly for NTT4x30Neon {
    #[inline(always)]
    fn znx_normalize_first_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        kn_normalize_first_step_carry_only(base2k, lsh, x, carry);
    }
}

impl ZnxNormalizeFirstStepAssign for NTT4x30Neon {
    #[inline(always)]
    fn znx_normalize_first_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        kn_normalize_first_step_assign(base2k, lsh, x, carry);
    }
}

impl ZnxNormalizeMiddleStepCarryOnly for NTT4x30Neon {
    #[inline(always)]
    fn znx_normalize_middle_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        kn_normalize_middle_step_carry_only(base2k, lsh, x, carry);
    }
}

impl ZnxNormalizeMiddleStepAssign for NTT4x30Neon {
    #[inline(always)]
    fn znx_normalize_middle_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        kn_normalize_middle_step_assign(base2k, lsh, x, carry);
    }
}

impl ZnxExtractDigitAddMul for NTT4x30Neon {
    #[inline(always)]
    fn znx_extract_digit_addmul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
        kn_extract_digit_addmul(base2k, lsh, res, src);
    }
}

impl ZnxNormalizeDigit for NTT4x30Neon {
    #[inline(always)]
    fn znx_normalize_digit(base2k: usize, res: &mut [i64], src: &mut [i64]) {
        kn_normalize_digit(base2k, res, src);
    }
}
