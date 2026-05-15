//! Single ring element (`Z[X]/(X^n+1)`) arithmetic for [`NTT120Neon`](super::NTT120Neon).
//!
//! Phase 3: add/sub/negate (and their assign variants) call the NEON kernels
//! from [`crate::neon::znx`]. Copy/zero stay on the reference functions and
//! the normalization helpers remain on `_ref`.

use poulpy_cpu_ref::reference::znx::{
    ZnxAdd, ZnxAddAssign, ZnxAutomorphism, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulAddPowerOfTwo, ZnxMulPowerOfTwo,
    ZnxMulPowerOfTwoAssign, ZnxNegate, ZnxNegateAssign, ZnxNormalizeDigit, ZnxNormalizeFinalStep, ZnxNormalizeFinalStepAssign,
    ZnxNormalizeFinalStepSub, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepAssign, ZnxNormalizeFirstStepCarryOnly,
    ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepAssign, ZnxNormalizeMiddleStepCarryOnly, ZnxNormalizeMiddleStepSub, ZnxRotate,
    ZnxSub, ZnxSubAssign, ZnxSubNegateAssign, ZnxSwitchRing, ZnxZero, znx_automorphism_ref, znx_copy_ref,
    znx_extract_digit_addmul_ref, znx_mul_add_power_of_two_ref, znx_mul_power_of_two_assign_ref, znx_mul_power_of_two_ref,
    znx_normalize_digit_ref, znx_normalize_final_step_assign_ref, znx_normalize_final_step_ref, znx_normalize_final_step_sub_ref,
    znx_normalize_first_step_assign_ref, znx_normalize_first_step_carry_only_ref, znx_normalize_first_step_ref,
    znx_normalize_middle_step_assign_ref, znx_normalize_middle_step_carry_only_ref, znx_normalize_middle_step_ref,
    znx_normalize_middle_step_sub_ref, znx_rotate, znx_switch_ring_ref, znx_zero_ref,
};

use super::NTT120Neon;

#[cfg(target_arch = "aarch64")]
use crate::neon::znx::{
    znx_add_assign_neon as kn_add_assign, znx_add_neon as kn_add, znx_negate_assign_neon as kn_negate_assign,
    znx_negate_neon as kn_negate, znx_sub_assign_neon as kn_sub_assign, znx_sub_negate_assign_neon as kn_sub_negate_assign,
    znx_sub_neon as kn_sub,
};
#[cfg(not(target_arch = "aarch64"))]
use poulpy_cpu_ref::reference::znx::{
    znx_add_assign_ref as kn_add_assign, znx_add_ref as kn_add, znx_negate_assign_ref as kn_negate_assign,
    znx_negate_ref as kn_negate, znx_sub_assign_ref as kn_sub_assign, znx_sub_negate_assign_ref as kn_sub_negate_assign,
    znx_sub_ref as kn_sub,
};

impl ZnxAdd for NTT120Neon {
    #[inline(always)]
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]) {
        kn_add(res, a, b);
    }
}

impl ZnxAddAssign for NTT120Neon {
    #[inline(always)]
    fn znx_add_assign(res: &mut [i64], a: &[i64]) {
        kn_add_assign(res, a);
    }
}

impl ZnxSub for NTT120Neon {
    #[inline(always)]
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]) {
        kn_sub(res, a, b);
    }
}

impl ZnxSubAssign for NTT120Neon {
    #[inline(always)]
    fn znx_sub_assign(res: &mut [i64], a: &[i64]) {
        kn_sub_assign(res, a);
    }
}

impl ZnxSubNegateAssign for NTT120Neon {
    #[inline(always)]
    fn znx_sub_negate_assign(res: &mut [i64], a: &[i64]) {
        kn_sub_negate_assign(res, a);
    }
}

impl ZnxMulAddPowerOfTwo for NTT120Neon {
    #[inline(always)]
    fn znx_muladd_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        znx_mul_add_power_of_two_ref(k, res, a);
    }
}

impl ZnxMulPowerOfTwo for NTT120Neon {
    #[inline(always)]
    fn znx_mul_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        znx_mul_power_of_two_ref(k, res, a);
    }
}

impl ZnxMulPowerOfTwoAssign for NTT120Neon {
    #[inline(always)]
    fn znx_mul_power_of_two_assign(k: i64, res: &mut [i64]) {
        znx_mul_power_of_two_assign_ref(k, res);
    }
}

impl ZnxAutomorphism for NTT120Neon {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        znx_automorphism_ref(p, res, a);
    }
}

impl ZnxCopy for NTT120Neon {
    #[inline(always)]
    fn znx_copy(res: &mut [i64], a: &[i64]) {
        znx_copy_ref(res, a);
    }
}

impl ZnxNegate for NTT120Neon {
    #[inline(always)]
    fn znx_negate(res: &mut [i64], src: &[i64]) {
        kn_negate(res, src);
    }
}

impl ZnxNegateAssign for NTT120Neon {
    #[inline(always)]
    fn znx_negate_assign(res: &mut [i64]) {
        kn_negate_assign(res);
    }
}

impl ZnxRotate for NTT120Neon {
    #[inline(always)]
    fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]) {
        znx_rotate::<Self>(p, res, src);
    }
}

impl ZnxZero for NTT120Neon {
    #[inline(always)]
    fn znx_zero(res: &mut [i64]) {
        znx_zero_ref(res);
    }
}

impl ZnxSwitchRing for NTT120Neon {
    #[inline(always)]
    fn znx_switch_ring(res: &mut [i64], a: &[i64]) {
        znx_switch_ring_ref(res, a);
    }
}

impl ZnxNormalizeFirstStep for NTT120Neon {
    #[inline(always)]
    fn znx_normalize_first_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_first_step_ref::<OVERWRITE>(base2k, lsh, x, a, carry);
    }
}

impl ZnxNormalizeMiddleStep for NTT120Neon {
    #[inline(always)]
    fn znx_normalize_middle_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_middle_step_ref::<OVERWRITE>(base2k, lsh, x, a, carry);
    }
}

impl ZnxNormalizeFinalStep for NTT120Neon {
    #[inline(always)]
    fn znx_normalize_final_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_final_step_ref::<OVERWRITE>(base2k, lsh, x, a, carry);
    }
}

impl ZnxNormalizeMiddleStepSub for NTT120Neon {
    #[inline(always)]
    fn znx_normalize_middle_step_sub(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_middle_step_sub_ref(base2k, lsh, x, a, carry);
    }
}

impl ZnxNormalizeFinalStepSub for NTT120Neon {
    #[inline(always)]
    fn znx_normalize_final_step_sub(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_final_step_sub_ref(base2k, lsh, x, a, carry);
    }
}

impl ZnxNormalizeFinalStepAssign for NTT120Neon {
    #[inline(always)]
    fn znx_normalize_final_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_final_step_assign_ref(base2k, lsh, x, carry);
    }
}

impl ZnxNormalizeFirstStepCarryOnly for NTT120Neon {
    #[inline(always)]
    fn znx_normalize_first_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        znx_normalize_first_step_carry_only_ref(base2k, lsh, x, carry);
    }
}

impl ZnxNormalizeFirstStepAssign for NTT120Neon {
    #[inline(always)]
    fn znx_normalize_first_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_first_step_assign_ref(base2k, lsh, x, carry);
    }
}

impl ZnxNormalizeMiddleStepCarryOnly for NTT120Neon {
    #[inline(always)]
    fn znx_normalize_middle_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        znx_normalize_middle_step_carry_only_ref(base2k, lsh, x, carry);
    }
}

impl ZnxNormalizeMiddleStepAssign for NTT120Neon {
    #[inline(always)]
    fn znx_normalize_middle_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_middle_step_assign_ref(base2k, lsh, x, carry);
    }
}

impl ZnxExtractDigitAddMul for NTT120Neon {
    #[inline(always)]
    fn znx_extract_digit_addmul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
        znx_extract_digit_addmul_ref(base2k, lsh, res, src);
    }
}

impl ZnxNormalizeDigit for NTT120Neon {
    #[inline(always)]
    fn znx_normalize_digit(base2k: usize, res: &mut [i64], src: &mut [i64]) {
        znx_normalize_digit_ref(base2k, res, src);
    }
}
