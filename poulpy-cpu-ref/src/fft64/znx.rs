//! Single ring element (`Z[X]/(X^n+1)`) arithmetic for [`FFT64Ref`](crate::FFT64Ref).
//!
//! Implements the `Znx*` traits from `crate::reference::znx`, covering
//! coefficient-wise addition, subtraction, negation, power-of-two multiplication,
//! Galois automorphisms (`X -> X^k`), rotation, ring switching, and multi-step
//! normalization (carry propagation across a base-2^k decomposition).
//!
//! These traits are **not** OEP traits (they are not `unsafe trait`) because the
//! `Znx` operations work on plain `&[i64]` slices with a single canonical memory
//! layout shared across all backends.
//!
//! Every implementation delegates directly to the corresponding `_ref` function
//! and is marked `#[inline(always)]` to eliminate call overhead.

use crate::reference::znx::{
    ZnxAdd, ZnxAddAssign, ZnxAutomorphism, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulAddPowerOfTwo, ZnxMulPowerOfTwo,
    ZnxMulPowerOfTwoAssign, ZnxNegate, ZnxNegateAssign, ZnxNormalizeDigit, ZnxNormalizeFinalStep, ZnxNormalizeFinalStepAssign,
    ZnxNormalizeFinalStepSub, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepAssign, ZnxNormalizeFirstStepCarryOnly,
    ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepAssign, ZnxNormalizeMiddleStepCarryOnly, ZnxNormalizeMiddleStepSub, ZnxRotate,
    ZnxSub, ZnxSubAssign, ZnxSubNegateAssign, ZnxSwitchRing, ZnxZero, znx_add_assign_ref, znx_add_ref, znx_automorphism_ref,
    znx_copy_ref, znx_extract_digit_addmul_ref, znx_mul_add_power_of_two_ref, znx_mul_power_of_two_assign_ref,
    znx_mul_power_of_two_ref, znx_negate_assign_ref, znx_negate_ref, znx_normalize_digit_ref,
    znx_normalize_final_step_assign_ref, znx_normalize_final_step_ref, znx_normalize_final_step_sub_ref,
    znx_normalize_first_step_assign_ref, znx_normalize_first_step_carry_only_ref, znx_normalize_first_step_ref,
    znx_normalize_middle_step_assign_ref, znx_normalize_middle_step_carry_only_ref, znx_normalize_middle_step_ref,
    znx_normalize_middle_step_sub_ref, znx_rotate, znx_sub_assign_ref, znx_sub_negate_assign_ref, znx_sub_ref,
    znx_switch_ring_ref, znx_zero_ref,
};

use super::FFT64Ref;

impl ZnxAdd for FFT64Ref {
    #[inline(always)]
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]) {
        znx_add_ref(res, a, b);
    }
}

impl ZnxAddAssign for FFT64Ref {
    #[inline(always)]
    fn znx_add_assign(res: &mut [i64], a: &[i64]) {
        znx_add_assign_ref(res, a);
    }
}

impl ZnxSub for FFT64Ref {
    #[inline(always)]
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]) {
        znx_sub_ref(res, a, b);
    }
}

impl ZnxSubAssign for FFT64Ref {
    #[inline(always)]
    fn znx_sub_assign(res: &mut [i64], a: &[i64]) {
        znx_sub_assign_ref(res, a);
    }
}

impl ZnxSubNegateAssign for FFT64Ref {
    #[inline(always)]
    fn znx_sub_negate_assign(res: &mut [i64], a: &[i64]) {
        znx_sub_negate_assign_ref(res, a);
    }
}

impl ZnxMulAddPowerOfTwo for FFT64Ref {
    #[inline(always)]
    fn znx_muladd_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        znx_mul_add_power_of_two_ref(k, res, a);
    }
}

impl ZnxMulPowerOfTwo for FFT64Ref {
    #[inline(always)]
    fn znx_mul_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        znx_mul_power_of_two_ref(k, res, a);
    }
}

impl ZnxMulPowerOfTwoAssign for FFT64Ref {
    #[inline(always)]
    fn znx_mul_power_of_two_assign(k: i64, res: &mut [i64]) {
        znx_mul_power_of_two_assign_ref(k, res);
    }
}

impl ZnxAutomorphism for FFT64Ref {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        znx_automorphism_ref(p, res, a);
    }
}

impl ZnxCopy for FFT64Ref {
    #[inline(always)]
    fn znx_copy(res: &mut [i64], a: &[i64]) {
        znx_copy_ref(res, a);
    }
}

impl ZnxNegate for FFT64Ref {
    #[inline(always)]
    fn znx_negate(res: &mut [i64], src: &[i64]) {
        znx_negate_ref(res, src);
    }
}

impl ZnxNegateAssign for FFT64Ref {
    #[inline(always)]
    fn znx_negate_assign(res: &mut [i64]) {
        znx_negate_assign_ref(res);
    }
}

impl ZnxRotate for FFT64Ref {
    #[inline(always)]
    fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]) {
        znx_rotate::<Self>(p, res, src);
    }
}

impl ZnxZero for FFT64Ref {
    #[inline(always)]
    fn znx_zero(res: &mut [i64]) {
        znx_zero_ref(res);
    }
}

impl ZnxSwitchRing for FFT64Ref {
    #[inline(always)]
    fn znx_switch_ring(res: &mut [i64], a: &[i64]) {
        znx_switch_ring_ref(res, a);
    }
}

impl ZnxNormalizeFirstStep for FFT64Ref {
    #[inline(always)]
    fn znx_normalize_first_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_first_step_ref::<OVERWRITE>(base2k, lsh, x, a, carry);
    }
}

impl ZnxNormalizeMiddleStep for FFT64Ref {
    #[inline(always)]
    fn znx_normalize_middle_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_middle_step_ref::<OVERWRITE>(base2k, lsh, x, a, carry);
    }
}

impl ZnxNormalizeFinalStep for FFT64Ref {
    #[inline(always)]
    fn znx_normalize_final_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_final_step_ref::<OVERWRITE>(base2k, lsh, x, a, carry);
    }
}

impl ZnxNormalizeMiddleStepSub for FFT64Ref {
    #[inline(always)]
    fn znx_normalize_middle_step_sub(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_middle_step_sub_ref(base2k, lsh, x, a, carry);
    }
}

impl ZnxNormalizeFinalStepSub for FFT64Ref {
    #[inline(always)]
    fn znx_normalize_final_step_sub(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_final_step_sub_ref(base2k, lsh, x, a, carry);
    }
}

impl ZnxNormalizeFinalStepAssign for FFT64Ref {
    #[inline(always)]
    fn znx_normalize_final_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_final_step_assign_ref(base2k, lsh, x, carry);
    }
}

impl ZnxNormalizeFirstStepCarryOnly for FFT64Ref {
    #[inline(always)]
    fn znx_normalize_first_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        znx_normalize_first_step_carry_only_ref(base2k, lsh, x, carry);
    }
}

impl ZnxNormalizeFirstStepAssign for FFT64Ref {
    #[inline(always)]
    fn znx_normalize_first_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_first_step_assign_ref(base2k, lsh, x, carry);
    }
}

impl ZnxNormalizeMiddleStepCarryOnly for FFT64Ref {
    #[inline(always)]
    fn znx_normalize_middle_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        znx_normalize_middle_step_carry_only_ref(base2k, lsh, x, carry);
    }
}

impl ZnxNormalizeMiddleStepAssign for FFT64Ref {
    #[inline(always)]
    fn znx_normalize_middle_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_middle_step_assign_ref(base2k, lsh, x, carry);
    }
}

impl ZnxExtractDigitAddMul for FFT64Ref {
    #[inline(always)]
    fn znx_extract_digit_addmul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
        znx_extract_digit_addmul_ref(base2k, lsh, res, src);
    }
}

impl ZnxNormalizeDigit for FFT64Ref {
    #[inline(always)]
    fn znx_normalize_digit(base2k: usize, res: &mut [i64], src: &mut [i64]) {
        znx_normalize_digit_ref(base2k, res, src);
    }
}
