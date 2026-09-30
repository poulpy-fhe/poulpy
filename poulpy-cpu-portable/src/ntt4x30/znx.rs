//! Single ring element (`Z[X]/(X^n+1)`) arithmetic for [`NTT4x30Portable`](super::NTT4x30Portable).
//!
//! Implements the `Znx*` traits from `crate::kernels::znx`. All implementations
//! delegate to the same `_ref` functions as `poulpy-cpu-portable` — these operate on plain
//! `&[i64]` slices, which are backend-independent.

use super::NTT4x30Portable;
use poulpy_hal::layouts::{ConjugateInvariant, Ring, Standard};

use crate::kernels::znx::{
    ZnxAdd, ZnxAddAssign, ZnxAutomorphism, ZnxAutomorphismRotate, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulAddPowerOfTwo,
    ZnxMulPowerOfTwo, ZnxMulPowerOfTwoAssign, ZnxNegate, ZnxNegateAssign, ZnxNormalizeDigit, ZnxNormalizeFinalStep,
    ZnxNormalizeFinalStepAssign, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepAssign, ZnxNormalizeFirstStepCarryOnly,
    ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepAssign, ZnxNormalizeMiddleStepCarryOnly, ZnxRotate, ZnxSub, ZnxSubAssign,
    ZnxSubNegateAssign, ZnxSwitchRing, ZnxZero, conjugate_invariant, standard, znx_add_assign_portable, znx_add_portable,
    znx_automorphism_rotate_portable, znx_copy_portable, znx_extract_digit_addmul_portable, znx_mul_add_power_of_two_portable,
    znx_mul_power_of_two_assign_portable, znx_mul_power_of_two_portable, znx_negate_assign_portable, znx_negate_portable,
    znx_normalize_digit_portable, znx_normalize_final_step_assign_portable, znx_normalize_final_step_portable,
    znx_normalize_first_step_assign_portable, znx_normalize_first_step_carry_only_portable, znx_normalize_first_step_portable,
    znx_normalize_middle_step_assign_portable, znx_normalize_middle_step_carry_only_portable, znx_normalize_middle_step_portable,
    znx_rotate, znx_sub_assign_portable, znx_sub_negate_assign_portable, znx_sub_portable, znx_switch_ring_portable,
    znx_zero_portable,
};

impl<R: Ring> ZnxAdd for NTT4x30Portable<R> {
    #[inline(always)]
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]) {
        znx_add_portable(res, a, b);
    }
}

impl<R: Ring> ZnxAddAssign for NTT4x30Portable<R> {
    #[inline(always)]
    fn znx_add_assign(res: &mut [i64], a: &[i64]) {
        znx_add_assign_portable(res, a);
    }
}

impl<R: Ring> ZnxSub for NTT4x30Portable<R> {
    #[inline(always)]
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]) {
        znx_sub_portable(res, a, b);
    }
}

impl<R: Ring> ZnxSubAssign for NTT4x30Portable<R> {
    #[inline(always)]
    fn znx_sub_assign(res: &mut [i64], a: &[i64]) {
        znx_sub_assign_portable(res, a);
    }
}

impl<R: Ring> ZnxSubNegateAssign for NTT4x30Portable<R> {
    #[inline(always)]
    fn znx_sub_negate_assign(res: &mut [i64], a: &[i64]) {
        znx_sub_negate_assign_portable(res, a);
    }
}

impl<R: Ring> ZnxMulAddPowerOfTwo for NTT4x30Portable<R> {
    #[inline(always)]
    fn znx_muladd_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        znx_mul_add_power_of_two_portable(k, res, a);
    }
}

impl<R: Ring> ZnxMulPowerOfTwo for NTT4x30Portable<R> {
    #[inline(always)]
    fn znx_mul_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        znx_mul_power_of_two_portable(k, res, a);
    }
}

impl<R: Ring> ZnxMulPowerOfTwoAssign for NTT4x30Portable<R> {
    #[inline(always)]
    fn znx_mul_power_of_two_assign(k: i64, res: &mut [i64]) {
        znx_mul_power_of_two_assign_portable(k, res);
    }
}

impl ZnxAutomorphism for NTT4x30Portable<Standard> {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        standard::znx_automorphism_portable(p, res, a);
    }

    #[inline(always)]
    fn znx_automorphism_i128(p: i64, res: &mut [i128], a: &[i128]) {
        standard::znx_automorphism_portable(p, res, a)
    }
}

impl ZnxAutomorphism for NTT4x30Portable<ConjugateInvariant> {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        conjugate_invariant::znx_automorphism_portable(p, res, a);
    }

    #[inline(always)]
    fn znx_automorphism_i128(p: i64, res: &mut [i128], a: &[i128]) {
        conjugate_invariant::znx_automorphism_portable(p, res, a);
    }
}

impl ZnxAutomorphismRotate for NTT4x30Portable<Standard> {
    #[inline(always)]
    fn znx_automorphism_rotate(p: i64, k: i64, res: &mut [i64], a: &[i64]) {
        znx_automorphism_rotate_portable(p, k, res, a);
    }
}

impl<R: Ring> ZnxCopy for NTT4x30Portable<R> {
    #[inline(always)]
    fn znx_copy(res: &mut [i64], a: &[i64]) {
        znx_copy_portable(res, a);
    }
}

impl<R: Ring> ZnxNegate for NTT4x30Portable<R> {
    #[inline(always)]
    fn znx_negate(res: &mut [i64], src: &[i64]) {
        znx_negate_portable(res, src);
    }
}

impl<R: Ring> ZnxNegateAssign for NTT4x30Portable<R> {
    #[inline(always)]
    fn znx_negate_assign(res: &mut [i64]) {
        znx_negate_assign_portable(res);
    }
}

impl<R: Ring> ZnxRotate for NTT4x30Portable<R> {
    #[inline(always)]
    fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]) {
        znx_rotate::<Self>(p, res, src);
    }
}

impl<R: Ring> ZnxZero for NTT4x30Portable<R> {
    #[inline(always)]
    fn znx_zero(res: &mut [i64]) {
        znx_zero_portable(res);
    }
}

impl<R: Ring> ZnxSwitchRing for NTT4x30Portable<R> {
    #[inline(always)]
    fn znx_switch_ring(res: &mut [i64], a: &[i64]) {
        znx_switch_ring_portable(res, a);
    }
}

impl<R: Ring> ZnxNormalizeFirstStep for NTT4x30Portable<R> {
    #[inline(always)]
    fn znx_normalize_first_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_first_step_portable::<OVERWRITE>(base2k, lsh, x, a, carry);
    }
}

impl<R: Ring> ZnxNormalizeMiddleStep for NTT4x30Portable<R> {
    #[inline(always)]
    fn znx_normalize_middle_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_middle_step_portable::<OVERWRITE>(base2k, lsh, x, a, carry);
    }
}

impl<R: Ring> ZnxNormalizeFinalStep for NTT4x30Portable<R> {
    #[inline(always)]
    fn znx_normalize_final_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_final_step_portable::<OVERWRITE>(base2k, lsh, x, a, carry);
    }
}

impl<R: Ring> ZnxNormalizeFinalStepAssign for NTT4x30Portable<R> {
    #[inline(always)]
    fn znx_normalize_final_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_final_step_assign_portable(base2k, lsh, x, carry);
    }
}

impl<R: Ring> ZnxNormalizeFirstStepCarryOnly for NTT4x30Portable<R> {
    #[inline(always)]
    fn znx_normalize_first_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        znx_normalize_first_step_carry_only_portable(base2k, lsh, x, carry);
    }
}

impl<R: Ring> ZnxNormalizeFirstStepAssign for NTT4x30Portable<R> {
    #[inline(always)]
    fn znx_normalize_first_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_first_step_assign_portable(base2k, lsh, x, carry);
    }
}

impl<R: Ring> ZnxNormalizeMiddleStepCarryOnly for NTT4x30Portable<R> {
    #[inline(always)]
    fn znx_normalize_middle_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        znx_normalize_middle_step_carry_only_portable(base2k, lsh, x, carry);
    }
}

impl<R: Ring> ZnxNormalizeMiddleStepAssign for NTT4x30Portable<R> {
    #[inline(always)]
    fn znx_normalize_middle_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_middle_step_assign_portable(base2k, lsh, x, carry);
    }
}

impl<R: Ring> ZnxExtractDigitAddMul for NTT4x30Portable<R> {
    #[inline(always)]
    fn znx_extract_digit_addmul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
        znx_extract_digit_addmul_portable(base2k, lsh, res, src);
    }
}

impl<R: Ring> crate::kernels::normalization::I64NormalizeOps for NTT4x30Portable<R> {}

impl<R: Ring> ZnxNormalizeDigit for NTT4x30Portable<R> {
    #[inline(always)]
    fn znx_normalize_digit(base2k: usize, res: &mut [i64], src: &mut [i64]) {
        znx_normalize_digit_portable(base2k, res, src);
    }
}
