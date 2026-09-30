//! Single ring element (`Z[X]/(X^n+1)`) arithmetic for [`FFT64Portable`](super::FFT64Portable).
//!
//! Implements the `Znx*` traits from `crate::kernels::znx`, covering
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

use super::FFT64Portable;
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

impl<R: Ring> ZnxAdd for FFT64Portable<R> {
    #[inline(always)]
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]) {
        znx_add_portable(res, a, b);
    }
}

impl<R: Ring> ZnxAddAssign for FFT64Portable<R> {
    #[inline(always)]
    fn znx_add_assign(res: &mut [i64], a: &[i64]) {
        znx_add_assign_portable(res, a);
    }
}

impl<R: Ring> ZnxSub for FFT64Portable<R> {
    #[inline(always)]
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]) {
        znx_sub_portable(res, a, b);
    }
}

impl<R: Ring> ZnxSubAssign for FFT64Portable<R> {
    #[inline(always)]
    fn znx_sub_assign(res: &mut [i64], a: &[i64]) {
        znx_sub_assign_portable(res, a);
    }
}

impl<R: Ring> ZnxSubNegateAssign for FFT64Portable<R> {
    #[inline(always)]
    fn znx_sub_negate_assign(res: &mut [i64], a: &[i64]) {
        znx_sub_negate_assign_portable(res, a);
    }
}

impl<R: Ring> ZnxMulAddPowerOfTwo for FFT64Portable<R> {
    #[inline(always)]
    fn znx_muladd_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        znx_mul_add_power_of_two_portable(k, res, a);
    }
}

impl<R: Ring> ZnxMulPowerOfTwo for FFT64Portable<R> {
    #[inline(always)]
    fn znx_mul_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        znx_mul_power_of_two_portable(k, res, a);
    }
}

impl<R: Ring> ZnxMulPowerOfTwoAssign for FFT64Portable<R> {
    #[inline(always)]
    fn znx_mul_power_of_two_assign(k: i64, res: &mut [i64]) {
        znx_mul_power_of_two_assign_portable(k, res);
    }
}

impl ZnxAutomorphism for FFT64Portable<Standard> {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        standard::znx_automorphism_portable(p, res, a);
    }

    #[inline(always)]
    fn znx_automorphism_i128(p: i64, res: &mut [i128], a: &[i128]) {
        standard::znx_automorphism_portable(p, res, a)
    }
}

impl ZnxAutomorphism for FFT64Portable<ConjugateInvariant> {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        conjugate_invariant::znx_automorphism_portable(p, res, a);
    }

    #[inline(always)]
    fn znx_automorphism_i128(p: i64, res: &mut [i128], a: &[i128]) {
        conjugate_invariant::znx_automorphism_portable(p, res, a);
    }
}

impl ZnxAutomorphismRotate for FFT64Portable<Standard> {
    #[inline(always)]
    fn znx_automorphism_rotate(p: i64, k: i64, res: &mut [i64], a: &[i64]) {
        znx_automorphism_rotate_portable(p, k, res, a);
    }
}

impl<R: Ring> ZnxCopy for FFT64Portable<R> {
    #[inline(always)]
    fn znx_copy(res: &mut [i64], a: &[i64]) {
        znx_copy_portable(res, a);
    }
}

impl<R: Ring> ZnxNegate for FFT64Portable<R> {
    #[inline(always)]
    fn znx_negate(res: &mut [i64], src: &[i64]) {
        znx_negate_portable(res, src);
    }
}

impl<R: Ring> ZnxNegateAssign for FFT64Portable<R> {
    #[inline(always)]
    fn znx_negate_assign(res: &mut [i64]) {
        znx_negate_assign_portable(res);
    }
}

impl<R: Ring> ZnxRotate for FFT64Portable<R> {
    #[inline(always)]
    fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]) {
        znx_rotate::<Self>(p, res, src);
    }
}

impl<R: Ring> ZnxZero for FFT64Portable<R> {
    #[inline(always)]
    fn znx_zero(res: &mut [i64]) {
        znx_zero_portable(res);
    }
}

impl<R: Ring> ZnxSwitchRing for FFT64Portable<R> {
    #[inline(always)]
    fn znx_switch_ring(res: &mut [i64], a: &[i64]) {
        znx_switch_ring_portable(res, a);
    }
}

impl<R: Ring> ZnxNormalizeFirstStep for FFT64Portable<R> {
    #[inline(always)]
    fn znx_normalize_first_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_first_step_portable::<OVERWRITE>(base2k, lsh, x, a, carry);
    }
}

impl<R: Ring> ZnxNormalizeMiddleStep for FFT64Portable<R> {
    #[inline(always)]
    fn znx_normalize_middle_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_middle_step_portable::<OVERWRITE>(base2k, lsh, x, a, carry);
    }
}

impl<R: Ring> ZnxNormalizeFinalStep for FFT64Portable<R> {
    #[inline(always)]
    fn znx_normalize_final_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_final_step_portable::<OVERWRITE>(base2k, lsh, x, a, carry);
    }
}

impl<R: Ring> ZnxNormalizeFinalStepAssign for FFT64Portable<R> {
    #[inline(always)]
    fn znx_normalize_final_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_final_step_assign_portable(base2k, lsh, x, carry);
    }
}

impl<R: Ring> ZnxNormalizeFirstStepCarryOnly for FFT64Portable<R> {
    #[inline(always)]
    fn znx_normalize_first_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        znx_normalize_first_step_carry_only_portable(base2k, lsh, x, carry);
    }
}

impl<R: Ring> ZnxNormalizeFirstStepAssign for FFT64Portable<R> {
    #[inline(always)]
    fn znx_normalize_first_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_first_step_assign_portable(base2k, lsh, x, carry);
    }
}

impl<R: Ring> ZnxNormalizeMiddleStepCarryOnly for FFT64Portable<R> {
    #[inline(always)]
    fn znx_normalize_middle_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        znx_normalize_middle_step_carry_only_portable(base2k, lsh, x, carry);
    }
}

impl<R: Ring> ZnxNormalizeMiddleStepAssign for FFT64Portable<R> {
    #[inline(always)]
    fn znx_normalize_middle_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_middle_step_assign_portable(base2k, lsh, x, carry);
    }
}

impl<R: Ring> ZnxExtractDigitAddMul for FFT64Portable<R> {
    #[inline(always)]
    fn znx_extract_digit_addmul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
        znx_extract_digit_addmul_portable(base2k, lsh, res, src);
    }
}

impl<R: Ring> crate::kernels::normalization::I64NormalizeOps for FFT64Portable<R> {}

impl<R: Ring> ZnxNormalizeDigit for FFT64Portable<R> {
    #[inline(always)]
    fn znx_normalize_digit(base2k: usize, res: &mut [i64], src: &mut [i64]) {
        znx_normalize_digit_portable(base2k, res, src);
    }
}

#[cfg(test)]
mod normalization_tests {
    #[test]
    fn test_normalization_kernels_bounded_inputs() {
        crate::test_suite::normalization::test_normalization_kernels::<super::FFT64Portable>();
        crate::test_suite::normalization::test_normalization_kernels::<super::super::super::NTT4x30Portable>();
    }
}
