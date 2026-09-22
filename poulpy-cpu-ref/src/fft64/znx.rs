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

use crate::FFT64RefBackend;
use crate::ring::CpuRing;

use crate::reference::znx::{
    ZnxAdd, ZnxAddAssign, ZnxAutomorphism, ZnxAutomorphismRotate, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulAddPowerOfTwo,
    ZnxMulPowerOfTwo, ZnxMulPowerOfTwoAssign, ZnxNegate, ZnxNegateAssign, ZnxNormalizeDigit, ZnxNormalizeFinalStep,
    ZnxNormalizeFinalStepAssign, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepAssign, ZnxNormalizeFirstStepCarryOnly,
    ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepAssign, ZnxNormalizeMiddleStepCarryOnly, ZnxRotate, ZnxSub, ZnxSubAssign,
    ZnxSubNegateAssign, ZnxSwitchRing, ZnxZero, znx_add_assign_ref, znx_add_ref, znx_automorphism_ref,
    znx_automorphism_rotate_ref, znx_copy_ref, znx_extract_digit_addmul_ref, znx_mul_add_power_of_two_ref,
    znx_mul_power_of_two_assign_ref, znx_mul_power_of_two_ref, znx_negate_assign_ref, znx_negate_ref, znx_normalize_digit_ref,
    znx_normalize_final_step_assign_ref, znx_normalize_final_step_ref, znx_normalize_first_step_assign_ref,
    znx_normalize_first_step_carry_only_ref, znx_normalize_first_step_ref, znx_normalize_middle_step_assign_ref,
    znx_normalize_middle_step_carry_only_ref, znx_normalize_middle_step_ref, znx_rotate, znx_sub_assign_ref,
    znx_sub_negate_assign_ref, znx_sub_ref, znx_switch_ring_ref, znx_zero_ref,
};

impl<R: CpuRing> ZnxAdd for FFT64RefBackend<R> {
    #[inline(always)]
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]) {
        znx_add_ref(res, a, b);
    }
}

impl<R: CpuRing> ZnxAddAssign for FFT64RefBackend<R> {
    #[inline(always)]
    fn znx_add_assign(res: &mut [i64], a: &[i64]) {
        znx_add_assign_ref(res, a);
    }
}

impl<R: CpuRing> ZnxSub for FFT64RefBackend<R> {
    #[inline(always)]
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]) {
        znx_sub_ref(res, a, b);
    }
}

impl<R: CpuRing> ZnxSubAssign for FFT64RefBackend<R> {
    #[inline(always)]
    fn znx_sub_assign(res: &mut [i64], a: &[i64]) {
        znx_sub_assign_ref(res, a);
    }
}

impl<R: CpuRing> ZnxSubNegateAssign for FFT64RefBackend<R> {
    #[inline(always)]
    fn znx_sub_negate_assign(res: &mut [i64], a: &[i64]) {
        znx_sub_negate_assign_ref(res, a);
    }
}

impl<R: CpuRing> ZnxMulAddPowerOfTwo for FFT64RefBackend<R> {
    #[inline(always)]
    fn znx_muladd_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        znx_mul_add_power_of_two_ref(k, res, a);
    }
}

impl<R: CpuRing> ZnxMulPowerOfTwo for FFT64RefBackend<R> {
    #[inline(always)]
    fn znx_mul_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        znx_mul_power_of_two_ref(k, res, a);
    }
}

impl<R: CpuRing> ZnxMulPowerOfTwoAssign for FFT64RefBackend<R> {
    #[inline(always)]
    fn znx_mul_power_of_two_assign(k: i64, res: &mut [i64]) {
        znx_mul_power_of_two_assign_ref(k, res);
    }
}

impl<R: CpuRing> ZnxAutomorphism for FFT64RefBackend<R> {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        znx_automorphism_ref(p, res, a);
    }
}

impl<R: CpuRing> ZnxAutomorphismRotate for FFT64RefBackend<R> {
    #[inline(always)]
    fn znx_automorphism_rotate(p: i64, k: i64, res: &mut [i64], a: &[i64]) {
        znx_automorphism_rotate_ref(p, k, res, a);
    }
}

impl<R: CpuRing> ZnxCopy for FFT64RefBackend<R> {
    #[inline(always)]
    fn znx_copy(res: &mut [i64], a: &[i64]) {
        znx_copy_ref(res, a);
    }
}

impl<R: CpuRing> ZnxNegate for FFT64RefBackend<R> {
    #[inline(always)]
    fn znx_negate(res: &mut [i64], src: &[i64]) {
        znx_negate_ref(res, src);
    }
}

impl<R: CpuRing> ZnxNegateAssign for FFT64RefBackend<R> {
    #[inline(always)]
    fn znx_negate_assign(res: &mut [i64]) {
        znx_negate_assign_ref(res);
    }
}

impl<R: CpuRing> ZnxRotate for FFT64RefBackend<R> {
    #[inline(always)]
    fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]) {
        znx_rotate::<Self>(p, res, src);
    }
}

impl<R: CpuRing> ZnxZero for FFT64RefBackend<R> {
    #[inline(always)]
    fn znx_zero(res: &mut [i64]) {
        znx_zero_ref(res);
    }
}

impl<R: CpuRing> ZnxSwitchRing for FFT64RefBackend<R> {
    #[inline(always)]
    fn znx_switch_ring(res: &mut [i64], a: &[i64]) {
        znx_switch_ring_ref(res, a);
    }
}

impl<R: CpuRing> ZnxNormalizeFirstStep for FFT64RefBackend<R> {
    #[inline(always)]
    fn znx_normalize_first_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_first_step_ref::<OVERWRITE>(base2k, lsh, x, a, carry);
    }
}

impl<R: CpuRing> ZnxNormalizeMiddleStep for FFT64RefBackend<R> {
    #[inline(always)]
    fn znx_normalize_middle_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_middle_step_ref::<OVERWRITE>(base2k, lsh, x, a, carry);
    }
}

impl<R: CpuRing> ZnxNormalizeFinalStep for FFT64RefBackend<R> {
    #[inline(always)]
    fn znx_normalize_final_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        znx_normalize_final_step_ref::<OVERWRITE>(base2k, lsh, x, a, carry);
    }
}

impl<R: CpuRing> ZnxNormalizeFinalStepAssign for FFT64RefBackend<R> {
    #[inline(always)]
    fn znx_normalize_final_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_final_step_assign_ref(base2k, lsh, x, carry);
    }
}

impl<R: CpuRing> ZnxNormalizeFirstStepCarryOnly for FFT64RefBackend<R> {
    #[inline(always)]
    fn znx_normalize_first_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        znx_normalize_first_step_carry_only_ref(base2k, lsh, x, carry);
    }
}

impl<R: CpuRing> ZnxNormalizeFirstStepAssign for FFT64RefBackend<R> {
    #[inline(always)]
    fn znx_normalize_first_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_first_step_assign_ref(base2k, lsh, x, carry);
    }
}

impl<R: CpuRing> ZnxNormalizeMiddleStepCarryOnly for FFT64RefBackend<R> {
    #[inline(always)]
    fn znx_normalize_middle_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        znx_normalize_middle_step_carry_only_ref(base2k, lsh, x, carry);
    }
}

impl<R: CpuRing> ZnxNormalizeMiddleStepAssign for FFT64RefBackend<R> {
    #[inline(always)]
    fn znx_normalize_middle_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        znx_normalize_middle_step_assign_ref(base2k, lsh, x, carry);
    }
}

impl<R: CpuRing> ZnxExtractDigitAddMul for FFT64RefBackend<R> {
    #[inline(always)]
    fn znx_extract_digit_addmul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
        znx_extract_digit_addmul_ref(base2k, lsh, res, src);
    }
}

impl<R: CpuRing> crate::reference::normalization::I64NormalizeOps for FFT64RefBackend<R> {}

impl<R: CpuRing> ZnxNormalizeDigit for FFT64RefBackend<R> {
    #[inline(always)]
    fn znx_normalize_digit(base2k: usize, res: &mut [i64], src: &mut [i64]) {
        znx_normalize_digit_ref(base2k, res, src);
    }
}

#[cfg(test)]
mod normalization_tests {
    #[test]
    fn test_normalization_kernels_bounded_inputs() {
        crate::test_suite::normalization::test_normalization_kernels::<crate::FFT64Ref>();
        crate::test_suite::normalization::test_normalization_kernels::<crate::NTT4x30Ref>();
    }
}
