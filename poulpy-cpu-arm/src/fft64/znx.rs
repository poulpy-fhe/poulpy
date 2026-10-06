//! `Znx*` trait impls for [`FFT64Neon`](super::FFT64Neon).

use poulpy_cpu_portable::kernels::znx::{
    ZnxAdd, ZnxAddAssign, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulAddPowerOfTwo, ZnxMulPowerOfTwo, ZnxMulPowerOfTwoAssign,
    ZnxNegate, ZnxNegateAssign, ZnxNormalizeDigit, ZnxNormalizeFinalStep, ZnxNormalizeFinalStepAssign, ZnxNormalizeFirstStep,
    ZnxNormalizeFirstStepAssign, ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepAssign,
    ZnxNormalizeMiddleStepCarryOnly, ZnxRotate, ZnxSub, ZnxSubAssign, ZnxSubNegateAssign, ZnxSwitchRing, ZnxZero,
    znx_copy_portable, znx_rotate_portable, znx_zero_portable,
};

use super::FFT64Neon;
use poulpy_hal::layouts::Ring;

#[cfg(target_arch = "aarch64")]
use crate::neon::{
    znx::{
        znx_add_assign_neon as kn_add_assign, znx_add_neon as kn_add, znx_mul_add_power_of_two_neon as kn_mul_add_p2,
        znx_mul_power_of_two_assign_neon as kn_mul_p2_assign, znx_mul_power_of_two_neon as kn_mul_p2,
        znx_negate_assign_neon as kn_negate_assign, znx_negate_neon as kn_negate, znx_sub_assign_neon as kn_sub_assign,
        znx_sub_negate_assign_neon as kn_sub_negate_assign, znx_sub_neon as kn_sub, znx_switch_ring_neon as kn_switch_ring,
    },
    znx_normalize::{
        znx_extract_digit_addmul_neon as kn_extract_digit_addmul, znx_normalize_digit_neon as kn_normalize_digit,
        znx_normalize_final_step_assign_neon as kn_normalize_final_step_assign,
        znx_normalize_final_step_neon as kn_normalize_final_step,
        znx_normalize_first_step_assign_neon as kn_normalize_first_step_assign,
        znx_normalize_first_step_carry_only_neon as kn_normalize_first_step_carry_only,
        znx_normalize_first_step_neon as kn_normalize_first_step,
        znx_normalize_middle_step_assign_neon as kn_normalize_middle_step_assign,
        znx_normalize_middle_step_carry_only_neon as kn_normalize_middle_step_carry_only,
        znx_normalize_middle_step_neon as kn_normalize_middle_step,
    },
};
#[cfg(not(target_arch = "aarch64"))]
use poulpy_cpu_portable::kernels::znx::{
    standard::znx_automorphism_portable as kn_automorphism, znx_add_assign_portable as kn_add_assign, znx_add_portable as kn_add,
    znx_automorphism_rotate_portable as kn_automorphism_rotate, znx_extract_digit_addmul_portable as kn_extract_digit_addmul,
    znx_mul_add_power_of_two_portable as kn_mul_add_p2, znx_mul_power_of_two_assign_portable as kn_mul_p2_assign,
    znx_mul_power_of_two_portable as kn_mul_p2, znx_negate_assign_portable as kn_negate_assign, znx_negate_portable as kn_negate,
    znx_normalize_digit_portable as kn_normalize_digit,
    znx_normalize_final_step_assign_portable as kn_normalize_final_step_assign,
    znx_normalize_final_step_portable as kn_normalize_final_step,
    znx_normalize_first_step_assign_portable as kn_normalize_first_step_assign,
    znx_normalize_first_step_carry_only_portable as kn_normalize_first_step_carry_only,
    znx_normalize_first_step_portable as kn_normalize_first_step,
    znx_normalize_middle_step_assign_portable as kn_normalize_middle_step_assign,
    znx_normalize_middle_step_carry_only_portable as kn_normalize_middle_step_carry_only,
    znx_normalize_middle_step_portable as kn_normalize_middle_step, znx_sub_assign_portable as kn_sub_assign,
    znx_sub_negate_assign_portable as kn_sub_negate_assign, znx_sub_portable as kn_sub,
    znx_switch_ring_portable as kn_switch_ring,
};

impl<R: Ring> ZnxAdd for FFT64Neon<R> {
    #[inline(always)]
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]) {
        kn_add(res, a, b);
    }
}

impl<R: Ring> ZnxAddAssign for FFT64Neon<R> {
    #[inline(always)]
    fn znx_add_assign(res: &mut [i64], a: &[i64]) {
        kn_add_assign(res, a);
    }
}

impl<R: Ring> ZnxSub for FFT64Neon<R> {
    #[inline(always)]
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]) {
        kn_sub(res, a, b);
    }
}

impl<R: Ring> ZnxSubAssign for FFT64Neon<R> {
    #[inline(always)]
    fn znx_sub_assign(res: &mut [i64], a: &[i64]) {
        kn_sub_assign(res, a);
    }
}

impl<R: Ring> ZnxSubNegateAssign for FFT64Neon<R> {
    #[inline(always)]
    fn znx_sub_negate_assign(res: &mut [i64], a: &[i64]) {
        kn_sub_negate_assign(res, a);
    }
}

impl<R: Ring> ZnxMulAddPowerOfTwo for FFT64Neon<R> {
    #[inline(always)]
    fn znx_muladd_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        kn_mul_add_p2(k, res, a);
    }
}

impl<R: Ring> ZnxMulPowerOfTwo for FFT64Neon<R> {
    #[inline(always)]
    fn znx_mul_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        kn_mul_p2(k, res, a);
    }
}

impl<R: Ring> ZnxMulPowerOfTwoAssign for FFT64Neon<R> {
    #[inline(always)]
    fn znx_mul_power_of_two_assign(k: i64, res: &mut [i64]) {
        kn_mul_p2_assign(k, res);
    }
}

impl<R: Ring> ZnxCopy for FFT64Neon<R> {
    #[inline(always)]
    fn znx_copy(res: &mut [i64], a: &[i64]) {
        znx_copy_portable(res, a);
    }
}

impl<R: Ring> ZnxNegate for FFT64Neon<R> {
    #[inline(always)]
    fn znx_negate(res: &mut [i64], src: &[i64]) {
        kn_negate(res, src);
    }
}

impl<R: Ring> ZnxNegateAssign for FFT64Neon<R> {
    #[inline(always)]
    fn znx_negate_assign(res: &mut [i64]) {
        kn_negate_assign(res);
    }
}

impl<R: Ring> ZnxRotate for FFT64Neon<R> {
    #[inline(always)]
    fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]) {
        znx_rotate_portable::<Self>(p, res, src);
    }
}

impl<R: Ring> ZnxZero for FFT64Neon<R> {
    #[inline(always)]
    fn znx_zero(res: &mut [i64]) {
        znx_zero_portable(res);
    }
}

impl<R: Ring> ZnxSwitchRing for FFT64Neon<R> {
    #[inline(always)]
    fn znx_switch_ring(res: &mut [i64], a: &[i64]) {
        kn_switch_ring(res, a);
    }
}

impl<R: Ring> ZnxNormalizeFirstStep for FFT64Neon<R> {
    #[inline(always)]
    fn znx_normalize_first_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        kn_normalize_first_step::<OVERWRITE>(base2k, lsh, x, a, carry);
    }
}

impl<R: Ring> ZnxNormalizeMiddleStep for FFT64Neon<R> {
    #[inline(always)]
    fn znx_normalize_middle_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        kn_normalize_middle_step::<OVERWRITE>(base2k, lsh, x, a, carry);
    }
}

impl<R: Ring> ZnxNormalizeFinalStep for FFT64Neon<R> {
    #[inline(always)]
    fn znx_normalize_final_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        kn_normalize_final_step::<OVERWRITE>(base2k, lsh, x, a, carry);
    }
}

impl<R: Ring> ZnxNormalizeFinalStepAssign for FFT64Neon<R> {
    #[inline(always)]
    fn znx_normalize_final_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        kn_normalize_final_step_assign(base2k, lsh, x, carry);
    }
}

impl<R: Ring> ZnxNormalizeFirstStepCarryOnly for FFT64Neon<R> {
    #[inline(always)]
    fn znx_normalize_first_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        kn_normalize_first_step_carry_only(base2k, lsh, x, carry);
    }
}

impl<R: Ring> ZnxNormalizeFirstStepAssign for FFT64Neon<R> {
    #[inline(always)]
    fn znx_normalize_first_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        kn_normalize_first_step_assign(base2k, lsh, x, carry);
    }
}

impl<R: Ring> ZnxNormalizeMiddleStepCarryOnly for FFT64Neon<R> {
    #[inline(always)]
    fn znx_normalize_middle_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        kn_normalize_middle_step_carry_only(base2k, lsh, x, carry);
    }
}

impl<R: Ring> ZnxNormalizeMiddleStepAssign for FFT64Neon<R> {
    #[inline(always)]
    fn znx_normalize_middle_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        kn_normalize_middle_step_assign(base2k, lsh, x, carry);
    }
}

impl<R: Ring> ZnxExtractDigitAddMul for FFT64Neon<R> {
    #[inline(always)]
    fn znx_extract_digit_addmul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
        kn_extract_digit_addmul(base2k, lsh, res, src);
    }
}

impl<R: Ring> poulpy_cpu_portable::kernels::normalization::I64NormalizeOps for FFT64Neon<R> {
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn znx_normalize_floor<const CARRY_IN: bool, const ROUND: bool>(base2k: usize, lsh: usize, a: &[i64], carry: &mut [i64]) {
        crate::neon::normalization_boundary::znx_normalize_floor_neon::<CARRY_IN, ROUND>(base2k, lsh, a, carry);
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn znx_normalize_round<const CARRY_IN: bool, const PAD: bool>(
        base2k: usize,
        lsh: usize,
        padding: usize,
        res: &mut [i64],
        a: &[i64],
        carry: &mut [i64],
    ) {
        crate::neon::normalization_boundary::znx_normalize_round_neon::<CARRY_IN, PAD>(base2k, lsh, padding, res, a, carry);
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn znx_normalize_round_assign<const CARRY_IN: bool>(
        base2k: usize,
        lsh: usize,
        padding: usize,
        res: &mut [i64],
        carry: &mut [i64],
    ) {
        crate::neon::normalization_boundary::znx_normalize_round_assign_neon::<CARRY_IN>(base2k, lsh, padding, res, carry);
    }

    #[inline(always)]
    fn znx_extract_digit_mul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
        #[cfg(target_arch = "aarch64")]
        crate::neon::znx_normalize::znx_extract_digit_mul_neon(base2k, lsh, res, src);
        #[cfg(not(target_arch = "aarch64"))]
        poulpy_cpu_portable::kernels::znx::znx_extract_digit_mul_portable(base2k, lsh, res, src);
    }

    #[inline(always)]
    fn znx_extract_digit_addmul_normalize<const OVERWRITE: bool>(
        base2k: usize,
        lsh: usize,
        res_base2k: usize,
        res: &mut [i64],
        src: &mut [i64],
        carry: &mut [i64],
    ) {
        #[cfg(target_arch = "aarch64")]
        crate::neon::znx_normalize::znx_extract_digit_addmul_normalize_neon::<OVERWRITE>(
            base2k, lsh, res_base2k, res, src, carry,
        );
        #[cfg(not(target_arch = "aarch64"))]
        poulpy_cpu_portable::kernels::znx::znx_extract_digit_addmul_normalize_portable::<OVERWRITE>(
            base2k, lsh, res_base2k, res, src, carry,
        );
    }
}

impl<R: Ring> ZnxNormalizeDigit for FFT64Neon<R> {
    #[inline(always)]
    fn znx_normalize_digit(base2k: usize, res: &mut [i64], src: &mut [i64]) {
        kn_normalize_digit(base2k, res, src);
    }
}
