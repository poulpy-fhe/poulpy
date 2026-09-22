//! Single ring element (`Z[X]/(X^n+1)`) arithmetic for [`NTT3x42Ifma`](super::NTT3x42Ifma).
//!
//! Implements the `Znx*` traits from `poulpy_cpu_ref::reference::znx`. All implementations
//! delegate to the AVX512-accelerated functions in `crate::znx_avx512` (same kernels used
//! by `FFT64Avx512`). These operate on plain `&[i64]` slices and are backend-independent.

#[cfg(feature = "enable-ifma")]
use crate::NTT3x42IfmaBackend;
use poulpy_cpu_ref::ring::CpuRing;

use poulpy_cpu_ref::reference::znx::{
    ZnxAdd, ZnxAddAssign, ZnxAutomorphism, ZnxAutomorphismRotate, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulAddPowerOfTwo,
    ZnxMulPowerOfTwo, ZnxMulPowerOfTwoAssign, ZnxNegate, ZnxNegateAssign, ZnxNormalizeDigit, ZnxNormalizeFinalStep,
    ZnxNormalizeFinalStepAssign, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepAssign, ZnxNormalizeFirstStepCarryOnly,
    ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepAssign, ZnxNormalizeMiddleStepCarryOnly, ZnxRotate, ZnxSub, ZnxSubAssign,
    ZnxSubNegateAssign, ZnxSwitchRing, ZnxZero, znx_copy_ref, znx_rotate, znx_zero_ref,
};

use crate::znx_avx512::{
    znx_add_assign_avx512, znx_add_avx512, znx_automorphism_avx512, znx_automorphism_rotate_avx512,
    znx_extract_digit_addmul_avx512, znx_mul_add_power_of_two_avx512, znx_mul_power_of_two_assign_avx512,
    znx_mul_power_of_two_avx512, znx_negate_assign_avx512, znx_negate_avx512, znx_normalize_digit_avx512,
    znx_normalize_final_step_assign_avx512, znx_normalize_final_step_avx512, znx_normalize_first_step_assign_avx512,
    znx_normalize_first_step_avx512, znx_normalize_first_step_carry_only_avx512, znx_normalize_middle_step_assign_avx512,
    znx_normalize_middle_step_avx512, znx_normalize_middle_step_carry_only_avx512, znx_sub_assign_avx512, znx_sub_avx512,
    znx_sub_negate_assign_avx512, znx_switch_ring_avx512,
};

impl<R: CpuRing> ZnxAdd for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]) {
        unsafe { znx_add_avx512(res, a, b) }
    }
}

impl<R: CpuRing> ZnxAddAssign for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_add_assign(res: &mut [i64], a: &[i64]) {
        unsafe { znx_add_assign_avx512(res, a) }
    }
}

impl<R: CpuRing> ZnxSub for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]) {
        unsafe { znx_sub_avx512(res, a, b) }
    }
}

impl<R: CpuRing> ZnxSubAssign for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_sub_assign(res: &mut [i64], a: &[i64]) {
        unsafe { znx_sub_assign_avx512(res, a) }
    }
}

impl<R: CpuRing> ZnxSubNegateAssign for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_sub_negate_assign(res: &mut [i64], a: &[i64]) {
        unsafe { znx_sub_negate_assign_avx512(res, a) }
    }
}

impl<R: CpuRing> ZnxMulAddPowerOfTwo for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_muladd_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        unsafe { znx_mul_add_power_of_two_avx512(k, res, a) }
    }
}

impl<R: CpuRing> ZnxMulPowerOfTwo for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_mul_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        unsafe { znx_mul_power_of_two_avx512(k, res, a) }
    }
}

impl<R: CpuRing> ZnxMulPowerOfTwoAssign for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_mul_power_of_two_assign(k: i64, res: &mut [i64]) {
        unsafe { znx_mul_power_of_two_assign_avx512(k, res) }
    }
}

impl<R: CpuRing> ZnxAutomorphism for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        unsafe { znx_automorphism_avx512(p, res, a) }
    }
}

impl<R: CpuRing> ZnxAutomorphismRotate for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_automorphism_rotate(p: i64, k: i64, res: &mut [i64], a: &[i64]) {
        unsafe { znx_automorphism_rotate_avx512(p, k, res, a) }
    }
}

impl<R: CpuRing> ZnxCopy for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_copy(res: &mut [i64], a: &[i64]) {
        znx_copy_ref(res, a);
    }
}

impl<R: CpuRing> ZnxNegate for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_negate(res: &mut [i64], src: &[i64]) {
        unsafe { znx_negate_avx512(res, src) }
    }
}

impl<R: CpuRing> ZnxNegateAssign for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_negate_assign(res: &mut [i64]) {
        unsafe { znx_negate_assign_avx512(res) }
    }
}

impl<R: CpuRing> ZnxRotate for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]) {
        znx_rotate::<Self>(p, res, src);
    }
}

impl<R: CpuRing> ZnxZero for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_zero(res: &mut [i64]) {
        znx_zero_ref(res);
    }
}

impl<R: CpuRing> ZnxSwitchRing for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_switch_ring(res: &mut [i64], a: &[i64]) {
        unsafe { znx_switch_ring_avx512(res, a) }
    }
}

impl<R: CpuRing> ZnxNormalizeFinalStep for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_normalize_final_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        unsafe { znx_normalize_final_step_avx512::<OVERWRITE>(base2k, lsh, x, a, carry) }
    }
}

impl<R: CpuRing> ZnxNormalizeFinalStepAssign for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_normalize_final_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        unsafe { znx_normalize_final_step_assign_avx512(base2k, lsh, x, carry) }
    }
}

impl<R: CpuRing> ZnxNormalizeFirstStep for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_normalize_first_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        unsafe { znx_normalize_first_step_avx512::<OVERWRITE>(base2k, lsh, x, a, carry) }
    }
}

impl<R: CpuRing> ZnxNormalizeFirstStepCarryOnly for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_normalize_first_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        unsafe { znx_normalize_first_step_carry_only_avx512(base2k, lsh, x, carry) }
    }
}

impl<R: CpuRing> ZnxNormalizeFirstStepAssign for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_normalize_first_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        unsafe { znx_normalize_first_step_assign_avx512(base2k, lsh, x, carry) }
    }
}

impl<R: CpuRing> ZnxNormalizeMiddleStep for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_normalize_middle_step<const OVERWRITE: bool>(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
        unsafe { znx_normalize_middle_step_avx512::<OVERWRITE>(base2k, lsh, x, a, carry) }
    }
}

impl<R: CpuRing> ZnxNormalizeMiddleStepCarryOnly for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_normalize_middle_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
        unsafe { znx_normalize_middle_step_carry_only_avx512(base2k, lsh, x, carry) }
    }
}

impl<R: CpuRing> ZnxNormalizeMiddleStepAssign for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_normalize_middle_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
        unsafe { znx_normalize_middle_step_assign_avx512(base2k, lsh, x, carry) }
    }
}

impl<R: CpuRing> ZnxExtractDigitAddMul for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_extract_digit_addmul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
        unsafe { znx_extract_digit_addmul_avx512(base2k, lsh, res, src) }
    }
}

impl<R: CpuRing> poulpy_cpu_ref::reference::normalization::I64NormalizeOps for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_normalize_floor<const CARRY_IN: bool, const ROUND: bool>(base2k: usize, lsh: usize, a: &[i64], carry: &mut [i64]) {
        assert!(a.len() >= carry.len());
        unsafe { crate::znx_avx512::znx_normalize_floor_avx512::<CARRY_IN, ROUND>(base2k, lsh, a, carry) }
    }

    #[inline(always)]
    fn znx_normalize_round<const CARRY_IN: bool, const PAD: bool>(
        base2k: usize,
        lsh: usize,
        padding: usize,
        res: &mut [i64],
        a: &[i64],
        carry: &mut [i64],
    ) {
        assert!(a.len() >= res.len() && carry.len() >= res.len());
        unsafe { crate::znx_avx512::znx_normalize_round_avx512::<CARRY_IN, PAD>(base2k, lsh, padding, res, a, carry) }
    }

    #[inline(always)]
    fn znx_normalize_round_assign<const CARRY_IN: bool>(
        base2k: usize,
        lsh: usize,
        padding: usize,
        res: &mut [i64],
        carry: &mut [i64],
    ) {
        assert!(carry.len() >= res.len());
        unsafe { crate::znx_avx512::znx_normalize_round_assign_avx512::<CARRY_IN>(base2k, lsh, padding, res, carry) }
    }

    #[inline(always)]
    fn znx_extract_digit_mul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
        unsafe {
            crate::znx_avx512::znx_extract_digit_mul_avx512(base2k, lsh, res, src);
        }
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
        unsafe {
            crate::znx_avx512::znx_extract_digit_addmul_normalize_avx512::<OVERWRITE>(base2k, lsh, res_base2k, res, src, carry);
        }
    }
}

impl<R: CpuRing> ZnxNormalizeDigit for NTT3x42IfmaBackend<R> {
    #[inline(always)]
    fn znx_normalize_digit(base2k: usize, res: &mut [i64], src: &mut [i64]) {
        unsafe { znx_normalize_digit_avx512(base2k, res, src) }
    }
}
