//! Single ring element (`Z[X]/(X^n+1)`) arithmetic for [`NTT3x42Ifma`](super::NTT3x42Ifma).
//!
//! Implements the `Znx*` traits from `poulpy_cpu_portable::kernels::znx`. All implementations
//! delegate to the AVX512-accelerated functions in `crate::znx_avx512` (same kernels used
//! by `FFT64Avx512`). These operate on plain `&[i64]` slices and are backend-independent.

use poulpy_cpu_portable::kernels::znx::{
    ZnxAdd, ZnxAddAssign, ZnxCopy, ZnxMulAddPowerOfTwo, ZnxMulPowerOfTwo, ZnxMulPowerOfTwoAssign, ZnxNegate, ZnxNegateAssign,
    ZnxRotate, ZnxSub, ZnxSubAssign, ZnxSubNegateAssign, ZnxSwitchRing, ZnxZero, znx_copy_portable, znx_rotate_portable,
    znx_zero_portable,
};

use crate::znx_avx512::{
    znx_add_assign_avx512, znx_add_avx512, znx_mul_add_power_of_two_avx512, znx_mul_power_of_two_assign_avx512,
    znx_mul_power_of_two_avx512, znx_negate_assign_avx512, znx_negate_avx512, znx_sub_assign_avx512, znx_sub_avx512,
    znx_sub_negate_assign_avx512, znx_switch_ring_avx512,
};

use super::NTT3x42Ifma;
use poulpy_hal::layouts::Ring;

impl<R: Ring> ZnxAdd for NTT3x42Ifma<R> {
    #[inline(always)]
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]) {
        unsafe { znx_add_avx512(res, a, b) }
    }
}

impl<R: Ring> ZnxAddAssign for NTT3x42Ifma<R> {
    #[inline(always)]
    fn znx_add_assign(res: &mut [i64], a: &[i64]) {
        unsafe { znx_add_assign_avx512(res, a) }
    }
}

impl<R: Ring> ZnxSub for NTT3x42Ifma<R> {
    #[inline(always)]
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]) {
        unsafe { znx_sub_avx512(res, a, b) }
    }
}

impl<R: Ring> ZnxSubAssign for NTT3x42Ifma<R> {
    #[inline(always)]
    fn znx_sub_assign(res: &mut [i64], a: &[i64]) {
        unsafe { znx_sub_assign_avx512(res, a) }
    }
}

impl<R: Ring> ZnxSubNegateAssign for NTT3x42Ifma<R> {
    #[inline(always)]
    fn znx_sub_negate_assign(res: &mut [i64], a: &[i64]) {
        unsafe { znx_sub_negate_assign_avx512(res, a) }
    }
}

impl<R: Ring> ZnxMulAddPowerOfTwo for NTT3x42Ifma<R> {
    #[inline(always)]
    fn znx_muladd_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        unsafe { znx_mul_add_power_of_two_avx512(k, res, a) }
    }
}

impl<R: Ring> ZnxMulPowerOfTwo for NTT3x42Ifma<R> {
    #[inline(always)]
    fn znx_mul_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        unsafe { znx_mul_power_of_two_avx512(k, res, a) }
    }
}

impl<R: Ring> ZnxMulPowerOfTwoAssign for NTT3x42Ifma<R> {
    #[inline(always)]
    fn znx_mul_power_of_two_assign(k: i64, res: &mut [i64]) {
        unsafe { znx_mul_power_of_two_assign_avx512(k, res) }
    }
}

impl<R: Ring> ZnxCopy for NTT3x42Ifma<R> {
    #[inline(always)]
    fn znx_copy(res: &mut [i64], a: &[i64]) {
        znx_copy_portable(res, a);
    }
}

impl<R: Ring> ZnxNegate for NTT3x42Ifma<R> {
    #[inline(always)]
    fn znx_negate(res: &mut [i64], src: &[i64]) {
        unsafe { znx_negate_avx512(res, src) }
    }
}

impl<R: Ring> ZnxNegateAssign for NTT3x42Ifma<R> {
    #[inline(always)]
    fn znx_negate_assign(res: &mut [i64]) {
        unsafe { znx_negate_assign_avx512(res) }
    }
}

impl<R: Ring> ZnxRotate for NTT3x42Ifma<R> {
    #[inline(always)]
    fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]) {
        znx_rotate_portable::<Self>(p, res, src);
    }
}

impl<R: Ring> ZnxZero for NTT3x42Ifma<R> {
    #[inline(always)]
    fn znx_zero(res: &mut [i64]) {
        znx_zero_portable(res);
    }
}

impl<R: Ring> ZnxSwitchRing for NTT3x42Ifma<R> {
    #[inline(always)]
    fn znx_switch_ring(res: &mut [i64], a: &[i64]) {
        unsafe { znx_switch_ring_avx512(res, a) }
    }
}

impl<R: Ring> poulpy_cpu_portable::kernels::znx_normalize::ZnxNormalize for NTT3x42Ifma<R> {}
