//! Single ring element (`Z[X]/(X^n+1)`) arithmetic for [`NTT4x30Avx`](super::NTT4x30Avx).
//!
//! Implements the `Znx*` traits from `poulpy_cpu_portable::kernels::znx`. All implementations
//! delegate to the AVX2-accelerated functions in `crate::znx_avx` (same kernels used
//! by `FFT64Avx`). These operate on plain `&[i64]` slices and are backend-independent.

use poulpy_cpu_portable::kernels::znx::{
    ZnxAdd, ZnxAddAssign, ZnxCopy, ZnxMulAddPowerOfTwo, ZnxMulPowerOfTwo, ZnxMulPowerOfTwoAssign, ZnxNegate, ZnxNegateAssign,
    ZnxRotate, ZnxSub, ZnxSubAssign, ZnxSubNegateAssign, ZnxSwitchRing, ZnxZero, znx_copy_portable, znx_rotate_portable,
    znx_zero_portable,
};
use poulpy_hal::layouts::Ring;

use crate::znx_avx::{
    znx_add_assign_avx, znx_add_avx, znx_mul_add_power_of_two_avx, znx_mul_power_of_two_assign_avx, znx_mul_power_of_two_avx,
    znx_negate_assign_avx, znx_negate_avx, znx_sub_assign_avx, znx_sub_avx, znx_sub_negate_assign_avx, znx_switch_ring_avx,
};

use super::NTT4x30Avx;

impl<R: Ring> ZnxAdd for NTT4x30Avx<R> {
    #[inline(always)]
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]) {
        unsafe { znx_add_avx(res, a, b) }
    }
}

impl<R: Ring> ZnxAddAssign for NTT4x30Avx<R> {
    #[inline(always)]
    fn znx_add_assign(res: &mut [i64], a: &[i64]) {
        unsafe { znx_add_assign_avx(res, a) }
    }
}

impl<R: Ring> ZnxSub for NTT4x30Avx<R> {
    #[inline(always)]
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]) {
        unsafe { znx_sub_avx(res, a, b) }
    }
}

impl<R: Ring> ZnxSubAssign for NTT4x30Avx<R> {
    #[inline(always)]
    fn znx_sub_assign(res: &mut [i64], a: &[i64]) {
        unsafe { znx_sub_assign_avx(res, a) }
    }
}

impl<R: Ring> ZnxSubNegateAssign for NTT4x30Avx<R> {
    #[inline(always)]
    fn znx_sub_negate_assign(res: &mut [i64], a: &[i64]) {
        unsafe { znx_sub_negate_assign_avx(res, a) }
    }
}

impl<R: Ring> ZnxMulAddPowerOfTwo for NTT4x30Avx<R> {
    #[inline(always)]
    fn znx_muladd_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        unsafe { znx_mul_add_power_of_two_avx(k, res, a) }
    }
}

impl<R: Ring> ZnxMulPowerOfTwo for NTT4x30Avx<R> {
    #[inline(always)]
    fn znx_mul_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        unsafe { znx_mul_power_of_two_avx(k, res, a) }
    }
}

impl<R: Ring> ZnxMulPowerOfTwoAssign for NTT4x30Avx<R> {
    #[inline(always)]
    fn znx_mul_power_of_two_assign(k: i64, res: &mut [i64]) {
        unsafe { znx_mul_power_of_two_assign_avx(k, res) }
    }
}

impl<R: Ring> ZnxCopy for NTT4x30Avx<R> {
    #[inline(always)]
    fn znx_copy(res: &mut [i64], a: &[i64]) {
        znx_copy_portable(res, a);
    }
}

impl<R: Ring> ZnxNegate for NTT4x30Avx<R> {
    #[inline(always)]
    fn znx_negate(res: &mut [i64], src: &[i64]) {
        unsafe { znx_negate_avx(res, src) }
    }
}

impl<R: Ring> ZnxNegateAssign for NTT4x30Avx<R> {
    #[inline(always)]
    fn znx_negate_assign(res: &mut [i64]) {
        unsafe { znx_negate_assign_avx(res) }
    }
}

impl<R: Ring> ZnxRotate for NTT4x30Avx<R> {
    #[inline(always)]
    fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]) {
        znx_rotate_portable::<Self>(p, res, src);
    }
}

impl<R: Ring> ZnxZero for NTT4x30Avx<R> {
    #[inline(always)]
    fn znx_zero(res: &mut [i64]) {
        znx_zero_portable(res);
    }
}

impl<R: Ring> ZnxSwitchRing for NTT4x30Avx<R> {
    #[inline(always)]
    fn znx_switch_ring(res: &mut [i64], a: &[i64]) {
        unsafe { znx_switch_ring_avx(res, a) }
    }
}

impl<R: Ring> poulpy_cpu_portable::kernels::znx_normalize::ZnxNormalize for NTT4x30Avx<R> {
    type WideLanes = poulpy_cpu_portable::kernels::znx_normalize::SimdLanes<16>;
}
