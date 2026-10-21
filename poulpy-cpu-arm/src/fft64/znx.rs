//! `Znx*` trait impls for [`FFT64Neon`](super::FFT64Neon).

use poulpy_cpu_portable::kernels::znx::{
    ZnxAdd, ZnxAddAssign, ZnxCopy, ZnxMulAddPowerOfTwo, ZnxMulPowerOfTwo, ZnxMulPowerOfTwoAssign, ZnxNegate, ZnxNegateAssign,
    ZnxRotate, ZnxSub, ZnxSubAssign, ZnxSubNegateAssign, ZnxSwitchRing, ZnxZero, znx_copy_portable, znx_rotate_portable,
    znx_zero_portable,
};

use super::FFT64Neon;
use poulpy_hal::layouts::Ring;

#[cfg(target_arch = "aarch64")]
use crate::neon::znx::{
    znx_add_assign_neon as kn_add_assign, znx_add_neon as kn_add, znx_mul_add_power_of_two_neon as kn_mul_add_p2,
    znx_mul_power_of_two_assign_neon as kn_mul_p2_assign, znx_mul_power_of_two_neon as kn_mul_p2,
    znx_negate_assign_neon as kn_negate_assign, znx_negate_neon as kn_negate, znx_sub_assign_neon as kn_sub_assign,
    znx_sub_negate_assign_neon as kn_sub_negate_assign, znx_sub_neon as kn_sub, znx_switch_ring_neon as kn_switch_ring,
};
#[cfg(not(target_arch = "aarch64"))]
use poulpy_cpu_portable::kernels::znx::{
    standard::znx_automorphism_portable as kn_automorphism, znx_add_assign_portable as kn_add_assign, znx_add_portable as kn_add,
    znx_automorphism_rotate_portable as kn_automorphism_rotate, znx_mul_add_power_of_two_portable as kn_mul_add_p2,
    znx_mul_power_of_two_assign_portable as kn_mul_p2_assign, znx_mul_power_of_two_portable as kn_mul_p2,
    znx_negate_assign_portable as kn_negate_assign, znx_negate_portable as kn_negate, znx_sub_assign_portable as kn_sub_assign,
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

impl<R: Ring> poulpy_cpu_portable::kernels::znx_normalize::ZnxNormalize for FFT64Neon<R> {}
