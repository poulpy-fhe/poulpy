//! Single ring element (`Z[X]/(X^n+1)`) arithmetic for [`NTT4x30Portable`](super::NTT4x30Portable).
//!
//! Implements the `Znx*` traits from `crate::kernels::znx`. All implementations
//! delegate to the shared `_portable` kernels, which operate on plain
//! `&[i64]` slices, which are backend-independent.

use super::NTT4x30Portable;
use poulpy_hal::layouts::{ConjugateInvariant, Standard};

use crate::kernels::znx::{
    ZnxAdd, ZnxAddAssign, ZnxAutomorphism, ZnxAutomorphismRotate, ZnxCopy, ZnxMulAddPowerOfTwo, ZnxMulPowerOfTwo,
    ZnxMulPowerOfTwoAssign, ZnxNegate, ZnxNegateAssign, ZnxRotate, ZnxSub, ZnxSubAssign, ZnxSubNegateAssign, ZnxSwitchRing,
    ZnxZero, conjugate_invariant, standard, znx_add_assign_portable, znx_add_portable, znx_automorphism_rotate_portable,
    znx_copy_portable, znx_mul_add_power_of_two_portable, znx_mul_power_of_two_assign_portable, znx_mul_power_of_two_portable,
    znx_negate_assign_portable, znx_negate_portable, znx_rotate_portable, znx_sub_assign_portable,
    znx_sub_negate_assign_portable, znx_sub_portable, znx_switch_ring_portable, znx_zero_portable,
};

impl ZnxAdd for NTT4x30Portable {
    #[inline(always)]
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]) {
        znx_add_portable(res, a, b);
    }
}

impl ZnxAddAssign for NTT4x30Portable {
    #[inline(always)]
    fn znx_add_assign(res: &mut [i64], a: &[i64]) {
        znx_add_assign_portable(res, a);
    }
}

impl ZnxSub for NTT4x30Portable {
    #[inline(always)]
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]) {
        znx_sub_portable(res, a, b);
    }
}

impl ZnxSubAssign for NTT4x30Portable {
    #[inline(always)]
    fn znx_sub_assign(res: &mut [i64], a: &[i64]) {
        znx_sub_assign_portable(res, a);
    }
}

impl ZnxSubNegateAssign for NTT4x30Portable {
    #[inline(always)]
    fn znx_sub_negate_assign(res: &mut [i64], a: &[i64]) {
        znx_sub_negate_assign_portable(res, a);
    }
}

impl ZnxMulAddPowerOfTwo for NTT4x30Portable {
    #[inline(always)]
    fn znx_muladd_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        znx_mul_add_power_of_two_portable(k, res, a);
    }
}

impl ZnxMulPowerOfTwo for NTT4x30Portable {
    #[inline(always)]
    fn znx_mul_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
        znx_mul_power_of_two_portable(k, res, a);
    }
}

impl ZnxMulPowerOfTwoAssign for NTT4x30Portable {
    #[inline(always)]
    fn znx_mul_power_of_two_assign(k: i64, res: &mut [i64]) {
        znx_mul_power_of_two_assign_portable(k, res);
    }
}

impl ZnxAutomorphism<Standard> for NTT4x30Portable {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        standard::znx_automorphism_portable(p, res, a);
    }

    #[inline(always)]
    fn znx_automorphism_i128(p: i64, res: &mut [i128], a: &[i128]) {
        standard::znx_automorphism_portable(p, res, a)
    }
}

impl ZnxAutomorphism<ConjugateInvariant> for NTT4x30Portable {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        conjugate_invariant::znx_automorphism_portable(p, res, a);
    }

    #[inline(always)]
    fn znx_automorphism_i128(p: i64, res: &mut [i128], a: &[i128]) {
        conjugate_invariant::znx_automorphism_portable(p, res, a);
    }
}

impl ZnxAutomorphismRotate<Standard> for NTT4x30Portable {
    #[inline(always)]
    fn znx_automorphism_rotate(p: i64, k: i64, res: &mut [i64], a: &[i64]) {
        znx_automorphism_rotate_portable(p, k, res, a);
    }
}

impl ZnxCopy for NTT4x30Portable {
    #[inline(always)]
    fn znx_copy(res: &mut [i64], a: &[i64]) {
        znx_copy_portable(res, a);
    }
}

impl ZnxNegate for NTT4x30Portable {
    #[inline(always)]
    fn znx_negate(res: &mut [i64], src: &[i64]) {
        znx_negate_portable(res, src);
    }
}

impl ZnxNegateAssign for NTT4x30Portable {
    #[inline(always)]
    fn znx_negate_assign(res: &mut [i64]) {
        znx_negate_assign_portable(res);
    }
}

impl ZnxRotate for NTT4x30Portable {
    #[inline(always)]
    fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]) {
        znx_rotate_portable::<Self>(p, res, src);
    }
}

impl ZnxZero for NTT4x30Portable {
    #[inline(always)]
    fn znx_zero(res: &mut [i64]) {
        znx_zero_portable(res);
    }
}

impl ZnxSwitchRing for NTT4x30Portable {
    #[inline(always)]
    fn znx_switch_ring(res: &mut [i64], a: &[i64]) {
        znx_switch_ring_portable(res, a);
    }
}

impl crate::kernels::znx_normalize::ZnxNormalize for NTT4x30Portable {}
