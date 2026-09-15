//! Scalar coefficient arithmetic for FFT64.

use crate::reference::znx::*;

use super::FFT64Oracle;

impl ZnxAdd for FFT64Oracle {
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]) {
        znx_add_ref(res, a, b);
    }
}

impl ZnxAddAssign for FFT64Oracle {
    fn znx_add_assign(res: &mut [i64], a: &[i64]) {
        znx_add_assign_ref(res, a);
    }
}

impl ZnxSub for FFT64Oracle {
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]) {
        znx_sub_ref(res, a, b);
    }
}

impl ZnxSubAssign for FFT64Oracle {
    fn znx_sub_assign(res: &mut [i64], a: &[i64]) {
        znx_sub_assign_ref(res, a);
    }
}

impl ZnxSubNegateAssign for FFT64Oracle {
    fn znx_sub_negate_assign(res: &mut [i64], a: &[i64]) {
        znx_sub_negate_assign_ref(res, a);
    }
}

impl ZnxAutomorphism for FFT64Oracle {
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        znx_automorphism_ref(p, res, a);
    }
}

impl ZnxCopy for FFT64Oracle {
    fn znx_copy(res: &mut [i64], a: &[i64]) {
        znx_copy_ref(res, a);
    }
}

impl ZnxNegate for FFT64Oracle {
    fn znx_negate(res: &mut [i64], src: &[i64]) {
        znx_negate_ref(res, src);
    }
}

impl ZnxNegateAssign for FFT64Oracle {
    fn znx_negate_assign(res: &mut [i64]) {
        znx_negate_assign_ref(res);
    }
}

impl ZnxRotate for FFT64Oracle {
    fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]) {
        znx_rotate::<Self>(p, res, src);
    }
}

impl ZnxZero for FFT64Oracle {
    fn znx_zero(res: &mut [i64]) {
        znx_zero_ref(res);
    }
}

impl ZnxSwitchRing for FFT64Oracle {
    fn znx_switch_ring(res: &mut [i64], a: &[i64]) {
        znx_switch_ring_ref(res, a);
    }
}
