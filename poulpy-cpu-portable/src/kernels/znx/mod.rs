mod add;
mod arithmetic;
mod automorphism_rotate;
pub mod conjugate_invariant;
mod copy;
mod mul;
mod neg;
mod rotate;
mod sampling;
pub mod standard;
mod sub;
mod switch_ring;
mod zero;

pub use add::*;
pub use arithmetic::*;
pub use automorphism_rotate::*;
pub use copy::*;
pub use mul::*;
pub use neg::*;
pub use rotate::*;
pub use sub::*;
pub use switch_ring::*;
pub use zero::*;

pub use sampling::*;

pub trait ZnxAdd {
    fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]);
}

pub trait ZnxAddAssign {
    fn znx_add_assign(res: &mut [i64], a: &[i64]);
}

pub trait ZnxSub {
    fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]);
}

pub trait ZnxSubAssign {
    fn znx_sub_assign(res: &mut [i64], a: &[i64]);
}

pub trait ZnxSubNegateAssign {
    fn znx_sub_negate_assign(res: &mut [i64], a: &[i64]);
}

pub trait ZnxAutomorphism {
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]);

    /// [`Self::znx_automorphism`] on `i128` big coefficients.
    fn znx_automorphism_i128(p: i64, res: &mut [i128], a: &[i128]);
}

pub trait ZnxAutomorphismRotate {
    /// Computes `res = X^k * auto(p, a)` (fused automorphism + rotation).
    fn znx_automorphism_rotate(p: i64, k: i64, res: &mut [i64], a: &[i64]);
}

pub trait ZnxCopy {
    fn znx_copy(res: &mut [i64], a: &[i64]);
}

pub trait ZnxNegate {
    fn znx_negate(res: &mut [i64], src: &[i64]);
}

pub trait ZnxNegateAssign {
    fn znx_negate_assign(res: &mut [i64]);
}

pub trait ZnxRotate {
    fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]);
}

pub trait ZnxZero {
    fn znx_zero(res: &mut [i64]);
}

pub trait ZnxMulPowerOfTwo {
    fn znx_mul_power_of_two(k: i64, res: &mut [i64], a: &[i64]);
}

pub trait ZnxMulAddPowerOfTwo {
    fn znx_muladd_power_of_two(k: i64, res: &mut [i64], a: &[i64]);
}

pub trait ZnxMulPowerOfTwoAssign {
    fn znx_mul_power_of_two_assign(k: i64, res: &mut [i64]);
}

pub trait ZnxSwitchRing {
    fn znx_switch_ring(res: &mut [i64], a: &[i64]);
}
