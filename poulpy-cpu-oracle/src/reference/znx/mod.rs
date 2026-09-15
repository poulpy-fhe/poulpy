mod add;
mod automorphism;
mod copy;
mod neg;
mod rotate;
mod sampling;
mod sub;
mod switch_ring;
mod zero;

pub use add::*;
pub use automorphism::*;
pub use copy::*;
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

pub trait ZnxSwitchRing {
    fn znx_switch_ring(res: &mut [i64], a: &[i64]);
}
