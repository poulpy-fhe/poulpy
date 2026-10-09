//! The automorphism `X -> X^p` in the DFT domain: a permutation of the slots,
//! computed by each thread from `p`.
//!
//! Slot `s` holds the value at `psi^(2 bitrev(s) + 1)`, so the image reads the
//! slot `t` with `2 bitrev(t) + 1 = p (2 bitrev(s) + 1) mod 2n`. Over the
//! conjugate-invariant ring the points are `alpha^(4 bitrev(s) + 1)` and the
//! exponent is taken modulo `4n`, for the sign of `p` that is `1 mod 4`: the
//! other sign maps to the conjugate points, where the values are the same.

use cubecl::prelude::*;

use super::{
    DftColumn,
    primes::{Q, sel4},
};
use crate::runtime::{GpuRuntime, Label, launch};

/// The substitution `X -> X^p` on DFT-domain vectors of degree `n`. It holds
/// no device state: the kernels derive the permutation from `mul`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AutomorphismPlan {
    pub(super) n: usize,
    /// `p` (or `-p`) modulo the order of the evaluation points.
    pub(super) mul: u32,
    /// log2 of the gap between the exponents of two points: 1, or 2 over the
    /// conjugate-invariant ring.
    pub(super) shift: u32,
}

impl AutomorphismPlan {
    pub(crate) fn new(n: usize, p: i64, invariant: bool) -> Self {
        assert!(n.is_power_of_two(), "ring degree must be a power of two, got {n}");
        assert!(p & 1 == 1, "automorphism exponent must be odd, got {p}");
        let shift = 1 + invariant as u32;
        let order = (n as i128) << shift;
        let mut mul = (p as i128).rem_euclid(order);
        if invariant && mul & 3 == 3 {
            mul = order - mul;
        }
        Self {
            n,
            mul: mul as u32,
            shift,
        }
    }
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn automorphism_kernel(
    res: &mut [u32],
    a: &[u32],
    log_n: u32,
    limbs: u32,
    res_base: u32,
    res_stride: u32,
    a_base: u32,
    a_stride: u32,
    a_size: u32,
    mul: u32,
    shift: u32,
    #[comptime] accumulate: bool,
    #[comptime] q: [u32; 4],
) {
    let pos = ABSOLUTE_POS as u32;
    let log_block = log_n + 2u32;
    if pos < (limbs << log_block) {
        let limb = pos >> log_block;
        let word = pos - (limb << log_block);
        let plane = word >> log_n;
        let slot = word - (plane << log_n);
        let mut x = 0u32;
        if limb < a_size {
            let up = 32u32 - log_n;
            let point = slot.reverse_bits() >> up;
            let exponent = (mul * ((point << shift) + 1u32)) & ((1u32 << (log_n + shift)) - 1u32);
            let from = (exponent >> shift).reverse_bits() >> up;
            x = a[(a_base + limb * a_stride + (plane << log_n) + from) as usize];
        }
        let index = res_base + limb * res_stride + word;
        if accumulate {
            let q = sel4(plane, q);
            let sum = x + res[index as usize];
            x = min(sum, sum - q);
        }
        res[index as usize] = x;
    }
}

/// `res = sigma(a)` over every limb of `res` (limbs past `a`'s are zero), or
/// `res += sigma(a)` over the limbs both have when `accumulate`. `res` and
/// `a` must not overlap.
pub(crate) fn automorphism<RT: GpuRuntime>(plan: &AutomorphismPlan, res: &DftColumn, a: &DftColumn, accumulate: bool) {
    assert!(
        res.n == plan.n && a.n == plan.n,
        "automorphism plan of degree {} applied at degrees {} and {}",
        plan.n,
        res.n,
        a.n
    );
    let limbs = if accumulate { res.size.min(a.size) } else { res.size };
    if limbs == 0 {
        return;
    }
    let units = 4 * res.n * limbs;
    launch!(
        RT, if accumulate { Label::DftAutomorphismAdd } else { Label::DftAutomorphism }, units * if accumulate { 12 } else { 8 }, automorphism_kernel, (units, RT::GROUP, RT::GROUP);
        [
            res.binding => u32,
            a.binding => u32,
        ];
        [
            res.n.ilog2(),
            limbs as u32,
            res.base,
            res.stride,
            a.base,
            a.stride,
            a.size as u32,
            plan.mul,
            plan.shift,
        ];
        [accumulate, Q]
    );
}
