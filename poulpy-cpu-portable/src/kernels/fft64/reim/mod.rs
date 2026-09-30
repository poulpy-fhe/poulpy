// ----------------------------------------------------------------------
// DISCLAIMER
//
// This module contains code that has been directly ported from the
// spqlios-arithmetic library
// (https://github.com/tfhe/spqlios-arithmetic), which is licensed
// under the Apache License, Version 2.0.
//
// The porting process from C to Rust was done with minimal changes
// in order to preserve the semantics and performance characteristics
// of the original implementation.
//
// Both Poulpy and spqlios-arithmetic are distributed under the terms
// of the Apache License, Version 2.0. See the LICENSE file for details.
//
// ----------------------------------------------------------------------

#![allow(bad_asm_style)]

mod conversion;
mod fft_portable;
mod fft_vec;
mod ifft_portable;
mod table_fft;
mod table_ifft;
mod zero;

pub use conversion::*;
pub use fft_portable::*;
pub use fft_vec::*;
pub use ifft_portable::*;
pub use table_fft::*;
pub use table_ifft::*;
pub use zero::*;

#[inline(always)]
pub(crate) fn as_arr<const SIZE: usize, R: Float + FloatConst>(x: &[R]) -> &[R; SIZE] {
    assert!(x.len() >= SIZE, "x.len():{} < size:{}", x.len(), SIZE);
    unsafe { &*(x.as_ptr() as *const [R; SIZE]) }
}

#[inline(always)]
pub(crate) fn as_arr_mut<const SIZE: usize, R: Float + FloatConst>(x: &mut [R]) -> &mut [R; SIZE] {
    assert!(x.len() >= SIZE);
    unsafe { &mut *(x.as_mut_ptr() as *mut [R; SIZE]) }
}

use rand_distr::num_traits::{Float, FloatConst};
#[inline(always)]
pub(crate) fn frac_rev_bits<R: Float + FloatConst>(x: usize) -> R {
    let half: R = R::from(0.5).unwrap();

    match x {
        0 => R::zero(),
        1 => half,
        _ => {
            if x.is_multiple_of(2) {
                frac_rev_bits::<R>(x >> 1) * half
            } else {
                frac_rev_bits::<R>(x >> 1) * half + half
            }
        }
    }
}

/// `(cos 2 pi t, sin 2 pi t)` from the platform trigonometry of the scalar.
#[inline(always)]
pub(crate) fn platform_root<R: Float + FloatConst>(turn: R) -> (R, R) {
    let angle = R::from(2).unwrap() * R::PI() * turn;
    (angle.cos(), angle.sin())
}

pub trait ReimFFTExecute<D, T> {
    fn reim_dft_execute(table: &D, data: &mut [T]);
}

pub trait ReimArith {
    fn reim_from_znx(res: &mut [f64], a: &[i64]) {
        reim_from_znx_i64_portable(res, a)
    }

    fn reim_to_znx(res: &mut [i64], divisor: f64, a: &[f64]) {
        reim_to_znx_i64_portable(res, divisor, a)
    }

    fn reim_to_znx_assign(res: &mut [f64], divisor: f64) {
        reim_to_znx_i64_assign_portable(res, divisor)
    }

    fn reim_add(res: &mut [f64], a: &[f64], b: &[f64]) {
        reim_add_portable(res, a, b)
    }

    fn reim_add_assign(res: &mut [f64], a: &[f64]) {
        reim_add_assign_portable(res, a)
    }

    fn reim_sub(res: &mut [f64], a: &[f64], b: &[f64]) {
        reim_sub_portable(res, a, b)
    }

    fn reim_sub_assign(res: &mut [f64], a: &[f64]) {
        reim_sub_assign_portable(res, a)
    }

    fn reim_sub_negate_assign(res: &mut [f64], a: &[f64]) {
        reim_sub_negate_assign_portable(res, a)
    }

    fn reim_negate(res: &mut [f64], a: &[f64]) {
        reim_negate_portable(res, a)
    }

    fn reim_negate_assign(res: &mut [f64]) {
        reim_negate_assign_portable(res)
    }

    fn reim_mul(res: &mut [f64], a: &[f64], b: &[f64]) {
        reim_mul_portable(res, a, b)
    }

    fn reim_mul_assign(res: &mut [f64], a: &[f64]) {
        reim_mul_assign_portable(res, a)
    }

    fn reim_real_mul(res: &mut [f64], a: &[f64], b: &[f64]) {
        reim_real_mul_portable(res, a, b)
    }

    fn reim_real_mul_assign(res: &mut [f64], a: &[f64]) {
        reim_real_mul_assign_portable(res, a)
    }

    fn reim_real_addmul(res: &mut [f64], a: &[f64], b: &[f64]) {
        reim_real_addmul_portable(res, a, b)
    }

    fn reim_addmul(res: &mut [f64], a: &[f64], b: &[f64]) {
        reim_addmul_portable(res, a, b)
    }

    fn reim_copy(res: &mut [f64], a: &[f64]) {
        reim_copy_portable(res, a)
    }

    fn reim_zero(res: &mut [f64]) {
        reim_zero_portable(res)
    }

    /// Complex-slot permutation of one limb: `res = tau_p(a)`.
    fn reim_automorphism(plan: &crate::kernels::fft64::vec_znx_dft::Fft64AutomorphismPlan, res: &mut [f64], a: &[f64]) {
        crate::kernels::fft64::standard::fft64_automorphism_portable(plan, res, a)
    }

    /// Complex-slot permutation of one limb: `res += tau_p(a)`.
    fn reim_automorphism_add(plan: &crate::kernels::fft64::vec_znx_dft::Fft64AutomorphismPlan, res: &mut [f64], a: &[f64]) {
        crate::kernels::fft64::standard::fft64_automorphism_add_portable(plan, res, a)
    }
}
