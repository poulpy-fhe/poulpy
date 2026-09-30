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

mod conversion;
mod fft_ref;
mod fft_vec;
mod ifft_ref;
mod table_fft;
mod table_ifft;
mod zero;

pub use conversion::*;
pub use fft_ref::*;
pub use fft_vec::*;
pub use ifft_ref::*;
pub use table_fft::*;
pub use table_ifft::*;
pub use zero::*;
pub trait ReimFFTExecute<D, T> {
    fn reim_dft_execute(table: &D, data: &mut [T]);
}

pub trait ReimArith {
    fn reim_from_znx(res: &mut [f64], a: &[i64]) {
        reim_from_znx_i64_ref(res, a)
    }

    fn reim_to_znx(res: &mut [i64], divisor: f64, a: &[f64]) {
        reim_to_znx_i64_ref(res, divisor, a)
    }

    fn reim_add(res: &mut [f64], a: &[f64], b: &[f64]) {
        reim_add_ref(res, a, b)
    }

    fn reim_add_assign(res: &mut [f64], a: &[f64]) {
        reim_add_assign_ref(res, a)
    }

    fn reim_sub(res: &mut [f64], a: &[f64], b: &[f64]) {
        reim_sub_ref(res, a, b)
    }

    fn reim_sub_assign(res: &mut [f64], a: &[f64]) {
        reim_sub_assign_ref(res, a)
    }

    fn reim_sub_negate_assign(res: &mut [f64], a: &[f64]) {
        reim_sub_negate_assign_ref(res, a)
    }

    fn reim_negate(res: &mut [f64], a: &[f64]) {
        reim_negate_ref(res, a)
    }

    fn reim_negate_assign(res: &mut [f64]) {
        reim_negate_assign_ref(res)
    }

    fn reim_copy(res: &mut [f64], a: &[f64]) {
        reim_copy_ref(res, a)
    }

    fn reim_zero(res: &mut [f64]) {
        reim_zero_ref(res)
    }
}
