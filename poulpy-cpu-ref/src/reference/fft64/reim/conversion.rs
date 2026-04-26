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

#[inline(always)]
pub fn reim_from_znx_i64_ref(res: &mut [f64], a: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len())
    }

    for i in 0..res.len() {
        res[i] = a[i] as f64
    }
}

#[inline(always)]
pub fn reim_from_znx_i64_masked_ref(res: &mut [f64], a: &[i64], mask: i64) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len())
    }

    for i in 0..res.len() {
        res[i] = (a[i] & mask) as f64
    }
}

#[inline(always)]
pub fn reim_to_znx_i64_ref(res: &mut [i64], divisor: f64, a: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len())
    }
    let inv_div = 1. / divisor;
    for i in 0..res.len() {
        res[i] = (a[i] * inv_div).round() as i64
    }
}

#[inline(always)]
pub fn reim_to_znx_i64_assign_ref(res: &mut [f64], divisor: f64) {
    let inv_div = 1. / divisor;
    for ri in res {
        *ri = f64::from_bits(((*ri * inv_div).round() as i64) as u64)
    }
}
