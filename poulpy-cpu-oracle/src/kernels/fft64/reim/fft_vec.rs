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

pub fn reim_add_ref(res: &mut [f64], a: &[f64], b: &[f64]) {
    {
        assert_eq!(a.len(), res.len());
        assert_eq!(b.len(), res.len());
    }

    for i in 0..res.len() {
        res[i] = a[i] + b[i]
    }
}

pub fn reim_add_assign_ref(res: &mut [f64], a: &[f64]) {
    {
        assert_eq!(a.len(), res.len());
    }

    for i in 0..res.len() {
        res[i] += a[i]
    }
}

pub fn reim_sub_ref(res: &mut [f64], a: &[f64], b: &[f64]) {
    {
        assert_eq!(a.len(), res.len());
        assert_eq!(b.len(), res.len());
    }

    for i in 0..res.len() {
        res[i] = a[i] - b[i]
    }
}

pub fn reim_sub_assign_ref(res: &mut [f64], a: &[f64]) {
    {
        assert_eq!(a.len(), res.len());
    }

    for i in 0..res.len() {
        res[i] -= a[i]
    }
}

pub fn reim_sub_negate_assign_ref(res: &mut [f64], a: &[f64]) {
    {
        assert_eq!(a.len(), res.len());
    }

    for i in 0..res.len() {
        res[i] = a[i] - res[i]
    }
}

pub fn reim_negate_ref(res: &mut [f64], a: &[f64]) {
    {
        assert_eq!(a.len(), res.len());
    }

    for i in 0..res.len() {
        res[i] = -a[i]
    }
}

pub fn reim_negate_assign_ref(res: &mut [f64]) {
    for ri in res {
        *ri = -*ri
    }
}
