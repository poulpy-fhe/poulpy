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
pub fn reim_add_ref(res: &mut [f64], a: &[f64], b: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
        assert_eq!(b.len(), res.len());
    }

    for i in 0..res.len() {
        res[i] = a[i] + b[i]
    }
}

#[inline(always)]
pub fn reim_add_assign_ref(res: &mut [f64], a: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
    }

    for i in 0..res.len() {
        res[i] += a[i]
    }
}

#[inline(always)]
pub fn reim_sub_ref(res: &mut [f64], a: &[f64], b: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
        assert_eq!(b.len(), res.len());
    }

    for i in 0..res.len() {
        res[i] = a[i] - b[i]
    }
}

#[inline(always)]
pub fn reim_sub_assign_ref(res: &mut [f64], a: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
    }

    for i in 0..res.len() {
        res[i] -= a[i]
    }
}

#[inline(always)]
pub fn reim_sub_negate_assign_ref(res: &mut [f64], a: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
    }

    for i in 0..res.len() {
        res[i] = a[i] - res[i]
    }
}

#[inline(always)]
pub fn reim_negate_ref(res: &mut [f64], a: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
    }

    for i in 0..res.len() {
        res[i] = -a[i]
    }
}

#[inline(always)]
pub fn reim_negate_assign_ref(res: &mut [f64]) {
    for ri in res {
        *ri = -*ri
    }
}

#[inline(always)]
pub fn reim_addmul_ref(res: &mut [f64], a: &[f64], b: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
        assert_eq!(b.len(), res.len());
    }

    let m: usize = res.len() >> 1;

    let (rr, ri) = res.split_at_mut(m);
    let (ar, ai) = a.split_at(m);
    let (br, bi) = b.split_at(m);

    for i in 0..m {
        let _ar: f64 = ar[i];
        let _ai: f64 = ai[i];
        let _br: f64 = br[i];
        let _bi: f64 = bi[i];
        let _rr: f64 = _ar * _br - _ai * _bi;
        let _ri: f64 = _ar * _bi + _ai * _br;
        rr[i] += _rr;
        ri[i] += _ri;
    }
}

#[inline(always)]
pub fn reim_mul_assign_ref(res: &mut [f64], a: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
    }

    let m: usize = res.len() >> 1;

    let (rr, ri) = res.split_at_mut(m);
    let (ar, ai) = a.split_at(m);

    for i in 0..m {
        let _ar: f64 = ar[i];
        let _ai: f64 = ai[i];
        let _br: f64 = rr[i];
        let _bi: f64 = ri[i];
        let _rr: f64 = _ar * _br - _ai * _bi;
        let _ri: f64 = _ar * _bi + _ai * _br;
        rr[i] = _rr;
        ri[i] = _ri;
    }
}

#[inline(always)]
pub fn reim_mul_ref(res: &mut [f64], a: &[f64], b: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
        assert_eq!(b.len(), res.len());
    }

    let m: usize = res.len() >> 1;

    let (rr, ri) = res.split_at_mut(m);
    let (ar, ai) = a.split_at(m);
    let (br, bi) = b.split_at(m);

    for i in 0..m {
        let _ar: f64 = ar[i];
        let _ai: f64 = ai[i];
        let _br: f64 = br[i];
        let _bi: f64 = bi[i];
        let _rr: f64 = _ar * _br - _ai * _bi;
        let _ri: f64 = _ar * _bi + _ai * _br;
        rr[i] = _rr;
        ri[i] = _ri;
    }
}
