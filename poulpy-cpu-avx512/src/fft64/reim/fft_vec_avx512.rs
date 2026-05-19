// ----------------------------------------------------------------------
// DISCLAIMER
//
// This module contains code adapted from the AVX2 / FMA C kernels of the
// spqlios-arithmetic library
// (https://github.com/tfhe/spqlios-arithmetic), which is licensed
// under the Apache License, Version 2.0.
//
// The 256-bit AVX2 originals were widened to 512-bit AVX-512 and translated
// to Rust intrinsics; algorithmic structure is preserved one-to-one with the
// spqlios sources to keep semantics identical.
//
// Both Poulpy and spqlios-arithmetic are distributed under the terms
// of the Apache License, Version 2.0. See the LICENSE file for details.
//
// ----------------------------------------------------------------------

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`).
#[target_feature(enable = "avx512f")]
pub unsafe fn reim_add_avx512(res: &mut [f64], a: &[f64], b: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
        assert_eq!(b.len(), res.len());
    }

    use std::arch::x86_64::{__m512d, _mm512_add_pd, _mm512_loadu_pd, _mm512_storeu_pd};

    let span: usize = res.len() >> 3;

    unsafe {
        let mut rr: *mut f64 = res.as_mut_ptr();
        let mut aa: *const f64 = a.as_ptr();
        let mut bb: *const f64 = b.as_ptr();

        for _ in 0..span {
            let a_512: __m512d = _mm512_loadu_pd(aa);
            let b_512: __m512d = _mm512_loadu_pd(bb);
            _mm512_storeu_pd(rr, _mm512_add_pd(a_512, b_512));
            rr = rr.add(8);
            aa = aa.add(8);
            bb = bb.add(8);
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`).
#[target_feature(enable = "avx512f")]
pub unsafe fn reim_add_assign_avx512(res: &mut [f64], a: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
    }

    use std::arch::x86_64::{__m512d, _mm512_add_pd, _mm512_loadu_pd, _mm512_storeu_pd};

    let span: usize = res.len() >> 3;

    unsafe {
        let mut rr: *mut f64 = res.as_mut_ptr();
        let mut aa: *const f64 = a.as_ptr();

        for _ in 0..span {
            let a_512: __m512d = _mm512_loadu_pd(aa);
            let r_512: __m512d = _mm512_loadu_pd(rr);
            _mm512_storeu_pd(rr, _mm512_add_pd(r_512, a_512));
            rr = rr.add(8);
            aa = aa.add(8);
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`).
#[target_feature(enable = "avx512f")]
pub unsafe fn reim_sub_avx512(res: &mut [f64], a: &[f64], b: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
        assert_eq!(b.len(), res.len());
    }

    use std::arch::x86_64::{__m512d, _mm512_loadu_pd, _mm512_storeu_pd, _mm512_sub_pd};

    let span: usize = res.len() >> 3;

    unsafe {
        let mut rr: *mut f64 = res.as_mut_ptr();
        let mut aa: *const f64 = a.as_ptr();
        let mut bb: *const f64 = b.as_ptr();

        for _ in 0..span {
            let a_512: __m512d = _mm512_loadu_pd(aa);
            let b_512: __m512d = _mm512_loadu_pd(bb);
            _mm512_storeu_pd(rr, _mm512_sub_pd(a_512, b_512));
            rr = rr.add(8);
            aa = aa.add(8);
            bb = bb.add(8);
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`).
#[target_feature(enable = "avx512f")]
pub unsafe fn reim_sub_assign_avx512(res: &mut [f64], a: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
    }

    use std::arch::x86_64::{__m512d, _mm512_loadu_pd, _mm512_storeu_pd, _mm512_sub_pd};

    let span: usize = res.len() >> 3;

    unsafe {
        let mut rr: *mut f64 = res.as_mut_ptr();
        let mut aa: *const f64 = a.as_ptr();

        for _ in 0..span {
            let a_512: __m512d = _mm512_loadu_pd(aa);
            let r_512: __m512d = _mm512_loadu_pd(rr);
            _mm512_storeu_pd(rr, _mm512_sub_pd(r_512, a_512));
            rr = rr.add(8);
            aa = aa.add(8);
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`).
#[target_feature(enable = "avx512f")]
pub unsafe fn reim_sub_negate_assign_avx512(res: &mut [f64], a: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
    }

    use std::arch::x86_64::{__m512d, _mm512_loadu_pd, _mm512_storeu_pd, _mm512_sub_pd};

    let span: usize = res.len() >> 3;

    unsafe {
        let mut rr: *mut f64 = res.as_mut_ptr();
        let mut aa: *const f64 = a.as_ptr();

        for _ in 0..span {
            let a_512: __m512d = _mm512_loadu_pd(aa);
            let r_512: __m512d = _mm512_loadu_pd(rr);
            _mm512_storeu_pd(rr, _mm512_sub_pd(a_512, r_512));
            rr = rr.add(8);
            aa = aa.add(8);
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`).
#[target_feature(enable = "avx512f")]
pub unsafe fn reim_negate_avx512(res: &mut [f64], a: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
    }

    use std::arch::x86_64::{__m512d, _mm512_loadu_pd, _mm512_setzero_pd, _mm512_storeu_pd, _mm512_sub_pd};

    let span: usize = res.len() >> 3;

    unsafe {
        let mut rr: *mut f64 = res.as_mut_ptr();
        let mut aa: *const f64 = a.as_ptr();

        let zero: __m512d = _mm512_setzero_pd();

        for _ in 0..span {
            let a_512: __m512d = _mm512_loadu_pd(aa);
            _mm512_storeu_pd(rr, _mm512_sub_pd(zero, a_512));
            rr = rr.add(8);
            aa = aa.add(8);
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`).
#[target_feature(enable = "avx512f")]
pub unsafe fn reim_negate_assign_avx512(res: &mut [f64]) {
    use std::arch::x86_64::{__m512d, _mm512_loadu_pd, _mm512_setzero_pd, _mm512_storeu_pd, _mm512_sub_pd};

    let span: usize = res.len() >> 3;

    unsafe {
        let mut rr: *mut f64 = res.as_mut_ptr();
        let zero: __m512d = _mm512_setzero_pd();

        for _ in 0..span {
            let r_512: __m512d = _mm512_loadu_pd(rr);
            _mm512_storeu_pd(rr, _mm512_sub_pd(zero, r_512));
            rr = rr.add(8);
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`).
#[target_feature(enable = "avx512f")]
pub unsafe fn reim_addmul_avx512(res: &mut [f64], a: &[f64], b: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
        assert_eq!(b.len(), res.len());
    }

    let m: usize = res.len() >> 1;

    let (rr, ri) = res.split_at_mut(m);
    let (ar, ai) = a.split_at(m);
    let (br, bi) = b.split_at(m);

    unsafe {
        let mut rr_ptr: *mut f64 = rr.as_mut_ptr();
        let mut ri_ptr: *mut f64 = ri.as_mut_ptr();
        let mut ar_ptr: *const f64 = ar.as_ptr();
        let mut ai_ptr: *const f64 = ai.as_ptr();
        let mut br_ptr: *const f64 = br.as_ptr();
        let mut bi_ptr: *const f64 = bi.as_ptr();

        use std::arch::x86_64::{__m512d, _mm512_fmadd_pd, _mm512_fmsub_pd, _mm512_loadu_pd, _mm512_storeu_pd};

        for _ in 0..(m >> 3) {
            let mut rr: __m512d = _mm512_loadu_pd(rr_ptr);
            let mut ri: __m512d = _mm512_loadu_pd(ri_ptr);
            let ar: __m512d = _mm512_loadu_pd(ar_ptr);
            let ai: __m512d = _mm512_loadu_pd(ai_ptr);
            let br: __m512d = _mm512_loadu_pd(br_ptr);
            let bi: __m512d = _mm512_loadu_pd(bi_ptr);

            rr = _mm512_fmsub_pd(ai, bi, rr);
            rr = _mm512_fmsub_pd(ar, br, rr);
            ri = _mm512_fmadd_pd(ar, bi, ri);
            ri = _mm512_fmadd_pd(ai, br, ri);

            _mm512_storeu_pd(rr_ptr, rr);
            _mm512_storeu_pd(ri_ptr, ri);

            rr_ptr = rr_ptr.add(8);
            ri_ptr = ri_ptr.add(8);
            ar_ptr = ar_ptr.add(8);
            ai_ptr = ai_ptr.add(8);
            br_ptr = br_ptr.add(8);
            bi_ptr = bi_ptr.add(8);
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`).
#[target_feature(enable = "avx512f")]
pub unsafe fn reim_mul_avx512(res: &mut [f64], a: &[f64], b: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
        assert_eq!(b.len(), res.len());
    }

    let m: usize = res.len() >> 1;

    let (rr, ri) = res.split_at_mut(m);
    let (ar, ai) = a.split_at(m);
    let (br, bi) = b.split_at(m);

    unsafe {
        let mut rr_ptr: *mut f64 = rr.as_mut_ptr();
        let mut ri_ptr: *mut f64 = ri.as_mut_ptr();
        let mut ar_ptr: *const f64 = ar.as_ptr();
        let mut ai_ptr: *const f64 = ai.as_ptr();
        let mut br_ptr: *const f64 = br.as_ptr();
        let mut bi_ptr: *const f64 = bi.as_ptr();

        use std::arch::x86_64::{__m512d, _mm512_fmadd_pd, _mm512_fmsub_pd, _mm512_loadu_pd, _mm512_mul_pd, _mm512_storeu_pd};

        for _ in 0..(m >> 3) {
            let ar: __m512d = _mm512_loadu_pd(ar_ptr);
            let ai: __m512d = _mm512_loadu_pd(ai_ptr);
            let br: __m512d = _mm512_loadu_pd(br_ptr);
            let bi: __m512d = _mm512_loadu_pd(bi_ptr);

            let t1: __m512d = _mm512_mul_pd(ai, bi);
            let t2: __m512d = _mm512_mul_pd(ar, bi);

            let rr: __m512d = _mm512_fmsub_pd(ar, br, t1);
            let ri: __m512d = _mm512_fmadd_pd(ai, br, t2);

            _mm512_storeu_pd(rr_ptr, rr);
            _mm512_storeu_pd(ri_ptr, ri);

            rr_ptr = rr_ptr.add(8);
            ri_ptr = ri_ptr.add(8);
            ar_ptr = ar_ptr.add(8);
            ai_ptr = ai_ptr.add(8);
            br_ptr = br_ptr.add(8);
            bi_ptr = bi_ptr.add(8);
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`).
#[target_feature(enable = "avx512f")]
pub unsafe fn reim_mul_assign_avx512(res: &mut [f64], a: &[f64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.len(), res.len());
    }

    let m: usize = res.len() >> 1;

    let (rr, ri) = res.split_at_mut(m);
    let (ar, ai) = a.split_at(m);

    unsafe {
        let mut rr_ptr: *mut f64 = rr.as_mut_ptr();
        let mut ri_ptr: *mut f64 = ri.as_mut_ptr();
        let mut ar_ptr: *const f64 = ar.as_ptr();
        let mut ai_ptr: *const f64 = ai.as_ptr();

        use std::arch::x86_64::{__m512d, _mm512_fmadd_pd, _mm512_fmsub_pd, _mm512_loadu_pd, _mm512_mul_pd, _mm512_storeu_pd};

        for _ in 0..(m >> 3) {
            let ar: __m512d = _mm512_loadu_pd(ar_ptr);
            let ai: __m512d = _mm512_loadu_pd(ai_ptr);
            let br: __m512d = _mm512_loadu_pd(rr_ptr);
            let bi: __m512d = _mm512_loadu_pd(ri_ptr);

            let t1: __m512d = _mm512_mul_pd(ai, bi);
            let t2: __m512d = _mm512_mul_pd(ar, bi);

            let rr = _mm512_fmsub_pd(ar, br, t1);
            let ri = _mm512_fmadd_pd(ai, br, t2);

            _mm512_storeu_pd(rr_ptr, rr);
            _mm512_storeu_pd(ri_ptr, ri);

            rr_ptr = rr_ptr.add(8);
            ri_ptr = ri_ptr.add(8);
            ar_ptr = ar_ptr.add(8);
            ai_ptr = ai_ptr.add(8);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, target_feature = "avx512f"))]
mod tests {
    use poulpy_cpu_ref::reference::fft64::reim::{
        reim_add_ref, reim_addmul_ref, reim_mul_ref, reim_negate_ref, reim_sub_negate_assign_ref, reim_sub_ref,
    };

    use super::*;

    fn reim_data(n: usize, seed: f64) -> Vec<f64> {
        (0..n).map(|i| (i as f64 * seed + 0.5) / n as f64).collect()
    }

    #[test]
    fn reim_add_avx512f_vs_ref() {
        let n = 64usize;
        let a = reim_data(n, 1.7);
        let b = reim_data(n, 2.3);
        let mut res_avx512 = vec![0f64; n];
        let mut res_ref = vec![0f64; n];
        unsafe { reim_add_avx512(&mut res_avx512, &a, &b) };
        reim_add_ref(&mut res_ref, &a, &b);
        assert_eq!(res_avx512, res_ref, "reim_add: AVX-512 vs ref mismatch");
    }

    #[test]
    fn reim_sub_avx512f_vs_ref() {
        let n = 64usize;
        let a = reim_data(n, 3.1);
        let b = reim_data(n, 1.4);
        let mut res_avx512 = vec![0f64; n];
        let mut res_ref = vec![0f64; n];
        unsafe { reim_sub_avx512(&mut res_avx512, &a, &b) };
        reim_sub_ref(&mut res_ref, &a, &b);
        assert_eq!(res_avx512, res_ref, "reim_sub: AVX-512 vs ref mismatch");
    }

    #[test]
    fn reim_negate_avx512f_vs_ref() {
        let n = 64usize;
        let a = reim_data(n, 2.9);
        let mut res_avx512 = vec![0f64; n];
        let mut res_ref = vec![0f64; n];
        unsafe { reim_negate_avx512(&mut res_avx512, &a) };
        reim_negate_ref(&mut res_ref, &a);
        assert_eq!(res_avx512, res_ref, "reim_negate: AVX-512 vs ref mismatch");
    }

    #[test]
    fn reim_mul_avx512f_vs_ref() {
        let n = 64usize;
        let a = reim_data(n, 1.3);
        let b = reim_data(n, 2.7);
        let mut res_avx512 = vec![0f64; n];
        let mut res_ref = vec![0f64; n];
        unsafe { reim_mul_avx512(&mut res_avx512, &a, &b) };
        reim_mul_ref(&mut res_ref, &a, &b);
        let tol = 1e-14f64;
        for i in 0..n {
            assert!(
                (res_avx512[i] - res_ref[i]).abs() <= tol,
                "reim_mul idx={i}: AVX-512={} ref={}",
                res_avx512[i],
                res_ref[i]
            );
        }
    }

    #[test]
    fn reim_addmul_avx512f_vs_ref() {
        let n = 64usize;
        let a = reim_data(n, 1.1);
        let b = reim_data(n, 2.2);
        let init = reim_data(n, 0.9);
        let mut res_avx512 = init.clone();
        let mut res_ref = init.clone();
        unsafe { reim_addmul_avx512(&mut res_avx512, &a, &b) };
        reim_addmul_ref(&mut res_ref, &a, &b);
        let tol = 1e-14f64;
        for i in 0..n {
            assert!(
                (res_avx512[i] - res_ref[i]).abs() <= tol,
                "reim_addmul idx={i}: AVX-512={} ref={}",
                res_avx512[i],
                res_ref[i]
            );
        }
    }

    #[test]
    fn reim_sub_negate_assign_avx512f_vs_ref() {
        let n = 64usize;
        let a = reim_data(n, 1.8);
        let init = reim_data(n, 3.3);
        let mut res_avx512 = init.clone();
        let mut res_ref = init.clone();
        unsafe { reim_sub_negate_assign_avx512(&mut res_avx512, &a) };
        reim_sub_negate_assign_ref(&mut res_ref, &a);
        assert_eq!(res_avx512, res_ref, "reim_sub_negate_assign: AVX-512 vs ref mismatch");
    }
}
