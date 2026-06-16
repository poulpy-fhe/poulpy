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

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[target_feature(enable = "avx")]
pub fn reim4_extract_1blk_from_reim_contiguous_avx(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
    use core::arch::x86_64::{__m256d, _mm256_loadu_pd, _mm256_storeu_pd};

    unsafe {
        let mut src_ptr: *const __m256d = src.as_ptr().add(blk << 2) as *const __m256d; // src + 4*blk
        let mut dst_ptr: *mut __m256d = dst.as_mut_ptr() as *mut __m256d;

        let step: usize = m >> 2;

        // Each iteration copies 4 doubles; advance src by m doubles each row
        for _ in 0..2 * rows {
            let v: __m256d = _mm256_loadu_pd(src_ptr as *const f64);
            _mm256_storeu_pd(dst_ptr as *mut f64, v);
            dst_ptr = dst_ptr.add(1);
            src_ptr = src_ptr.add(step);
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[target_feature(enable = "avx")]
pub fn reim4_save_1blk_to_reim_contiguous_avx(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
    use core::arch::x86_64::{__m256d, _mm256_loadu_pd, _mm256_storeu_pd};

    unsafe {
        let mut src_ptr: *const __m256d = src.as_ptr() as *const __m256d;
        let mut dst_ptr: *mut __m256d = dst.as_mut_ptr().add(blk << 2) as *mut __m256d; // dst + 4*blk

        let step: usize = m >> 2;

        // Each iteration copies 4 doubles; advance dst by m doubles each row
        for _ in 0..2 * rows {
            let v: __m256d = _mm256_loadu_pd(src_ptr as *const f64);
            _mm256_storeu_pd(dst_ptr as *mut f64, v);
            dst_ptr = dst_ptr.add(step);
            src_ptr = src_ptr.add(1);
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[target_feature(enable = "avx2,fma")]
pub fn reim4_save_1blk_to_reim_avx<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
    use core::arch::x86_64::{__m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_storeu_pd};
    unsafe {
        let off: usize = blk * 4;
        let src_ptr: *const f64 = src.as_ptr();

        let s0: __m256d = _mm256_loadu_pd(src_ptr);
        let s1: __m256d = _mm256_loadu_pd(src_ptr.add(4));

        let d0_ptr: *mut f64 = dst.as_mut_ptr().add(off);
        let d1_ptr: *mut f64 = d0_ptr.add(m);

        if OVERWRITE {
            // Use normal stores: the produced data is reused immediately by the
            // next operation in the pipeline (e.g. VMP -> SVP/add/sub), so
            // non-temporal stores would evict still-hot cache lines.
            _mm256_storeu_pd(d0_ptr, s0);
            _mm256_storeu_pd(d1_ptr, s1);
        } else {
            let d0: __m256d = _mm256_loadu_pd(d0_ptr);
            let d1: __m256d = _mm256_loadu_pd(d1_ptr);
            _mm256_storeu_pd(d0_ptr, _mm256_add_pd(d0, s0));
            _mm256_storeu_pd(d1_ptr, _mm256_add_pd(d1, s1));
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[target_feature(enable = "avx2,fma")]
pub fn reim4_save_2blk_to_reim_avx<const OVERWRITE: bool>(
    m: usize,        //
    blk: usize,      // block index
    dst: &mut [f64], //
    src: &[f64],     // 16 doubles [re1(4), im1(4), re2(4), im2(4)]
) {
    use core::arch::x86_64::{__m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_storeu_pd};
    unsafe {
        let off: usize = blk * 4;
        let src_ptr: *const f64 = src.as_ptr();

        let d0_ptr: *mut f64 = dst.as_mut_ptr().add(off);
        let d1_ptr: *mut f64 = d0_ptr.add(m);
        let d2_ptr: *mut f64 = d1_ptr.add(m);
        let d3_ptr: *mut f64 = d2_ptr.add(m);

        let s0: __m256d = _mm256_loadu_pd(src_ptr);
        let s1: __m256d = _mm256_loadu_pd(src_ptr.add(4));
        let s2: __m256d = _mm256_loadu_pd(src_ptr.add(8));
        let s3: __m256d = _mm256_loadu_pd(src_ptr.add(12));

        if OVERWRITE {
            // Use normal stores: the produced data is reused immediately by the
            // next operation in the pipeline (e.g. VMP -> SVP/add/sub), so
            // non-temporal stores would evict still-hot cache lines.
            _mm256_storeu_pd(d0_ptr, s0);
            _mm256_storeu_pd(d1_ptr, s1);
            _mm256_storeu_pd(d2_ptr, s2);
            _mm256_storeu_pd(d3_ptr, s3);
        } else {
            let d0: __m256d = _mm256_loadu_pd(d0_ptr);
            let d1: __m256d = _mm256_loadu_pd(d1_ptr);
            let d2: __m256d = _mm256_loadu_pd(d2_ptr);
            let d3: __m256d = _mm256_loadu_pd(d3_ptr);
            _mm256_storeu_pd(d0_ptr, _mm256_add_pd(d0, s0));
            _mm256_storeu_pd(d1_ptr, _mm256_add_pd(d1, s1));
            _mm256_storeu_pd(d2_ptr, _mm256_add_pd(d2, s2));
            _mm256_storeu_pd(d3_ptr, _mm256_add_pd(d3, s3));
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[target_feature(enable = "avx2", enable = "fma")]
pub fn reim4_vec_mat1col_product_avx(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
    use core::arch::x86_64::{__m256d, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_setzero_pd, _mm256_storeu_pd};

    #[cfg(debug_assertions)]
    {
        assert!(dst.len() >= 8, "dst must have at least 8 doubles");
        assert!(u.len() >= nrows * 8, "u must be at least nrows * 8 doubles");
        assert!(v.len() >= nrows * 8, "v must be at least nrows * 8 doubles");
    }

    unsafe {
        use std::arch::x86_64::{_mm256_add_pd, _mm256_sub_pd};

        let mut re1: __m256d = _mm256_setzero_pd();
        let mut im1: __m256d = _mm256_setzero_pd();
        let mut re2: __m256d = _mm256_setzero_pd();
        let mut im2: __m256d = _mm256_setzero_pd();

        let mut u_ptr: *const f64 = u.as_ptr();
        let mut v_ptr: *const f64 = v.as_ptr();

        for _ in 0..nrows {
            let ur: __m256d = _mm256_loadu_pd(u_ptr);
            let ui: __m256d = _mm256_loadu_pd(u_ptr.add(4));
            let vr: __m256d = _mm256_loadu_pd(v_ptr);
            let vi: __m256d = _mm256_loadu_pd(v_ptr.add(4));

            // re1 = re1 + ur*vr;
            re1 = _mm256_fmadd_pd(ur, vr, re1);
            // im1 = im1 + ur*d;
            im1 = _mm256_fmadd_pd(ur, vi, im1);
            // re2 = re2 + ui*d;
            re2 = _mm256_fmadd_pd(ui, vi, re2);
            // im2 = im2 + ui*vr;
            im2 = _mm256_fmadd_pd(ui, vr, im2);

            u_ptr = u_ptr.add(8);
            v_ptr = v_ptr.add(8);
        }

        // re1 - re2
        _mm256_storeu_pd(dst.as_mut_ptr(), _mm256_sub_pd(re1, re2));

        // im1 + im2
        _mm256_storeu_pd(dst.as_mut_ptr().add(4), _mm256_add_pd(im1, im2));
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[target_feature(enable = "avx2", enable = "fma")]
pub fn reim4_vec_mat2cols_product_avx(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
    use core::arch::x86_64::{__m256d, _mm256_fmadd_pd, _mm256_fmsub_pd, _mm256_loadu_pd, _mm256_setzero_pd, _mm256_storeu_pd};

    #[cfg(debug_assertions)]
    {
        assert!(dst.len() >= 8, "dst must be at least 8 doubles but is {}", dst.len());
        assert!(
            u.len() >= nrows * 8,
            "u must be at least nrows={} * 8 doubles but is {}",
            nrows,
            u.len()
        );
        assert!(
            v.len() >= nrows * 16,
            "v must be at least nrows={} * 16 doubles but is {}",
            nrows,
            v.len()
        );
    }

    unsafe {
        let mut re1: __m256d = _mm256_setzero_pd();
        let mut im1: __m256d = _mm256_setzero_pd();
        let mut re2: __m256d = _mm256_setzero_pd();
        let mut im2: __m256d = _mm256_setzero_pd();

        let mut u_ptr: *const f64 = u.as_ptr();
        let mut v_ptr: *const f64 = v.as_ptr();

        for _ in 0..nrows {
            let ur: __m256d = _mm256_loadu_pd(u_ptr);
            let ui: __m256d = _mm256_loadu_pd(u_ptr.add(4));

            let ar: __m256d = _mm256_loadu_pd(v_ptr);
            let ai: __m256d = _mm256_loadu_pd(v_ptr.add(4));
            let br: __m256d = _mm256_loadu_pd(v_ptr.add(8));
            let bi: __m256d = _mm256_loadu_pd(v_ptr.add(12));

            // re1 = ui*ai - re1; re2 =  ui*bi - re2;
            re1 = _mm256_fmsub_pd(ui, ai, re1);
            re2 = _mm256_fmsub_pd(ui, bi, re2);
            // im1 = ur*ai + im1; im2 =  ur*bi + im2;
            im1 = _mm256_fmadd_pd(ur, ai, im1);
            im2 = _mm256_fmadd_pd(ur, bi, im2);
            // re1 = ur*ar - re1; re2 = ur*br - re2;
            re1 = _mm256_fmsub_pd(ur, ar, re1);
            re2 = _mm256_fmsub_pd(ur, br, re2);
            // im1 = ui*ar + im1; im2 = ui*br + im2;
            im1 = _mm256_fmadd_pd(ui, ar, im1);
            im2 = _mm256_fmadd_pd(ui, br, im2);

            u_ptr = u_ptr.add(8);
            v_ptr = v_ptr.add(16);
        }

        _mm256_storeu_pd(dst.as_mut_ptr(), re1);
        _mm256_storeu_pd(dst.as_mut_ptr().add(4), im1);
        _mm256_storeu_pd(dst.as_mut_ptr().add(8), re2);
        _mm256_storeu_pd(dst.as_mut_ptr().add(12), im2);
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[target_feature(enable = "avx2", enable = "fma")]
pub fn reim4_vec_mat2cols_2ndcol_product_avx(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
    use core::arch::x86_64::{__m256d, _mm256_fmadd_pd, _mm256_fmsub_pd, _mm256_loadu_pd, _mm256_setzero_pd, _mm256_storeu_pd};

    #[cfg(debug_assertions)]
    {
        assert_eq!(dst.len(), 16, "dst must have 16 doubles");
        assert!(u.len() >= nrows * 8, "u must be at least nrows * 8 doubles");
        assert!(v.len() >= nrows * 16, "v must be at least nrows * 16 doubles");
    }

    unsafe {
        let mut re1: __m256d = _mm256_setzero_pd();
        let mut im1: __m256d = _mm256_setzero_pd();

        let mut u_ptr: *const f64 = u.as_ptr();
        let mut v_ptr: *const f64 = v.as_ptr().add(8); // Offset to 2nd column

        for _ in 0..nrows {
            let ur: __m256d = _mm256_loadu_pd(u_ptr);
            let ui: __m256d = _mm256_loadu_pd(u_ptr.add(4));

            let ar: __m256d = _mm256_loadu_pd(v_ptr);
            let ai: __m256d = _mm256_loadu_pd(v_ptr.add(4));

            // re1 = ui*ai - re1;
            re1 = _mm256_fmsub_pd(ui, ai, re1);
            // im1 = im1 + ur*ai;
            im1 = _mm256_fmadd_pd(ur, ai, im1);
            // re1 = ur*ar - re1;
            re1 = _mm256_fmsub_pd(ur, ar, re1);
            // im1 = im1 + ui*ar;
            im1 = _mm256_fmadd_pd(ui, ar, im1);

            u_ptr = u_ptr.add(8);
            v_ptr = v_ptr.add(16);
        }

        _mm256_storeu_pd(dst.as_mut_ptr(), re1);
        _mm256_storeu_pd(dst.as_mut_ptr().add(4), im1);
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA (e.g. `is_x86_feature_detected!("avx2")`).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn reim4_convolution_1coeff_avx(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
    use core::arch::x86_64::{__m256d, _mm256_fmadd_pd, _mm256_fmsub_pd, _mm256_loadu_pd, _mm256_setzero_pd, _mm256_storeu_pd};

    unsafe {
        // Scalar guard — same semantics as reference implementation
        if k >= a_size + b_size {
            let zero: __m256d = _mm256_setzero_pd();
            let dst_ptr: *mut f64 = dst.as_mut_ptr();
            _mm256_storeu_pd(dst_ptr, zero);
            _mm256_storeu_pd(dst_ptr.add(4), zero);
            return;
        }

        let j_min: usize = k.saturating_sub(a_size - 1);
        let j_max: usize = (k + 1).min(b_size);

        // acc_re = dst[0..4], acc_im = dst[4..8]
        let mut acc_re: __m256d = _mm256_setzero_pd();
        let mut acc_im: __m256d = _mm256_setzero_pd();

        let mut a_ptr: *const f64 = a.as_ptr().add(8 * (k - j_min));
        let mut b_ptr: *const f64 = b.as_ptr().add(8 * j_min);

        for _ in 0..j_max - j_min {
            // Load a[(k - j)]
            let ar: __m256d = _mm256_loadu_pd(a_ptr);
            let ai: __m256d = _mm256_loadu_pd(a_ptr.add(4));

            // Load b[j]
            let br: __m256d = _mm256_loadu_pd(b_ptr);
            let bi: __m256d = _mm256_loadu_pd(b_ptr.add(4));

            // acc_re = ai*bi - acc_re
            acc_re = _mm256_fmsub_pd(ai, bi, acc_re);
            // acc_im = ar*bi - acc_im
            acc_im = _mm256_fmadd_pd(ar, bi, acc_im);
            // acc_re = ar*br - acc_re
            acc_re = _mm256_fmsub_pd(ar, br, acc_re);
            // acc_im = acc_im + ai*br
            acc_im = _mm256_fmadd_pd(ai, br, acc_im);

            a_ptr = a_ptr.sub(8);
            b_ptr = b_ptr.add(8);
        }

        // Store accumulators into dst
        _mm256_storeu_pd(dst.as_mut_ptr(), acc_re);
        _mm256_storeu_pd(dst.as_mut_ptr().add(4), acc_im);
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA (e.g. `is_x86_feature_detected!("avx2")`).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn reim4_convolution_2coeffs_avx(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
    use core::arch::x86_64::{__m256d, _mm256_fmadd_pd, _mm256_fnmadd_pd, _mm256_loadu_pd, _mm256_setzero_pd, _mm256_storeu_pd};

    debug_assert!(a.len() >= 8 * a_size);
    debug_assert!(b.len() >= 8 * b_size);

    let k0: usize = k;
    let k1: usize = k + 1;
    let bound: usize = a_size + b_size;

    // Since k is a multiple of two, if either k0 or k1 are out of range,
    // both are.
    if k0 >= bound {
        unsafe {
            let zero: __m256d = _mm256_setzero_pd();
            let dst_ptr: *mut f64 = dst.as_mut_ptr();
            _mm256_storeu_pd(dst_ptr, zero);
            _mm256_storeu_pd(dst_ptr.add(4), zero);
            _mm256_storeu_pd(dst_ptr.add(8), zero);
            _mm256_storeu_pd(dst_ptr.add(12), zero);
        }
        return;
    }

    unsafe {
        let mut acc_re_k0: __m256d = _mm256_setzero_pd();
        let mut acc_im_k0: __m256d = _mm256_setzero_pd();
        let mut acc_re_k1: __m256d = _mm256_setzero_pd();
        let mut acc_im_k1: __m256d = _mm256_setzero_pd();

        let j0_min: usize = (k0 + 1).saturating_sub(a_size);
        let j0_max: usize = (k0 + 1).min(b_size);

        if k1 >= bound {
            let mut a_k0_ptr: *const f64 = a.as_ptr().add(8 * (k0 - j0_min));
            let mut b_ptr: *const f64 = b.as_ptr().add(8 * j0_min);

            // Region 1: contributions to k0 only, j ∈ [j0_min, j1_min)
            for _ in 0..j0_max - j0_min {
                let ar: __m256d = _mm256_loadu_pd(a_k0_ptr);
                let ai: __m256d = _mm256_loadu_pd(a_k0_ptr.add(4));
                let br: __m256d = _mm256_loadu_pd(b_ptr);
                let bi: __m256d = _mm256_loadu_pd(b_ptr.add(4));

                acc_re_k0 = _mm256_fmadd_pd(ar, br, acc_re_k0);
                acc_re_k0 = _mm256_fnmadd_pd(ai, bi, acc_re_k0);
                acc_im_k0 = _mm256_fmadd_pd(ar, bi, acc_im_k0);
                acc_im_k0 = _mm256_fmadd_pd(ai, br, acc_im_k0);

                a_k0_ptr = a_k0_ptr.sub(8);
                b_ptr = b_ptr.add(8);
            }
        } else {
            let j1_min: usize = (k1 + 1).saturating_sub(a_size);
            let j1_max: usize = (k1 + 1).min(b_size);

            let mut a_k0_ptr: *const f64 = a.as_ptr().add(8 * (k0 - j0_min));
            let mut a_k1_ptr: *const f64 = a.as_ptr().add(8 * (k1 - j1_min));
            let mut b_ptr: *const f64 = b.as_ptr().add(8 * j0_min);

            // Region 1: contributions to k0 only, j ∈ [j0_min, j1_min)
            for _ in 0..j1_min - j0_min {
                let ar: __m256d = _mm256_loadu_pd(a_k0_ptr);
                let ai: __m256d = _mm256_loadu_pd(a_k0_ptr.add(4));
                let br: __m256d = _mm256_loadu_pd(b_ptr);
                let bi: __m256d = _mm256_loadu_pd(b_ptr.add(4));

                acc_re_k0 = _mm256_fmadd_pd(ar, br, acc_re_k0);
                acc_re_k0 = _mm256_fnmadd_pd(ai, bi, acc_re_k0);
                acc_im_k0 = _mm256_fmadd_pd(ar, bi, acc_im_k0);
                acc_im_k0 = _mm256_fmadd_pd(ai, br, acc_im_k0);

                a_k0_ptr = a_k0_ptr.sub(8);
                b_ptr = b_ptr.add(8);
            }

            // Region 2: overlap, contributions to both k0 and k1, j ∈ [j1_min, j0_max)
            // We can save one load on b.
            for _ in 0..j0_max - j1_min {
                let ar0: __m256d = _mm256_loadu_pd(a_k0_ptr);
                let ai0: __m256d = _mm256_loadu_pd(a_k0_ptr.add(4));
                let ar1: __m256d = _mm256_loadu_pd(a_k1_ptr);
                let ai1: __m256d = _mm256_loadu_pd(a_k1_ptr.add(4));
                let br: __m256d = _mm256_loadu_pd(b_ptr);
                let bi: __m256d = _mm256_loadu_pd(b_ptr.add(4));

                // k0
                acc_re_k0 = _mm256_fmadd_pd(ar0, br, acc_re_k0);
                acc_re_k0 = _mm256_fnmadd_pd(ai0, bi, acc_re_k0);
                acc_im_k0 = _mm256_fmadd_pd(ar0, bi, acc_im_k0);
                acc_im_k0 = _mm256_fmadd_pd(ai0, br, acc_im_k0);

                // k1
                acc_re_k1 = _mm256_fmadd_pd(ar1, br, acc_re_k1);
                acc_re_k1 = _mm256_fnmadd_pd(ai1, bi, acc_re_k1);
                acc_im_k1 = _mm256_fmadd_pd(ar1, bi, acc_im_k1);
                acc_im_k1 = _mm256_fmadd_pd(ai1, br, acc_im_k1);

                a_k0_ptr = a_k0_ptr.sub(8);
                a_k1_ptr = a_k1_ptr.sub(8);
                b_ptr = b_ptr.add(8);
            }

            // Region 3: contributions to k1 only, j ∈ [j0_max, j1_max)
            for _ in 0..j1_max - j0_max {
                let ar1: __m256d = _mm256_loadu_pd(a_k1_ptr);
                let ai1: __m256d = _mm256_loadu_pd(a_k1_ptr.add(4));
                let br: __m256d = _mm256_loadu_pd(b_ptr);
                let bi: __m256d = _mm256_loadu_pd(b_ptr.add(4));

                acc_re_k1 = _mm256_fmadd_pd(ar1, br, acc_re_k1);
                acc_re_k1 = _mm256_fnmadd_pd(ai1, bi, acc_re_k1);
                acc_im_k1 = _mm256_fmadd_pd(ar1, bi, acc_im_k1);
                acc_im_k1 = _mm256_fmadd_pd(ai1, br, acc_im_k1);

                a_k1_ptr = a_k1_ptr.sub(8);
                b_ptr = b_ptr.add(8);
            }
        }

        // Store both coefficients
        let dst_ptr = dst.as_mut_ptr();
        _mm256_storeu_pd(dst_ptr, acc_re_k0);
        _mm256_storeu_pd(dst_ptr.add(4), acc_im_k0);
        _mm256_storeu_pd(dst_ptr.add(8), acc_re_k1);
        _mm256_storeu_pd(dst_ptr.add(12), acc_im_k1);
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA (e.g. `is_x86_feature_detected!("avx2")`).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn reim4_convolution_by_real_const_1coeff_avx(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64]) {
    use core::arch::x86_64::{__m256d, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd};

    unsafe {
        let b_size: usize = b.len();

        if k >= a_size + b_size {
            let zero: __m256d = _mm256_setzero_pd();
            let dst_ptr: *mut f64 = dst.as_mut_ptr();
            _mm256_storeu_pd(dst_ptr, zero);
            _mm256_storeu_pd(dst_ptr.add(4), zero);
            return;
        }

        let j_min: usize = k.saturating_sub(a_size - 1);
        let j_max: usize = (k + 1).min(b_size);

        // acc_re = dst[0..4], acc_im = dst[4..8]
        let mut acc_re: __m256d = _mm256_setzero_pd();
        let mut acc_im: __m256d = _mm256_setzero_pd();

        let mut a_ptr: *const f64 = a.as_ptr().add(8 * (k - j_min));
        let mut b_ptr: *const f64 = b.as_ptr().add(j_min);

        for _ in 0..j_max - j_min {
            // Load a[(k - j)]
            let ar: __m256d = _mm256_loadu_pd(a_ptr);
            let ai: __m256d = _mm256_loadu_pd(a_ptr.add(4));

            // Load scalar b[j] and broadcast
            let br: __m256d = _mm256_set1_pd(*b_ptr);

            // Complex * real:
            // re += ar * br
            // im += ai * br
            acc_re = _mm256_fmadd_pd(ar, br, acc_re);
            acc_im = _mm256_fmadd_pd(ai, br, acc_im);

            a_ptr = a_ptr.sub(8);
            b_ptr = b_ptr.add(1);
        }

        // Store accumulators into dst
        _mm256_storeu_pd(dst.as_mut_ptr(), acc_re);
        _mm256_storeu_pd(dst.as_mut_ptr().add(4), acc_im);
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA (e.g. `is_x86_feature_detected!("avx2")`).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn reim4_convolution_by_real_const_2coeffs_avx(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64]) {
    use core::arch::x86_64::{__m256d, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd};

    let b_size: usize = b.len();

    debug_assert!(a.len() >= 8 * a_size);

    let k0: usize = k;
    let k1: usize = k + 1;
    let bound: usize = a_size + b_size;

    // Since k is a multiple of two, if either k0 or k1 are out of range,
    // both are.
    if k0 >= bound {
        unsafe {
            let zero: __m256d = _mm256_setzero_pd();
            let dst_ptr: *mut f64 = dst.as_mut_ptr();
            _mm256_storeu_pd(dst_ptr, zero);
            _mm256_storeu_pd(dst_ptr.add(4), zero);
            _mm256_storeu_pd(dst_ptr.add(8), zero);
            _mm256_storeu_pd(dst_ptr.add(12), zero);
        }
        return;
    }

    unsafe {
        let mut acc_re_k0: __m256d = _mm256_setzero_pd();
        let mut acc_im_k0: __m256d = _mm256_setzero_pd();
        let mut acc_re_k1: __m256d = _mm256_setzero_pd();
        let mut acc_im_k1: __m256d = _mm256_setzero_pd();

        let j0_min: usize = (k0 + 1).saturating_sub(a_size);
        let j0_max: usize = (k0 + 1).min(b_size);

        if k1 >= bound {
            let mut a_k0_ptr: *const f64 = a.as_ptr().add(8 * (k0 - j0_min));
            let mut b_ptr: *const f64 = b.as_ptr().add(j0_min);

            // Contributions to k0 only
            for _ in 0..j0_max - j0_min {
                let ar: __m256d = _mm256_loadu_pd(a_k0_ptr);
                let ai: __m256d = _mm256_loadu_pd(a_k0_ptr.add(4));
                let br: __m256d = _mm256_set1_pd(*b_ptr);

                // complex * real
                acc_re_k0 = _mm256_fmadd_pd(ar, br, acc_re_k0);
                acc_im_k0 = _mm256_fmadd_pd(ai, br, acc_im_k0);

                a_k0_ptr = a_k0_ptr.sub(8);
                b_ptr = b_ptr.add(1);
            }
        } else {
            let j1_min: usize = (k1 + 1).saturating_sub(a_size);
            let j1_max: usize = (k1 + 1).min(b_size);

            let mut a_k0_ptr: *const f64 = a.as_ptr().add(8 * (k0 - j0_min));
            let mut a_k1_ptr: *const f64 = a.as_ptr().add(8 * (k1 - j1_min));
            let mut b_ptr: *const f64 = b.as_ptr().add(j0_min);

            // Region 1: k0 only, j ∈ [j0_min, j1_min)
            for _ in 0..j1_min - j0_min {
                let ar0: __m256d = _mm256_loadu_pd(a_k0_ptr);
                let ai0: __m256d = _mm256_loadu_pd(a_k0_ptr.add(4));
                let br: __m256d = _mm256_set1_pd(*b_ptr);

                acc_re_k0 = _mm256_fmadd_pd(ar0, br, acc_re_k0);
                acc_im_k0 = _mm256_fmadd_pd(ai0, br, acc_im_k0);

                a_k0_ptr = a_k0_ptr.sub(8);
                b_ptr = b_ptr.add(1);
            }

            // Region 2: overlap, contributions to both k0 and k1, j ∈ [j1_min, j0_max)
            // Still “save one load on b”: we broadcast once and reuse.
            for _ in 0..j0_max - j1_min {
                let ar0: __m256d = _mm256_loadu_pd(a_k0_ptr);
                let ai0: __m256d = _mm256_loadu_pd(a_k0_ptr.add(4));
                let ar1: __m256d = _mm256_loadu_pd(a_k1_ptr);
                let ai1: __m256d = _mm256_loadu_pd(a_k1_ptr.add(4));
                let br: __m256d = _mm256_set1_pd(*b_ptr);

                // k0
                acc_re_k0 = _mm256_fmadd_pd(ar0, br, acc_re_k0);
                acc_im_k0 = _mm256_fmadd_pd(ai0, br, acc_im_k0);

                // k1
                acc_re_k1 = _mm256_fmadd_pd(ar1, br, acc_re_k1);
                acc_im_k1 = _mm256_fmadd_pd(ai1, br, acc_im_k1);

                a_k0_ptr = a_k0_ptr.sub(8);
                a_k1_ptr = a_k1_ptr.sub(8);
                b_ptr = b_ptr.add(1);
            }

            // Region 3: k1 only, j ∈ [j0_max, j1_max)
            for _ in 0..j1_max - j0_max {
                let ar1: __m256d = _mm256_loadu_pd(a_k1_ptr);
                let ai1: __m256d = _mm256_loadu_pd(a_k1_ptr.add(4));
                let br: __m256d = _mm256_set1_pd(*b_ptr);

                acc_re_k1 = _mm256_fmadd_pd(ar1, br, acc_re_k1);
                acc_im_k1 = _mm256_fmadd_pd(ai1, br, acc_im_k1);

                a_k1_ptr = a_k1_ptr.sub(8);
                b_ptr = b_ptr.add(1);
            }
        }

        // Store both coefficients
        let dst_ptr = dst.as_mut_ptr();
        _mm256_storeu_pd(dst_ptr, acc_re_k0);
        _mm256_storeu_pd(dst_ptr.add(4), acc_im_k0);
        _mm256_storeu_pd(dst_ptr.add(8), acc_re_k1);
        _mm256_storeu_pd(dst_ptr.add(12), acc_im_k1);
    }
}

/// Full-row convolution, tiling outputs by 3: the widest tile whose working
/// set (two accumulators and one window row pair per output, plus the shared
/// b row, `4T + 2 = 14` registers) fits AVX2's 16 ymm registers without
/// spilling. Per-output accumulation order matches
/// `reim4_convolution_2coeffs_avx`.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA (e.g. `is_x86_feature_detected!("avx2")`).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn reim4_convolution_avx(
    dst: &mut [f64],
    dst_size: usize,
    offset: usize,
    a: &[f64],
    a_size: usize,
    b: &[f64],
    b_size: usize,
) {
    debug_assert!(a_size > 0);
    debug_assert!(b_size > 0);
    debug_assert!(dst.len() >= 8 * dst_size);
    debug_assert!(a.len() >= 8 * a_size);
    debug_assert!(b.len() >= 8 * b_size);

    unsafe {
        let mut k: usize = 0;
        while k + 3 <= dst_size {
            reim4_convolution_tile3_avx(k + offset, dst.as_mut_ptr().add(8 * k), a, a_size, b, b_size);
            k += 3;
        }
        while k < dst_size {
            let dst_blk: &mut [f64; 8] = &mut *(dst.as_mut_ptr().add(8 * k) as *mut [f64; 8]);
            reim4_convolution_1coeff_avx(k + offset, dst_blk, a, a_size, b, b_size);
            k += 1;
        }
    }
}

/// One tile of three consecutive output blocks `k_abs..k_abs+3`: head/tail
/// edges with partial `t` coverage around a full region sliding the `a` window.
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn reim4_convolution_tile3_avx(k_abs: usize, dst: *mut f64, a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
    use core::arch::x86_64::{__m256d, _mm256_fmadd_pd, _mm256_fnmadd_pd, _mm256_loadu_pd, _mm256_setzero_pd, _mm256_storeu_pd};

    unsafe {
        let j_start: usize = (k_abs + 1).saturating_sub(a_size).min(b_size);
        let j_end: usize = (k_abs + 3).min(b_size);
        let j_full_start: usize = (k_abs + 3).saturating_sub(a_size).max(j_start).min(j_end);
        let j_full_end: usize = (k_abs + 1).min(j_end).max(j_full_start);

        let mut acc_re0: __m256d = _mm256_setzero_pd();
        let mut acc_im0: __m256d = _mm256_setzero_pd();
        let mut acc_re1: __m256d = _mm256_setzero_pd();
        let mut acc_im1: __m256d = _mm256_setzero_pd();
        let mut acc_re2: __m256d = _mm256_setzero_pd();
        let mut acc_im2: __m256d = _mm256_setzero_pd();

        let edge = |j: usize,
                    acc_re0: &mut __m256d,
                    acc_im0: &mut __m256d,
                    acc_re1: &mut __m256d,
                    acc_im1: &mut __m256d,
                    acc_re2: &mut __m256d,
                    acc_im2: &mut __m256d| {
            let br: __m256d = _mm256_loadu_pd(b.as_ptr().add(8 * j));
            let bi: __m256d = _mm256_loadu_pd(b.as_ptr().add(8 * j + 4));
            if j <= k_abs && k_abs - j < a_size {
                let a_ptr: *const f64 = a.as_ptr().add(8 * (k_abs - j));
                let ar: __m256d = _mm256_loadu_pd(a_ptr);
                let ai: __m256d = _mm256_loadu_pd(a_ptr.add(4));
                *acc_re0 = _mm256_fmadd_pd(ar, br, *acc_re0);
                *acc_re0 = _mm256_fnmadd_pd(ai, bi, *acc_re0);
                *acc_im0 = _mm256_fmadd_pd(ar, bi, *acc_im0);
                *acc_im0 = _mm256_fmadd_pd(ai, br, *acc_im0);
            }
            if j <= k_abs + 1 && k_abs + 1 - j < a_size {
                let a_ptr: *const f64 = a.as_ptr().add(8 * (k_abs + 1 - j));
                let ar: __m256d = _mm256_loadu_pd(a_ptr);
                let ai: __m256d = _mm256_loadu_pd(a_ptr.add(4));
                *acc_re1 = _mm256_fmadd_pd(ar, br, *acc_re1);
                *acc_re1 = _mm256_fnmadd_pd(ai, bi, *acc_re1);
                *acc_im1 = _mm256_fmadd_pd(ar, bi, *acc_im1);
                *acc_im1 = _mm256_fmadd_pd(ai, br, *acc_im1);
            }
            if j <= k_abs + 2 && k_abs + 2 - j < a_size {
                let a_ptr: *const f64 = a.as_ptr().add(8 * (k_abs + 2 - j));
                let ar: __m256d = _mm256_loadu_pd(a_ptr);
                let ai: __m256d = _mm256_loadu_pd(a_ptr.add(4));
                *acc_re2 = _mm256_fmadd_pd(ar, br, *acc_re2);
                *acc_re2 = _mm256_fnmadd_pd(ai, bi, *acc_re2);
                *acc_im2 = _mm256_fmadd_pd(ar, bi, *acc_im2);
                *acc_im2 = _mm256_fmadd_pd(ai, br, *acc_im2);
            }
        };

        for j in j_start..j_full_start {
            edge(
                j,
                &mut acc_re0,
                &mut acc_im0,
                &mut acc_re1,
                &mut acc_im1,
                &mut acc_re2,
                &mut acc_im2,
            );
        }

        if j_full_start < j_full_end {
            let mut a_ptr: *const f64 = a.as_ptr().add(8 * (k_abs - j_full_start));
            let mut b_ptr: *const f64 = b.as_ptr().add(8 * j_full_start);

            let mut ar0: __m256d = _mm256_loadu_pd(a_ptr);
            let mut ai0: __m256d = _mm256_loadu_pd(a_ptr.add(4));
            let mut ar1: __m256d = _mm256_loadu_pd(a_ptr.add(8));
            let mut ai1: __m256d = _mm256_loadu_pd(a_ptr.add(12));
            let mut ar2: __m256d = _mm256_loadu_pd(a_ptr.add(16));
            let mut ai2: __m256d = _mm256_loadu_pd(a_ptr.add(20));

            let mut j: usize = j_full_start;
            loop {
                let br: __m256d = _mm256_loadu_pd(b_ptr);
                let bi: __m256d = _mm256_loadu_pd(b_ptr.add(4));

                acc_re0 = _mm256_fmadd_pd(ar0, br, acc_re0);
                acc_re0 = _mm256_fnmadd_pd(ai0, bi, acc_re0);
                acc_im0 = _mm256_fmadd_pd(ar0, bi, acc_im0);
                acc_im0 = _mm256_fmadd_pd(ai0, br, acc_im0);
                acc_re1 = _mm256_fmadd_pd(ar1, br, acc_re1);
                acc_re1 = _mm256_fnmadd_pd(ai1, bi, acc_re1);
                acc_im1 = _mm256_fmadd_pd(ar1, bi, acc_im1);
                acc_im1 = _mm256_fmadd_pd(ai1, br, acc_im1);
                acc_re2 = _mm256_fmadd_pd(ar2, br, acc_re2);
                acc_re2 = _mm256_fnmadd_pd(ai2, bi, acc_re2);
                acc_im2 = _mm256_fmadd_pd(ar2, bi, acc_im2);
                acc_im2 = _mm256_fmadd_pd(ai2, br, acc_im2);

                j += 1;
                if j == j_full_end {
                    break;
                }
                ar2 = ar1;
                ai2 = ai1;
                ar1 = ar0;
                ai1 = ai0;
                a_ptr = a_ptr.sub(8);
                ar0 = _mm256_loadu_pd(a_ptr);
                ai0 = _mm256_loadu_pd(a_ptr.add(4));
                b_ptr = b_ptr.add(8);
            }
        }

        for j in j_full_end..j_end {
            edge(
                j,
                &mut acc_re0,
                &mut acc_im0,
                &mut acc_re1,
                &mut acc_im1,
                &mut acc_re2,
                &mut acc_im2,
            );
        }

        _mm256_storeu_pd(dst, acc_re0);
        _mm256_storeu_pd(dst.add(4), acc_im0);
        _mm256_storeu_pd(dst.add(8), acc_re1);
        _mm256_storeu_pd(dst.add(12), acc_im1);
        _mm256_storeu_pd(dst.add(16), acc_re2);
        _mm256_storeu_pd(dst.add(20), acc_im2);
    }
}

/// Column-level convolution over all `m/4` blocks; AVX2 counterpart of
/// `reim4_convolution_apply_avx512`. Accumulation order matches
/// `reim4_convolution_2coeffs_avx`.
///
/// `dst` layout: limb `k` re-half at `dst[dst_stride*k + 4*blk]`, im-half `m`
/// further; `dst_stride` is `2m` for a one-column destination, `2m * cols` for
/// a column of a column-interleaved `VecZnxDft`.
/// `tmp` must hold at least `8 * (a_size + 4 + b_size + 16 * min_size)` f64.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA (e.g. `is_x86_feature_detected!("avx2")`).
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn reim4_convolution_apply_avx(
    m: usize,
    min_size: usize,
    offset: usize,
    dst: &mut [f64],
    dst_stride: usize,
    a: &[f64],
    a_size: usize,
    b: &[f64],
    b_size: usize,
    tmp: &mut [f64],
) {
    unsafe {
        reim4_convolution_apply_core_avx::<false, false>(m, min_size, offset, dst, dst_stride, a, a, a_size, b, b, b_size, tmp)
    }
}

/// Pairwise variant of [`reim4_convolution_apply_avx`]: `(a0 + a1) ⊛ (b0 + b1)`.
///
/// `tmp` must hold at least `8 * (a_size + 4 + b_size + 16 * min_size)` f64.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA (e.g. `is_x86_feature_detected!("avx2")`).
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn reim4_convolution_pairwise_apply_avx(
    m: usize,
    min_size: usize,
    offset: usize,
    dst: &mut [f64],
    dst_stride: usize,
    a0: &[f64],
    a1: &[f64],
    a_size: usize,
    b0: &[f64],
    b1: &[f64],
    b_size: usize,
    tmp: &mut [f64],
) {
    unsafe {
        reim4_convolution_apply_core_avx::<true, false>(m, min_size, offset, dst, dst_stride, a0, a1, a_size, b0, b1, b_size, tmp)
    }
}

/// Accumulating variant of [`reim4_convolution_apply_avx`]: `dst += a ⊛ b`,
/// leaving limbs beyond `min_size` untouched.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA (e.g. `is_x86_feature_detected!("avx2")`).
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn reim4_convolution_apply_accumulate_avx(
    m: usize,
    min_size: usize,
    offset: usize,
    dst: &mut [f64],
    dst_stride: usize,
    a: &[f64],
    a_size: usize,
    b: &[f64],
    b_size: usize,
    tmp: &mut [f64],
) {
    unsafe {
        reim4_convolution_apply_core_avx::<false, true>(m, min_size, offset, dst, dst_stride, a, a, a_size, b, b, b_size, tmp)
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn reim4_convolution_apply_core_avx<const PAIRWISE: bool, const ACC: bool>(
    m: usize,
    min_size: usize,
    offset: usize,
    dst: &mut [f64],
    dst_stride: usize,
    a0: &[f64],
    a1: &[f64],
    a_size: usize,
    b0: &[f64],
    b1: &[f64],
    b_size: usize,
    tmp: &mut [f64],
) {
    use core::arch::x86_64::{
        __m256d, _mm256_add_pd, _mm256_fmadd_pd, _mm256_fnmadd_pd, _mm256_loadu_pd, _mm256_setzero_pd, _mm256_storeu_pd,
    };

    debug_assert!(a_size > 0);
    debug_assert!(b_size > 0);
    debug_assert!(m.is_multiple_of(4));
    debug_assert!(tmp.len() >= 8 * (a_size + 4 + b_size * (PAIRWISE as usize) + 16 * min_size));
    debug_assert!(a0.len() >= (m / 4) * 8 * a_size);
    debug_assert!(b0.len() >= (m / 4) * 8 * b_size);
    debug_assert!(dst_stride >= 2 * m);
    debug_assert!(dst.len() >= dst_stride * (min_size - 1) + 2 * m);

    const GROUP: usize = 16;

    unsafe {
        let (a_pad, rest) = tmp.split_at_mut(8 * (a_size + 4));
        let (b_sum, stage) = rest.split_at_mut(if PAIRWISE { 8 * b_size } else { 0 });
        let stage: &mut [f64] = &mut stage[..8 * GROUP * min_size];

        // Pad rows are zeroed once and never overwritten.
        let zero: __m256d = _mm256_setzero_pd();
        for h in 0..4 {
            _mm256_storeu_pd(a_pad.as_mut_ptr().add(4 * h), zero);
            _mm256_storeu_pd(a_pad.as_mut_ptr().add(8 * (a_size + 2) + 4 * h), zero);
        }

        let n_tiles: usize = min_size.div_ceil(3);
        let dst_ptr: *mut f64 = dst.as_mut_ptr();
        let n_blocks: usize = m / 4;

        for blk in 0..n_blocks {
            let stage_ptr: *mut f64 = stage.as_mut_ptr().add(8 * min_size * (blk % GROUP));

            let a_blk: *const f64 = a0.as_ptr().add(blk * 8 * a_size);
            let b_blk: *const f64 = b0.as_ptr().add(blk * 8 * b_size);
            let b_src: *const f64 = if PAIRWISE {
                let a1_blk: *const f64 = a1.as_ptr().add(blk * 8 * a_size);
                for r in 0..a_size {
                    let lo: __m256d = _mm256_add_pd(_mm256_loadu_pd(a_blk.add(8 * r)), _mm256_loadu_pd(a1_blk.add(8 * r)));
                    let hi: __m256d =
                        _mm256_add_pd(_mm256_loadu_pd(a_blk.add(8 * r + 4)), _mm256_loadu_pd(a1_blk.add(8 * r + 4)));
                    _mm256_storeu_pd(a_pad.as_mut_ptr().add(8 * (2 + r)), lo);
                    _mm256_storeu_pd(a_pad.as_mut_ptr().add(8 * (2 + r) + 4), hi);
                }
                let b1_blk: *const f64 = b1.as_ptr().add(blk * 8 * b_size);
                for r in 0..b_size {
                    let lo: __m256d = _mm256_add_pd(_mm256_loadu_pd(b_blk.add(8 * r)), _mm256_loadu_pd(b1_blk.add(8 * r)));
                    let hi: __m256d =
                        _mm256_add_pd(_mm256_loadu_pd(b_blk.add(8 * r + 4)), _mm256_loadu_pd(b1_blk.add(8 * r + 4)));
                    _mm256_storeu_pd(b_sum.as_mut_ptr().add(8 * r), lo);
                    _mm256_storeu_pd(b_sum.as_mut_ptr().add(8 * r + 4), hi);
                }
                b_sum.as_ptr()
            } else {
                for r in 0..a_size {
                    _mm256_storeu_pd(a_pad.as_mut_ptr().add(8 * (2 + r)), _mm256_loadu_pd(a_blk.add(8 * r)));
                    _mm256_storeu_pd(a_pad.as_mut_ptr().add(8 * (2 + r) + 4), _mm256_loadu_pd(a_blk.add(8 * r + 4)));
                }
                b_blk
            };

            for tile in 0..n_tiles {
                let k0: usize = offset + 3 * tile;
                let j_start: usize = (k0 + 1).saturating_sub(a_size).min(b_size);
                let j_end: usize = (k0 + 3).min(b_size);

                let mut acc_re0: __m256d = _mm256_setzero_pd();
                let mut acc_im0: __m256d = _mm256_setzero_pd();
                let mut acc_re1: __m256d = _mm256_setzero_pd();
                let mut acc_im1: __m256d = _mm256_setzero_pd();
                let mut acc_re2: __m256d = _mm256_setzero_pd();
                let mut acc_im2: __m256d = _mm256_setzero_pd();

                if j_start < j_end {
                    // w_t = padded row (k0 + t - j) + 2, sliding down one row per j.
                    let mut a_ptr: *const f64 = a_pad.as_ptr().add(8 * (k0 - j_start + 2));
                    let mut b_ptr: *const f64 = b_src.add(8 * j_start);

                    let mut wr0: __m256d = _mm256_loadu_pd(a_ptr);
                    let mut wi0: __m256d = _mm256_loadu_pd(a_ptr.add(4));
                    let mut wr1: __m256d = _mm256_loadu_pd(a_ptr.add(8));
                    let mut wi1: __m256d = _mm256_loadu_pd(a_ptr.add(12));
                    let mut wr2: __m256d = _mm256_loadu_pd(a_ptr.add(16));
                    let mut wi2: __m256d = _mm256_loadu_pd(a_ptr.add(20));

                    let mut j: usize = j_start;
                    loop {
                        let br: __m256d = _mm256_loadu_pd(b_ptr);
                        let bi: __m256d = _mm256_loadu_pd(b_ptr.add(4));

                        acc_re0 = _mm256_fmadd_pd(wr0, br, acc_re0);
                        acc_re0 = _mm256_fnmadd_pd(wi0, bi, acc_re0);
                        acc_im0 = _mm256_fmadd_pd(wr0, bi, acc_im0);
                        acc_im0 = _mm256_fmadd_pd(wi0, br, acc_im0);
                        acc_re1 = _mm256_fmadd_pd(wr1, br, acc_re1);
                        acc_re1 = _mm256_fnmadd_pd(wi1, bi, acc_re1);
                        acc_im1 = _mm256_fmadd_pd(wr1, bi, acc_im1);
                        acc_im1 = _mm256_fmadd_pd(wi1, br, acc_im1);
                        acc_re2 = _mm256_fmadd_pd(wr2, br, acc_re2);
                        acc_re2 = _mm256_fnmadd_pd(wi2, bi, acc_re2);
                        acc_im2 = _mm256_fmadd_pd(wr2, bi, acc_im2);
                        acc_im2 = _mm256_fmadd_pd(wi2, br, acc_im2);

                        j += 1;
                        if j == j_end {
                            break;
                        }
                        wr2 = wr1;
                        wi2 = wi1;
                        wr1 = wr0;
                        wi1 = wi0;
                        a_ptr = a_ptr.sub(8);
                        wr0 = _mm256_loadu_pd(a_ptr);
                        wi0 = _mm256_loadu_pd(a_ptr.add(4));
                        b_ptr = b_ptr.add(8);
                    }
                }

                let k_rel: usize = 3 * tile;
                let out: *mut f64 = stage_ptr.add(8 * k_rel);
                _mm256_storeu_pd(out, acc_re0);
                _mm256_storeu_pd(out.add(4), acc_im0);
                if k_rel + 1 < min_size {
                    _mm256_storeu_pd(out.add(8), acc_re1);
                    _mm256_storeu_pd(out.add(12), acc_im1);
                }
                if k_rel + 2 < min_size {
                    _mm256_storeu_pd(out.add(16), acc_re2);
                    _mm256_storeu_pd(out.add(20), acc_im2);
                }
            }

            // Flush the group: per limb, consecutive line stores.
            let in_group: usize = (blk % GROUP) + 1;
            if in_group == GROUP || blk == n_blocks - 1 {
                let grp_base: usize = blk + 1 - in_group;
                let stage_base: *const f64 = stage.as_ptr();
                for k in 0..min_size {
                    let row: *const f64 = stage_base.add(8 * k);
                    let out: *mut f64 = dst_ptr.add(dst_stride * k + 4 * grp_base);
                    for p in 0..in_group {
                        let mut re: __m256d = _mm256_loadu_pd(row.add(8 * min_size * p));
                        let mut im: __m256d = _mm256_loadu_pd(row.add(8 * min_size * p + 4));
                        if ACC {
                            re = _mm256_add_pd(_mm256_loadu_pd(out.add(4 * p)), re);
                            im = _mm256_add_pd(_mm256_loadu_pd(out.add(m + 4 * p)), im);
                        }
                        _mm256_storeu_pd(out.add(4 * p), re);
                        _mm256_storeu_pd(out.add(m + 4 * p), im);
                    }
                }
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, target_feature = "avx2"))]
mod tests {
    use poulpy_cpu_ref::reference::fft64::reim4::{
        reim4_convolution_1coeff_ref, reim4_convolution_2coeffs_ref, reim4_extract_1blk_from_reim_contiguous_ref,
        reim4_save_1blk_to_reim_contiguous_ref, reim4_vec_mat1col_product_ref, reim4_vec_mat2cols_product_ref,
    };

    use super::*;

    fn reim4_data(size: usize, seed: f64) -> Vec<f64> {
        (0..size * 8).map(|i| (i as f64 * seed + 0.5) / size as f64).collect()
    }

    /// AVX extract+save round-trip matches reference.
    #[test]
    fn reim4_extract_save_1blk_avx_vs_ref() {
        let m = 8usize; // multiple of 4
        let rows = 2usize;
        let blk = 0usize;

        let src: Vec<f64> = (0..2 * rows * m).map(|i| i as f64 + 1.0).collect();
        let mut dst_avx = vec![0f64; 2 * rows * 4];
        let mut dst_ref = vec![0f64; 2 * rows * 4];

        unsafe { reim4_extract_1blk_from_reim_contiguous_avx(m, rows, blk, &mut dst_avx, &src) };
        reim4_extract_1blk_from_reim_contiguous_ref(m, rows, blk, &mut dst_ref, &src);

        assert_eq!(dst_avx, dst_ref, "reim4_extract_1blk: AVX vs ref mismatch");

        // Also verify save round-trip
        let mut out_avx = vec![0f64; 2 * rows * m];
        let mut out_ref = vec![0f64; 2 * rows * m];
        unsafe { reim4_save_1blk_to_reim_contiguous_avx(m, rows, blk, &mut out_avx, &dst_avx) };
        reim4_save_1blk_to_reim_contiguous_ref(m, rows, blk, &mut out_ref, &dst_ref);

        assert_eq!(out_avx, out_ref, "reim4_save_1blk: AVX vs ref mismatch");
    }

    /// AVX `reim4_vec_mat1col_product` matches reference.
    #[test]
    fn reim4_vec_mat1col_product_avx_vs_ref() {
        let nrows = 8usize;
        let u = reim4_data(nrows, 1.3);
        let v = reim4_data(nrows, 2.7);
        let mut dst_avx = vec![0f64; 8];
        let mut dst_ref = vec![0f64; 8];

        unsafe { reim4_vec_mat1col_product_avx(nrows, &mut dst_avx, &u, &v) };
        reim4_vec_mat1col_product_ref(nrows, &mut dst_ref, &u, &v);

        let tol = 1e-12f64;
        for i in 0..8 {
            assert!(
                (dst_avx[i] - dst_ref[i]).abs() <= tol,
                "mat1col idx={i}: AVX={} ref={}",
                dst_avx[i],
                dst_ref[i]
            );
        }
    }

    /// AVX `reim4_vec_mat2cols_product` matches reference.
    #[test]
    fn reim4_vec_mat2cols_product_avx_vs_ref() {
        let nrows = 8usize;
        let u = reim4_data(nrows, 1.1);
        let v: Vec<f64> = (0..nrows * 16).map(|i| i as f64 * 0.07 + 0.1).collect();
        let mut dst_avx = vec![0f64; 16];
        let mut dst_ref = vec![0f64; 16];

        unsafe { reim4_vec_mat2cols_product_avx(nrows, &mut dst_avx, &u, &v) };
        reim4_vec_mat2cols_product_ref(nrows, &mut dst_ref, &u, &v);

        let tol = 1e-12f64;
        for i in 0..16 {
            assert!(
                (dst_avx[i] - dst_ref[i]).abs() <= tol,
                "mat2cols idx={i}: AVX={} ref={}",
                dst_avx[i],
                dst_ref[i]
            );
        }
    }

    /// AVX `reim4_convolution_1coeff` matches reference for all k values.
    #[test]
    fn reim4_convolution_1coeff_avx_vs_ref() {
        let a_size = 4usize;
        let b_size = 4usize;
        let a = reim4_data(a_size, 1.5);
        let b = reim4_data(b_size, 2.1);

        for k in 0..a_size + b_size + 1 {
            let mut dst_avx = [0f64; 8];
            let mut dst_ref = [0f64; 8];
            unsafe { reim4_convolution_1coeff_avx(k, &mut dst_avx, &a, a_size, &b, b_size) };
            reim4_convolution_1coeff_ref(k, &mut dst_ref, &a, a_size, &b, b_size);
            let tol = 1e-12f64;
            for i in 0..8 {
                assert!(
                    (dst_avx[i] - dst_ref[i]).abs() <= tol,
                    "conv1coeff k={k} i={i}: AVX={} ref={}",
                    dst_avx[i],
                    dst_ref[i]
                );
            }
        }
    }

    /// AVX `reim4_convolution_2coeffs` matches reference for all k values.
    #[test]
    fn reim4_convolution_2coeffs_avx_vs_ref() {
        let a_size = 4usize;
        let b_size = 4usize;
        let a = reim4_data(a_size, 1.7);
        let b = reim4_data(b_size, 2.3);

        for k in 0..a_size + b_size + 1 {
            let mut dst_avx = [0f64; 16];
            let mut dst_ref = [0f64; 16];
            unsafe { reim4_convolution_2coeffs_avx(k, &mut dst_avx, &a, a_size, &b, b_size) };
            reim4_convolution_2coeffs_ref(k, &mut dst_ref, &a, a_size, &b, b_size);
            let tol = 1e-12f64;
            for i in 0..16 {
                assert!(
                    (dst_avx[i] - dst_ref[i]).abs() <= tol,
                    "conv2coeffs k={k} i={i}: AVX={} ref={}",
                    dst_avx[i],
                    dst_ref[i]
                );
            }
        }
    }

    /// Column kernel matches the per-coefficient reference.
    #[test]
    fn reim4_convolution_apply_avx_vs_ref() {
        for &(a_size, b_size) in &[(1usize, 1usize), (2, 3), (4, 4), (5, 2), (14, 14), (14, 1), (3, 14), (7, 9)] {
            let bound = a_size + b_size - 1;
            for &m in &[4usize, 16, 144] {
                let n_blk = m / 4;
                let a: Vec<f64> = (0..n_blk * 8 * a_size).map(|i| (i as f64 * 0.13 + 1.5).sin()).collect();
                let b: Vec<f64> = (0..n_blk * 8 * b_size).map(|i| (i as f64 * 0.07 + 2.1).cos()).collect();
                for offset in [0usize, 1, bound / 2, bound.saturating_sub(1), bound] {
                    for min_size in [1usize, 2, 3, 4, 5, 8, bound, bound + 2] {
                        let mut dst_fused = vec![f64::NAN; 2 * m * min_size];
                        let mut dst_ref = vec![f64::NAN; 2 * m * min_size];
                        let mut tmp = vec![0f64; 8 * (a_size + 4 + b_size + 16 * min_size)];

                        unsafe {
                            reim4_convolution_apply_avx(
                                m,
                                min_size,
                                offset,
                                &mut dst_fused,
                                2 * m,
                                &a,
                                a_size,
                                &b,
                                b_size,
                                &mut tmp,
                            )
                        };

                        let mut blk_out = vec![0f64; 8 * min_size];
                        for blk in 0..n_blk {
                            for k in 0..min_size {
                                let out: &mut [f64; 8] = (&mut blk_out[8 * k..8 * k + 8]).try_into().unwrap();
                                reim4_convolution_1coeff_ref(
                                    k + offset,
                                    out,
                                    &a[blk * 8 * a_size..],
                                    a_size,
                                    &b[blk * 8 * b_size..],
                                    b_size,
                                );
                            }
                            reim4_save_1blk_to_reim_contiguous_ref(m, min_size, blk, &mut dst_ref, &blk_out);
                        }

                        let tol = 1e-12f64;
                        for i in 0..2 * m * min_size {
                            assert!(
                                (dst_fused[i] - dst_ref[i]).abs() <= tol,
                                "conv_apply a={a_size} b={b_size} m={m} offset={offset} min={min_size} i={i}: fused={} ref={}",
                                dst_fused[i],
                                dst_ref[i]
                            );
                        }
                    }
                }
            }
        }
    }

    /// Full-row `reim4_convolution` matches the per-coefficient reference.
    #[test]
    fn reim4_convolution_full_avx_vs_ref() {
        for &(a_size, b_size) in &[(1usize, 1usize), (2, 3), (4, 4), (5, 2), (14, 14), (14, 1), (3, 14), (7, 9)] {
            let a = reim4_data(a_size, 1.5);
            let b = reim4_data(b_size, 2.1);
            let bound = a_size + b_size - 1;
            for offset in [0usize, 1, 3, bound / 2, bound.saturating_sub(1), bound] {
                for dst_size in [1usize, 2, 3, 4, 5, 7, 8, bound, bound + 2] {
                    let mut dst_avx = vec![f64::NAN; 8 * dst_size];
                    let mut dst_ref = vec![f64::NAN; 8 * dst_size];

                    unsafe { reim4_convolution_avx(&mut dst_avx, dst_size, offset, &a, a_size, &b, b_size) };
                    for k in 0..dst_size {
                        let blk: &mut [f64; 8] = (&mut dst_ref[8 * k..8 * k + 8]).try_into().unwrap();
                        reim4_convolution_1coeff_ref(k + offset, blk, &a, a_size, &b, b_size);
                    }

                    let tol = 1e-12f64;
                    for i in 0..8 * dst_size {
                        assert!(
                            (dst_avx[i] - dst_ref[i]).abs() <= tol,
                            "conv_full a={a_size} b={b_size} offset={offset} dst={dst_size} i={i}: AVX={} ref={}",
                            dst_avx[i],
                            dst_ref[i]
                        );
                    }
                }
            }
        }
    }
}
