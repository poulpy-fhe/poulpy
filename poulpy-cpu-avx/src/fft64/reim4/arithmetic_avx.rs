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
    use core::arch::x86_64::{__m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_storeu_pd, _mm256_stream_pd};
    unsafe {
        let off: usize = blk * 4;
        let src_ptr: *const f64 = src.as_ptr();

        let s0: __m256d = _mm256_loadu_pd(src_ptr);
        let s1: __m256d = _mm256_loadu_pd(src_ptr.add(4));

        let d0_ptr: *mut f64 = dst.as_mut_ptr().add(off);
        let d1_ptr: *mut f64 = d0_ptr.add(m);

        if OVERWRITE {
            // `dst` allocations are 64-byte aligned and each block starts on a
            // 32-byte boundary, so overwrite-mode can use non-temporal stores.
            _mm256_stream_pd(d0_ptr, s0);
            _mm256_stream_pd(d1_ptr, s1);
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
    use core::arch::x86_64::{__m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_storeu_pd, _mm256_stream_pd};
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
            // `dst` allocations are 64-byte aligned and each block starts on a
            // 32-byte boundary, so overwrite-mode can use non-temporal stores.
            _mm256_stream_pd(d0_ptr, s0);
            _mm256_stream_pd(d1_ptr, s1);
            _mm256_stream_pd(d2_ptr, s2);
            _mm256_stream_pd(d3_ptr, s3);
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
}
