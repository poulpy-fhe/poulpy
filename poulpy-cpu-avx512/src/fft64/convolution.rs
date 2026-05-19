/// # Safety
/// Caller must ensure the CPU supports AVX-512F.
/// Assumes all inputs fit in i32 (so i32×i32→i64 is exact).
#[target_feature(enable = "avx512f")]
pub unsafe fn i64_convolution_by_const_1coeff_avx512(k: usize, dst: &mut [i64; 8], a: &[i64], a_size: usize, b: &[i64]) {
    use core::arch::x86_64::{
        __m512i, _mm512_add_epi64, _mm512_loadu_si512, _mm512_mul_epi32, _mm512_set1_epi32, _mm512_setzero_si512,
        _mm512_storeu_si512,
    };

    dst.fill(0);

    let b_size = b.len();
    if k >= a_size + b_size {
        return;
    }

    let j_min = k.saturating_sub(a_size - 1);
    let j_max = (k + 1).min(b_size);

    unsafe {
        // Single 512-bit accumulator for all 8 outputs
        let mut acc: __m512i = _mm512_setzero_si512();

        let mut a_ptr: *const i64 = a.as_ptr().add(8 * (k - j_min));
        let mut b_ptr: *const i64 = b.as_ptr().add(j_min);

        for _ in 0..(j_max - j_min) {
            // Broadcast scalar b[j] as i32
            let br: __m512i = _mm512_set1_epi32(*b_ptr as i32);

            // Load 8×i64 in one shot
            let a_vec: __m512i = _mm512_loadu_si512(a_ptr as *const __m512i);

            let prod: __m512i = _mm512_mul_epi32(a_vec, br);

            acc = _mm512_add_epi64(acc, prod);

            a_ptr = a_ptr.sub(8);
            b_ptr = b_ptr.add(1);
        }

        // Store final result
        _mm512_storeu_si512(dst.as_mut_ptr() as *mut __m512i, acc);
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX-512F.
/// Assumes all values in `a` and `b` fit in i32 (so i32×i32→i64 is exact).
#[target_feature(enable = "avx512f")]
pub unsafe fn i64_convolution_by_const_2coeffs_avx512(
    k: usize,
    dst: &mut [i64; 16],
    a: &[i64],
    a_size: usize,
    b: &[i64], // real scalars, stride-1
) {
    use core::arch::x86_64::{
        __m512i, _mm512_add_epi64, _mm512_loadu_si512, _mm512_mul_epi32, _mm512_set1_epi32, _mm512_setzero_si512,
        _mm512_storeu_si512,
    };

    let b_size: usize = b.len();

    debug_assert!(a.len() >= 8 * a_size);

    let k0: usize = k;
    let k1: usize = k + 1;
    let bound: usize = a_size + b_size;

    if k0 >= bound {
        unsafe {
            let zero: __m512i = _mm512_setzero_si512();
            let dst_ptr: *mut i64 = dst.as_mut_ptr();
            _mm512_storeu_si512(dst_ptr as *mut __m512i, zero);
            _mm512_storeu_si512(dst_ptr.add(8) as *mut __m512i, zero);
        }
        return;
    }

    unsafe {
        let mut acc_k0: __m512i = _mm512_setzero_si512();
        let mut acc_k1: __m512i = _mm512_setzero_si512();

        let j0_min: usize = (k0 + 1).saturating_sub(a_size);
        let j0_max: usize = (k0 + 1).min(b_size);

        if k1 >= bound {
            let mut a_k0_ptr: *const i64 = a.as_ptr().add(8 * (k0 - j0_min));
            let mut b_ptr: *const i64 = b.as_ptr().add(j0_min);

            // Contributions to k0 only
            for _ in 0..j0_max - j0_min {
                // Broadcast b[j] as i32
                let br: __m512i = _mm512_set1_epi32(*b_ptr as i32);

                // Load 8×i64 in one shot
                let a_k0: __m512i = _mm512_loadu_si512(a_k0_ptr as *const __m512i);

                acc_k0 = _mm512_add_epi64(acc_k0, _mm512_mul_epi32(a_k0, br));

                a_k0_ptr = a_k0_ptr.sub(8);
                b_ptr = b_ptr.add(1);
            }
        } else {
            let j1_min: usize = (k1 + 1).saturating_sub(a_size);
            let j1_max: usize = (k1 + 1).min(b_size);

            let mut a_k0_ptr: *const i64 = a.as_ptr().add(8 * (k0 - j0_min));
            let mut a_k1_ptr: *const i64 = a.as_ptr().add(8 * (k1 - j1_min));
            let mut b_ptr: *const i64 = b.as_ptr().add(j0_min);

            // Region 1: k0 only, j in [j0_min, j1_min)
            for _ in 0..j1_min - j0_min {
                let br: __m512i = _mm512_set1_epi32(*b_ptr as i32);

                let a_k0: __m512i = _mm512_loadu_si512(a_k0_ptr as *const __m512i);

                acc_k0 = _mm512_add_epi64(acc_k0, _mm512_mul_epi32(a_k0, br));

                a_k0_ptr = a_k0_ptr.sub(8);
                b_ptr = b_ptr.add(1);
            }

            // Region 2: overlap, contributions to both k0 and k1, j in [j1_min, j0_max)
            // Broadcast once and reuse.
            for _ in 0..j0_max - j1_min {
                let br: __m512i = _mm512_set1_epi32(*b_ptr as i32);

                let a_k0: __m512i = _mm512_loadu_si512(a_k0_ptr as *const __m512i);
                let a_k1: __m512i = _mm512_loadu_si512(a_k1_ptr as *const __m512i);

                // k0
                acc_k0 = _mm512_add_epi64(acc_k0, _mm512_mul_epi32(a_k0, br));

                // k1
                acc_k1 = _mm512_add_epi64(acc_k1, _mm512_mul_epi32(a_k1, br));

                a_k0_ptr = a_k0_ptr.sub(8);
                a_k1_ptr = a_k1_ptr.sub(8);
                b_ptr = b_ptr.add(1);
            }

            // Region 3: k1 only, j in [j0_max, j1_max)
            for _ in 0..j1_max - j0_max {
                let br: __m512i = _mm512_set1_epi32(*b_ptr as i32);

                let a_k1: __m512i = _mm512_loadu_si512(a_k1_ptr as *const __m512i);

                acc_k1 = _mm512_add_epi64(acc_k1, _mm512_mul_epi32(a_k1, br));

                a_k1_ptr = a_k1_ptr.sub(8);
                b_ptr = b_ptr.add(1);
            }
        }

        let dst_ptr: *mut i64 = dst.as_mut_ptr();
        _mm512_storeu_si512(dst_ptr as *mut __m512i, acc_k0);
        _mm512_storeu_si512(dst_ptr.add(8) as *mut __m512i, acc_k1);
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`).
#[target_feature(enable = "avx512f")]
pub fn i64_extract_1blk_contiguous_avx512(n: usize, offset: usize, rows: usize, blk: usize, dst: &mut [i64], src: &[i64]) {
    use core::arch::x86_64::{__m512i, _mm512_loadu_si512, _mm512_storeu_si512};

    unsafe {
        let mut src_ptr: *const i64 = src.as_ptr().add(offset + (blk << 3));
        let mut dst_ptr: *mut i64 = dst.as_mut_ptr();

        let step: usize = n; // advance by n i64 each row

        // Each iteration copies 8 i64 = one __m512i
        for _ in 0..rows {
            let v: __m512i = _mm512_loadu_si512(src_ptr as *const __m512i);
            _mm512_storeu_si512(dst_ptr as *mut __m512i, v);
            dst_ptr = dst_ptr.add(8);
            src_ptr = src_ptr.add(step);
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`).
#[target_feature(enable = "avx512f")]
pub fn i64_save_1blk_contiguous_avx512(n: usize, offset: usize, rows: usize, blk: usize, dst: &mut [i64], src: &[i64]) {
    use core::arch::x86_64::{__m512i, _mm512_loadu_si512, _mm512_storeu_si512};

    unsafe {
        let mut src_ptr: *const i64 = src.as_ptr();
        let mut dst_ptr: *mut i64 = dst.as_mut_ptr().add(offset + (blk << 3));

        let step: usize = n; // advance by n i64 each row

        // Each iteration copies 8 i64 = one __m512i
        for _ in 0..rows {
            let v: __m512i = _mm512_loadu_si512(src_ptr as *const __m512i);
            _mm512_storeu_si512(dst_ptr as *mut __m512i, v);
            dst_ptr = dst_ptr.add(step);
            src_ptr = src_ptr.add(8);
        }
    }
}
