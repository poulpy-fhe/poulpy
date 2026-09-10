//! AVX2 helpers for FFT64 convolution by-constant kernels.

/// # Safety
/// Caller must ensure the CPU supports AVX2.
/// Assumes all inputs fit in i32 (so i32×i32→i64 is exact).
#[target_feature(enable = "avx2")]
pub unsafe fn i64_convolution_by_const_1coeff_avx(k: usize, dst: &mut [i64; 8], a: &[i64], a_size: usize, b: &[i64]) {
    use core::arch::x86_64::{
        __m256i, _mm256_add_epi64, _mm256_loadu_si256, _mm256_mul_epi32, _mm256_set1_epi32, _mm256_setzero_si256,
        _mm256_storeu_si256,
    };

    dst.fill(0);

    let b_size = b.len();
    if k >= a_size + b_size {
        return;
    }

    let j_min = k.saturating_sub(a_size - 1);
    let j_max = (k + 1).min(b_size);

    unsafe {
        // Two accumulators = 8 outputs total
        let mut acc_lo: __m256i = _mm256_setzero_si256(); // dst[0..4)
        let mut acc_hi: __m256i = _mm256_setzero_si256(); // dst[4..8)

        let mut a_ptr: *const i64 = a.as_ptr().add(8 * (k - j_min));
        let mut b_ptr: *const i64 = b.as_ptr().add(j_min);

        for _ in 0..(j_max - j_min) {
            // Broadcast scalar b[j] as i32
            let br: __m256i = _mm256_set1_epi32(*b_ptr as i32);

            // ---- lower half: a[0..4) ----
            let a_lo: __m256i = _mm256_loadu_si256(a_ptr as *const __m256i);

            let prod_lo: __m256i = _mm256_mul_epi32(a_lo, br);

            acc_lo = _mm256_add_epi64(acc_lo, prod_lo);

            // ---- upper half: a[4..8) ----
            let a_hi: __m256i = _mm256_loadu_si256(a_ptr.add(4) as *const __m256i);

            let prod_hi: __m256i = _mm256_mul_epi32(a_hi, br);

            acc_hi = _mm256_add_epi64(acc_hi, prod_hi);

            a_ptr = a_ptr.sub(8);
            b_ptr = b_ptr.add(1);
        }

        // Store final result
        _mm256_storeu_si256(dst.as_mut_ptr() as *mut __m256i, acc_lo);
        _mm256_storeu_si256(dst.as_mut_ptr().add(4) as *mut __m256i, acc_hi);
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2.
/// Assumes all values in `a` and `b` fit in i32 (so i32×i32→i64 is exact).
#[target_feature(enable = "avx2")]
pub unsafe fn i64_convolution_by_real_const_2coeffs_avx(
    k: usize,
    dst: &mut [i64; 16],
    a: &[i64],
    a_size: usize,
    b: &[i64], // real scalars, stride-1
) {
    use core::arch::x86_64::{
        __m256i, _mm256_add_epi64, _mm256_loadu_si256, _mm256_mul_epi32, _mm256_set1_epi32, _mm256_setzero_si256,
        _mm256_storeu_si256,
    };

    let b_size: usize = b.len();

    assert!(a.len() >= 8 * a_size);

    let k0: usize = k;
    let k1: usize = k + 1;
    let bound: usize = a_size + b_size;

    if k0 >= bound {
        unsafe {
            let zero: __m256i = _mm256_setzero_si256();
            let dst_ptr: *mut i64 = dst.as_mut_ptr();
            _mm256_storeu_si256(dst_ptr as *mut __m256i, zero);
            _mm256_storeu_si256(dst_ptr.add(4) as *mut __m256i, zero);
            _mm256_storeu_si256(dst_ptr.add(8) as *mut __m256i, zero);
            _mm256_storeu_si256(dst_ptr.add(12) as *mut __m256i, zero);
        }
        return;
    }

    unsafe {
        let mut acc_lo_k0: __m256i = _mm256_setzero_si256();
        let mut acc_hi_k0: __m256i = _mm256_setzero_si256();
        let mut acc_lo_k1: __m256i = _mm256_setzero_si256();
        let mut acc_hi_k1: __m256i = _mm256_setzero_si256();

        let j0_min: usize = (k0 + 1).saturating_sub(a_size);
        let j0_max: usize = (k0 + 1).min(b_size);

        if k1 >= bound {
            let mut a_k0_ptr: *const i64 = a.as_ptr().add(8 * (k0 - j0_min));
            let mut b_ptr: *const i64 = b.as_ptr().add(j0_min);

            // Contributions to k0 only
            for _ in 0..j0_max - j0_min {
                // Broadcast b[j] as i32
                let br: __m256i = _mm256_set1_epi32(*b_ptr as i32);

                // Load 4×i64 (low half) and 4×i64 (high half)
                let a_lo_k0: __m256i = _mm256_loadu_si256(a_k0_ptr as *const __m256i);
                let a_hi_k0: __m256i = _mm256_loadu_si256(a_k0_ptr.add(4) as *const __m256i);

                acc_lo_k0 = _mm256_add_epi64(acc_lo_k0, _mm256_mul_epi32(a_lo_k0, br));
                acc_hi_k0 = _mm256_add_epi64(acc_hi_k0, _mm256_mul_epi32(a_hi_k0, br));

                a_k0_ptr = a_k0_ptr.sub(8);
                b_ptr = b_ptr.add(1);
            }
        } else {
            let j1_min: usize = (k1 + 1).saturating_sub(a_size);
            let j1_max: usize = (k1 + 1).min(b_size);

            let mut a_k0_ptr: *const i64 = a.as_ptr().add(8 * (k0 - j0_min));
            let mut a_k1_ptr: *const i64 = a.as_ptr().add(8 * (k1 - j1_min));
            let mut b_ptr: *const i64 = b.as_ptr().add(j0_min);

            // Region 1: k0 only, j ∈ [j0_min, j1_min)
            for _ in 0..j1_min - j0_min {
                let br: __m256i = _mm256_set1_epi32(*b_ptr as i32);

                let a_k0_lo: __m256i = _mm256_loadu_si256(a_k0_ptr as *const __m256i);
                let a_k0_hi: __m256i = _mm256_loadu_si256(a_k0_ptr.add(4) as *const __m256i);

                acc_lo_k0 = _mm256_add_epi64(acc_lo_k0, _mm256_mul_epi32(a_k0_lo, br));
                acc_hi_k0 = _mm256_add_epi64(acc_hi_k0, _mm256_mul_epi32(a_k0_hi, br));

                a_k0_ptr = a_k0_ptr.sub(8);
                b_ptr = b_ptr.add(1);
            }

            // Region 2: overlap, contributions to both k0 and k1, j ∈ [j1_min, j0_max)
            // Save one load on b: broadcast once and reuse.
            for _ in 0..j0_max - j1_min {
                let br: __m256i = _mm256_set1_epi32(*b_ptr as i32);

                let a_lo_k0: __m256i = _mm256_loadu_si256(a_k0_ptr as *const __m256i);
                let a_hi_k0: __m256i = _mm256_loadu_si256(a_k0_ptr.add(4) as *const __m256i);
                let a_lo_k1: __m256i = _mm256_loadu_si256(a_k1_ptr as *const __m256i);
                let a_hi_k1: __m256i = _mm256_loadu_si256(a_k1_ptr.add(4) as *const __m256i);

                // k0
                acc_lo_k0 = _mm256_add_epi64(acc_lo_k0, _mm256_mul_epi32(a_lo_k0, br));
                acc_hi_k0 = _mm256_add_epi64(acc_hi_k0, _mm256_mul_epi32(a_hi_k0, br));

                // k1
                acc_lo_k1 = _mm256_add_epi64(acc_lo_k1, _mm256_mul_epi32(a_lo_k1, br));
                acc_hi_k1 = _mm256_add_epi64(acc_hi_k1, _mm256_mul_epi32(a_hi_k1, br));

                a_k0_ptr = a_k0_ptr.sub(8);
                a_k1_ptr = a_k1_ptr.sub(8);
                b_ptr = b_ptr.add(1);
            }

            // Region 3: k1 only, j ∈ [j0_max, j1_max)
            for _ in 0..j1_max - j0_max {
                let br: __m256i = _mm256_set1_epi32(*b_ptr as i32);

                let a_lo_k1: __m256i = _mm256_loadu_si256(a_k1_ptr as *const __m256i);
                let a_hi_k1: __m256i = _mm256_loadu_si256(a_k1_ptr.add(4) as *const __m256i);

                acc_lo_k1 = _mm256_add_epi64(acc_lo_k1, _mm256_mul_epi32(a_lo_k1, br));
                acc_hi_k1 = _mm256_add_epi64(acc_hi_k1, _mm256_mul_epi32(a_hi_k1, br));

                a_k1_ptr = a_k1_ptr.sub(8);
                b_ptr = b_ptr.add(1);
            }
        }

        let dst_ptr: *mut i64 = dst.as_mut_ptr();
        _mm256_storeu_si256(dst_ptr as *mut __m256i, acc_lo_k0);
        _mm256_storeu_si256(dst_ptr.add(4) as *mut __m256i, acc_hi_k0);
        _mm256_storeu_si256(dst_ptr.add(8) as *mut __m256i, acc_lo_k1);
        _mm256_storeu_si256(dst_ptr.add(12) as *mut __m256i, acc_hi_k1);
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[target_feature(enable = "avx")]
pub fn i64_extract_1blk_contiguous_avx(n: usize, offset: usize, rows: usize, blk: usize, dst: &mut [i64], src: &[i64]) {
    use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_storeu_si256};

    unsafe {
        let mut src_ptr: *const __m256i = src.as_ptr().add(offset + (blk << 3)) as *const __m256i; // src + 8*blk
        let mut dst_ptr: *mut __m256i = dst.as_mut_ptr() as *mut __m256i;

        let step: usize = n >> 2;

        // Each iteration copies 8 i64; advance src by n i64 each row
        for _ in 0..rows {
            let v: __m256i = _mm256_loadu_si256(src_ptr);
            _mm256_storeu_si256(dst_ptr, v);
            let v: __m256i = _mm256_loadu_si256(src_ptr.add(1));
            _mm256_storeu_si256(dst_ptr.add(1), v);
            dst_ptr = dst_ptr.add(2);
            src_ptr = src_ptr.add(step);
        }
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
#[target_feature(enable = "avx")]
pub fn i64_save_1blk_contiguous_avx(n: usize, offset: usize, rows: usize, blk: usize, dst: &mut [i64], src: &[i64]) {
    use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_storeu_si256};

    unsafe {
        let mut src_ptr: *const __m256i = src.as_ptr() as *const __m256i;
        let mut dst_ptr: *mut __m256i = dst.as_mut_ptr().add(offset + (blk << 3)) as *mut __m256i; // dst + 8*blk

        let step: usize = n >> 2;

        // Each iteration copies 8 i64; advance dst by n i64 each row
        for _ in 0..rows {
            let v: __m256i = _mm256_loadu_si256(src_ptr);
            _mm256_storeu_si256(dst_ptr, v);
            let v: __m256i = _mm256_loadu_si256(src_ptr.add(1));
            _mm256_storeu_si256(dst_ptr.add(1), v);
            dst_ptr = dst_ptr.add(step);
            src_ptr = src_ptr.add(2);
        }
    }
}
