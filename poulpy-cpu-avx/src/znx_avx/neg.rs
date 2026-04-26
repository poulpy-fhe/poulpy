/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
/// all inputs must have the same length and must not alias.
#[target_feature(enable = "avx2")]
pub fn znx_negate_avx(res: &mut [i64], src: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), src.len())
    }

    let n: usize = res.len();

    use std::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_setzero_si256, _mm256_storeu_si256, _mm256_sub_epi64};
    let span: usize = n >> 2;

    unsafe {
        let mut aa: *const __m256i = src.as_ptr() as *const __m256i;
        let mut rr: *mut __m256i = res.as_mut_ptr() as *mut __m256i;
        let zero: __m256i = _mm256_setzero_si256();
        for _ in 0..span {
            let v: __m256i = _mm256_loadu_si256(aa);
            let neg: __m256i = _mm256_sub_epi64(zero, v);
            _mm256_storeu_si256(rr, neg);
            aa = aa.add(1);
            rr = rr.add(1);
        }
    }

    if !res.len().is_multiple_of(4) {
        use poulpy_cpu_ref::reference::znx::znx_negate_ref;

        znx_negate_ref(&mut res[span << 2..], &src[span << 2..])
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
/// all inputs must have the same length and must not alias.
#[target_feature(enable = "avx2")]
pub fn znx_negate_assign_avx(res: &mut [i64]) {
    let n: usize = res.len();

    use std::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_setzero_si256, _mm256_storeu_si256, _mm256_sub_epi64};
    let span: usize = n >> 2;

    unsafe {
        let mut rr: *mut __m256i = res.as_mut_ptr() as *mut __m256i;
        let zero: __m256i = _mm256_setzero_si256();
        for _ in 0..span {
            let v: __m256i = _mm256_loadu_si256(rr);
            let neg: __m256i = _mm256_sub_epi64(zero, v);
            _mm256_storeu_si256(rr, neg);
            rr = rr.add(1);
        }
    }

    if !res.len().is_multiple_of(4) {
        use poulpy_cpu_ref::reference::znx::znx_negate_assign_ref;

        znx_negate_assign_ref(&mut res[span << 2..])
    }
}
