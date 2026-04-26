/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
/// all inputs must have the same length and must not alias.
#[target_feature(enable = "avx2")]
pub fn znx_sub_avx(res: &mut [i64], a: &[i64], b: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len());
        assert_eq!(res.len(), b.len());
    }

    use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_storeu_si256, _mm256_sub_epi64};

    let n: usize = res.len();

    let span: usize = n >> 2;

    let mut rr: *mut __m256i = res.as_mut_ptr() as *mut __m256i;
    let mut aa: *const __m256i = a.as_ptr() as *const __m256i;
    let mut bb: *const __m256i = b.as_ptr() as *const __m256i;

    unsafe {
        for _ in 0..span {
            let sum: __m256i = _mm256_sub_epi64(_mm256_loadu_si256(aa), _mm256_loadu_si256(bb));
            _mm256_storeu_si256(rr, sum);
            rr = rr.add(1);
            aa = aa.add(1);
            bb = bb.add(1);
        }
    }

    // tail
    if !res.len().is_multiple_of(4) {
        use poulpy_cpu_ref::reference::znx::znx_sub_ref;

        znx_sub_ref(&mut res[span << 2..], &a[span << 2..], &b[span << 2..]);
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
/// all inputs must have the same length and must not alias.
#[target_feature(enable = "avx2")]
pub fn znx_sub_assign_avx(res: &mut [i64], a: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len());
    }

    use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_storeu_si256, _mm256_sub_epi64};

    let n: usize = res.len();

    let span: usize = n >> 2;

    let mut rr: *mut __m256i = res.as_mut_ptr() as *mut __m256i;
    let mut aa: *const __m256i = a.as_ptr() as *const __m256i;

    unsafe {
        for _ in 0..span {
            let sum: __m256i = _mm256_sub_epi64(_mm256_loadu_si256(rr), _mm256_loadu_si256(aa));
            _mm256_storeu_si256(rr, sum);
            rr = rr.add(1);
            aa = aa.add(1);
        }
    }

    // tail
    if !res.len().is_multiple_of(4) {
        use poulpy_cpu_ref::reference::znx::znx_sub_assign_ref;

        znx_sub_assign_ref(&mut res[span << 2..], &a[span << 2..]);
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
/// all inputs must have the same length and must not alias.
#[target_feature(enable = "avx2")]
pub fn znx_sub_negate_assign_avx(res: &mut [i64], a: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len());
    }

    use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_storeu_si256, _mm256_sub_epi64};

    let n: usize = res.len();

    let span: usize = n >> 2;

    let mut rr: *mut __m256i = res.as_mut_ptr() as *mut __m256i;
    let mut aa: *const __m256i = a.as_ptr() as *const __m256i;

    unsafe {
        for _ in 0..span {
            let sum: __m256i = _mm256_sub_epi64(_mm256_loadu_si256(aa), _mm256_loadu_si256(rr));
            _mm256_storeu_si256(rr, sum);
            rr = rr.add(1);
            aa = aa.add(1);
        }
    }

    // tail
    if !res.len().is_multiple_of(4) {
        use poulpy_cpu_ref::reference::znx::znx_sub_negate_assign_ref;

        znx_sub_negate_assign_ref(&mut res[span << 2..], &a[span << 2..]);
    }
}
