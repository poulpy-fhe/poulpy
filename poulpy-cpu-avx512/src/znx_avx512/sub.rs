use core::arch::x86_64::{__m512i, _mm512_loadu_si512, _mm512_storeu_si512, _mm512_sub_epi64};

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`);
/// all inputs must have the same length and must not alias.
#[target_feature(enable = "avx512f")]
pub unsafe fn znx_sub_avx512(res: &mut [i64], a: &[i64], b: &[i64]) {
    debug_assert_eq!(res.len(), a.len());
    debug_assert_eq!(res.len(), b.len());

    let n = res.len();
    let span = n >> 3;

    unsafe {
        let mut rr = res.as_mut_ptr() as *mut __m512i;
        let mut aa = a.as_ptr() as *const __m512i;
        let mut bb = b.as_ptr() as *const __m512i;

        for _ in 0..span {
            _mm512_storeu_si512(rr, _mm512_sub_epi64(_mm512_loadu_si512(aa), _mm512_loadu_si512(bb)));
            rr = rr.add(1);
            aa = aa.add(1);
            bb = bb.add(1);
        }
    }

    let tail = span << 3;
    for i in tail..n {
        res[i] = a[i].wrapping_sub(b[i]);
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`);
/// all inputs must have the same length and must not alias.
#[target_feature(enable = "avx512f")]
pub unsafe fn znx_sub_assign_avx512(res: &mut [i64], a: &[i64]) {
    debug_assert_eq!(res.len(), a.len());

    let n = res.len();
    let span = n >> 3;

    unsafe {
        let mut rr = res.as_mut_ptr() as *mut __m512i;
        let mut aa = a.as_ptr() as *const __m512i;

        for _ in 0..span {
            _mm512_storeu_si512(
                rr,
                _mm512_sub_epi64(_mm512_loadu_si512(rr as *const _), _mm512_loadu_si512(aa)),
            );
            rr = rr.add(1);
            aa = aa.add(1);
        }
    }

    let tail = span << 3;
    for i in tail..n {
        res[i] = res[i].wrapping_sub(a[i]);
    }
}

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`);
/// all inputs must have the same length and must not alias.
#[target_feature(enable = "avx512f")]
pub unsafe fn znx_sub_negate_assign_avx512(res: &mut [i64], a: &[i64]) {
    debug_assert_eq!(res.len(), a.len());

    let n = res.len();
    let span = n >> 3;

    unsafe {
        let mut rr = res.as_mut_ptr() as *mut __m512i;
        let mut aa = a.as_ptr() as *const __m512i;

        for _ in 0..span {
            _mm512_storeu_si512(
                rr,
                _mm512_sub_epi64(_mm512_loadu_si512(aa), _mm512_loadu_si512(rr as *const _)),
            );
            rr = rr.add(1);
            aa = aa.add(1);
        }
    }

    let tail = span << 3;
    for i in tail..n {
        res[i] = a[i].wrapping_sub(res[i]);
    }
}
