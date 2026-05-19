use core::arch::x86_64::{__m512i, _mm512_loadu_si512, _mm512_setzero_si512, _mm512_storeu_si512, _mm512_sub_epi64};

/// AVX-512 vectorised `res[i] = -src[i]`.
#[target_feature(enable = "avx512f")]
pub unsafe fn znx_negate_avx512(res: &mut [i64], src: &[i64]) {
    debug_assert_eq!(res.len(), src.len());

    let n = res.len();
    let span = n >> 3;

    unsafe {
        let zero = _mm512_setzero_si512();
        let mut rr = res.as_mut_ptr() as *mut __m512i;
        let mut ss = src.as_ptr() as *const __m512i;

        for _ in 0..span {
            _mm512_storeu_si512(rr, _mm512_sub_epi64(zero, _mm512_loadu_si512(ss)));
            rr = rr.add(1);
            ss = ss.add(1);
        }
    }

    let tail = span << 3;
    for i in tail..n {
        res[i] = src[i].wrapping_neg();
    }
}

/// AVX-512 vectorised `res[i] = -res[i]`.
#[target_feature(enable = "avx512f")]
pub unsafe fn znx_negate_assign_avx512(res: &mut [i64]) {
    let n = res.len();
    let span = n >> 3;

    unsafe {
        let zero = _mm512_setzero_si512();
        let mut rr = res.as_mut_ptr() as *mut __m512i;

        for _ in 0..span {
            _mm512_storeu_si512(rr, _mm512_sub_epi64(zero, _mm512_loadu_si512(rr as *const _)));
            rr = rr.add(1);
        }
    }

    for v in &mut res[(span << 3)..n] {
        *v = v.wrapping_neg();
    }
}
