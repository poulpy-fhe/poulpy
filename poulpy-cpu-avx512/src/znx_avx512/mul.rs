use core::arch::x86_64::{
    __m128i, __m512i, _mm_cvtsi32_si128, _mm512_add_epi64, _mm512_loadu_si512, _mm512_set1_epi64, _mm512_sll_epi64,
    _mm512_sra_epi64, _mm512_srli_epi64, _mm512_storeu_si512, _mm512_sub_epi64,
};

use super::znx_add_assign_avx512;

/// Multiply/divide by a power of two with rounding matching [poulpy_cpu_ref::reference::znx::znx_mul_power_of_two_ref].
///
/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`);
/// all inputs must have the same length and must not alias.
#[target_feature(enable = "avx512f")]
pub unsafe fn znx_mul_power_of_two_avx512(k: i64, res: &mut [i64], a: &[i64]) {
    debug_assert_eq!(res.len(), a.len());

    let n = res.len();
    if n == 0 {
        return;
    }
    if k == 0 {
        res.copy_from_slice(a);
        return;
    }

    let span = n >> 3;

    if k > 0 {
        assert!((1..=63).contains(&k));
        unsafe {
            let cnt: __m128i = _mm_cvtsi32_si128(k as i32);
            let mut rr = res.as_mut_ptr() as *mut __m512i;
            let mut aa = a.as_ptr() as *const __m512i;
            for _ in 0..span {
                _mm512_storeu_si512(rr, _mm512_sll_epi64(_mm512_loadu_si512(aa), cnt));
                rr = rr.add(1);
                aa = aa.add(1);
            }
        }
        for i in (span << 3)..n {
            res[i] = a[i].wrapping_shl(k as u32);
        }
        return;
    }

    // k < 0: arithmetic right shift with rounding
    let kp = -k;
    assert!((1..=63).contains(&kp));
    unsafe {
        let cnt: __m128i = _mm_cvtsi32_si128(kp as i32);
        let bias_base: __m512i = _mm512_set1_epi64(1_i64 << (kp - 1));
        let mut rr = res.as_mut_ptr() as *mut __m512i;
        let mut aa = a.as_ptr() as *const __m512i;

        for _ in 0..span {
            let x = _mm512_loadu_si512(aa);
            let sign_bit = _mm512_srli_epi64(x, 63);
            let bias = _mm512_sub_epi64(bias_base, sign_bit);
            let y = _mm512_sra_epi64(_mm512_add_epi64(x, bias), cnt);
            _mm512_storeu_si512(rr, y);
            rr = rr.add(1);
            aa = aa.add(1);
        }
    }

    for i in (span << 3)..n {
        let x = a[i];
        let sign_bit = (x as u64 >> 63) as i64;
        let bias = (1_i64 << (kp - 1)) - sign_bit;
        res[i] = (x + bias) >> kp;
    }
}

/// Multiply/divide inplace by a power of two with rounding matching
/// [poulpy_cpu_ref::reference::znx::znx_mul_power_of_two_assign_ref].
///
/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`);
/// all inputs must have the same length and must not alias.
#[target_feature(enable = "avx512f")]
pub unsafe fn znx_mul_power_of_two_assign_avx512(k: i64, res: &mut [i64]) {
    let n = res.len();
    if n == 0 || k == 0 {
        return;
    }

    let span = n >> 3;

    if k > 0 {
        assert!((1..=63).contains(&k));
        unsafe {
            let cnt: __m128i = _mm_cvtsi32_si128(k as i32);
            let mut rr = res.as_mut_ptr() as *mut __m512i;
            for _ in 0..span {
                _mm512_storeu_si512(rr, _mm512_sll_epi64(_mm512_loadu_si512(rr as *const _), cnt));
                rr = rr.add(1);
            }
        }
        for v in &mut res[(span << 3)..n] {
            *v = v.wrapping_shl(k as u32);
        }
        return;
    }

    let kp = -k;
    assert!((1..=63).contains(&kp));
    unsafe {
        let cnt: __m128i = _mm_cvtsi32_si128(kp as i32);
        let bias_base: __m512i = _mm512_set1_epi64(1_i64 << (kp - 1));
        let mut rr = res.as_mut_ptr() as *mut __m512i;

        for _ in 0..span {
            let x = _mm512_loadu_si512(rr as *const _);
            let sign_bit = _mm512_srli_epi64(x, 63);
            let bias = _mm512_sub_epi64(bias_base, sign_bit);
            _mm512_storeu_si512(rr, _mm512_sra_epi64(_mm512_add_epi64(x, bias), cnt));
            rr = rr.add(1);
        }
    }

    for v in &mut res[(span << 3)..n] {
        let x = *v;
        let sign_bit = (x as u64 >> 63) as i64;
        let bias = (1_i64 << (kp - 1)) - sign_bit;
        *v = (x + bias) >> kp;
    }
}

/// Multiply/divide by a power of two and add on the result with rounding matching
/// [poulpy_cpu_ref::reference::znx::znx_mul_power_of_two_assign_ref].
///
/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`);
/// all inputs must have the same length and must not alias.
#[target_feature(enable = "avx512f")]
pub unsafe fn znx_mul_add_power_of_two_avx512(k: i64, res: &mut [i64], a: &[i64]) {
    debug_assert_eq!(res.len(), a.len());

    let n = res.len();
    if n == 0 {
        return;
    }
    if k == 0 {
        unsafe { znx_add_assign_avx512(res, a) };
        return;
    }

    let span = n >> 3;

    if k > 0 {
        assert!((1..=63).contains(&k));
        unsafe {
            let cnt: __m128i = _mm_cvtsi32_si128(k as i32);
            let mut rr = res.as_mut_ptr() as *mut __m512i;
            let mut aa = a.as_ptr() as *const __m512i;
            for _ in 0..span {
                let x = _mm512_loadu_si512(aa);
                let y = _mm512_loadu_si512(rr as *const _);
                _mm512_storeu_si512(rr, _mm512_add_epi64(y, _mm512_sll_epi64(x, cnt)));
                rr = rr.add(1);
                aa = aa.add(1);
            }
        }
        for i in (span << 3)..n {
            res[i] = res[i].wrapping_add(a[i].wrapping_shl(k as u32));
        }
        return;
    }

    let kp = -k;
    assert!((1..=63).contains(&kp));
    unsafe {
        let cnt: __m128i = _mm_cvtsi32_si128(kp as i32);
        let bias_base: __m512i = _mm512_set1_epi64(1_i64 << (kp - 1));
        let mut rr = res.as_mut_ptr() as *mut __m512i;
        let mut aa = a.as_ptr() as *const __m512i;

        for _ in 0..span {
            let x = _mm512_loadu_si512(aa);
            let y = _mm512_loadu_si512(rr as *const _);
            let sign_bit = _mm512_srli_epi64(x, 63);
            let bias = _mm512_sub_epi64(bias_base, sign_bit);
            let shifted = _mm512_sra_epi64(_mm512_add_epi64(x, bias), cnt);
            _mm512_storeu_si512(rr, _mm512_add_epi64(y, shifted));
            rr = rr.add(1);
            aa = aa.add(1);
        }
    }

    for i in (span << 3)..n {
        let x = a[i];
        let sign_bit = (x as u64 >> 63) as i64;
        let bias = (1_i64 << (kp - 1)) - sign_bit;
        res[i] = res[i].wrapping_add((x + bias) >> kp);
    }
}
