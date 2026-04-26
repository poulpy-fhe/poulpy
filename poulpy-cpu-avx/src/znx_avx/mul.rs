/// Multiply/divide by a power of two with rounding matching [poulpy_cpu_ref::reference::znx::znx_mul_power_of_two_ref].
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
/// all inputs must have the same length and must not alias.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn znx_mul_power_of_two_avx(k: i64, res: &mut [i64], a: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len());
    }

    use core::arch::x86_64::{
        __m128i, __m256i, _mm_cvtsi32_si128, _mm256_add_epi64, _mm256_and_si256, _mm256_cmpgt_epi64, _mm256_loadu_si256,
        _mm256_or_si256, _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_sll_epi64, _mm256_srl_epi64, _mm256_srli_epi64,
        _mm256_storeu_si256, _mm256_sub_epi64,
    };

    let n: usize = res.len();

    if n == 0 {
        return;
    }

    if k == 0 {
        use poulpy_cpu_ref::reference::znx::znx_copy_ref;
        znx_copy_ref(res, a);
        return;
    }

    let span: usize = n >> 2; // number of 256-bit chunks

    unsafe {
        let mut rr: *mut __m256i = res.as_mut_ptr() as *mut __m256i;
        let mut aa: *const __m256i = a.as_ptr() as *const __m256i;

        if k > 0 {
            // Left shift by k (variable count).
            #[cfg(debug_assertions)]
            {
                debug_assert!(k <= 63);
            }
            let cnt128: __m128i = _mm_cvtsi32_si128(k as i32);
            for _ in 0..span {
                let x: __m256i = _mm256_loadu_si256(aa);
                let y: __m256i = _mm256_sll_epi64(x, cnt128);
                _mm256_storeu_si256(rr, y);
                rr = rr.add(1);
                aa = aa.add(1);
            }

            // tail
            if !n.is_multiple_of(4) {
                use poulpy_cpu_ref::reference::znx::znx_mul_power_of_two_ref;

                znx_mul_power_of_two_ref(k, &mut res[span << 2..], &a[span << 2..]);
            }
            return;
        }

        // k < 0  => arithmetic right shift with rounding:
        // for each x:
        //   sign_bit = (x >> 63) & 1
        //   bias = (1<<(kp-1)) - sign_bit
        //   t = x + bias
        //   y = t >> kp   (arithmetic)
        let kp = -k;
        assert!((1..=63).contains(&kp), "kp must be in [1, 63], got {}", kp);

        let cnt_right: __m128i = _mm_cvtsi32_si128(kp as i32);
        let bias_base: __m256i = _mm256_set1_epi64x(1_i64 << (kp - 1));
        let top_mask: __m256i = _mm256_set1_epi64x(-1_i64 << (64 - kp)); // high kp bits
        let zero: __m256i = _mm256_setzero_si256();

        for _ in 0..span {
            let x = _mm256_loadu_si256(aa);

            // bias = (1 << (kp-1)) - sign_bit
            let sign_bit_x: __m256i = _mm256_srli_epi64(x, 63);
            let bias: __m256i = _mm256_sub_epi64(bias_base, sign_bit_x);

            // t = x + bias
            let t: __m256i = _mm256_add_epi64(x, bias);

            // logical shift
            let lsr: __m256i = _mm256_srl_epi64(t, cnt_right);

            // sign extension
            let neg: __m256i = _mm256_cmpgt_epi64(zero, t);
            let fill: __m256i = _mm256_and_si256(neg, top_mask);
            let y: __m256i = _mm256_or_si256(lsr, fill);

            _mm256_storeu_si256(rr, y);
            rr = rr.add(1);
            aa = aa.add(1);
        }
    }

    // tail
    if !n.is_multiple_of(4) {
        use poulpy_cpu_ref::reference::znx::znx_mul_power_of_two_ref;

        znx_mul_power_of_two_ref(k, &mut res[span << 2..], &a[span << 2..]);
    }
}

/// Multiply/divide inplace by a power of two with rounding matching [poulpy_cpu_ref::reference::znx::znx_mul_power_of_two_assign_ref].
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
/// all inputs must have the same length and must not alias.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn znx_mul_power_of_two_assign_avx(k: i64, res: &mut [i64]) {
    use core::arch::x86_64::{
        __m128i, __m256i, _mm_cvtsi32_si128, _mm256_add_epi64, _mm256_and_si256, _mm256_cmpgt_epi64, _mm256_loadu_si256,
        _mm256_or_si256, _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_sll_epi64, _mm256_srl_epi64, _mm256_srli_epi64,
        _mm256_storeu_si256, _mm256_sub_epi64,
    };

    let n: usize = res.len();

    if n == 0 {
        return;
    }

    if k == 0 {
        return;
    }

    let span: usize = n >> 2; // number of 256-bit chunks

    unsafe {
        let mut rr: *mut __m256i = res.as_mut_ptr() as *mut __m256i;

        if k > 0 {
            // Left shift by k (variable count).
            #[cfg(debug_assertions)]
            {
                debug_assert!(k <= 63);
            }
            let cnt128: __m128i = _mm_cvtsi32_si128(k as i32);
            for _ in 0..span {
                let x: __m256i = _mm256_loadu_si256(rr);
                let y: __m256i = _mm256_sll_epi64(x, cnt128);
                _mm256_storeu_si256(rr, y);
                rr = rr.add(1);
            }

            // tail
            if !n.is_multiple_of(4) {
                use poulpy_cpu_ref::reference::znx::znx_mul_power_of_two_assign_ref;
                znx_mul_power_of_two_assign_ref(k, &mut res[span << 2..]);
            }
            return;
        }

        // k < 0  => arithmetic right shift with rounding:
        // for each x:
        //   sign_bit = (x >> 63) & 1
        //   bias = (1<<(kp-1)) - sign_bit
        //   t = x + bias
        //   y = t >> kp   (arithmetic)
        let kp = -k;
        assert!((1..=63).contains(&kp), "kp must be in [1, 63], got {}", kp);

        let cnt_right: __m128i = _mm_cvtsi32_si128(kp as i32);
        let bias_base: __m256i = _mm256_set1_epi64x(1_i64 << (kp - 1));
        let top_mask: __m256i = _mm256_set1_epi64x(-1_i64 << (64 - kp)); // high kp bits
        let zero: __m256i = _mm256_setzero_si256();

        for _ in 0..span {
            let x = _mm256_loadu_si256(rr);

            // bias = (1 << (kp-1)) - sign_bit
            let sign_bit_x: __m256i = _mm256_srli_epi64(x, 63);
            let bias: __m256i = _mm256_sub_epi64(bias_base, sign_bit_x);

            // t = x + bias
            let t: __m256i = _mm256_add_epi64(x, bias);

            // logical shift
            let lsr: __m256i = _mm256_srl_epi64(t, cnt_right);

            // sign extension
            let neg: __m256i = _mm256_cmpgt_epi64(zero, t);
            let fill: __m256i = _mm256_and_si256(neg, top_mask);
            let y: __m256i = _mm256_or_si256(lsr, fill);

            _mm256_storeu_si256(rr, y);
            rr = rr.add(1);
        }
    }

    // tail
    if !n.is_multiple_of(4) {
        use poulpy_cpu_ref::reference::znx::znx_mul_power_of_two_assign_ref;
        znx_mul_power_of_two_assign_ref(k, &mut res[span << 2..]);
    }
}

/// Multiply/divide by a power of two and add on the result with rounding matching [poulpy_cpu_ref::reference::znx::znx_mul_power_of_two_assign_ref].
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
/// all inputs must have the same length and must not alias.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn znx_mul_add_power_of_two_avx(k: i64, res: &mut [i64], a: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len());
    }

    use core::arch::x86_64::{
        __m128i, __m256i, _mm_cvtsi32_si128, _mm256_add_epi64, _mm256_and_si256, _mm256_cmpgt_epi64, _mm256_loadu_si256,
        _mm256_or_si256, _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_sll_epi64, _mm256_srl_epi64, _mm256_srli_epi64,
        _mm256_storeu_si256, _mm256_sub_epi64,
    };

    let n: usize = res.len();

    if n == 0 {
        return;
    }

    if k == 0 {
        use crate::znx_avx::znx_add_assign_avx;

        znx_add_assign_avx(res, a);
        return;
    }

    let span: usize = n >> 2; // number of 256-bit chunks

    unsafe {
        let mut rr: *mut __m256i = res.as_mut_ptr() as *mut __m256i;
        let mut aa: *const __m256i = a.as_ptr() as *const __m256i;

        if k > 0 {
            // Left shift by k (variable count).
            #[cfg(debug_assertions)]
            {
                debug_assert!(k <= 63);
            }
            let cnt128: __m128i = _mm_cvtsi32_si128(k as i32);
            for _ in 0..span {
                let x: __m256i = _mm256_loadu_si256(aa);
                let y: __m256i = _mm256_loadu_si256(rr);
                _mm256_storeu_si256(rr, _mm256_add_epi64(y, _mm256_sll_epi64(x, cnt128)));
                rr = rr.add(1);
                aa = aa.add(1);
            }

            // tail
            if !n.is_multiple_of(4) {
                use poulpy_cpu_ref::reference::znx::znx_mul_add_power_of_two_ref;

                znx_mul_add_power_of_two_ref(k, &mut res[span << 2..], &a[span << 2..]);
            }
            return;
        }

        // k < 0  => arithmetic right shift with rounding:
        // for each x:
        //   sign_bit = (x >> 63) & 1
        //   bias = (1<<(kp-1)) - sign_bit
        //   t = x + bias
        //   y = t >> kp   (arithmetic)
        let kp = -k;
        assert!((1..=63).contains(&kp), "kp must be in [1, 63], got {}", kp);

        let cnt_right: __m128i = _mm_cvtsi32_si128(kp as i32);
        let bias_base: __m256i = _mm256_set1_epi64x(1_i64 << (kp - 1));
        let top_mask: __m256i = _mm256_set1_epi64x(-1_i64 << (64 - kp)); // high kp bits
        let zero: __m256i = _mm256_setzero_si256();

        for _ in 0..span {
            let x: __m256i = _mm256_loadu_si256(aa);
            let y: __m256i = _mm256_loadu_si256(rr);

            // bias = (1 << (kp-1)) - sign_bit
            let sign_bit_x: __m256i = _mm256_srli_epi64(x, 63);
            let bias: __m256i = _mm256_sub_epi64(bias_base, sign_bit_x);

            // t = x + bias
            let t: __m256i = _mm256_add_epi64(x, bias);

            // logical shift
            let lsr: __m256i = _mm256_srl_epi64(t, cnt_right);

            // sign extension
            let neg: __m256i = _mm256_cmpgt_epi64(zero, t);
            let fill: __m256i = _mm256_and_si256(neg, top_mask);
            let out: __m256i = _mm256_or_si256(lsr, fill);

            _mm256_storeu_si256(rr, _mm256_add_epi64(y, out));
            rr = rr.add(1);
            aa = aa.add(1);
        }
    }

    // tail
    if !n.is_multiple_of(4) {
        use poulpy_cpu_ref::reference::znx::znx_mul_add_power_of_two_ref;
        znx_mul_add_power_of_two_ref(k, &mut res[span << 2..], &a[span << 2..]);
    }
}
