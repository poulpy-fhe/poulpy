use core::arch::x86_64::*;

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`);
/// all inputs must have compatible ring dimensions and must not alias.
#[target_feature(enable = "avx512f")]
pub unsafe fn znx_switch_ring_avx512(res: &mut [i64], a: &[i64]) {
    let (n_in, n_out) = (a.len(), res.len());

    debug_assert!(n_in.is_power_of_two());
    debug_assert!(n_in.max(n_out).is_multiple_of(n_in.min(n_out)));

    if n_in == n_out {
        res.copy_from_slice(a);
        return;
    }

    if n_in > n_out {
        // Downsample: res[k] = a[k * gap]
        let gap = n_in / n_out;
        let span = n_out >> 3;

        unsafe {
            let step = _mm512_setr_epi64(
                0,
                gap as i64,
                2 * gap as i64,
                3 * gap as i64,
                4 * gap as i64,
                5 * gap as i64,
                6 * gap as i64,
                7 * gap as i64,
            );
            let bump = _mm512_set1_epi64(8 * gap as i64);
            let a_ptr = a.as_ptr();

            let mut rr = res.as_mut_ptr() as *mut __m512i;
            let mut base = _mm512_setzero_si512();

            for _ in 0..span {
                let idx = _mm512_add_epi64(base, step);
                let v = _mm512_i64gather_epi64(idx, a_ptr, 8);
                _mm512_storeu_si512(rr, v);
                base = _mm512_add_epi64(base, bump);
                rr = rr.add(1);
            }
        }

        for k in (span << 3)..n_out {
            res[k] = a[k * gap];
        }
    } else {
        // Upsample: res[k * gap] = a[k], rest zero
        let gap = n_out / n_in;

        poulpy_cpu_ref::reference::znx::znx_zero_ref(res);

        let span = n_in >> 3;

        unsafe {
            let step = _mm512_setr_epi64(
                0,
                gap as i64,
                2 * gap as i64,
                3 * gap as i64,
                4 * gap as i64,
                5 * gap as i64,
                6 * gap as i64,
                7 * gap as i64,
            );
            let bump = _mm512_set1_epi64(8 * gap as i64);

            let mut aa = a.as_ptr() as *const __m512i;
            let res_ptr = res.as_mut_ptr();
            let mut base = _mm512_setzero_si512();

            for _ in 0..span {
                let v = _mm512_loadu_si512(aa);
                let idx = _mm512_add_epi64(base, step);
                _mm512_i64scatter_epi64(res_ptr, idx, v, 8);
                base = _mm512_add_epi64(base, bump);
                aa = aa.add(1);
            }
        }

        for k in (span << 3)..n_in {
            res[k * gap] = a[k];
        }
    }
}
