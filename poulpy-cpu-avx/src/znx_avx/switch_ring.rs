#[target_feature(enable = "avx2")]
pub unsafe fn znx_switch_ring_avx(res: &mut [i64], a: &[i64]) {
    unsafe {
        use core::arch::x86_64::*;

        let (n_in, n_out) = (a.len(), res.len());

        {
            assert!(n_in.is_power_of_two());
            assert!(n_in.max(n_out).is_multiple_of(n_in.min(n_out)))
        }

        if n_in == n_out {
            use poulpy_cpu_ref::reference::znx::znx_copy_ref;

            znx_copy_ref(res, a);
            return;
        }

        if n_in > n_out {
            // Downsample: res[k] = a[k * gap_in], contiguous stores
            let gap_in: usize = n_in / n_out;

            // index vector: [0*gap, 1*gap, 2*gap, 3*gap] * gap_in
            let step: __m256i = _mm256_setr_epi64x(0, gap_in as i64, 2 * gap_in as i64, 3 * gap_in as i64);

            let span: usize = n_out >> 2;
            let bump: __m256i = _mm256_set1_epi64x(4 * gap_in as i64);

            let mut res_4xi64: *mut __m256i = res.as_mut_ptr() as *mut __m256i;
            let a_ptr: *const i64 = a.as_ptr();

            let mut base: __m256i = _mm256_setzero_si256(); // starts at 0*gap

            for _ in 0..span {
                // idx = base + step
                let idx: __m256i = _mm256_add_epi64(base, step);

                // gather 4 spaced i64 (scale=8 bytes)
                let v: __m256i = _mm256_i64gather_epi64(a_ptr, idx, 8);

                // store contiguously
                _mm256_storeu_si256(res_4xi64, v);

                base = _mm256_add_epi64(base, bump);
                res_4xi64 = res_4xi64.add(1);
            }
        } else {
            // Upsample: res[k * gap_out] = a[k], i.e. res has holes;

            use poulpy_cpu_ref::reference::znx::znx_zero_ref;
            let gap_out = n_out / n_in;

            // zero then scatter scalar stores
            znx_zero_ref(res);

            let mut a_4xi64: *const __m256i = a.as_ptr() as *const __m256i;

            for i in (0..n_in).step_by(4) {
                // Load contiguously 4 inputs
                let v = _mm256_loadu_si256(a_4xi64);

                // extract 4 lanes (pextrq). This is still the best we can do on AVX2.
                let x0: i64 = _mm256_extract_epi64(v, 0);
                let x1: i64 = _mm256_extract_epi64(v, 1);
                let x2: i64 = _mm256_extract_epi64(v, 2);
                let x3: i64 = _mm256_extract_epi64(v, 3);

                // starting output pointer for this group
                let mut p: *mut i64 = res.as_mut_ptr().add(i * gap_out);

                // four strided stores with pointer bump (avoid mul each time)
                *p = x0;
                p = p.add(gap_out);
                *p = x1;
                p = p.add(gap_out);
                *p = x2;
                p = p.add(gap_out);
                *p = x3;

                a_4xi64 = a_4xi64.add(1)
            }
        }
    }
}
