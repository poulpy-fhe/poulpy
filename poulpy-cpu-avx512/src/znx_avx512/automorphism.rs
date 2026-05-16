use core::arch::x86_64::*;

#[inline]
fn inv_mod_pow2(p: usize, bits: u32) -> usize {
    debug_assert!(p % 2 == 1);
    let mut x: usize = 1usize;
    let mut i: u32 = 1;
    while i < bits {
        x = x.wrapping_mul(2usize.wrapping_sub(p.wrapping_mul(x)));
        i <<= 1;
    }
    x & ((1usize << bits) - 1)
}

/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`);
/// all inputs must have the same length and must not alias.
#[target_feature(enable = "avx512f")]
pub unsafe fn znx_automorphism_avx512(p: i64, res: &mut [i64], a: &[i64]) {
    debug_assert_eq!(res.len(), a.len());
    let n = res.len();
    if n == 0 {
        return;
    }
    assert!(n.is_power_of_two());
    debug_assert!(p & 1 == 1);

    if n < 8 {
        use poulpy_cpu_ref::reference::znx::znx_automorphism_ref;
        znx_automorphism_ref(p, res, a);
        return;
    }

    let two_n = n << 1;
    let span = n >> 3;
    let bits = (two_n as u64).trailing_zeros();
    let mask_2n = two_n - 1;
    let mask_1n = n - 1;

    let p_2n = (((p & mask_2n as i64) + two_n as i64) as usize) & mask_2n;
    let inv = inv_mod_pow2(p_2n, bits);

    unsafe {
        let n_minus1_vec = _mm512_set1_epi64((n as i64) - 1);
        let mask_2n_vec = _mm512_set1_epi64(mask_2n as i64);
        let mask_1n_vec = _mm512_set1_epi64(mask_1n as i64);

        let lane_offsets = _mm512_set_epi64(
            ((inv * 7) & mask_2n) as i64,
            ((inv * 6) & mask_2n) as i64,
            ((inv * 5) & mask_2n) as i64,
            ((inv * 4) & mask_2n) as i64,
            ((inv * 3) & mask_2n) as i64,
            ((inv * 2) & mask_2n) as i64,
            inv as i64,
            0i64,
        );

        let mut t_base: usize = 0;
        let step = (inv << 3) & mask_2n;

        let mut rr = res.as_mut_ptr() as *mut __m512i;
        let aa = a.as_ptr();

        for _ in 0..span {
            let t_base_vec = _mm512_set1_epi64(t_base as i64);
            let t_vec = _mm512_and_si512(_mm512_add_epi64(t_base_vec, lane_offsets), mask_2n_vec);
            let idx_vec = _mm512_and_si512(t_vec, mask_1n_vec);

            let sign_k: __mmask8 = _mm512_cmpgt_epi64_mask(t_vec, n_minus1_vec);
            let sign_mask: __m512i = _mm512_movm_epi64(sign_k);

            let vals = _mm512_i64gather_epi64(idx_vec, aa, 8);
            let out = _mm512_sub_epi64(_mm512_xor_si512(vals, sign_mask), sign_mask);

            _mm512_storeu_si512(rr, out);
            rr = rr.add(1);
            t_base = (t_base + step) & mask_2n;
        }
    }
}

#[cfg(test)]
mod tests {
    use poulpy_cpu_ref::reference::znx::znx_automorphism_ref;

    use super::*;

    #[target_feature(enable = "avx512f")]
    unsafe fn test_znx_automorphism_internal() {
        let a: [i64; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let p: i64 = -5;
        let mut r0 = vec![0i64; a.len()];
        let mut r1 = vec![0i64; a.len()];
        unsafe {
            znx_automorphism_ref(p, &mut r0, &a);
            znx_automorphism_avx512(p, &mut r1, &a);
        }
        assert_eq!(r0, r1);
    }

    #[test]
    fn test_znx_automorphism_avx512() {
        unsafe {
            test_znx_automorphism_internal();
        }
    }
}
