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

/// Fused automorphism + rotation: computes `res = X^k * auto(p, a)`.
///
/// Identical to [`znx_automorphism_avx512`](super::znx_automorphism_avx512) except
/// the running gather index starts at `t_base = (-k·p⁻¹) mod 2n` instead of `0`,
/// shifting every output position by `k` (the rotation). See the AVX2 variant for
/// the derivation.
///
/// # Safety
/// Caller must ensure the CPU supports AVX-512F (e.g., via `is_x86_feature_detected!("avx512f")`);
/// all inputs must have the same length and must not alias.
#[target_feature(enable = "avx512f")]
pub unsafe fn znx_automorphism_rotate_avx512(p: i64, k: i64, res: &mut [i64], a: &[i64]) {
    debug_assert_eq!(res.len(), a.len());
    let n = res.len();
    if n == 0 {
        return;
    }
    assert!(n.is_power_of_two());
    debug_assert!(p & 1 == 1);

    if n < 8 {
        use poulpy_cpu_ref::reference::znx::znx_automorphism_rotate_ref;
        znx_automorphism_rotate_ref(p, k, res, a);
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

        // Rotation shift: start at (-k * inv) mod 2n so that the gathered position
        // for global index j becomes (j - k) * inv mod 2n.
        let k_2n: usize = (((k & mask_2n as i64) + two_n as i64) as usize) & mask_2n;
        let mut t_base: usize = (two_n - ((k_2n.wrapping_mul(inv)) & mask_2n)) & mask_2n;
        let step = (inv << 3) & mask_2n;

        let mut rr = res.as_mut_ptr() as *mut __m512i;
        let aa = a.as_ptr();

        for _ in 0..span {
            let t_base_vec = _mm512_set1_epi64(t_base as i64);
            let t_vec = _mm512_and_si512(_mm512_add_epi64(t_base_vec, lane_offsets), mask_2n_vec);
            let idx_vec = _mm512_and_si512(t_vec, mask_1n_vec);

            let sign_k: __mmask8 = _mm512_cmpgt_epi64_mask(t_vec, n_minus1_vec);

            let vals = _mm512_i64gather_epi64(idx_vec, aa, 8);
            // Conditional negate under `sign_k` (0 - vals in flagged lanes). Stays
            // within AVX-512F: `_mm512_movm_epi64` + xor/sub would pull in AVX-512DQ,
            // which is outside this crate's compile-time baseline and blocks inlining.
            let out = _mm512_mask_sub_epi64(vals, sign_k, _mm512_setzero_si512(), vals);

            _mm512_storeu_si512(rr, out);
            rr = rr.add(1);
            t_base = (t_base + step) & mask_2n;
        }
    }
}

#[cfg(test)]
mod tests {
    use poulpy_cpu_ref::reference::znx::znx_automorphism_rotate_ref;

    use super::*;

    #[target_feature(enable = "avx512f")]
    unsafe fn test_znx_automorphism_rotate_internal() {
        let a: [i64; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        for p in [1i64, 3, 5, -5, 7, -11] {
            for k in [0i64, 1, 3, 8, 15, -1, -7, 31] {
                let mut r0 = vec![0i64; a.len()];
                let mut r1 = vec![0i64; a.len()];
                unsafe {
                    znx_automorphism_rotate_ref(p, k, &mut r0, &a);
                    znx_automorphism_rotate_avx512(p, k, &mut r1, &a);
                }
                assert_eq!(r0, r1, "mismatch for p={p}, k={k}");
            }
        }
    }

    #[test]
    fn test_znx_automorphism_rotate_avx512() {
        if !std::is_x86_feature_detected!("avx512f") {
            eprintln!("skipping: CPU lacks avx512f");
            return;
        }
        unsafe {
            test_znx_automorphism_rotate_internal();
        }
    }
}
