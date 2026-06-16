use core::arch::x86_64::*;

#[inline]
fn inv_mod_pow2(p: usize, bits: u32) -> usize {
    // Compute p^{-1} mod 2^bits (p must be odd) through Hensel lifting.
    debug_assert!(p % 2 == 1);
    let mut x: usize = 1usize; // inverse mod 2
    let mut i: u32 = 1;
    while i < bits {
        // x <- x * (2 - p*x)  mod 2^(2^i)  (wrapping arithmetic)
        x = x.wrapping_mul(2usize.wrapping_sub(p.wrapping_mul(x)));
        i <<= 1;
    }
    x & ((1usize << bits) - 1)
}

/// Fused automorphism + rotation: computes `res = X^k * auto(p, a)`.
///
/// The automorphism `auto(p, ·)` gather form computes `out[j] = sign(t)·a[t & (n-1)]`
/// with `t = j·p⁻¹ mod 2n`. Multiplying by `X^k` shifts every output index by `k`,
/// i.e. it replaces `t` by `(j - k)·p⁻¹ mod 2n`. Since the lane step is unchanged,
/// this is exactly the automorphism kernel started at `t_base = (-k·p⁻¹) mod 2n`
/// instead of `0`.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 (e.g., via `is_x86_feature_detected!("avx2")`);
/// all inputs must have the same length and must not alias.
#[target_feature(enable = "avx2", enable = "fma")]
pub fn znx_automorphism_rotate_avx(p: i64, k: i64, res: &mut [i64], a: &[i64]) {
    debug_assert_eq!(res.len(), a.len());
    let n: usize = res.len();
    if n == 0 {
        return;
    }
    assert!(n.is_power_of_two(), "Polynomial degree {} must be power of 2", n);
    debug_assert!(p & 1 == 1, "p must be odd (invertible mod 2n)");

    if n < 4 {
        use poulpy_cpu_ref::reference::znx::znx_automorphism_rotate_ref;

        znx_automorphism_rotate_ref(p, k, res, a);
        return;
    }

    unsafe {
        let two_n: usize = n << 1;
        let span: usize = n >> 2;
        let bits: u32 = (two_n as u64).trailing_zeros();
        let mask_2n: usize = two_n - 1;
        let mask_1n: usize = n - 1;

        // p mod 2n (positive)
        let p_2n: usize = (((p & mask_2n as i64) + two_n as i64) as usize) & mask_2n;

        // p^-1 mod 2n
        let inv: usize = inv_mod_pow2(p_2n, bits);

        // Broadcast constants
        let n_minus1_vec: __m256i = _mm256_set1_epi64x((n as i64) - 1);
        let mask_2n_vec: __m256i = _mm256_set1_epi64x(mask_2n as i64);
        let mask_1n_vec: __m256i = _mm256_set1_epi64x(mask_1n as i64);

        // Lane offsets [0, inv, 2*inv, 3*inv] (mod 2n)
        let lane_offsets: __m256i =
            _mm256_set_epi64x(((inv * 3) & mask_2n) as i64, ((inv * 2) & mask_2n) as i64, inv as i64, 0i64);

        // Rotation shift: start the running index at (-k * inv) mod 2n so that the
        // gathered position for global index j becomes (j - k) * inv mod 2n.
        let k_2n: usize = (((k & mask_2n as i64) + two_n as i64) as usize) & mask_2n;
        let mut t_base: usize = (two_n - ((k_2n.wrapping_mul(inv)) & mask_2n)) & mask_2n;
        let step: usize = (inv << 2) & mask_2n;

        let mut rr: *mut __m256i = res.as_mut_ptr() as *mut __m256i;
        let aa: *const i64 = a.as_ptr();

        for _ in 0..span {
            // t_vec = (t_base + [0, inv, 2*inv, 3*inv]) & (2n-1)
            let t_base_vec: __m256i = _mm256_set1_epi64x(t_base as i64);
            let t_vec: __m256i = _mm256_and_si256(_mm256_add_epi64(t_base_vec, lane_offsets), mask_2n_vec);

            // idx = t_vec & (n-1)
            let idx_vec: __m256i = _mm256_and_si256(t_vec, mask_1n_vec);

            // sign = t >= n ? -1 : 0  (mask of all-ones where negate)
            let sign_mask: __m256i = _mm256_cmpgt_epi64(t_vec, n_minus1_vec);

            // gather a[idx] (scale = 8 bytes per i64)
            let vals: __m256i = _mm256_i64gather_epi64(aa, idx_vec, 8);

            // Conditional negate: (vals ^ sign_mask) - sign_mask
            let vals_x: __m256i = _mm256_xor_si256(vals, sign_mask);
            let out: __m256i = _mm256_sub_epi64(vals_x, sign_mask);

            // store to res[j..j+4]
            _mm256_storeu_si256(rr, out);

            // advance
            rr = rr.add(1);
            t_base = (t_base + step) & mask_2n;
        }
    }
}

#[cfg(test)]
mod tests {
    use poulpy_cpu_ref::reference::znx::znx_automorphism_rotate_ref;

    use super::*;

    #[allow(dead_code)]
    #[target_feature(enable = "avx2", enable = "fma")]
    fn test_znx_automorphism_rotate_internal() {
        let a: [i64; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

        for p in [1i64, 3, 5, -5, 7, -11] {
            for k in [0i64, 1, 3, 8, 15, -1, -7, 31] {
                let mut r0: Vec<i64> = vec![0i64; a.len()];
                let mut r1: Vec<i64> = vec![0i64; a.len()];

                znx_automorphism_rotate_ref(p, k, &mut r0, &a);
                znx_automorphism_rotate_avx(p, k, &mut r1, &a);

                assert_eq!(r0, r1, "mismatch for p={p}, k={k}");
            }
        }
    }

    #[test]
    fn test_znx_automorphism_rotate_avx() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("skipping: CPU lacks avx2");
            return;
        };
        unsafe {
            test_znx_automorphism_rotate_internal();
        }
    }
}
