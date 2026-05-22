//! NEON sign-extending i64 → i128 helpers for `VecZnxBig` arithmetic.
//!
//! Mirrors `vi128_{from_small,neg_from_small}_avx2` in
//! `poulpy-cpu-avx/src/ntt120/vec_znx_big_avx.rs`. NEON has 128-bit registers
//! (2 × i64 lanes), so each iteration processes two i128 coefficients: load
//! 4 × i64, sign-extend low → hi via `vshrq_n_s64::<63>`, optionally negate,
//! re-interleave, store. Tail of `n % 2` handled scalar.
//!
//! The remaining i128 ops (add / sub / negate family) fall back to the
//! `I128BigOps` trait defaults because NEON has no native 128-bit add and no
//! widening i64×i64, so the hand kernel didn't beat scalar autovec.
//!
//! Wired into the `I128BigOps` impl in `crate::ntt120::vec_znx_big` under
//! `#[cfg(target_arch = "aarch64")]`.

use core::arch::aarch64::{
    int64x2_t, uint64x2_t, vaddq_s64, vdupq_n_s64, vld1q_s64, vreinterpretq_s64_u64, vreinterpretq_u64_s64, vst1q_s64, vsubq_s64,
    vsubq_u64, vzip1q_s64, vzip2q_s64,
};

// ─── private split-i128 primitives (operate on 2 coefficients per call) ───

#[inline(always)]
unsafe fn load2_i64_as_i128(p: *const i64) -> (uint64x2_t, int64x2_t) {
    unsafe {
        let lo_s: int64x2_t = vld1q_s64(p); // [a0, a1] sign-extended via lane copy
        let hi: int64x2_t = vshrq_helper_arith_63(lo_s);
        (vreinterpretq_u64_s64(lo_s), hi)
    }
}

/// Arithmetic right shift by 63 to sign-extend each lane (sign bit replicated).
#[inline(always)]
unsafe fn vshrq_helper_arith_63(v: int64x2_t) -> int64x2_t {
    unsafe { core::arch::aarch64::vshrq_n_s64::<63>(v) }
}

#[inline(always)]
unsafe fn store2_i128(p: *mut i128, lo: uint64x2_t, hi: int64x2_t) {
    unsafe {
        let lo_s: int64x2_t = vreinterpretq_s64_u64(lo);
        let v0: int64x2_t = vzip1q_s64(lo_s, hi); // [lo0, hi0]
        let v1: int64x2_t = vzip2q_s64(lo_s, hi); // [lo1, hi1]
        vst1q_s64(p as *mut i64, v0);
        vst1q_s64((p as *mut i64).add(2), v1);
    }
}

/// `(lo_r, hi_r) = -(lo_a, hi_a)` over 2 lanes of i128: `(0, 0) - (lo_a, hi_a)`.
#[inline(always)]
unsafe fn neg2_i128(lo_a: uint64x2_t, hi_a: int64x2_t) -> (uint64x2_t, int64x2_t) {
    unsafe {
        let zero_u: uint64x2_t = core::mem::transmute([0u64, 0u64]);
        let zero_s: int64x2_t = vdupq_n_s64(0);
        let lo_r: uint64x2_t = vsubq_u64(zero_u, lo_a);
        // borrow mask: -1 where lo_a != 0.
        let borrow_mask: int64x2_t = vreinterpretq_s64_u64(core::arch::aarch64::vcltq_u64(zero_u, lo_a));
        let hi_r: int64x2_t = vaddq_s64(vsubq_s64(zero_s, hi_a), borrow_mask);
        (lo_r, hi_r)
    }
}

// ─── public NEON kernels ──────────────────────────────────────────────────

/// `res[i] = a[i] as i128` for `n` elements (sign-extend i64 to i128).
pub(crate) fn vi128_from_small_neon(n: usize, res: &mut [i128], a: &[i64]) {
    debug_assert!(res.len() >= n && a.len() >= n);
    let chunks = n >> 1;
    let mut rr = res.as_mut_ptr();
    let mut aa = a.as_ptr();
    unsafe {
        for _ in 0..chunks {
            let (lo_a, hi_a) = load2_i64_as_i128(aa);
            store2_i128(rr, lo_a, hi_a);
            rr = rr.add(2);
            aa = aa.add(2);
        }
    }
    let tail = chunks << 1;
    for i in tail..n {
        res[i] = a[i] as i128;
    }
}

/// `res[i] = -(a[i] as i128)` for `n` elements (sign-extend then negate).
pub(crate) fn vi128_neg_from_small_neon(n: usize, res: &mut [i128], a: &[i64]) {
    debug_assert!(res.len() >= n && a.len() >= n);
    let chunks = n >> 1;
    let mut rr = res.as_mut_ptr();
    let mut aa = a.as_ptr();
    unsafe {
        for _ in 0..chunks {
            let (lo_a, hi_a) = load2_i64_as_i128(aa);
            let (lo_r, hi_r) = neg2_i128(lo_a, hi_a);
            store2_i128(rr, lo_r, hi_r);
            rr = rr.add(2);
            aa = aa.add(2);
        }
    }
    let tail = chunks << 1;
    for i in tail..n {
        res[i] = (a[i] as i128).wrapping_neg();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    /// Sizes exercising the SIMD body (n >= 2, even) and the scalar tail (n % 2 != 0).
    const SIZES: &[usize] = &[0, 1, 2, 3, 4, 5, 7, 8, 16, 17, 64, 65];

    fn rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(0x5a5a_5a5a_5a5a_5a5a)
    }

    fn random_i64_vec(rng: &mut ChaCha8Rng, n: usize) -> Vec<i64> {
        (0..n).map(|_| rng.random::<i64>()).collect()
    }

    #[test]
    fn vi128_from_small_matches_scalar() {
        let mut rng = rng();
        for &n in SIZES {
            let a = random_i64_vec(&mut rng, n);
            let want: Vec<i128> = a.iter().map(|&x| x as i128).collect();
            let mut got = vec![0i128; n];
            vi128_from_small_neon(n, &mut got, &a);
            assert_eq!(got, want, "vi128_from_small_neon mismatch at n={n}");
        }
    }

    #[test]
    fn vi128_neg_from_small_matches_scalar() {
        let mut rng = rng();
        for &n in SIZES {
            let a = random_i64_vec(&mut rng, n);
            let want: Vec<i128> = a.iter().map(|&x| (x as i128).wrapping_neg()).collect();
            let mut got = vec![0i128; n];
            vi128_neg_from_small_neon(n, &mut got, &a);
            assert_eq!(got, want, "vi128_neg_from_small_neon mismatch at n={n}");
        }
    }

    /// i64 sign-extension boundaries.
    #[test]
    fn vi128_small_sign_extend_boundaries() {
        let small = vec![i64::MIN, -1, 0, 1, i64::MAX, i64::MIN + 1, i64::MAX - 1, 42];
        let n = small.len();
        let mut got = vec![0i128; n];

        let want: Vec<i128> = small.iter().map(|&x| x as i128).collect();
        vi128_from_small_neon(n, &mut got, &small);
        assert_eq!(got, want, "boundary from_small");

        let want: Vec<i128> = small.iter().map(|&x| (x as i128).wrapping_neg()).collect();
        vi128_neg_from_small_neon(n, &mut got, &small);
        assert_eq!(got, want, "boundary neg_from_small");
    }
}
