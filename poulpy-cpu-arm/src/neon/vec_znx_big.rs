//! NEON i128 paired-register helpers for `VecZnxBig` arithmetic.
//!
//! Mirrors the `vi128_*_avx2` family in
//! `poulpy-cpu-avx/src/ntt120/vec_znx_big_avx.rs`. NEON has 128-bit registers
//! (2 × i64 lanes), so each block iteration processes two i128 coefficients:
//! load 4 × i64, deinterleave into `(lo_vec, hi_vec)`, run the add/sub/neg
//! logic with a NEON unsigned compare for the carry/borrow, re-interleave,
//! store. Tail of `n % 2` handled scalar via `i128::wrapping_*`.
//!
//! Wired into the `I128BigOps` impl in `crate::ntt120::vec_znx_big` under
//! `#[cfg(target_arch = "aarch64")]`.

use core::arch::aarch64::{
    int64x2_t, uint64x2_t, vaddq_s64, vaddq_u64, vcltq_u64, vdupq_n_s64, vld1q_s64, vreinterpretq_s64_u64, vreinterpretq_u64_s64,
    vst1q_s64, vsubq_s64, vsubq_u64, vuzp1q_u64, vuzp2q_s64, vzip1q_s64, vzip2q_s64,
};

// ─── private split-i128 primitives (operate on 2 coefficients per call) ───

#[inline(always)]
unsafe fn load2_i128(p: *const i128) -> (uint64x2_t, int64x2_t) {
    unsafe {
        let v0: int64x2_t = vld1q_s64(p as *const i64); // [lo0, hi0]
        let v1: int64x2_t = vld1q_s64((p as *const i64).add(2)); // [lo1, hi1]
        let lo: uint64x2_t = vuzp1q_u64(vreinterpretq_u64_s64(v0), vreinterpretq_u64_s64(v1));
        let hi: int64x2_t = vuzp2q_s64(v0, v1);
        (lo, hi)
    }
}

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
    unsafe {
        // NEON has no immediate >>63 for int64x2_t directly via the signed
        // arithmetic right-shift intrinsic in stable; use vshrq_n_s64 with
        // the const generic.
        core::arch::aarch64::vshrq_n_s64::<63>(v)
    }
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

/// `(lo_r, hi_r) = (lo_a, hi_a) + (lo_b, hi_b)` over 2 lanes of i128.
#[inline(always)]
unsafe fn add2_i128(lo_a: uint64x2_t, hi_a: int64x2_t, lo_b: uint64x2_t, hi_b: int64x2_t) -> (uint64x2_t, int64x2_t) {
    unsafe {
        let lo_r: uint64x2_t = vaddq_u64(lo_a, lo_b);
        // carry mask: lanes where lo_r < lo_a (unsigned overflow) = -1.
        let carry_mask: int64x2_t = vreinterpretq_s64_u64(vcltq_u64(lo_r, lo_a));
        // hi_r = hi_a + hi_b + carry; carry_mask is -1 on carry, 0 otherwise,
        // so subtract it (subtracting -1 = adding 1).
        let hi_r: int64x2_t = vsubq_s64(vaddq_s64(hi_a, hi_b), carry_mask);
        (lo_r, hi_r)
    }
}

/// `(lo_r, hi_r) = (lo_a, hi_a) - (lo_b, hi_b)` over 2 lanes of i128.
#[inline(always)]
unsafe fn sub2_i128(lo_a: uint64x2_t, hi_a: int64x2_t, lo_b: uint64x2_t, hi_b: int64x2_t) -> (uint64x2_t, int64x2_t) {
    unsafe {
        let lo_r: uint64x2_t = vsubq_u64(lo_a, lo_b);
        // borrow mask: lanes where lo_a < lo_b = -1.
        let borrow_mask: int64x2_t = vreinterpretq_s64_u64(vcltq_u64(lo_a, lo_b));
        // hi_r = hi_a - hi_b - borrow; borrow_mask is -1 on borrow, so adding
        // it subtracts 1.
        let hi_r: int64x2_t = vaddq_s64(vsubq_s64(hi_a, hi_b), borrow_mask);
        (lo_r, hi_r)
    }
}

/// `(lo_r, hi_r) = -(lo_a, hi_a)` over 2 lanes of i128.
#[inline(always)]
unsafe fn neg2_i128(lo_a: uint64x2_t, hi_a: int64x2_t) -> (uint64x2_t, int64x2_t) {
    unsafe {
        // Equivalent to (0, 0) - (lo_a, hi_a).
        let zero_u: uint64x2_t = core::mem::transmute([0u64, 0u64]);
        let zero_s: int64x2_t = vdupq_n_s64(0);
        sub2_i128(zero_u, zero_s, lo_a, hi_a)
    }
}

// ─── public NEON kernels ──────────────────────────────────────────────────

/// `res[i] = a[i].wrapping_add(b[i])` for `n` i128 elements.
pub(crate) fn vi128_add_neon(n: usize, res: &mut [i128], a: &[i128], b: &[i128]) {
    debug_assert!(res.len() >= n && a.len() >= n && b.len() >= n);
    let chunks = n >> 1;
    let mut rr = res.as_mut_ptr();
    let mut aa = a.as_ptr();
    let mut bb = b.as_ptr();
    unsafe {
        for _ in 0..chunks {
            let (lo_a, hi_a) = load2_i128(aa);
            let (lo_b, hi_b) = load2_i128(bb);
            let (lo_r, hi_r) = add2_i128(lo_a, hi_a, lo_b, hi_b);
            store2_i128(rr, lo_r, hi_r);
            rr = rr.add(2);
            aa = aa.add(2);
            bb = bb.add(2);
        }
    }
    let tail = chunks << 1;
    for i in tail..n {
        res[i] = a[i].wrapping_add(b[i]);
    }
}

/// `res[i] = res[i].wrapping_add(a[i])` for `n` i128 elements.
pub(crate) fn vi128_add_assign_neon(n: usize, res: &mut [i128], a: &[i128]) {
    debug_assert!(res.len() >= n && a.len() >= n);
    let chunks = n >> 1;
    let mut rr = res.as_mut_ptr();
    let mut aa = a.as_ptr();
    unsafe {
        for _ in 0..chunks {
            let (lo_r, hi_r) = load2_i128(rr);
            let (lo_a, hi_a) = load2_i128(aa);
            let (lo_n, hi_n) = add2_i128(lo_r, hi_r, lo_a, hi_a);
            store2_i128(rr, lo_n, hi_n);
            rr = rr.add(2);
            aa = aa.add(2);
        }
    }
    let tail = chunks << 1;
    for i in tail..n {
        res[i] = res[i].wrapping_add(a[i]);
    }
}

/// `res[i] = a[i].wrapping_add(b[i] as i128)` for `n` elements (`b` is `i64`).
pub(crate) fn vi128_add_small_neon(n: usize, res: &mut [i128], a: &[i128], b: &[i64]) {
    debug_assert!(res.len() >= n && a.len() >= n && b.len() >= n);
    let chunks = n >> 1;
    let mut rr = res.as_mut_ptr();
    let mut aa = a.as_ptr();
    let mut bb = b.as_ptr();
    unsafe {
        for _ in 0..chunks {
            let (lo_a, hi_a) = load2_i128(aa);
            let (lo_b, hi_b) = load2_i64_as_i128(bb);
            let (lo_r, hi_r) = add2_i128(lo_a, hi_a, lo_b, hi_b);
            store2_i128(rr, lo_r, hi_r);
            rr = rr.add(2);
            aa = aa.add(2);
            bb = bb.add(2);
        }
    }
    let tail = chunks << 1;
    for i in tail..n {
        res[i] = a[i].wrapping_add(b[i] as i128);
    }
}

/// `res[i] = res[i].wrapping_add(a[i] as i128)` for `n` elements (`a` is `i64`).
pub(crate) fn vi128_add_small_assign_neon(n: usize, res: &mut [i128], a: &[i64]) {
    debug_assert!(res.len() >= n && a.len() >= n);
    let chunks = n >> 1;
    let mut rr = res.as_mut_ptr();
    let mut aa = a.as_ptr();
    unsafe {
        for _ in 0..chunks {
            let (lo_r, hi_r) = load2_i128(rr);
            let (lo_a, hi_a) = load2_i64_as_i128(aa);
            let (lo_n, hi_n) = add2_i128(lo_r, hi_r, lo_a, hi_a);
            store2_i128(rr, lo_n, hi_n);
            rr = rr.add(2);
            aa = aa.add(2);
        }
    }
    let tail = chunks << 1;
    for i in tail..n {
        res[i] = res[i].wrapping_add(a[i] as i128);
    }
}

/// `res[i] = a[i].wrapping_sub(b[i])` for `n` i128 elements.
pub(crate) fn vi128_sub_neon(n: usize, res: &mut [i128], a: &[i128], b: &[i128]) {
    debug_assert!(res.len() >= n && a.len() >= n && b.len() >= n);
    let chunks = n >> 1;
    let mut rr = res.as_mut_ptr();
    let mut aa = a.as_ptr();
    let mut bb = b.as_ptr();
    unsafe {
        for _ in 0..chunks {
            let (lo_a, hi_a) = load2_i128(aa);
            let (lo_b, hi_b) = load2_i128(bb);
            let (lo_r, hi_r) = sub2_i128(lo_a, hi_a, lo_b, hi_b);
            store2_i128(rr, lo_r, hi_r);
            rr = rr.add(2);
            aa = aa.add(2);
            bb = bb.add(2);
        }
    }
    let tail = chunks << 1;
    for i in tail..n {
        res[i] = a[i].wrapping_sub(b[i]);
    }
}

/// `res[i] = res[i].wrapping_sub(a[i])` for `n` i128 elements.
pub(crate) fn vi128_sub_assign_neon(n: usize, res: &mut [i128], a: &[i128]) {
    debug_assert!(res.len() >= n && a.len() >= n);
    let chunks = n >> 1;
    let mut rr = res.as_mut_ptr();
    let mut aa = a.as_ptr();
    unsafe {
        for _ in 0..chunks {
            let (lo_r, hi_r) = load2_i128(rr);
            let (lo_a, hi_a) = load2_i128(aa);
            let (lo_n, hi_n) = sub2_i128(lo_r, hi_r, lo_a, hi_a);
            store2_i128(rr, lo_n, hi_n);
            rr = rr.add(2);
            aa = aa.add(2);
        }
    }
    let tail = chunks << 1;
    for i in tail..n {
        res[i] = res[i].wrapping_sub(a[i]);
    }
}

/// `res[i] = a[i].wrapping_sub(res[i])` for `n` i128 elements.
pub(crate) fn vi128_sub_negate_assign_neon(n: usize, res: &mut [i128], a: &[i128]) {
    debug_assert!(res.len() >= n && a.len() >= n);
    let chunks = n >> 1;
    let mut rr = res.as_mut_ptr();
    let mut aa = a.as_ptr();
    unsafe {
        for _ in 0..chunks {
            let (lo_r, hi_r) = load2_i128(rr);
            let (lo_a, hi_a) = load2_i128(aa);
            let (lo_n, hi_n) = sub2_i128(lo_a, hi_a, lo_r, hi_r);
            store2_i128(rr, lo_n, hi_n);
            rr = rr.add(2);
            aa = aa.add(2);
        }
    }
    let tail = chunks << 1;
    for i in tail..n {
        res[i] = a[i].wrapping_sub(res[i]);
    }
}

/// `res[i] = (a[i] as i128).wrapping_sub(b[i])` for `n` elements (`a` is `i64`).
pub(crate) fn vi128_sub_small_a_neon(n: usize, res: &mut [i128], a: &[i64], b: &[i128]) {
    debug_assert!(res.len() >= n && a.len() >= n && b.len() >= n);
    let chunks = n >> 1;
    let mut rr = res.as_mut_ptr();
    let mut aa = a.as_ptr();
    let mut bb = b.as_ptr();
    unsafe {
        for _ in 0..chunks {
            let (lo_a, hi_a) = load2_i64_as_i128(aa);
            let (lo_b, hi_b) = load2_i128(bb);
            let (lo_r, hi_r) = sub2_i128(lo_a, hi_a, lo_b, hi_b);
            store2_i128(rr, lo_r, hi_r);
            rr = rr.add(2);
            aa = aa.add(2);
            bb = bb.add(2);
        }
    }
    let tail = chunks << 1;
    for i in tail..n {
        res[i] = (a[i] as i128).wrapping_sub(b[i]);
    }
}

/// `res[i] = a[i].wrapping_sub(b[i] as i128)` for `n` elements (`b` is `i64`).
pub(crate) fn vi128_sub_small_b_neon(n: usize, res: &mut [i128], a: &[i128], b: &[i64]) {
    debug_assert!(res.len() >= n && a.len() >= n && b.len() >= n);
    let chunks = n >> 1;
    let mut rr = res.as_mut_ptr();
    let mut aa = a.as_ptr();
    let mut bb = b.as_ptr();
    unsafe {
        for _ in 0..chunks {
            let (lo_a, hi_a) = load2_i128(aa);
            let (lo_b, hi_b) = load2_i64_as_i128(bb);
            let (lo_r, hi_r) = sub2_i128(lo_a, hi_a, lo_b, hi_b);
            store2_i128(rr, lo_r, hi_r);
            rr = rr.add(2);
            aa = aa.add(2);
            bb = bb.add(2);
        }
    }
    let tail = chunks << 1;
    for i in tail..n {
        res[i] = a[i].wrapping_sub(b[i] as i128);
    }
}

/// `res[i] = res[i].wrapping_sub(a[i] as i128)` for `n` elements (`a` is `i64`).
pub(crate) fn vi128_sub_small_assign_neon(n: usize, res: &mut [i128], a: &[i64]) {
    debug_assert!(res.len() >= n && a.len() >= n);
    let chunks = n >> 1;
    let mut rr = res.as_mut_ptr();
    let mut aa = a.as_ptr();
    unsafe {
        for _ in 0..chunks {
            let (lo_r, hi_r) = load2_i128(rr);
            let (lo_a, hi_a) = load2_i64_as_i128(aa);
            let (lo_n, hi_n) = sub2_i128(lo_r, hi_r, lo_a, hi_a);
            store2_i128(rr, lo_n, hi_n);
            rr = rr.add(2);
            aa = aa.add(2);
        }
    }
    let tail = chunks << 1;
    for i in tail..n {
        res[i] = res[i].wrapping_sub(a[i] as i128);
    }
}

/// `res[i] = (a[i] as i128).wrapping_sub(res[i])` for `n` elements (`a` is `i64`).
pub(crate) fn vi128_sub_small_negate_assign_neon(n: usize, res: &mut [i128], a: &[i64]) {
    debug_assert!(res.len() >= n && a.len() >= n);
    let chunks = n >> 1;
    let mut rr = res.as_mut_ptr();
    let mut aa = a.as_ptr();
    unsafe {
        for _ in 0..chunks {
            let (lo_r, hi_r) = load2_i128(rr);
            let (lo_a, hi_a) = load2_i64_as_i128(aa);
            let (lo_n, hi_n) = sub2_i128(lo_a, hi_a, lo_r, hi_r);
            store2_i128(rr, lo_n, hi_n);
            rr = rr.add(2);
            aa = aa.add(2);
        }
    }
    let tail = chunks << 1;
    for i in tail..n {
        res[i] = (a[i] as i128).wrapping_sub(res[i]);
    }
}

/// `res[i] = a[i].wrapping_neg()` for `n` i128 elements.
pub(crate) fn vi128_negate_neon(n: usize, res: &mut [i128], a: &[i128]) {
    debug_assert!(res.len() >= n && a.len() >= n);
    let chunks = n >> 1;
    let mut rr = res.as_mut_ptr();
    let mut aa = a.as_ptr();
    unsafe {
        for _ in 0..chunks {
            let (lo_a, hi_a) = load2_i128(aa);
            let (lo_r, hi_r) = neg2_i128(lo_a, hi_a);
            store2_i128(rr, lo_r, hi_r);
            rr = rr.add(2);
            aa = aa.add(2);
        }
    }
    let tail = chunks << 1;
    for i in tail..n {
        res[i] = a[i].wrapping_neg();
    }
}

/// `res[i] = res[i].wrapping_neg()` for `n` i128 elements.
pub(crate) fn vi128_negate_assign_neon(n: usize, res: &mut [i128]) {
    debug_assert!(res.len() >= n);
    let chunks = n >> 1;
    let mut rr = res.as_mut_ptr();
    unsafe {
        for _ in 0..chunks {
            let (lo_r, hi_r) = load2_i128(rr);
            let (lo_n, hi_n) = neg2_i128(lo_r, hi_r);
            store2_i128(rr, lo_n, hi_n);
            rr = rr.add(2);
        }
    }
    let tail = chunks << 1;
    for r in res.iter_mut().take(n).skip(tail) {
        *r = r.wrapping_neg();
    }
}

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

    /// Sizes exercising the SIMD body (n >= 2, even) and the scalar tail
    /// (n % 2 != 0), plus carry-edge boundary cases tested separately below.
    const SIZES: &[usize] = &[0, 1, 2, 3, 4, 5, 7, 8, 16, 17, 64, 65];

    fn rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(0x5a5a_5a5a_5a5a_5a5a)
    }

    fn random_i128_vec(rng: &mut ChaCha8Rng, n: usize) -> Vec<i128> {
        (0..n)
            .map(|_| {
                let lo: u64 = rng.random();
                let hi: u64 = rng.random();
                ((hi as u128) << 64 | lo as u128) as i128
            })
            .collect()
    }

    fn random_i64_vec(rng: &mut ChaCha8Rng, n: usize) -> Vec<i64> {
        (0..n).map(|_| rng.random::<i64>()).collect()
    }

    #[test]
    fn vi128_add_matches_scalar() {
        let mut rng = rng();
        for &n in SIZES {
            let a = random_i128_vec(&mut rng, n);
            let b = random_i128_vec(&mut rng, n);
            let want: Vec<i128> = a.iter().zip(b.iter()).map(|(x, y)| x.wrapping_add(*y)).collect();
            let mut got = vec![0i128; n];
            vi128_add_neon(n, &mut got, &a, &b);
            assert_eq!(got, want, "vi128_add_neon mismatch at n={n}");
        }
    }

    #[test]
    fn vi128_add_assign_matches_scalar() {
        let mut rng = rng();
        for &n in SIZES {
            let r0 = random_i128_vec(&mut rng, n);
            let a = random_i128_vec(&mut rng, n);
            let want: Vec<i128> = r0.iter().zip(a.iter()).map(|(x, y)| x.wrapping_add(*y)).collect();
            let mut got = r0;
            vi128_add_assign_neon(n, &mut got, &a);
            assert_eq!(got, want, "vi128_add_assign_neon mismatch at n={n}");
        }
    }

    #[test]
    fn vi128_add_small_matches_scalar() {
        let mut rng = rng();
        for &n in SIZES {
            let a = random_i128_vec(&mut rng, n);
            let b = random_i64_vec(&mut rng, n);
            let want: Vec<i128> = a.iter().zip(b.iter()).map(|(x, &y)| x.wrapping_add(y as i128)).collect();
            let mut got = vec![0i128; n];
            vi128_add_small_neon(n, &mut got, &a, &b);
            assert_eq!(got, want, "vi128_add_small_neon mismatch at n={n}");
        }
    }

    #[test]
    fn vi128_add_small_assign_matches_scalar() {
        let mut rng = rng();
        for &n in SIZES {
            let r0 = random_i128_vec(&mut rng, n);
            let a = random_i64_vec(&mut rng, n);
            let want: Vec<i128> = r0.iter().zip(a.iter()).map(|(x, &y)| x.wrapping_add(y as i128)).collect();
            let mut got = r0;
            vi128_add_small_assign_neon(n, &mut got, &a);
            assert_eq!(got, want, "vi128_add_small_assign_neon mismatch at n={n}");
        }
    }

    #[test]
    fn vi128_sub_matches_scalar() {
        let mut rng = rng();
        for &n in SIZES {
            let a = random_i128_vec(&mut rng, n);
            let b = random_i128_vec(&mut rng, n);
            let want: Vec<i128> = a.iter().zip(b.iter()).map(|(x, y)| x.wrapping_sub(*y)).collect();
            let mut got = vec![0i128; n];
            vi128_sub_neon(n, &mut got, &a, &b);
            assert_eq!(got, want, "vi128_sub_neon mismatch at n={n}");
        }
    }

    #[test]
    fn vi128_sub_assign_matches_scalar() {
        let mut rng = rng();
        for &n in SIZES {
            let r0 = random_i128_vec(&mut rng, n);
            let a = random_i128_vec(&mut rng, n);
            let want: Vec<i128> = r0.iter().zip(a.iter()).map(|(x, y)| x.wrapping_sub(*y)).collect();
            let mut got = r0;
            vi128_sub_assign_neon(n, &mut got, &a);
            assert_eq!(got, want, "vi128_sub_assign_neon mismatch at n={n}");
        }
    }

    #[test]
    fn vi128_sub_negate_assign_matches_scalar() {
        let mut rng = rng();
        for &n in SIZES {
            let r0 = random_i128_vec(&mut rng, n);
            let a = random_i128_vec(&mut rng, n);
            let want: Vec<i128> = a.iter().zip(r0.iter()).map(|(x, y)| x.wrapping_sub(*y)).collect();
            let mut got = r0;
            vi128_sub_negate_assign_neon(n, &mut got, &a);
            assert_eq!(got, want, "vi128_sub_negate_assign_neon mismatch at n={n}");
        }
    }

    #[test]
    fn vi128_sub_small_a_matches_scalar() {
        let mut rng = rng();
        for &n in SIZES {
            let a = random_i64_vec(&mut rng, n);
            let b = random_i128_vec(&mut rng, n);
            let want: Vec<i128> = a.iter().zip(b.iter()).map(|(&x, y)| (x as i128).wrapping_sub(*y)).collect();
            let mut got = vec![0i128; n];
            vi128_sub_small_a_neon(n, &mut got, &a, &b);
            assert_eq!(got, want, "vi128_sub_small_a_neon mismatch at n={n}");
        }
    }

    #[test]
    fn vi128_sub_small_b_matches_scalar() {
        let mut rng = rng();
        for &n in SIZES {
            let a = random_i128_vec(&mut rng, n);
            let b = random_i64_vec(&mut rng, n);
            let want: Vec<i128> = a.iter().zip(b.iter()).map(|(x, &y)| x.wrapping_sub(y as i128)).collect();
            let mut got = vec![0i128; n];
            vi128_sub_small_b_neon(n, &mut got, &a, &b);
            assert_eq!(got, want, "vi128_sub_small_b_neon mismatch at n={n}");
        }
    }

    #[test]
    fn vi128_sub_small_assign_matches_scalar() {
        let mut rng = rng();
        for &n in SIZES {
            let r0 = random_i128_vec(&mut rng, n);
            let a = random_i64_vec(&mut rng, n);
            let want: Vec<i128> = r0.iter().zip(a.iter()).map(|(x, &y)| x.wrapping_sub(y as i128)).collect();
            let mut got = r0;
            vi128_sub_small_assign_neon(n, &mut got, &a);
            assert_eq!(got, want, "vi128_sub_small_assign_neon mismatch at n={n}");
        }
    }

    #[test]
    fn vi128_sub_small_negate_assign_matches_scalar() {
        let mut rng = rng();
        for &n in SIZES {
            let r0 = random_i128_vec(&mut rng, n);
            let a = random_i64_vec(&mut rng, n);
            let want: Vec<i128> = r0.iter().zip(a.iter()).map(|(x, &y)| (y as i128).wrapping_sub(*x)).collect();
            let mut got = r0;
            vi128_sub_small_negate_assign_neon(n, &mut got, &a);
            assert_eq!(got, want, "vi128_sub_small_negate_assign_neon mismatch at n={n}");
        }
    }

    #[test]
    fn vi128_negate_matches_scalar() {
        let mut rng = rng();
        for &n in SIZES {
            let a = random_i128_vec(&mut rng, n);
            let want: Vec<i128> = a.iter().map(|x| x.wrapping_neg()).collect();
            let mut got = vec![0i128; n];
            vi128_negate_neon(n, &mut got, &a);
            assert_eq!(got, want, "vi128_negate_neon mismatch at n={n}");
        }
    }

    #[test]
    fn vi128_negate_assign_matches_scalar() {
        let mut rng = rng();
        for &n in SIZES {
            let r0 = random_i128_vec(&mut rng, n);
            let want: Vec<i128> = r0.iter().map(|x| x.wrapping_neg()).collect();
            let mut got = r0;
            vi128_negate_assign_neon(n, &mut got);
            assert_eq!(got, want, "vi128_negate_assign_neon mismatch at n={n}");
        }
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

    /// Boundary inputs around carry/borrow edges and i128/i64 limits.
    #[test]
    fn vi128_carry_borrow_boundaries() {
        let lo_max: i128 = u64::MAX as i128; // hi=0, lo=u64::MAX
        let one: i128 = 1;
        let neg_one: i128 = -1; // hi=-1, lo=u64::MAX
        let hi_only: i128 = 1i128 << 64;
        let i64_min_ext: i128 = i64::MIN as i128;
        let i64_max_ext: i128 = i64::MAX as i128;

        let a = vec![lo_max, neg_one, i64_min_ext, i64_max_ext, hi_only, 0, i128::MIN, i128::MAX];
        let b = vec![one, one, neg_one, one, neg_one, neg_one, neg_one, one];
        let n = a.len();

        let mut got = vec![0i128; n];
        let want: Vec<i128> = a.iter().zip(b.iter()).map(|(x, y)| x.wrapping_add(*y)).collect();
        vi128_add_neon(n, &mut got, &a, &b);
        assert_eq!(got, want, "boundary add");

        let want: Vec<i128> = a.iter().zip(b.iter()).map(|(x, y)| x.wrapping_sub(*y)).collect();
        vi128_sub_neon(n, &mut got, &a, &b);
        assert_eq!(got, want, "boundary sub");

        let want: Vec<i128> = a.iter().map(|x| x.wrapping_neg()).collect();
        vi128_negate_neon(n, &mut got, &a);
        assert_eq!(got, want, "boundary negate");
    }

    /// i64 sign-extension boundaries.
    #[test]
    fn vi128_small_sign_extend_boundaries() {
        let small = vec![i64::MIN, -1, 0, 1, i64::MAX, i64::MIN + 1, i64::MAX - 1, 42];
        let big = vec![0i128, 1, -1, 1i128 << 80, -(1i128 << 80), i128::MIN, i128::MAX, 0];
        let n = small.len();

        let mut got = vec![0i128; n];

        let want: Vec<i128> = big
            .iter()
            .zip(small.iter())
            .map(|(x, &y)| x.wrapping_add(y as i128))
            .collect();
        vi128_add_small_neon(n, &mut got, &big, &small);
        assert_eq!(got, want, "boundary add_small");

        let want: Vec<i128> = small
            .iter()
            .zip(big.iter())
            .map(|(&x, y)| (x as i128).wrapping_sub(*y))
            .collect();
        vi128_sub_small_a_neon(n, &mut got, &small, &big);
        assert_eq!(got, want, "boundary sub_small_a");

        let want: Vec<i128> = big
            .iter()
            .zip(small.iter())
            .map(|(x, &y)| x.wrapping_sub(y as i128))
            .collect();
        vi128_sub_small_b_neon(n, &mut got, &big, &small);
        assert_eq!(got, want, "boundary sub_small_b");

        let want: Vec<i128> = small.iter().map(|&x| x as i128).collect();
        vi128_from_small_neon(n, &mut got, &small);
        assert_eq!(got, want, "boundary from_small");

        let want: Vec<i128> = small.iter().map(|&x| (x as i128).wrapping_neg()).collect();
        vi128_neg_from_small_neon(n, &mut got, &small);
        assert_eq!(got, want, "boundary neg_from_small");
    }
}
