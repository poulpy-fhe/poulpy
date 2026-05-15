//! NEON i64 kernels for `Znx*` ring-element arithmetic.
//!
//! Mirrors `poulpy-cpu-avx/src/znx_avx/{add,sub,neg}.rs`. NEON is 128-bit so
//! each kernel processes two `int64x2_t` registers per iteration (4 i64 per
//! block), preserving the AVX backend's `n >> 2` block stride for side-by-side
//! review. The remainder `n % 4` is delegated to the portable reference
//! functions.
//!
//! Aliasing: `_assign` variants tolerate `res == a` (each loop iteration
//! reads its registers before storing them back). All other variants assume
//! disjoint slices, matching the reference contract.
//!
//! Wired into `crate::fft64::znx` and `crate::ntt120::znx` under
//! `#[cfg(target_arch = "aarch64")]`.

use core::arch::aarch64::{int64x2_t, vaddq_s64, vld1q_s64, vnegq_s64, vst1q_s64, vsubq_s64};

use poulpy_cpu_ref::reference::znx::{
    znx_add_assign_ref, znx_add_ref, znx_negate_assign_ref, znx_negate_ref, znx_sub_assign_ref, znx_sub_negate_assign_ref,
    znx_sub_ref,
};

/// `res[i] = a[i].wrapping_add(b[i])` for all `i`.
///
/// All slices must have the same length. Aliasing across slices is undefined.
pub(crate) fn znx_add_neon(res: &mut [i64], a: &[i64], b: &[i64]) {
    debug_assert_eq!(res.len(), a.len());
    debug_assert_eq!(res.len(), b.len());

    let n = res.len();
    let span = n >> 2;

    let mut rr = res.as_mut_ptr();
    let mut aa = a.as_ptr();
    let mut bb = b.as_ptr();

    unsafe {
        for _ in 0..span {
            let s0: int64x2_t = vaddq_s64(vld1q_s64(aa), vld1q_s64(bb));
            vst1q_s64(rr, s0);
            let s1: int64x2_t = vaddq_s64(vld1q_s64(aa.add(2)), vld1q_s64(bb.add(2)));
            vst1q_s64(rr.add(2), s1);
            rr = rr.add(4);
            aa = aa.add(4);
            bb = bb.add(4);
        }
    }

    let tail = span << 2;
    if tail < n {
        znx_add_ref(&mut res[tail..], &a[tail..], &b[tail..]);
    }
}

/// `res[i] = res[i].wrapping_add(a[i])` for all `i`.
pub(crate) fn znx_add_assign_neon(res: &mut [i64], a: &[i64]) {
    debug_assert_eq!(res.len(), a.len());

    let n = res.len();
    let span = n >> 2;

    let mut rr = res.as_mut_ptr();
    let mut aa = a.as_ptr();

    unsafe {
        for _ in 0..span {
            let s0: int64x2_t = vaddq_s64(vld1q_s64(rr), vld1q_s64(aa));
            vst1q_s64(rr, s0);
            let s1: int64x2_t = vaddq_s64(vld1q_s64(rr.add(2)), vld1q_s64(aa.add(2)));
            vst1q_s64(rr.add(2), s1);
            rr = rr.add(4);
            aa = aa.add(4);
        }
    }

    let tail = span << 2;
    if tail < n {
        znx_add_assign_ref(&mut res[tail..], &a[tail..]);
    }
}

/// `res[i] = a[i].wrapping_sub(b[i])` for all `i`.
pub(crate) fn znx_sub_neon(res: &mut [i64], a: &[i64], b: &[i64]) {
    debug_assert_eq!(res.len(), a.len());
    debug_assert_eq!(res.len(), b.len());

    let n = res.len();
    let span = n >> 2;

    let mut rr = res.as_mut_ptr();
    let mut aa = a.as_ptr();
    let mut bb = b.as_ptr();

    unsafe {
        for _ in 0..span {
            let s0: int64x2_t = vsubq_s64(vld1q_s64(aa), vld1q_s64(bb));
            vst1q_s64(rr, s0);
            let s1: int64x2_t = vsubq_s64(vld1q_s64(aa.add(2)), vld1q_s64(bb.add(2)));
            vst1q_s64(rr.add(2), s1);
            rr = rr.add(4);
            aa = aa.add(4);
            bb = bb.add(4);
        }
    }

    let tail = span << 2;
    if tail < n {
        znx_sub_ref(&mut res[tail..], &a[tail..], &b[tail..]);
    }
}

/// `res[i] = res[i].wrapping_sub(a[i])` for all `i`.
pub(crate) fn znx_sub_assign_neon(res: &mut [i64], a: &[i64]) {
    debug_assert_eq!(res.len(), a.len());

    let n = res.len();
    let span = n >> 2;

    let mut rr = res.as_mut_ptr();
    let mut aa = a.as_ptr();

    unsafe {
        for _ in 0..span {
            let s0: int64x2_t = vsubq_s64(vld1q_s64(rr), vld1q_s64(aa));
            vst1q_s64(rr, s0);
            let s1: int64x2_t = vsubq_s64(vld1q_s64(rr.add(2)), vld1q_s64(aa.add(2)));
            vst1q_s64(rr.add(2), s1);
            rr = rr.add(4);
            aa = aa.add(4);
        }
    }

    let tail = span << 2;
    if tail < n {
        znx_sub_assign_ref(&mut res[tail..], &a[tail..]);
    }
}

/// `res[i] = a[i].wrapping_sub(res[i])` for all `i`.
pub(crate) fn znx_sub_negate_assign_neon(res: &mut [i64], a: &[i64]) {
    debug_assert_eq!(res.len(), a.len());

    let n = res.len();
    let span = n >> 2;

    let mut rr = res.as_mut_ptr();
    let mut aa = a.as_ptr();

    unsafe {
        for _ in 0..span {
            let s0: int64x2_t = vsubq_s64(vld1q_s64(aa), vld1q_s64(rr));
            vst1q_s64(rr, s0);
            let s1: int64x2_t = vsubq_s64(vld1q_s64(aa.add(2)), vld1q_s64(rr.add(2)));
            vst1q_s64(rr.add(2), s1);
            rr = rr.add(4);
            aa = aa.add(4);
        }
    }

    let tail = span << 2;
    if tail < n {
        znx_sub_negate_assign_ref(&mut res[tail..], &a[tail..]);
    }
}

/// `res[i] = a[i].wrapping_neg()` for all `i`.
pub(crate) fn znx_negate_neon(res: &mut [i64], a: &[i64]) {
    debug_assert_eq!(res.len(), a.len());

    let n = res.len();
    let span = n >> 2;

    let mut rr = res.as_mut_ptr();
    let mut aa = a.as_ptr();

    unsafe {
        for _ in 0..span {
            let s0: int64x2_t = vnegq_s64(vld1q_s64(aa));
            vst1q_s64(rr, s0);
            let s1: int64x2_t = vnegq_s64(vld1q_s64(aa.add(2)));
            vst1q_s64(rr.add(2), s1);
            rr = rr.add(4);
            aa = aa.add(4);
        }
    }

    let tail = span << 2;
    if tail < n {
        znx_negate_ref(&mut res[tail..], &a[tail..]);
    }
}

/// `res[i] = res[i].wrapping_neg()` for all `i`.
pub(crate) fn znx_negate_assign_neon(res: &mut [i64]) {
    let n = res.len();
    let span = n >> 2;

    let mut rr = res.as_mut_ptr();

    unsafe {
        for _ in 0..span {
            let s0: int64x2_t = vnegq_s64(vld1q_s64(rr));
            vst1q_s64(rr, s0);
            let s1: int64x2_t = vnegq_s64(vld1q_s64(rr.add(2)));
            vst1q_s64(rr.add(2), s1);
            rr = rr.add(4);
        }
    }

    let tail = span << 2;
    if tail < n {
        znx_negate_assign_ref(&mut res[tail..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    /// Sizes that exercise the SIMD body (n >= 4, multiple of 4) and scalar
    /// tail (n % 4 != 0), plus boundary cases.
    const SIZES: &[usize] = &[0, 1, 2, 3, 4, 5, 7, 8, 16, 17, 256, 257];

    fn rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(0xa5a5_a5a5_a5a5_a5a5)
    }

    /// Random `i64` values bounded to `[-2^60, 2^60)`. The scalar reference
    /// `znx_*_ref` uses non-wrapping `+`/`-`, which panics in debug mode on
    /// overflow; bounding to 2^60 keeps `add`/`sub` results safely below
    /// `i64::MAX/MIN` while still exercising both signs and the high bits
    /// the NEON kernel must propagate.
    fn random_vec(rng: &mut ChaCha8Rng, n: usize) -> Vec<i64> {
        (0..n).map(|_| rng.random::<i64>() >> 3).collect()
    }

    #[test]
    fn test_znx_add_neon_matches_ref() {
        let mut rng = rng();
        for &n in SIZES {
            let a = random_vec(&mut rng, n);
            let b = random_vec(&mut rng, n);
            let mut got = vec![0i64; n];
            let mut want = vec![0i64; n];
            znx_add_neon(&mut got, &a, &b);
            znx_add_ref(&mut want, &a, &b);
            assert_eq!(got, want, "znx_add_neon mismatch at n={n}");
        }
    }

    #[test]
    fn test_znx_add_assign_neon_matches_ref() {
        let mut rng = rng();
        for &n in SIZES {
            let a = random_vec(&mut rng, n);
            let r0 = random_vec(&mut rng, n);
            let mut got = r0.clone();
            let mut want = r0;
            znx_add_assign_neon(&mut got, &a);
            znx_add_assign_ref(&mut want, &a);
            assert_eq!(got, want, "znx_add_assign_neon mismatch at n={n}");
        }
    }

    #[test]
    fn test_znx_sub_neon_matches_ref() {
        let mut rng = rng();
        for &n in SIZES {
            let a = random_vec(&mut rng, n);
            let b = random_vec(&mut rng, n);
            let mut got = vec![0i64; n];
            let mut want = vec![0i64; n];
            znx_sub_neon(&mut got, &a, &b);
            znx_sub_ref(&mut want, &a, &b);
            assert_eq!(got, want, "znx_sub_neon mismatch at n={n}");
        }
    }

    #[test]
    fn test_znx_sub_assign_neon_matches_ref() {
        let mut rng = rng();
        for &n in SIZES {
            let a = random_vec(&mut rng, n);
            let r0 = random_vec(&mut rng, n);
            let mut got = r0.clone();
            let mut want = r0;
            znx_sub_assign_neon(&mut got, &a);
            znx_sub_assign_ref(&mut want, &a);
            assert_eq!(got, want, "znx_sub_assign_neon mismatch at n={n}");
        }
    }

    #[test]
    fn test_znx_sub_negate_assign_neon_matches_ref() {
        let mut rng = rng();
        for &n in SIZES {
            let a = random_vec(&mut rng, n);
            let r0 = random_vec(&mut rng, n);
            let mut got = r0.clone();
            let mut want = r0;
            znx_sub_negate_assign_neon(&mut got, &a);
            znx_sub_negate_assign_ref(&mut want, &a);
            assert_eq!(got, want, "znx_sub_negate_assign_neon mismatch at n={n}");
        }
    }

    #[test]
    fn test_znx_negate_neon_matches_ref() {
        let mut rng = rng();
        for &n in SIZES {
            let a = random_vec(&mut rng, n);
            let mut got = vec![0i64; n];
            let mut want = vec![0i64; n];
            znx_negate_neon(&mut got, &a);
            znx_negate_ref(&mut want, &a);
            assert_eq!(got, want, "znx_negate_neon mismatch at n={n}");
        }
    }

    #[test]
    fn test_znx_negate_assign_neon_matches_ref() {
        let mut rng = rng();
        for &n in SIZES {
            let r0 = random_vec(&mut rng, n);
            let mut got = r0.clone();
            let mut want = r0;
            znx_negate_assign_neon(&mut got);
            znx_negate_assign_ref(&mut want);
            assert_eq!(got, want, "znx_negate_assign_neon mismatch at n={n}");
        }
    }

    /// Boundary inputs near `±2^62` (well within the non-overflowing range
    /// of the scalar reference). The NEON kernels share semantics with the
    /// reference for any in-range input, so this test pins the high-bit
    /// path without inviting reference-side overflow panics.
    #[test]
    fn test_znx_boundary_values() {
        const HI: i64 = 1i64 << 62;
        let a = vec![HI, -HI, 0, -1, 1, HI - 1, -(HI - 1), 42];
        let b = vec![1, -1, HI, -HI, 0, -42, 42, -(HI - 5)];
        let n = a.len();

        let mut got = vec![0i64; n];
        let mut want = vec![0i64; n];

        znx_add_neon(&mut got, &a, &b);
        znx_add_ref(&mut want, &a, &b);
        assert_eq!(got, want);

        znx_sub_neon(&mut got, &a, &b);
        znx_sub_ref(&mut want, &a, &b);
        assert_eq!(got, want);

        znx_negate_neon(&mut got, &a);
        znx_negate_ref(&mut want, &a);
        assert_eq!(got, want);
    }
}
