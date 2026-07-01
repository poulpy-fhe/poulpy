//! NEON kernels for q120b lazy modular arithmetic (NTT120 backend).

use core::arch::aarch64::{uint64x2_t, vaddq_u64, vbslq_u64, vcgeq_u64, vld1q_u64, vst1q_u64, vsubq_u64};
use poulpy_cpu_ref::reference::ntt120::types::Q_SHIFTED;

/// Lazy reduction: bring each lane of `x ∈ [0, 2·q_s)` into `[0, q_s)`.
/// Subtracts `q_s` from each lane where `x >= q_s` (unsigned).
#[inline(always)]
unsafe fn lazy_reduce(x: uint64x2_t, q_s: uint64x2_t) -> uint64x2_t {
    unsafe {
        let mask = vcgeq_u64(x, q_s);
        vbslq_u64(mask, vsubq_u64(x, q_s), x)
    }
}

/// `res[j*4..j*4+4] = lazy(a[…]) + lazy(b[…])` for `n` q120b coefficients.
pub(crate) fn ntt_add_neon(n: usize, res: &mut [u64], a: &[u64], b: &[u64]) {
    debug_assert!(res.len() >= 4 * n && a.len() >= 4 * n && b.len() >= 4 * n);
    unsafe {
        let q_lo: uint64x2_t = vld1q_u64(Q_SHIFTED.as_ptr());
        let q_hi: uint64x2_t = vld1q_u64(Q_SHIFTED.as_ptr().add(2));
        let mut rr = res.as_mut_ptr();
        let mut aa = a.as_ptr();
        let mut bb = b.as_ptr();
        for _ in 0..n {
            let av_lo = lazy_reduce(vld1q_u64(aa), q_lo);
            let av_hi = lazy_reduce(vld1q_u64(aa.add(2)), q_hi);
            let bv_lo = lazy_reduce(vld1q_u64(bb), q_lo);
            let bv_hi = lazy_reduce(vld1q_u64(bb.add(2)), q_hi);
            vst1q_u64(rr, vaddq_u64(av_lo, bv_lo));
            vst1q_u64(rr.add(2), vaddq_u64(av_hi, bv_hi));
            rr = rr.add(4);
            aa = aa.add(4);
            bb = bb.add(4);
        }
    }
}

/// `res[…] = lazy(res[…]) + lazy(a[…])` for `n` q120b coefficients.
pub(crate) fn ntt_add_assign_neon(n: usize, res: &mut [u64], a: &[u64]) {
    debug_assert!(res.len() >= 4 * n && a.len() >= 4 * n);
    unsafe {
        let q_lo: uint64x2_t = vld1q_u64(Q_SHIFTED.as_ptr());
        let q_hi: uint64x2_t = vld1q_u64(Q_SHIFTED.as_ptr().add(2));
        let mut rr = res.as_mut_ptr();
        let mut aa = a.as_ptr();
        for _ in 0..n {
            let rv_lo = lazy_reduce(vld1q_u64(rr), q_lo);
            let rv_hi = lazy_reduce(vld1q_u64(rr.add(2)), q_hi);
            let av_lo = lazy_reduce(vld1q_u64(aa), q_lo);
            let av_hi = lazy_reduce(vld1q_u64(aa.add(2)), q_hi);
            vst1q_u64(rr, vaddq_u64(rv_lo, av_lo));
            vst1q_u64(rr.add(2), vaddq_u64(rv_hi, av_hi));
            rr = rr.add(4);
            aa = aa.add(4);
        }
    }
}

/// `res[…] = lazy(a[…]) + (q_s − lazy(b[…]))` for `n` q120b coefficients.
pub(crate) fn ntt_sub_neon(n: usize, res: &mut [u64], a: &[u64], b: &[u64]) {
    debug_assert!(res.len() >= 4 * n && a.len() >= 4 * n && b.len() >= 4 * n);
    unsafe {
        let q_lo: uint64x2_t = vld1q_u64(Q_SHIFTED.as_ptr());
        let q_hi: uint64x2_t = vld1q_u64(Q_SHIFTED.as_ptr().add(2));
        let mut rr = res.as_mut_ptr();
        let mut aa = a.as_ptr();
        let mut bb = b.as_ptr();
        for _ in 0..n {
            let av_lo = lazy_reduce(vld1q_u64(aa), q_lo);
            let av_hi = lazy_reduce(vld1q_u64(aa.add(2)), q_hi);
            let bv_lo = lazy_reduce(vld1q_u64(bb), q_lo);
            let bv_hi = lazy_reduce(vld1q_u64(bb.add(2)), q_hi);
            vst1q_u64(rr, vaddq_u64(av_lo, vsubq_u64(q_lo, bv_lo)));
            vst1q_u64(rr.add(2), vaddq_u64(av_hi, vsubq_u64(q_hi, bv_hi)));
            rr = rr.add(4);
            aa = aa.add(4);
            bb = bb.add(4);
        }
    }
}

/// `res[…] = lazy(res[…]) + (q_s − lazy(a[…]))` for `n` q120b coefficients.
pub(crate) fn ntt_sub_assign_neon(n: usize, res: &mut [u64], a: &[u64]) {
    debug_assert!(res.len() >= 4 * n && a.len() >= 4 * n);
    unsafe {
        let q_lo: uint64x2_t = vld1q_u64(Q_SHIFTED.as_ptr());
        let q_hi: uint64x2_t = vld1q_u64(Q_SHIFTED.as_ptr().add(2));
        let mut rr = res.as_mut_ptr();
        let mut aa = a.as_ptr();
        for _ in 0..n {
            let rv_lo = lazy_reduce(vld1q_u64(rr), q_lo);
            let rv_hi = lazy_reduce(vld1q_u64(rr.add(2)), q_hi);
            let av_lo = lazy_reduce(vld1q_u64(aa), q_lo);
            let av_hi = lazy_reduce(vld1q_u64(aa.add(2)), q_hi);
            vst1q_u64(rr, vaddq_u64(rv_lo, vsubq_u64(q_lo, av_lo)));
            vst1q_u64(rr.add(2), vaddq_u64(rv_hi, vsubq_u64(q_hi, av_hi)));
            rr = rr.add(4);
            aa = aa.add(4);
        }
    }
}

/// `res[…] = lazy(a[…]) + (q_s − lazy(res[…]))` for `n` q120b coefficients.
pub(crate) fn ntt_sub_negate_assign_neon(n: usize, res: &mut [u64], a: &[u64]) {
    debug_assert!(res.len() >= 4 * n && a.len() >= 4 * n);
    unsafe {
        let q_lo: uint64x2_t = vld1q_u64(Q_SHIFTED.as_ptr());
        let q_hi: uint64x2_t = vld1q_u64(Q_SHIFTED.as_ptr().add(2));
        let mut rr = res.as_mut_ptr();
        let mut aa = a.as_ptr();
        for _ in 0..n {
            let rv_lo = lazy_reduce(vld1q_u64(rr), q_lo);
            let rv_hi = lazy_reduce(vld1q_u64(rr.add(2)), q_hi);
            let av_lo = lazy_reduce(vld1q_u64(aa), q_lo);
            let av_hi = lazy_reduce(vld1q_u64(aa.add(2)), q_hi);
            vst1q_u64(rr, vaddq_u64(av_lo, vsubq_u64(q_lo, rv_lo)));
            vst1q_u64(rr.add(2), vaddq_u64(av_hi, vsubq_u64(q_hi, rv_hi)));
            rr = rr.add(4);
            aa = aa.add(4);
        }
    }
}

/// `res[…] = q_s − lazy(a[…])` for `n` q120b coefficients.
/// **Output range**: For a zero input the result is `Q_SHIFTED[k]` (≡ 0 mod `Q[k]`),
/// not `0`. Use `val % Q[k] == 0`, not `val == 0`, to test for zero.
pub(crate) fn ntt_negate_neon(n: usize, res: &mut [u64], a: &[u64]) {
    debug_assert!(res.len() >= 4 * n && a.len() >= 4 * n);
    unsafe {
        let q_lo: uint64x2_t = vld1q_u64(Q_SHIFTED.as_ptr());
        let q_hi: uint64x2_t = vld1q_u64(Q_SHIFTED.as_ptr().add(2));
        let mut rr = res.as_mut_ptr();
        let mut aa = a.as_ptr();
        for _ in 0..n {
            let av_lo = lazy_reduce(vld1q_u64(aa), q_lo);
            let av_hi = lazy_reduce(vld1q_u64(aa.add(2)), q_hi);
            vst1q_u64(rr, vsubq_u64(q_lo, av_lo));
            vst1q_u64(rr.add(2), vsubq_u64(q_hi, av_hi));
            rr = rr.add(4);
            aa = aa.add(4);
        }
    }
}

/// `res[…] = q_s − lazy(res[…])` for `n` q120b coefficients.
/// **Output range**: same as [`ntt_negate_neon`].
pub(crate) fn ntt_negate_assign_neon(n: usize, res: &mut [u64]) {
    debug_assert!(res.len() >= 4 * n);
    unsafe {
        let q_lo: uint64x2_t = vld1q_u64(Q_SHIFTED.as_ptr());
        let q_hi: uint64x2_t = vld1q_u64(Q_SHIFTED.as_ptr().add(2));
        let mut rr = res.as_mut_ptr();
        for _ in 0..n {
            let rv_lo = lazy_reduce(vld1q_u64(rr), q_lo);
            let rv_hi = lazy_reduce(vld1q_u64(rr.add(2)), q_hi);
            vst1q_u64(rr, vsubq_u64(q_lo, rv_lo));
            vst1q_u64(rr.add(2), vsubq_u64(q_hi, rv_hi));
            rr = rr.add(4);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    /// Number of q120b coefficients; total u64 length is 4 × n.
    const COEFF_COUNTS: &[usize] = &[1, 2, 4, 8, 64, 257];

    fn rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(0xc0c0_c0c0_c0c0_c0c0)
    }

    /// Sample u64 lane values in `[0, 2 * Q_SHIFTED[k])`, i.e. the lazy
    /// invariant produced by `accum_to_q120b`.
    fn random_lazy_q120b(rng: &mut ChaCha8Rng, n: usize) -> Vec<u64> {
        (0..4 * n)
            .map(|i| {
                let k = i % 4;
                let bound = Q_SHIFTED[k] << 1;
                rng.random::<u64>() % bound
            })
            .collect()
    }

    /// Scalar lazy reference: per-lane `(x % q_s)`. Matches the formulas in
    /// `poulpy-cpu-ref/src/ntt120/prim.rs`.
    fn lazy_lane(x: u64, q_s: u64) -> u64 {
        x % q_s
    }

    // The reference loops below intentionally index `Q_SHIFTED[k]` to keep the
    // scalar shape parallel to the NEON layout (4 prime lanes per
    // coefficient). `clippy::needless_range_loop` would obscure that
    // structure if rewritten as `iter().enumerate()`.
    #[allow(clippy::needless_range_loop)]
    fn ntt_add_ref(n: usize, res: &mut [u64], a: &[u64], b: &[u64]) {
        for j in 0..n {
            for k in 0..4 {
                let i = 4 * j + k;
                res[i] = lazy_lane(a[i], Q_SHIFTED[k]) + lazy_lane(b[i], Q_SHIFTED[k]);
            }
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn ntt_add_assign_ref_(n: usize, res: &mut [u64], a: &[u64]) {
        for j in 0..n {
            for k in 0..4 {
                let i = 4 * j + k;
                res[i] = lazy_lane(res[i], Q_SHIFTED[k]) + lazy_lane(a[i], Q_SHIFTED[k]);
            }
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn ntt_sub_ref(n: usize, res: &mut [u64], a: &[u64], b: &[u64]) {
        for j in 0..n {
            for k in 0..4 {
                let i = 4 * j + k;
                res[i] = lazy_lane(a[i], Q_SHIFTED[k]) + (Q_SHIFTED[k] - lazy_lane(b[i], Q_SHIFTED[k]));
            }
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn ntt_sub_assign_ref_(n: usize, res: &mut [u64], a: &[u64]) {
        for j in 0..n {
            for k in 0..4 {
                let i = 4 * j + k;
                res[i] = lazy_lane(res[i], Q_SHIFTED[k]) + (Q_SHIFTED[k] - lazy_lane(a[i], Q_SHIFTED[k]));
            }
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn ntt_sub_negate_assign_ref_(n: usize, res: &mut [u64], a: &[u64]) {
        for j in 0..n {
            for k in 0..4 {
                let i = 4 * j + k;
                res[i] = lazy_lane(a[i], Q_SHIFTED[k]) + (Q_SHIFTED[k] - lazy_lane(res[i], Q_SHIFTED[k]));
            }
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn ntt_negate_ref(n: usize, res: &mut [u64], a: &[u64]) {
        for j in 0..n {
            for k in 0..4 {
                let i = 4 * j + k;
                res[i] = Q_SHIFTED[k] - lazy_lane(a[i], Q_SHIFTED[k]);
            }
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn ntt_negate_assign_ref_(n: usize, res: &mut [u64]) {
        for j in 0..n {
            for k in 0..4 {
                let i = 4 * j + k;
                res[i] = Q_SHIFTED[k] - lazy_lane(res[i], Q_SHIFTED[k]);
            }
        }
    }

    #[test]
    fn ntt_add_neon_matches_ref() {
        let mut rng = rng();
        for &n in COEFF_COUNTS {
            let a = random_lazy_q120b(&mut rng, n);
            let b = random_lazy_q120b(&mut rng, n);
            let mut got = vec![0u64; 4 * n];
            let mut want = vec![0u64; 4 * n];
            ntt_add_neon(n, &mut got, &a, &b);
            ntt_add_ref(n, &mut want, &a, &b);
            assert_eq!(got, want, "ntt_add_neon mismatch at n={n}");
        }
    }

    #[test]
    fn ntt_add_assign_neon_matches_ref() {
        let mut rng = rng();
        for &n in COEFF_COUNTS {
            let r0 = random_lazy_q120b(&mut rng, n);
            let a = random_lazy_q120b(&mut rng, n);
            let mut got = r0.clone();
            let mut want = r0;
            ntt_add_assign_neon(n, &mut got, &a);
            ntt_add_assign_ref_(n, &mut want, &a);
            assert_eq!(got, want, "ntt_add_assign_neon mismatch at n={n}");
        }
    }

    #[test]
    fn ntt_sub_neon_matches_ref() {
        let mut rng = rng();
        for &n in COEFF_COUNTS {
            let a = random_lazy_q120b(&mut rng, n);
            let b = random_lazy_q120b(&mut rng, n);
            let mut got = vec![0u64; 4 * n];
            let mut want = vec![0u64; 4 * n];
            ntt_sub_neon(n, &mut got, &a, &b);
            ntt_sub_ref(n, &mut want, &a, &b);
            assert_eq!(got, want, "ntt_sub_neon mismatch at n={n}");
        }
    }

    #[test]
    fn ntt_sub_assign_neon_matches_ref() {
        let mut rng = rng();
        for &n in COEFF_COUNTS {
            let r0 = random_lazy_q120b(&mut rng, n);
            let a = random_lazy_q120b(&mut rng, n);
            let mut got = r0.clone();
            let mut want = r0;
            ntt_sub_assign_neon(n, &mut got, &a);
            ntt_sub_assign_ref_(n, &mut want, &a);
            assert_eq!(got, want, "ntt_sub_assign_neon mismatch at n={n}");
        }
    }

    #[test]
    fn ntt_sub_negate_assign_neon_matches_ref() {
        let mut rng = rng();
        for &n in COEFF_COUNTS {
            let r0 = random_lazy_q120b(&mut rng, n);
            let a = random_lazy_q120b(&mut rng, n);
            let mut got = r0.clone();
            let mut want = r0;
            ntt_sub_negate_assign_neon(n, &mut got, &a);
            ntt_sub_negate_assign_ref_(n, &mut want, &a);
            assert_eq!(got, want, "ntt_sub_negate_assign_neon mismatch at n={n}");
        }
    }

    #[test]
    fn ntt_negate_neon_matches_ref() {
        let mut rng = rng();
        for &n in COEFF_COUNTS {
            let a = random_lazy_q120b(&mut rng, n);
            let mut got = vec![0u64; 4 * n];
            let mut want = vec![0u64; 4 * n];
            ntt_negate_neon(n, &mut got, &a);
            ntt_negate_ref(n, &mut want, &a);
            assert_eq!(got, want, "ntt_negate_neon mismatch at n={n}");
        }
    }

    #[test]
    fn ntt_negate_assign_neon_matches_ref() {
        let mut rng = rng();
        for &n in COEFF_COUNTS {
            let r0 = random_lazy_q120b(&mut rng, n);
            let mut got = r0.clone();
            let mut want = r0;
            ntt_negate_assign_neon(n, &mut got);
            ntt_negate_assign_ref_(n, &mut want);
            assert_eq!(got, want, "ntt_negate_assign_neon mismatch at n={n}");
        }
    }

    /// Boundary cases: zero, exactly q_s, just below 2*q_s.
    #[test]
    fn ntt_lazy_reduce_boundaries() {
        let n = 4usize;
        let mut a = vec![0u64; 4 * n];
        for j in 0..n {
            for k in 0..4 {
                let q = Q_SHIFTED[k];
                let val = match j {
                    0 => 0u64,
                    1 => q - 1,
                    2 => q,
                    _ => 2 * q - 1,
                };
                a[4 * j + k] = val;
            }
        }
        let b = a.clone();
        let mut got = vec![0u64; 4 * n];
        let mut want = vec![0u64; 4 * n];

        ntt_add_neon(n, &mut got, &a, &b);
        ntt_add_ref(n, &mut want, &a, &b);
        assert_eq!(got, want, "boundary add");

        ntt_sub_neon(n, &mut got, &a, &b);
        ntt_sub_ref(n, &mut want, &a, &b);
        assert_eq!(got, want, "boundary sub");

        ntt_negate_neon(n, &mut got, &a);
        ntt_negate_ref(n, &mut want, &a);
        assert_eq!(got, want, "boundary negate");
    }
}
