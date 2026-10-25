//! NEON pointwise REIM arithmetic for the FFT64 backend.
//!
//! Layout: a complex vector of length `2m` is stored as `[re_0..re_{m-1},
//! im_0..im_{m-1}]` (split, not interleaved).

use core::arch::aarch64::{
    float64x2_t, vaddq_f64, vcvtaq_s64_f64, vcvtq_f64_s64, vdupq_n_f64, vfmaq_f64, vfmsq_f64, vld1q_f64, vld1q_s64, vmulq_f64,
    vnegq_f64, vst1q_f64, vst1q_s64, vsubq_f64,
};

#[allow(unused_imports)]
use poulpy_cpu_portable::kernels::fft64::reim::{
    reim_add_assign_portable, reim_add_portable, reim_addmul_portable, reim_from_znx_i64_portable, reim_mul_assign_portable,
    reim_mul_portable, reim_negate_assign_portable, reim_negate_portable, reim_sub_assign_portable,
    reim_sub_negate_assign_portable, reim_sub_portable, reim_to_znx_i64_assign_portable, reim_to_znx_i64_portable,
};

/// `res[i] = a[i] + b[i]` for all `i`.
/// Kept for the unit test below; the `FFT64Neon` `ReimArith::reim_add` impl
/// routes to the autovec reference because the hand-NEON loop is memory
/// bandwidth bound at large `n` and the autovec wins.
#[allow(dead_code)]
pub(crate) fn reim_add_neon(res: &mut [f64], a: &[f64], b: &[f64]) {
    assert_eq!(res.len(), a.len());
    assert_eq!(res.len(), b.len());
    let n = res.len();
    let span = n >> 2;
    unsafe {
        let mut rr = res.as_mut_ptr();
        let mut aa = a.as_ptr();
        let mut bb = b.as_ptr();
        for _ in 0..span {
            let s0: float64x2_t = vaddq_f64(vld1q_f64(aa), vld1q_f64(bb));
            vst1q_f64(rr, s0);
            let s1: float64x2_t = vaddq_f64(vld1q_f64(aa.add(2)), vld1q_f64(bb.add(2)));
            vst1q_f64(rr.add(2), s1);
            rr = rr.add(4);
            aa = aa.add(4);
            bb = bb.add(4);
        }
    }
    let tail = span << 2;
    if tail < n {
        reim_add_portable(&mut res[tail..], &a[tail..], &b[tail..]);
    }
}

/// `res[i] = res[i] + a[i]` for all `i`.
/// See `reim_add_neon`: kept for tests, dispatched to the ref by `FFT64Neon`.
#[allow(dead_code)]
pub(crate) fn reim_add_assign_neon(res: &mut [f64], a: &[f64]) {
    assert_eq!(res.len(), a.len());
    let n = res.len();
    let span = n >> 2;
    unsafe {
        let mut rr = res.as_mut_ptr();
        let mut aa = a.as_ptr();
        for _ in 0..span {
            let s0: float64x2_t = vaddq_f64(vld1q_f64(rr), vld1q_f64(aa));
            vst1q_f64(rr, s0);
            let s1: float64x2_t = vaddq_f64(vld1q_f64(rr.add(2)), vld1q_f64(aa.add(2)));
            vst1q_f64(rr.add(2), s1);
            rr = rr.add(4);
            aa = aa.add(4);
        }
    }
    let tail = span << 2;
    if tail < n {
        reim_add_assign_portable(&mut res[tail..], &a[tail..]);
    }
}

/// `res[i] = a[i] - b[i]` for all `i`.
pub(crate) fn reim_sub_neon(res: &mut [f64], a: &[f64], b: &[f64]) {
    assert_eq!(res.len(), a.len());
    assert_eq!(res.len(), b.len());
    let n = res.len();
    let span = n >> 2;
    unsafe {
        let mut rr = res.as_mut_ptr();
        let mut aa = a.as_ptr();
        let mut bb = b.as_ptr();
        for _ in 0..span {
            let s0: float64x2_t = vsubq_f64(vld1q_f64(aa), vld1q_f64(bb));
            vst1q_f64(rr, s0);
            let s1: float64x2_t = vsubq_f64(vld1q_f64(aa.add(2)), vld1q_f64(bb.add(2)));
            vst1q_f64(rr.add(2), s1);
            rr = rr.add(4);
            aa = aa.add(4);
            bb = bb.add(4);
        }
    }
    let tail = span << 2;
    if tail < n {
        reim_sub_portable(&mut res[tail..], &a[tail..], &b[tail..]);
    }
}

/// `res[i] = res[i] - a[i]` for all `i`.
pub(crate) fn reim_sub_assign_neon(res: &mut [f64], a: &[f64]) {
    assert_eq!(res.len(), a.len());
    let n = res.len();
    let span = n >> 2;
    unsafe {
        let mut rr = res.as_mut_ptr();
        let mut aa = a.as_ptr();
        for _ in 0..span {
            let s0: float64x2_t = vsubq_f64(vld1q_f64(rr), vld1q_f64(aa));
            vst1q_f64(rr, s0);
            let s1: float64x2_t = vsubq_f64(vld1q_f64(rr.add(2)), vld1q_f64(aa.add(2)));
            vst1q_f64(rr.add(2), s1);
            rr = rr.add(4);
            aa = aa.add(4);
        }
    }
    let tail = span << 2;
    if tail < n {
        reim_sub_assign_portable(&mut res[tail..], &a[tail..]);
    }
}

/// `res[i] = a[i] - res[i]` for all `i`.
pub(crate) fn reim_sub_negate_assign_neon(res: &mut [f64], a: &[f64]) {
    assert_eq!(res.len(), a.len());
    let n = res.len();
    let span = n >> 2;
    unsafe {
        let mut rr = res.as_mut_ptr();
        let mut aa = a.as_ptr();
        for _ in 0..span {
            let s0: float64x2_t = vsubq_f64(vld1q_f64(aa), vld1q_f64(rr));
            vst1q_f64(rr, s0);
            let s1: float64x2_t = vsubq_f64(vld1q_f64(aa.add(2)), vld1q_f64(rr.add(2)));
            vst1q_f64(rr.add(2), s1);
            rr = rr.add(4);
            aa = aa.add(4);
        }
    }
    let tail = span << 2;
    if tail < n {
        reim_sub_negate_assign_portable(&mut res[tail..], &a[tail..]);
    }
}

/// `res[i] = -a[i]` for all `i`.
pub(crate) fn reim_negate_neon(res: &mut [f64], a: &[f64]) {
    assert_eq!(res.len(), a.len());
    let n = res.len();
    let span = n >> 2;
    unsafe {
        let mut rr = res.as_mut_ptr();
        let mut aa = a.as_ptr();
        for _ in 0..span {
            let s0: float64x2_t = vnegq_f64(vld1q_f64(aa));
            vst1q_f64(rr, s0);
            let s1: float64x2_t = vnegq_f64(vld1q_f64(aa.add(2)));
            vst1q_f64(rr.add(2), s1);
            rr = rr.add(4);
            aa = aa.add(4);
        }
    }
    let tail = span << 2;
    if tail < n {
        reim_negate_portable(&mut res[tail..], &a[tail..]);
    }
}

/// `res[i] = -res[i]` for all `i`.
pub(crate) fn reim_negate_assign_neon(res: &mut [f64]) {
    let n = res.len();
    let span = n >> 2;
    unsafe {
        let mut rr = res.as_mut_ptr();
        for _ in 0..span {
            let s0: float64x2_t = vnegq_f64(vld1q_f64(rr));
            vst1q_f64(rr, s0);
            let s1: float64x2_t = vnegq_f64(vld1q_f64(rr.add(2)));
            vst1q_f64(rr.add(2), s1);
            rr = rr.add(4);
        }
    }
    let tail = span << 2;
    if tail < n {
        reim_negate_assign_portable(&mut res[tail..]);
    }
}

/// Complex multiply: `res = a * b` over `m` complex points (split layout).
/// `(ar + i·ai) * (br + i·bi) = (ar·br − ai·bi) + i·(ar·bi + ai·br)`.
pub(crate) fn reim_mul_neon(res: &mut [f64], a: &[f64], b: &[f64]) {
    assert_eq!(res.len(), a.len());
    assert_eq!(res.len(), b.len());
    let m = res.len() >> 1;
    let span = m >> 2;
    let (rr, ri) = res.split_at_mut(m);
    let (ar, ai) = a.split_at(m);
    let (br, bi) = b.split_at(m);
    unsafe {
        let mut rr_ptr = rr.as_mut_ptr();
        let mut ri_ptr = ri.as_mut_ptr();
        let mut ar_ptr = ar.as_ptr();
        let mut ai_ptr = ai.as_ptr();
        let mut br_ptr = br.as_ptr();
        let mut bi_ptr = bi.as_ptr();
        for _ in 0..span {
            let chunk = |off: usize, rr_p: *mut f64, ri_p: *mut f64| {
                let ar_v = vld1q_f64(ar_ptr.add(off));
                let ai_v = vld1q_f64(ai_ptr.add(off));
                let br_v = vld1q_f64(br_ptr.add(off));
                let bi_v = vld1q_f64(bi_ptr.add(off));
                // rr = ar·br − ai·bi  (seed with ar·br; vfmsq subtracts ai·bi)
                let t1 = vmulq_f64(ar_v, br_v);
                let rr_out = vfmsq_f64(t1, ai_v, bi_v);
                vst1q_f64(rr_p.add(off), rr_out);
                // ri = ar·bi + ai·br
                let t2 = vmulq_f64(ar_v, bi_v);
                let ri_out = vfmaq_f64(t2, ai_v, br_v); // t2 + ai·br
                vst1q_f64(ri_p.add(off), ri_out);
            };
            chunk(0, rr_ptr, ri_ptr);
            chunk(2, rr_ptr, ri_ptr);
            rr_ptr = rr_ptr.add(4);
            ri_ptr = ri_ptr.add(4);
            ar_ptr = ar_ptr.add(4);
            ai_ptr = ai_ptr.add(4);
            br_ptr = br_ptr.add(4);
            bi_ptr = bi_ptr.add(4);
        }
    }
    let tail = span << 2;
    if tail < m {
        // Recombine the tail slices: split halves on m, but tail starts at
        // `tail` within each half; reim_mul_portable expects the original full
        // layout. Reconstruct by slicing relative to the original buffers.
        let n = res.len();
        let lo = tail;
        let hi_off = m + tail;
        let res_tail = &mut res[lo..]; // covers re_tail.. (length m-tail) then im
        // res_tail length = (m - tail) + (m - tail) = 2*(m - tail)? No:
        // res_tail = res[lo..n] = [re_tail..re_{m-1}, im_0..im_{m-1}].
        // We need only the slice equivalent to a "full reim vector" for the
        // tail elements. Using reim_mul_portable directly on the unaligned
        // residual is unsafe — fall back to per-element scalar.
        let _ = (res_tail, hi_off, n); // unused: explicit per-element fallback below.
        for i in tail..m {
            let ar_v = a[i];
            let ai_v = a[m + i];
            let br_v = b[i];
            let bi_v = b[m + i];
            res[i] = ar_v * br_v - ai_v * bi_v;
            res[m + i] = ar_v * bi_v + ai_v * br_v;
        }
    }
}

/// Complex multiply in place: `res *= a`. Mirrors `reim_mul_assign_avx2_fma` at
/// `fft_vec_avx2_fma.rs:317`.
pub(crate) fn reim_mul_assign_neon(res: &mut [f64], a: &[f64]) {
    assert_eq!(res.len(), a.len());
    let m = res.len() >> 1;
    let span = m >> 2;
    let (rr, ri) = res.split_at_mut(m);
    let (ar, ai) = a.split_at(m);
    unsafe {
        let mut rr_ptr = rr.as_mut_ptr();
        let mut ri_ptr = ri.as_mut_ptr();
        let mut ar_ptr = ar.as_ptr();
        let mut ai_ptr = ai.as_ptr();
        for _ in 0..span {
            for off in [0usize, 2] {
                let ar_v = vld1q_f64(ar_ptr.add(off));
                let ai_v = vld1q_f64(ai_ptr.add(off));
                let br_v = vld1q_f64(rr_ptr.add(off));
                let bi_v = vld1q_f64(ri_ptr.add(off));
                let t1 = vmulq_f64(ar_v, br_v);
                let rr_out = vfmsq_f64(t1, ai_v, bi_v);
                vst1q_f64(rr_ptr.add(off), rr_out);
                let t2 = vmulq_f64(ar_v, bi_v);
                let ri_out = vfmaq_f64(t2, ai_v, br_v);
                vst1q_f64(ri_ptr.add(off), ri_out);
            }
            rr_ptr = rr_ptr.add(4);
            ri_ptr = ri_ptr.add(4);
            ar_ptr = ar_ptr.add(4);
            ai_ptr = ai_ptr.add(4);
        }
    }
    let tail = span << 2;
    if tail < m {
        // Per-element scalar tail (see reim_mul_neon for rationale).
        for i in tail..m {
            let ar_v = a[i];
            let ai_v = a[m + i];
            let br_v = res[i];
            let bi_v = res[m + i];
            res[i] = ar_v * br_v - ai_v * bi_v;
            res[m + i] = ar_v * bi_v + ai_v * br_v;
        }
    }
    // Note: cannot use reim_mul_assign_portable for the tail because the
    // reference function assumes a full split-layout slice, but our
    // remainder is at an offset within `res` and `a`. Fall through is
    // exact scalar.
    let _ = reim_mul_assign_portable; // suppress unused-import on the cfg path
}

/// Complex addmul: `res += a * b`. Mirrors `reim_addmul_avx2_fma` at
/// `fft_vec_avx2_fma.rs:214`.
pub(crate) fn reim_addmul_neon(res: &mut [f64], a: &[f64], b: &[f64]) {
    assert_eq!(res.len(), a.len());
    assert_eq!(res.len(), b.len());
    let m = res.len() >> 1;
    let span = m >> 2;
    let (rr, ri) = res.split_at_mut(m);
    let (ar, ai) = a.split_at(m);
    let (br, bi) = b.split_at(m);
    unsafe {
        let mut rr_ptr = rr.as_mut_ptr();
        let mut ri_ptr = ri.as_mut_ptr();
        let mut ar_ptr = ar.as_ptr();
        let mut ai_ptr = ai.as_ptr();
        let mut br_ptr = br.as_ptr();
        let mut bi_ptr = bi.as_ptr();
        for _ in 0..span {
            for off in [0usize, 2] {
                let ar_v = vld1q_f64(ar_ptr.add(off));
                let ai_v = vld1q_f64(ai_ptr.add(off));
                let br_v = vld1q_f64(br_ptr.add(off));
                let bi_v = vld1q_f64(bi_ptr.add(off));
                let mut rr_v = vld1q_f64(rr_ptr.add(off));
                let mut ri_v = vld1q_f64(ri_ptr.add(off));
                // rr += ar·br − ai·bi
                rr_v = vfmaq_f64(rr_v, ar_v, br_v);
                rr_v = vfmsq_f64(rr_v, ai_v, bi_v);
                // ri += ar·bi + ai·br
                ri_v = vfmaq_f64(ri_v, ar_v, bi_v);
                ri_v = vfmaq_f64(ri_v, ai_v, br_v);
                vst1q_f64(rr_ptr.add(off), rr_v);
                vst1q_f64(ri_ptr.add(off), ri_v);
            }
            rr_ptr = rr_ptr.add(4);
            ri_ptr = ri_ptr.add(4);
            ar_ptr = ar_ptr.add(4);
            ai_ptr = ai_ptr.add(4);
            br_ptr = br_ptr.add(4);
            bi_ptr = bi_ptr.add(4);
        }
    }
    let tail = span << 2;
    if tail < m {
        for i in tail..m {
            let ar_v = a[i];
            let ai_v = a[m + i];
            let br_v = b[i];
            let bi_v = b[m + i];
            res[i] += ar_v * br_v - ai_v * bi_v;
            res[m + i] += ar_v * bi_v + ai_v * br_v;
        }
    }
    let _ = reim_addmul_portable;
}

/// `i64 → f64` conversion, rounded to nearest even as the `as f64` cast of the reference.
///
/// Uses the native conversion, so every `i64` is accepted.
pub(crate) fn reim_from_znx_i64_neon(res: &mut [f64], a: &[i64]) {
    assert_eq!(res.len(), a.len());
    let n = res.len();
    let span = n >> 2;
    unsafe {
        let (r, a) = (res.as_mut_ptr(), a.as_ptr());
        for i in 0..span {
            let o = 4 * i;
            vst1q_f64(r.add(o), vcvtq_f64_s64(vld1q_s64(a.add(o))));
            vst1q_f64(r.add(o + 2), vcvtq_f64_s64(vld1q_s64(a.add(o + 2))));
        }
    }
    let tail = span << 2;
    if tail < n {
        reim_from_znx_i64_portable(&mut res[tail..], &a[tail..]);
    }
}

/// `f64 → i64` conversion of `a / divisor`, rounded to nearest with ties away from zero.
///
/// The native conversion computes `(a * (1 / divisor)).round() as i64` as the reference does, saturation included.
pub(crate) fn reim_to_znx_i64_neon(res: &mut [i64], divisor: f64, a: &[f64]) {
    assert_eq!(res.len(), a.len());
    let n = res.len();
    let span = n >> 2;
    unsafe {
        let inv = vdupq_n_f64(1. / divisor);
        let (r, a) = (res.as_mut_ptr(), a.as_ptr());
        for i in 0..span {
            let o = 4 * i;
            vst1q_s64(r.add(o), vcvtaq_s64_f64(vmulq_f64(vld1q_f64(a.add(o)), inv)));
            vst1q_s64(r.add(o + 2), vcvtaq_s64_f64(vmulq_f64(vld1q_f64(a.add(o + 2)), inv)));
        }
    }
    let tail = span << 2;
    if tail < n {
        reim_to_znx_i64_portable(&mut res[tail..], divisor, &a[tail..]);
    }
}

/// In-place variant: read `f64`, write `i64` (reinterpreted) into the same buffer.
pub(crate) fn reim_to_znx_i64_assign_neon(res: &mut [f64], divisor: f64) {
    let n = res.len();
    let span = n >> 2;
    unsafe {
        let inv = vdupq_n_f64(1. / divisor);
        let p = res.as_mut_ptr();
        for i in 0..span {
            let o = 4 * i;
            let lo = vcvtaq_s64_f64(vmulq_f64(vld1q_f64(p.add(o)), inv));
            let hi = vcvtaq_s64_f64(vmulq_f64(vld1q_f64(p.add(o + 2)), inv));
            vst1q_s64(p.add(o) as *mut i64, lo);
            vst1q_s64(p.add(o + 2) as *mut i64, hi);
        }
    }
    let tail = span << 2;
    if tail < n {
        reim_to_znx_i64_assign_portable(&mut res[tail..], divisor);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn reim_to_znx_rounding_boundaries() {
        poulpy_cpu_portable::test_suite::reim_conversion::test_reim_to_znx_rounding::<crate::FFT64Neon>();
    }

    use super::*;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    /// Sizes exercising both the SIMD body and the scalar tail. `reim_mul`
    /// expects even length, so all sizes here are even.
    const SIZES: &[usize] = &[2, 4, 6, 8, 16, 18, 64, 66, 256, 258];

    fn rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(0xfeed_beef_cafe_babe)
    }

    fn random_f64(rng: &mut ChaCha8Rng, n: usize) -> Vec<f64> {
        (0..n).map(|_| rng.random::<f64>() * 1e6 - 5e5).collect()
    }

    /// Bit-exact check: NEON kernel must match the scalar reference for
    /// pointwise add/sub/negate (no FMA, so no rounding divergence).
    #[test]
    fn reim_add_neon_exact_vs_ref() {
        let mut r = rng();
        for &n in SIZES {
            let a = random_f64(&mut r, n);
            let b = random_f64(&mut r, n);
            let mut got = vec![0f64; n];
            let mut want = vec![0f64; n];
            reim_add_neon(&mut got, &a, &b);
            reim_add_portable(&mut want, &a, &b);
            assert_eq!(got, want, "reim_add_neon n={n}");
        }
    }

    #[test]
    fn reim_sub_neon_exact_vs_ref() {
        let mut r = rng();
        for &n in SIZES {
            let a = random_f64(&mut r, n);
            let b = random_f64(&mut r, n);
            let mut got = vec![0f64; n];
            let mut want = vec![0f64; n];
            reim_sub_neon(&mut got, &a, &b);
            reim_sub_portable(&mut want, &a, &b);
            assert_eq!(got, want, "reim_sub_neon n={n}");
        }
    }

    #[test]
    fn reim_negate_neon_exact_vs_ref() {
        let mut r = rng();
        for &n in SIZES {
            let a = random_f64(&mut r, n);
            let mut got = vec![0f64; n];
            let mut want = vec![0f64; n];
            reim_negate_neon(&mut got, &a);
            reim_negate_portable(&mut want, &a);
            assert_eq!(got, want, "reim_negate_neon n={n}");
        }
    }

    /// Tolerance check: NEON `reim_mul` and `reim_addmul` use FMA, which the
    /// scalar reference does not — so results may differ in the last bit.
    /// Allow a tiny ULP-relative tolerance.
    fn close_enough(got: &[f64], want: &[f64], tag: &str) {
        const REL_TOL: f64 = 1e-12;
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            let denom = w.abs().max(1.0);
            let diff = (g - w).abs();
            assert!(diff / denom < REL_TOL, "{tag}: idx={i} got={g} want={w} diff={diff}");
        }
    }

    #[test]
    fn reim_mul_neon_close_to_ref() {
        let mut r = rng();
        for &n in SIZES {
            let a = random_f64(&mut r, n);
            let b = random_f64(&mut r, n);
            let mut got = vec![0f64; n];
            let mut want = vec![0f64; n];
            reim_mul_neon(&mut got, &a, &b);
            reim_mul_portable(&mut want, &a, &b);
            close_enough(&got, &want, &format!("reim_mul_neon n={n}"));
        }
    }

    #[test]
    fn reim_from_znx_neon_exact_vs_ref() {
        let mut r = rng();
        for &n in SIZES {
            // The whole range, with the values whose conversion rounds.
            let a: Vec<i64> = (0..n)
                .map(|i| match i % 6 {
                    0 => i64::MAX,
                    1 => i64::MIN,
                    2 => (1i64 << 53) + 1,
                    3 => -(1i64 << 53) - 3,
                    _ => r.random::<i64>(),
                })
                .collect();
            let mut got = vec![0f64; n];
            let mut want = vec![0f64; n];
            reim_from_znx_i64_neon(&mut got, &a);
            reim_from_znx_i64_portable(&mut want, &a);
            assert_eq!(got, want, "reim_from_znx_i64_neon n={n}");
        }
    }

    #[test]
    fn reim_addmul_neon_close_to_ref() {
        let mut r = rng();
        for &n in SIZES {
            let a = random_f64(&mut r, n);
            let b = random_f64(&mut r, n);
            let r0 = random_f64(&mut r, n);
            let mut got = r0.clone();
            let mut want = r0;
            reim_addmul_neon(&mut got, &a, &b);
            reim_addmul_portable(&mut want, &a, &b);
            close_enough(&got, &want, &format!("reim_addmul_neon n={n}"));
        }
    }
}
