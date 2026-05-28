//! NEON i64 kernels for `Znx*` ring-element arithmetic.
//!
//! `_assign` variants tolerate `res == a`; other variants assume disjoint slices.

use core::arch::aarch64::{
    int64x2_t, vaddq_s64, vandq_s64, vcgtq_s64, vdupq_n_s64, veorq_s64, vgetq_lane_s64, vld1q_s64, vnegq_s64,
    vreinterpretq_s64_u64, vreinterpretq_u64_s64, vsetq_lane_s64, vshlq_s64, vshrq_n_u64, vst1q_s64, vsubq_s64,
};

use poulpy_cpu_ref::reference::znx::{
    znx_add_assign_ref, znx_add_ref, znx_automorphism_ref, znx_copy_ref, znx_mul_add_power_of_two_ref,
    znx_mul_power_of_two_assign_ref, znx_mul_power_of_two_ref, znx_negate_assign_ref, znx_negate_ref, znx_sub_assign_ref,
    znx_sub_negate_assign_ref, znx_sub_ref, znx_switch_ring_ref, znx_zero_ref,
};

/// `res[i] = a[i].wrapping_add(b[i])` for all `i`.
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

#[inline]
fn inv_mod_pow2(p: usize, bits: u32) -> usize {
    debug_assert!(p % 2 == 1);
    let mut x: usize = 1;
    let mut i: u32 = 1;
    while i < bits {
        x = x.wrapping_mul(2usize.wrapping_sub(p.wrapping_mul(x)));
        i <<= 1;
    }
    x & ((1usize << bits) - 1)
}

/// `res[i] = (-1)^{(i*p > n)} * a[(i*p) mod 2n]`, the negacyclic automorphism by `p`.
pub(crate) fn znx_automorphism_neon(p: i64, res: &mut [i64], a: &[i64]) {
    debug_assert_eq!(res.len(), a.len());
    let n: usize = res.len();
    if n == 0 {
        return;
    }
    assert!(n.is_power_of_two(), "Polynomial degree {} must be power of 2", n);
    debug_assert!(p & 1 == 1, "p must be odd (invertible mod 2n)");

    if n < 4 {
        znx_automorphism_ref(p, res, a);
        return;
    }

    let two_n: usize = n << 1;
    let span: usize = n >> 2;
    let bits: u32 = (two_n as u64).trailing_zeros();
    let mask_2n: usize = two_n - 1;
    let mask_1n: usize = n - 1;

    let p_2n: usize = (((p & mask_2n as i64) + two_n as i64) as usize) & mask_2n;
    let inv: usize = inv_mod_pow2(p_2n, bits);

    unsafe {
        let mask_2n_v: int64x2_t = vdupq_n_s64(mask_2n as i64);
        let mask_1n_v: int64x2_t = vdupq_n_s64(mask_1n as i64);
        let n_minus1_v: int64x2_t = vdupq_n_s64((n as i64) - 1);

        // lane offsets: [0, inv] and [2*inv, 3*inv] (mod 2n)
        let off01: int64x2_t = vsetq_lane_s64::<1>(inv as i64, vdupq_n_s64(0));
        let off23: int64x2_t = vsetq_lane_s64::<1>(((inv * 3) & mask_2n) as i64, vdupq_n_s64(((inv * 2) & mask_2n) as i64));

        let mut t_base: usize = 0;
        let step: usize = (inv << 2) & mask_2n;

        let mut rr: *mut i64 = res.as_mut_ptr();
        let aa: *const i64 = a.as_ptr();

        for _ in 0..span {
            let tb: int64x2_t = vdupq_n_s64(t_base as i64);
            let t01: int64x2_t = vandq_s64(vaddq_s64(tb, off01), mask_2n_v);
            let t23: int64x2_t = vandq_s64(vaddq_s64(tb, off23), mask_2n_v);

            let idx01: int64x2_t = vandq_s64(t01, mask_1n_v);
            let idx23: int64x2_t = vandq_s64(t23, mask_1n_v);

            let i0 = vgetq_lane_s64::<0>(idx01) as usize;
            let i1 = vgetq_lane_s64::<1>(idx01) as usize;
            let i2 = vgetq_lane_s64::<0>(idx23) as usize;
            let i3 = vgetq_lane_s64::<1>(idx23) as usize;

            let v0 = *aa.add(i0);
            let v1 = *aa.add(i1);
            let v2 = *aa.add(i2);
            let v3 = *aa.add(i3);
            let vals01: int64x2_t = vsetq_lane_s64::<1>(v1, vdupq_n_s64(v0));
            let vals23: int64x2_t = vsetq_lane_s64::<1>(v3, vdupq_n_s64(v2));

            // sign = (t >= n) ? -1 : 0  (cmpgt against n-1)
            let sign01: int64x2_t = vreinterpretq_s64_u64(vcgtq_s64(t01, n_minus1_v));
            let sign23: int64x2_t = vreinterpretq_s64_u64(vcgtq_s64(t23, n_minus1_v));

            // conditional negate: (val ^ sign) - sign
            let out01: int64x2_t = vsubq_s64(veorq_s64(vals01, sign01), sign01);
            let out23: int64x2_t = vsubq_s64(veorq_s64(vals23, sign23), sign23);

            vst1q_s64(rr, out01);
            vst1q_s64(rr.add(2), out23);

            rr = rr.add(4);
            t_base = (t_base + step) & mask_2n;
        }
    }
}

/// `res[k] = a[k * gap]` (downsample) or `res[k * gap] = a[k]`, else zero (upsample).
pub(crate) fn znx_switch_ring_neon(res: &mut [i64], a: &[i64]) {
    let (n_in, n_out) = (a.len(), res.len());
    debug_assert!(n_in.is_power_of_two());
    debug_assert!(n_in.max(n_out).is_multiple_of(n_in.min(n_out)));

    if n_in == n_out {
        znx_copy_ref(res, a);
        return;
    }

    if n_in > n_out {
        let gap = n_in / n_out;
        let span = n_out >> 2;
        let aa = a.as_ptr();
        let mut rr = res.as_mut_ptr();
        unsafe {
            let mut base: usize = 0;
            for _ in 0..span {
                let v01: int64x2_t = vsetq_lane_s64::<1>(*aa.add(base + gap), vdupq_n_s64(*aa.add(base)));
                let v23: int64x2_t = vsetq_lane_s64::<1>(*aa.add(base + 3 * gap), vdupq_n_s64(*aa.add(base + 2 * gap)));
                vst1q_s64(rr, v01);
                vst1q_s64(rr.add(2), v23);
                rr = rr.add(4);
                base += 4 * gap;
            }
        }
        let tail = span << 2;
        if tail < n_out {
            znx_switch_ring_ref(&mut res[tail..], &a[tail * gap..]);
        }
    } else {
        let gap = n_out / n_in;
        znx_zero_ref(res);
        let span = n_in >> 2;
        let mut aa = a.as_ptr();
        let rr = res.as_mut_ptr();
        unsafe {
            for i in 0..span {
                let v01: int64x2_t = vld1q_s64(aa);
                let v23: int64x2_t = vld1q_s64(aa.add(2));
                let base = (i << 2) * gap;
                *rr.add(base) = vgetq_lane_s64::<0>(v01);
                *rr.add(base + gap) = vgetq_lane_s64::<1>(v01);
                *rr.add(base + 2 * gap) = vgetq_lane_s64::<0>(v23);
                *rr.add(base + 3 * gap) = vgetq_lane_s64::<1>(v23);
                aa = aa.add(4);
            }
            let tail = span << 2;
            for (i, &v) in a.iter().enumerate().skip(tail) {
                *rr.add(i * gap) = v;
            }
        }
    }
}

/// Arithmetic right shift with rounding bias: `(x + bias) >> kp`, `bias = (1<<(kp-1)) - (x>>63)`.
/// kp in `[1, 63]`; `cnt_right = vdupq_n_s64(-kp)`.
#[inline(always)]
unsafe fn rshift_round_neon(x: int64x2_t, bias_base: int64x2_t, cnt_right: int64x2_t) -> int64x2_t {
    unsafe {
        let sign_bit: int64x2_t = vreinterpretq_s64_u64(vshrq_n_u64::<63>(vreinterpretq_u64_s64(x)));
        let bias: int64x2_t = vsubq_s64(bias_base, sign_bit);
        let t: int64x2_t = vaddq_s64(x, bias);
        vshlq_s64(t, cnt_right)
    }
}

/// `res[i] = a[i] << k` (k > 0) or `a[i] >>_rounded |k|` (k < 0).
pub(crate) fn znx_mul_power_of_two_neon(k: i64, res: &mut [i64], a: &[i64]) {
    debug_assert_eq!(res.len(), a.len());
    let n = res.len();
    if n == 0 {
        return;
    }
    if k == 0 {
        znx_copy_ref(res, a);
        return;
    }
    let span = n >> 2;
    unsafe {
        let mut rr = res.as_mut_ptr();
        let mut aa = a.as_ptr();
        if k > 0 {
            debug_assert!(k <= 63);
            let cnt: int64x2_t = vdupq_n_s64(k);
            for _ in 0..span {
                vst1q_s64(rr, vshlq_s64(vld1q_s64(aa), cnt));
                vst1q_s64(rr.add(2), vshlq_s64(vld1q_s64(aa.add(2)), cnt));
                rr = rr.add(4);
                aa = aa.add(4);
            }
        } else {
            let kp = -k;
            assert!((1..=63).contains(&kp));
            let cnt_right: int64x2_t = vdupq_n_s64(-kp);
            let bias_base: int64x2_t = vdupq_n_s64(1_i64 << (kp - 1));
            for _ in 0..span {
                vst1q_s64(rr, rshift_round_neon(vld1q_s64(aa), bias_base, cnt_right));
                vst1q_s64(rr.add(2), rshift_round_neon(vld1q_s64(aa.add(2)), bias_base, cnt_right));
                rr = rr.add(4);
                aa = aa.add(4);
            }
        }
    }
    let tail = span << 2;
    if tail < n {
        znx_mul_power_of_two_ref(k, &mut res[tail..], &a[tail..]);
    }
}

/// In-place variant of [`znx_mul_power_of_two_neon`].
pub(crate) fn znx_mul_power_of_two_assign_neon(k: i64, res: &mut [i64]) {
    let n = res.len();
    if n == 0 || k == 0 {
        return;
    }
    let span = n >> 2;
    unsafe {
        let mut rr = res.as_mut_ptr();
        if k > 0 {
            debug_assert!(k <= 63);
            let cnt: int64x2_t = vdupq_n_s64(k);
            for _ in 0..span {
                vst1q_s64(rr, vshlq_s64(vld1q_s64(rr), cnt));
                vst1q_s64(rr.add(2), vshlq_s64(vld1q_s64(rr.add(2)), cnt));
                rr = rr.add(4);
            }
        } else {
            let kp = -k;
            assert!((1..=63).contains(&kp));
            let cnt_right: int64x2_t = vdupq_n_s64(-kp);
            let bias_base: int64x2_t = vdupq_n_s64(1_i64 << (kp - 1));
            for _ in 0..span {
                vst1q_s64(rr, rshift_round_neon(vld1q_s64(rr), bias_base, cnt_right));
                vst1q_s64(rr.add(2), rshift_round_neon(vld1q_s64(rr.add(2)), bias_base, cnt_right));
                rr = rr.add(4);
            }
        }
    }
    let tail = span << 2;
    if tail < n {
        znx_mul_power_of_two_assign_ref(k, &mut res[tail..]);
    }
}

/// `res[i] += a[i] << k` (or `>> |k|` rounded) — fused multiply-add by power of two.
pub(crate) fn znx_mul_add_power_of_two_neon(k: i64, res: &mut [i64], a: &[i64]) {
    debug_assert_eq!(res.len(), a.len());
    let n = res.len();
    if n == 0 {
        return;
    }
    if k == 0 {
        znx_add_assign_neon(res, a);
        return;
    }
    let span = n >> 2;
    unsafe {
        let mut rr = res.as_mut_ptr();
        let mut aa = a.as_ptr();
        if k > 0 {
            debug_assert!(k <= 63);
            let cnt: int64x2_t = vdupq_n_s64(k);
            for _ in 0..span {
                vst1q_s64(rr, vaddq_s64(vld1q_s64(rr), vshlq_s64(vld1q_s64(aa), cnt)));
                vst1q_s64(
                    rr.add(2),
                    vaddq_s64(vld1q_s64(rr.add(2)), vshlq_s64(vld1q_s64(aa.add(2)), cnt)),
                );
                rr = rr.add(4);
                aa = aa.add(4);
            }
        } else {
            let kp = -k;
            assert!((1..=63).contains(&kp));
            let cnt_right: int64x2_t = vdupq_n_s64(-kp);
            let bias_base: int64x2_t = vdupq_n_s64(1_i64 << (kp - 1));
            for _ in 0..span {
                vst1q_s64(
                    rr,
                    vaddq_s64(vld1q_s64(rr), rshift_round_neon(vld1q_s64(aa), bias_base, cnt_right)),
                );
                vst1q_s64(
                    rr.add(2),
                    vaddq_s64(
                        vld1q_s64(rr.add(2)),
                        rshift_round_neon(vld1q_s64(aa.add(2)), bias_base, cnt_right),
                    ),
                );
                rr = rr.add(4);
                aa = aa.add(4);
            }
        }
    }
    let tail = span << 2;
    if tail < n {
        znx_mul_add_power_of_two_ref(k, &mut res[tail..], &a[tail..]);
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
