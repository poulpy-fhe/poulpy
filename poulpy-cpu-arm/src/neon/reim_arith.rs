//! NEON pointwise REIM arithmetic for the FFT64 backend.
//!
//! Layout: a complex vector of length `2m` is stored as `[re_0..re_{m-1},
//! im_0..im_{m-1}]` (split, not interleaved).

use core::arch::aarch64::{
    float64x2_t, int64x2_t, vaddq_f64, vaddq_s64, vandq_s64, vandq_u64, vdupq_n_f64, vdupq_n_s64, vdupq_n_u64, veorq_s64,
    vfmaq_f64, vfmsq_f64, vld1q_f64, vld1q_s64, vmulq_f64, vnegq_f64, vorrq_s64, vorrq_u64, vreinterpretq_f64_u64,
    vreinterpretq_s64_f64, vreinterpretq_s64_u64, vreinterpretq_u64_f64, vreinterpretq_u64_s64, vshlq_u64, vshrq_n_s64,
    vshrq_n_u64, vst1q_f64, vst1q_s64, vsubq_f64, vsubq_s64,
};

#[allow(unused_imports)]
use poulpy_cpu_ref::reference::fft64::reim::{
    reim_add_assign_ref, reim_add_ref, reim_addmul_ref, reim_from_znx_i64_masked_ref, reim_from_znx_i64_ref, reim_mul_assign_ref,
    reim_mul_ref, reim_negate_assign_ref, reim_negate_ref, reim_sub_assign_ref, reim_sub_negate_assign_ref, reim_sub_ref,
    reim_to_znx_i64_assign_ref, reim_to_znx_i64_ref,
};

/// `res[i] = a[i] + b[i]` for all `i`.
/// Kept for the unit test below; the `FFT64Neon` `ReimArith::reim_add` impl
/// routes to the autovec reference because the hand-NEON loop is memory
/// bandwidth bound at large `n` and the autovec wins.
#[allow(dead_code)]
pub(crate) fn reim_add_neon(res: &mut [f64], a: &[f64], b: &[f64]) {
    debug_assert_eq!(res.len(), a.len());
    debug_assert_eq!(res.len(), b.len());
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
        reim_add_ref(&mut res[tail..], &a[tail..], &b[tail..]);
    }
}

/// `res[i] = res[i] + a[i]` for all `i`.
/// See `reim_add_neon`: kept for tests, dispatched to the ref by `FFT64Neon`.
#[allow(dead_code)]
pub(crate) fn reim_add_assign_neon(res: &mut [f64], a: &[f64]) {
    debug_assert_eq!(res.len(), a.len());
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
        reim_add_assign_ref(&mut res[tail..], &a[tail..]);
    }
}

/// `res[i] = a[i] - b[i]` for all `i`.
pub(crate) fn reim_sub_neon(res: &mut [f64], a: &[f64], b: &[f64]) {
    debug_assert_eq!(res.len(), a.len());
    debug_assert_eq!(res.len(), b.len());
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
        reim_sub_ref(&mut res[tail..], &a[tail..], &b[tail..]);
    }
}

/// `res[i] = res[i] - a[i]` for all `i`.
pub(crate) fn reim_sub_assign_neon(res: &mut [f64], a: &[f64]) {
    debug_assert_eq!(res.len(), a.len());
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
        reim_sub_assign_ref(&mut res[tail..], &a[tail..]);
    }
}

/// `res[i] = a[i] - res[i]` for all `i`.
pub(crate) fn reim_sub_negate_assign_neon(res: &mut [f64], a: &[f64]) {
    debug_assert_eq!(res.len(), a.len());
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
        reim_sub_negate_assign_ref(&mut res[tail..], &a[tail..]);
    }
}

/// `res[i] = -a[i]` for all `i`.
pub(crate) fn reim_negate_neon(res: &mut [f64], a: &[f64]) {
    debug_assert_eq!(res.len(), a.len());
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
        reim_negate_ref(&mut res[tail..], &a[tail..]);
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
        reim_negate_assign_ref(&mut res[tail..]);
    }
}

/// Complex multiply: `res = a * b` over `m` complex points (split layout).
/// `(ar + i·ai) * (br + i·bi) = (ar·br − ai·bi) + i·(ar·bi + ai·br)`.
pub(crate) fn reim_mul_neon(res: &mut [f64], a: &[f64], b: &[f64]) {
    debug_assert_eq!(res.len(), a.len());
    debug_assert_eq!(res.len(), b.len());
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
        // `tail` within each half; reim_mul_ref expects the original full
        // layout. Reconstruct by slicing relative to the original buffers.
        let n = res.len();
        let lo = tail;
        let hi_off = m + tail;
        let res_tail = &mut res[lo..]; // covers re_tail.. (length m-tail) then im
        // res_tail length = (m - tail) + (m - tail) = 2*(m - tail)? No:
        // res_tail = res[lo..n] = [re_tail..re_{m-1}, im_0..im_{m-1}].
        // We need only the slice equivalent to a "full reim vector" for the
        // tail elements. Using reim_mul_ref directly on the unaligned
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
    debug_assert_eq!(res.len(), a.len());
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
    // Note: cannot use reim_mul_assign_ref for the tail because the
    // reference function assumes a full split-layout slice, but our
    // remainder is at an offset within `res` and `a`. Fall through is
    // exact scalar.
    let _ = reim_mul_assign_ref; // suppress unused-import on the cfg path
}

/// Complex addmul: `res += a * b`. Mirrors `reim_addmul_avx2_fma` at
/// `fft_vec_avx2_fma.rs:214`.
pub(crate) fn reim_addmul_neon(res: &mut [f64], a: &[f64], b: &[f64]) {
    debug_assert_eq!(res.len(), a.len());
    debug_assert_eq!(res.len(), b.len());
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
    let _ = reim_addmul_ref;
}

/// `i64 → f64` exact conversion for `|a[i]| < 2^50` via IEEE 754 bit trick.
///
/// Caller must ensure `|a[i]| <= 2^50 - 1`; debug builds assert.
pub(crate) fn reim_from_znx_i64_bnd50_neon(res: &mut [f64], a: &[i64]) {
    debug_assert_eq!(res.len(), a.len());
    #[cfg(debug_assertions)]
    {
        const BOUND: i64 = (1i64 << 50) - 1;
        for (i, &val) in a.iter().enumerate() {
            assert!(
                val.abs() <= BOUND,
                "reim_from_znx_i64_bnd50_neon: a[{i}] = {val} exceeds 2^50-1"
            );
        }
    }
    let n = res.len();
    let span = n >> 2;
    unsafe {
        let expo: f64 = (1i64 << 52) as f64;
        let add_cst: i64 = 1i64 << 51;
        let sub_cst: f64 = (3i64 << 51) as f64;

        let expo_v = vreinterpretq_u64_f64(vdupq_n_f64(expo));
        let add_cst_v = vdupq_n_s64(add_cst);
        let sub_cst_v = vdupq_n_f64(sub_cst);

        let mut res_ptr = res.as_mut_ptr();
        let mut a_ptr = a.as_ptr();

        for _ in 0..span {
            // Process 4 lanes via 2 NEON registers.
            let lo = vaddq_s64(vld1q_s64(a_ptr), add_cst_v);
            let hi = vaddq_s64(vld1q_s64(a_ptr.add(2)), add_cst_v);
            let mut lo_f = vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_s64(lo), expo_v));
            let mut hi_f = vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_s64(hi), expo_v));
            lo_f = vsubq_f64(lo_f, sub_cst_v);
            hi_f = vsubq_f64(hi_f, sub_cst_v);
            vst1q_f64(res_ptr, lo_f);
            vst1q_f64(res_ptr.add(2), hi_f);
            res_ptr = res_ptr.add(4);
            a_ptr = a_ptr.add(4);
        }
    }
    let tail = span << 2;
    if tail < n {
        reim_from_znx_i64_ref(&mut res[tail..], &a[tail..]);
    }
}

/// Masked variant: `(a[i] & mask) → f64`. Mirrors `reim_from_znx_i64_masked_bnd50_fma`.
pub(crate) fn reim_from_znx_i64_masked_bnd50_neon(res: &mut [f64], a: &[i64], mask: i64) {
    debug_assert_eq!(res.len(), a.len());
    #[cfg(debug_assertions)]
    {
        const BOUND: i64 = (1i64 << 50) - 1;
        for (i, &val) in a.iter().enumerate() {
            let masked = val & mask;
            assert!(
                masked.abs() <= BOUND,
                "reim_from_znx_i64_masked_bnd50_neon: (a[{i}] & mask) = {masked} exceeds 2^50-1"
            );
        }
    }
    let n = res.len();
    let span = n >> 2;
    unsafe {
        let expo: f64 = (1i64 << 52) as f64;
        let add_cst: i64 = 1i64 << 51;
        let sub_cst: f64 = (3i64 << 51) as f64;

        let expo_v = vreinterpretq_u64_f64(vdupq_n_f64(expo));
        let add_cst_v = vdupq_n_s64(add_cst);
        let sub_cst_v = vdupq_n_f64(sub_cst);
        let mask_v = vdupq_n_s64(mask);

        let mut res_ptr = res.as_mut_ptr();
        let mut a_ptr = a.as_ptr();

        for _ in 0..span {
            let lo_raw = vandq_s64(vld1q_s64(a_ptr), mask_v);
            let hi_raw = vandq_s64(vld1q_s64(a_ptr.add(2)), mask_v);
            let lo = vaddq_s64(lo_raw, add_cst_v);
            let hi = vaddq_s64(hi_raw, add_cst_v);
            let mut lo_f = vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_s64(lo), expo_v));
            let mut hi_f = vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_s64(hi), expo_v));
            lo_f = vsubq_f64(lo_f, sub_cst_v);
            hi_f = vsubq_f64(hi_f, sub_cst_v);
            vst1q_f64(res_ptr, lo_f);
            vst1q_f64(res_ptr.add(2), hi_f);
            res_ptr = res_ptr.add(4);
            a_ptr = a_ptr.add(4);
        }
    }
    let tail = span << 2;
    if tail < n {
        reim_from_znx_i64_masked_ref(&mut res[tail..], &a[tail..], mask);
    }
}

/// Shared per-lane body: round and convert one f64 lane vector to i64 via
/// IEEE 754 exponent-diff bit manipulation. Mirrors the inner block of
/// `reim_to_znx_i64_bnd63_avx2_fma` at `conversion.rs:223`. Caller supplies
/// the broadcast constants to avoid recomputation in the hot loop.
/// Bound: caller guarantees `|a / divisor| < 2^62` (output fits in i64).
#[inline(always)]
unsafe fn reim_to_znx_chunk(
    a_f: float64x2_t,
    sign_mask_f: float64x2_t,
    offset_f: float64x2_t,
    expo_mask: int64x2_t,
    mantissa_mask: int64x2_t,
    mantissa_msb: int64x2_t,
    divi_bits: int64x2_t,
) -> int64x2_t {
    unsafe {
        // a_round = a + sign(a) * (divisor / 2)
        let asign = vreinterpretq_s64_u64(vandq_u64(vreinterpretq_u64_f64(a_f), vreinterpretq_u64_f64(sign_mask_f)));
        let bias = vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_s64(asign), vreinterpretq_u64_f64(offset_f)));
        let a_round = vaddq_f64(a_f, bias);

        // sign_full = -1 if a was negative, else 0 (used for two's-complement negate)
        let sign_full = vsubq_s64(
            vdupq_n_s64(0),
            vreinterpretq_s64_u64(vshrq_n_u64::<63>(vreinterpretq_u64_s64(asign))),
        );

        // exp = (a as u64) & expo_mask
        let a_bits = vreinterpretq_s64_f64(a_round);
        let a0exp = vandq_s64(a_bits, expo_mask);

        // shift_signed = (a0exp - divi_bits) >>_arith 52  (positive = left, negative = right amount)
        let exp_diff = vsubq_s64(a0exp, divi_bits);
        let shift_signed = vshrq_n_s64::<52>(exp_diff);

        // mantissa = (a as u64) & mantissa_mask | mantissa_msb
        let a0pos_u = vorrq_u64(
            vandq_u64(vreinterpretq_u64_s64(a_bits), vreinterpretq_u64_s64(mantissa_mask)),
            vreinterpretq_u64_s64(mantissa_msb),
        );

        // out = vshlq_u64(mantissa, shift_signed) — handles both directions in one op.
        let out_u = vshlq_u64(a0pos_u, shift_signed);

        // Apply sign: out = (out ^ sign_full) - sign_full
        let out_s = vreinterpretq_s64_u64(out_u);
        vsubq_s64(veorq_s64(out_s, sign_full), sign_full)
    }
}

/// `f64 → i64` conversion with rounding-divide by `divisor`. Bound: output
/// must fit in i64.
pub(crate) fn reim_to_znx_i64_bnd63_neon(res: &mut [i64], divisor: f64, a: &[f64]) {
    debug_assert_eq!(res.len(), a.len());
    let n = res.len();
    let span = n >> 2;

    let sign_mask: u64 = 0x8000_0000_0000_0000;
    let expo_mask: u64 = 0x7FF0_0000_0000_0000;
    let mantissa_mask: u64 = (i64::MAX as u64) ^ expo_mask;
    let mantissa_msb: u64 = 0x0010_0000_0000_0000;
    let divi_bits_f: f64 = divisor * (1i64 << 52) as f64;
    let offset: f64 = divisor / 2.0;

    unsafe {
        let sign_mask_f = vreinterpretq_f64_u64(vdupq_n_u64(sign_mask));
        let expo_mask_v = vreinterpretq_s64_u64(vdupq_n_u64(expo_mask));
        let mantissa_mask_v = vreinterpretq_s64_u64(vdupq_n_u64(mantissa_mask));
        let mantissa_msb_v = vreinterpretq_s64_u64(vdupq_n_u64(mantissa_msb));
        let offset_f = vdupq_n_f64(offset);
        let divi_bits_v = vreinterpretq_s64_f64(vdupq_n_f64(divi_bits_f));

        let mut res_ptr = res.as_mut_ptr();
        let mut a_ptr = a.as_ptr();

        for _ in 0..span {
            let lo = reim_to_znx_chunk(
                vld1q_f64(a_ptr),
                sign_mask_f,
                offset_f,
                expo_mask_v,
                mantissa_mask_v,
                mantissa_msb_v,
                divi_bits_v,
            );
            let hi = reim_to_znx_chunk(
                vld1q_f64(a_ptr.add(2)),
                sign_mask_f,
                offset_f,
                expo_mask_v,
                mantissa_mask_v,
                mantissa_msb_v,
                divi_bits_v,
            );
            vst1q_s64(res_ptr, lo);
            vst1q_s64(res_ptr.add(2), hi);
            res_ptr = res_ptr.add(4);
            a_ptr = a_ptr.add(4);
        }
    }
    let tail = span << 2;
    if tail < n {
        reim_to_znx_i64_ref(&mut res[tail..], divisor, &a[tail..]);
    }
}

/// In-place variant: read `f64`, write `i64` (reinterpreted) into the same buffer.
pub(crate) fn reim_to_znx_i64_assign_bnd63_neon(res: &mut [f64], divisor: f64) {
    let n = res.len();
    let span = n >> 2;

    let sign_mask: u64 = 0x8000_0000_0000_0000;
    let expo_mask: u64 = 0x7FF0_0000_0000_0000;
    let mantissa_mask: u64 = (i64::MAX as u64) ^ expo_mask;
    let mantissa_msb: u64 = 0x0010_0000_0000_0000;
    let divi_bits_f: f64 = divisor * (1i64 << 52) as f64;
    let offset: f64 = divisor / 2.0;

    unsafe {
        let sign_mask_f = vreinterpretq_f64_u64(vdupq_n_u64(sign_mask));
        let expo_mask_v = vreinterpretq_s64_u64(vdupq_n_u64(expo_mask));
        let mantissa_mask_v = vreinterpretq_s64_u64(vdupq_n_u64(mantissa_mask));
        let mantissa_msb_v = vreinterpretq_s64_u64(vdupq_n_u64(mantissa_msb));
        let offset_f = vdupq_n_f64(offset);
        let divi_bits_v = vreinterpretq_s64_f64(vdupq_n_f64(divi_bits_f));

        let mut ptr_f = res.as_mut_ptr();
        let mut ptr_i = res.as_mut_ptr() as *mut i64;

        for _ in 0..span {
            let lo = reim_to_znx_chunk(
                vld1q_f64(ptr_f),
                sign_mask_f,
                offset_f,
                expo_mask_v,
                mantissa_mask_v,
                mantissa_msb_v,
                divi_bits_v,
            );
            let hi = reim_to_znx_chunk(
                vld1q_f64(ptr_f.add(2)),
                sign_mask_f,
                offset_f,
                expo_mask_v,
                mantissa_mask_v,
                mantissa_msb_v,
                divi_bits_v,
            );
            vst1q_s64(ptr_i, lo);
            vst1q_s64(ptr_i.add(2), hi);
            ptr_f = ptr_f.add(4);
            ptr_i = ptr_i.add(4);
        }
    }
    let tail = span << 2;
    if tail < n {
        reim_to_znx_i64_assign_ref(&mut res[tail..], divisor);
    }
}

// suppress: reused for tail handling on builds without orrq/eorq detection helpers
#[allow(dead_code)]
fn _unused() {
    let _ = vorrq_s64;
}

#[cfg(test)]
mod tests {
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
            reim_add_ref(&mut want, &a, &b);
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
            reim_sub_ref(&mut want, &a, &b);
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
            reim_negate_ref(&mut want, &a);
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
            reim_mul_ref(&mut want, &a, &b);
            close_enough(&got, &want, &format!("reim_mul_neon n={n}"));
        }
    }

    #[test]
    fn reim_from_znx_neon_exact_vs_ref() {
        let mut r = rng();
        for &n in SIZES {
            let a: Vec<i64> = (0..n)
                .map(|_| (r.random::<u64>() & ((1u64 << 50) - 1)) as i64 - (1i64 << 49))
                .collect();
            let mut got = vec![0f64; n];
            let mut want = vec![0f64; n];
            reim_from_znx_i64_bnd50_neon(&mut got, &a);
            reim_from_znx_i64_ref(&mut want, &a);
            assert_eq!(got, want, "reim_from_znx_i64_bnd50_neon n={n}");
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
            reim_addmul_ref(&mut want, &a, &b);
            close_enough(&got, &want, &format!("reim_addmul_neon n={n}"));
        }
    }
}
