//! NEON kernels for the base-2^k normalize / digit-extract family.

use core::arch::aarch64::{int64x2_t, vaddq_s64, vandq_s64, vdupq_n_s64, veorq_s64, vld1q_s64, vshlq_s64, vst1q_s64, vsubq_s64};

use poulpy_cpu_ref::reference::znx::{
    znx_extract_digit_addmul_ref, znx_normalize_digit_ref, znx_normalize_final_step_assign_ref, znx_normalize_final_step_ref,
    znx_normalize_final_step_sub_ref, znx_normalize_first_step_assign_ref, znx_normalize_first_step_carry_only_ref,
    znx_normalize_first_step_ref, znx_normalize_middle_step_assign_ref, znx_normalize_middle_step_carry_only_ref,
    znx_normalize_middle_step_ref, znx_normalize_middle_step_sub_ref,
};

/// `(mask_k, sign_k, cnt_neg)` with `cnt_neg = -base2k` for `vshlq_s64` arithmetic right shift.
#[inline(always)]
unsafe fn normalize_consts_neon(base2k: usize) -> (int64x2_t, int64x2_t, int64x2_t) {
    debug_assert!((1..=63).contains(&base2k));
    let mask_k: i64 = ((1u64 << base2k) - 1) as i64;
    let sign_k: i64 = (1u64 << (base2k - 1)) as i64;
    unsafe { (vdupq_n_s64(mask_k), vdupq_n_s64(sign_k), vdupq_n_s64(-(base2k as i64))) }
}

/// `digit = ((x & mask_k) ^ sign_k) - sign_k` — sign-extends the low `base2k` bits.
#[inline(always)]
unsafe fn get_digit_neon(x: int64x2_t, mask_k: int64x2_t, sign_k: int64x2_t) -> int64x2_t {
    unsafe {
        let low = vandq_s64(x, mask_k);
        vsubq_s64(veorq_s64(low, sign_k), sign_k)
    }
}

/// `carry = (x - digit) >>_arith base2k`.
#[inline(always)]
unsafe fn get_carry_neon(x: int64x2_t, digit: int64x2_t, cnt_neg: int64x2_t) -> int64x2_t {
    unsafe { vshlq_s64(vsubq_s64(x, digit), cnt_neg) }
}

/// `res += digit(src) << lsh` ; `src = carry`.
pub(crate) fn znx_extract_digit_addmul_neon(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
    debug_assert_eq!(res.len(), src.len());
    let n = res.len();
    let span = n >> 2;
    unsafe {
        let mut rr = res.as_mut_ptr();
        let mut ss = src.as_mut_ptr();
        let (mask, sign, cnt_neg) = normalize_consts_neon(base2k);
        let lsh_v: int64x2_t = vdupq_n_s64(lsh as i64);
        for _ in 0..span {
            let s0 = vld1q_s64(ss);
            let s1 = vld1q_s64(ss.add(2));
            let d0 = get_digit_neon(s0, mask, sign);
            let d1 = get_digit_neon(s1, mask, sign);
            let c0 = get_carry_neon(s0, d0, cnt_neg);
            let c1 = get_carry_neon(s1, d1, cnt_neg);
            let r0 = vaddq_s64(vld1q_s64(rr), vshlq_s64(d0, lsh_v));
            let r1 = vaddq_s64(vld1q_s64(rr.add(2)), vshlq_s64(d1, lsh_v));
            vst1q_s64(rr, r0);
            vst1q_s64(rr.add(2), r1);
            vst1q_s64(ss, c0);
            vst1q_s64(ss.add(2), c1);
            rr = rr.add(4);
            ss = ss.add(4);
        }
    }
    let tail = span << 2;
    if tail < n {
        znx_extract_digit_addmul_ref(base2k, lsh, &mut res[tail..], &mut src[tail..]);
    }
}

/// `res = digit(res)` ; `src += carry(res)`.
pub(crate) fn znx_normalize_digit_neon(base2k: usize, res: &mut [i64], src: &mut [i64]) {
    debug_assert_eq!(res.len(), src.len());
    let n = res.len();
    let span = n >> 2;
    unsafe {
        let mut rr = res.as_mut_ptr();
        let mut ss = src.as_mut_ptr();
        let (mask, sign, cnt_neg) = normalize_consts_neon(base2k);
        for _ in 0..span {
            let r0 = vld1q_s64(rr);
            let r1 = vld1q_s64(rr.add(2));
            let d0 = get_digit_neon(r0, mask, sign);
            let d1 = get_digit_neon(r1, mask, sign);
            let c0 = get_carry_neon(r0, d0, cnt_neg);
            let c1 = get_carry_neon(r1, d1, cnt_neg);
            let s0 = vaddq_s64(vld1q_s64(ss), c0);
            let s1 = vaddq_s64(vld1q_s64(ss.add(2)), c1);
            vst1q_s64(rr, d0);
            vst1q_s64(rr.add(2), d1);
            vst1q_s64(ss, s0);
            vst1q_s64(ss.add(2), s1);
            rr = rr.add(4);
            ss = ss.add(4);
        }
    }
    let tail = span << 2;
    if tail < n {
        znx_normalize_digit_ref(base2k, &mut res[tail..], &mut src[tail..]);
    }
}

/// First step (carry-only): `carry = carry(x, base2k - lsh if lsh else base2k)`.
pub(crate) fn znx_normalize_first_step_carry_only_neon(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
    debug_assert!(x.len() <= carry.len());
    debug_assert!(lsh < base2k);
    let n = x.len();
    let span = n >> 2;
    unsafe {
        let mut xx = x.as_ptr();
        let mut cc = carry.as_mut_ptr();
        let (mask, sign, cnt_neg) = normalize_consts_neon(if lsh == 0 { base2k } else { base2k - lsh });
        for _ in 0..span {
            let x0 = vld1q_s64(xx);
            let x1 = vld1q_s64(xx.add(2));
            let d0 = get_digit_neon(x0, mask, sign);
            let d1 = get_digit_neon(x1, mask, sign);
            vst1q_s64(cc, get_carry_neon(x0, d0, cnt_neg));
            vst1q_s64(cc.add(2), get_carry_neon(x1, d1, cnt_neg));
            xx = xx.add(4);
            cc = cc.add(4);
        }
    }
    let tail = span << 2;
    if tail < n {
        znx_normalize_first_step_carry_only_ref(base2k, lsh, &x[tail..], &mut carry[tail..]);
    }
}

/// First step (in-place): `x = digit(x) << lsh` ; `carry = carry(x)`.
pub(crate) fn znx_normalize_first_step_assign_neon(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
    debug_assert!(x.len() <= carry.len());
    debug_assert!(lsh < base2k);
    let n = x.len();
    let span = n >> 2;
    unsafe {
        let mut xx = x.as_mut_ptr();
        let mut cc = carry.as_mut_ptr();
        if lsh == 0 {
            let (mask, sign, cnt_neg) = normalize_consts_neon(base2k);
            for _ in 0..span {
                let x0 = vld1q_s64(xx);
                let x1 = vld1q_s64(xx.add(2));
                let d0 = get_digit_neon(x0, mask, sign);
                let d1 = get_digit_neon(x1, mask, sign);
                vst1q_s64(xx, d0);
                vst1q_s64(xx.add(2), d1);
                vst1q_s64(cc, get_carry_neon(x0, d0, cnt_neg));
                vst1q_s64(cc.add(2), get_carry_neon(x1, d1, cnt_neg));
                xx = xx.add(4);
                cc = cc.add(4);
            }
        } else {
            let (mask, sign, cnt_neg) = normalize_consts_neon(base2k - lsh);
            let lsh_v: int64x2_t = vdupq_n_s64(lsh as i64);
            for _ in 0..span {
                let x0 = vld1q_s64(xx);
                let x1 = vld1q_s64(xx.add(2));
                let d0 = get_digit_neon(x0, mask, sign);
                let d1 = get_digit_neon(x1, mask, sign);
                vst1q_s64(xx, vshlq_s64(d0, lsh_v));
                vst1q_s64(xx.add(2), vshlq_s64(d1, lsh_v));
                vst1q_s64(cc, get_carry_neon(x0, d0, cnt_neg));
                vst1q_s64(cc.add(2), get_carry_neon(x1, d1, cnt_neg));
                xx = xx.add(4);
                cc = cc.add(4);
            }
        }
    }
    let tail = span << 2;
    if tail < n {
        znx_normalize_first_step_assign_ref(base2k, lsh, &mut x[tail..], &mut carry[tail..]);
    }
}

/// First step (generic): `x = (OVERWRITE ? digit(a) << lsh : x + digit(a) << lsh)` ; `carry = carry(a)`.
pub(crate) fn znx_normalize_first_step_neon<const OVERWRITE: bool>(
    base2k: usize,
    lsh: usize,
    x: &mut [i64],
    a: &[i64],
    carry: &mut [i64],
) {
    debug_assert_eq!(x.len(), a.len());
    debug_assert!(x.len() <= carry.len());
    debug_assert!(lsh < base2k);
    let n = x.len();
    let span = n >> 2;
    unsafe {
        let mut xx = x.as_mut_ptr();
        let mut aa = a.as_ptr();
        let mut cc = carry.as_mut_ptr();
        if lsh == 0 {
            let (mask, sign, cnt_neg) = normalize_consts_neon(base2k);
            for _ in 0..span {
                let a0 = vld1q_s64(aa);
                let a1 = vld1q_s64(aa.add(2));
                let d0 = get_digit_neon(a0, mask, sign);
                let d1 = get_digit_neon(a1, mask, sign);
                let c0 = get_carry_neon(a0, d0, cnt_neg);
                let c1 = get_carry_neon(a1, d1, cnt_neg);
                if OVERWRITE {
                    vst1q_s64(xx, d0);
                    vst1q_s64(xx.add(2), d1);
                } else {
                    vst1q_s64(xx, vaddq_s64(vld1q_s64(xx), d0));
                    vst1q_s64(xx.add(2), vaddq_s64(vld1q_s64(xx.add(2)), d1));
                }
                vst1q_s64(cc, c0);
                vst1q_s64(cc.add(2), c1);
                xx = xx.add(4);
                aa = aa.add(4);
                cc = cc.add(4);
            }
        } else {
            let (mask, sign, cnt_neg) = normalize_consts_neon(base2k - lsh);
            let lsh_v: int64x2_t = vdupq_n_s64(lsh as i64);
            for _ in 0..span {
                let a0 = vld1q_s64(aa);
                let a1 = vld1q_s64(aa.add(2));
                let d0 = get_digit_neon(a0, mask, sign);
                let d1 = get_digit_neon(a1, mask, sign);
                let c0 = get_carry_neon(a0, d0, cnt_neg);
                let c1 = get_carry_neon(a1, d1, cnt_neg);
                let s0 = vshlq_s64(d0, lsh_v);
                let s1 = vshlq_s64(d1, lsh_v);
                if OVERWRITE {
                    vst1q_s64(xx, s0);
                    vst1q_s64(xx.add(2), s1);
                } else {
                    vst1q_s64(xx, vaddq_s64(vld1q_s64(xx), s0));
                    vst1q_s64(xx.add(2), vaddq_s64(vld1q_s64(xx.add(2)), s1));
                }
                vst1q_s64(cc, c0);
                vst1q_s64(cc.add(2), c1);
                xx = xx.add(4);
                aa = aa.add(4);
                cc = cc.add(4);
            }
        }
    }
    let tail = span << 2;
    if tail < n {
        znx_normalize_first_step_ref::<OVERWRITE>(base2k, lsh, &mut x[tail..], &a[tail..], &mut carry[tail..]);
    }
}

/// Two-pass middle step body: returns `(new_x, new_carry_out)` given `(x, carry_in)`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn middle_chunk(
    x: int64x2_t,
    cv: int64x2_t,
    mask: int64x2_t,
    sign: int64x2_t,
    cnt_neg: int64x2_t,
    mask_lsh: int64x2_t,
    sign_lsh: int64x2_t,
    cnt_neg_lsh: int64x2_t,
    lsh_v: int64x2_t,
    has_lsh: bool,
) -> (int64x2_t, int64x2_t) {
    unsafe {
        let (d0, c0) = if has_lsh {
            let d = get_digit_neon(x, mask_lsh, sign_lsh);
            let c = get_carry_neon(x, d, cnt_neg_lsh);
            (vshlq_s64(d, lsh_v), c)
        } else {
            let d = get_digit_neon(x, mask, sign);
            let c = get_carry_neon(x, d, cnt_neg);
            (d, c)
        };
        let s = vaddq_s64(d0, cv);
        let x1 = get_digit_neon(s, mask, sign);
        let c1 = get_carry_neon(s, x1, cnt_neg);
        (x1, vaddq_s64(c0, c1))
    }
}

/// Middle step (in-place): two-pass digit/carry chain on `x` with `carry` accumulator.
pub(crate) fn znx_normalize_middle_step_assign_neon(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
    debug_assert!(x.len() <= carry.len());
    debug_assert!(lsh < base2k);
    let n = x.len();
    let span = n >> 2;
    unsafe {
        let mut xx = x.as_mut_ptr();
        let mut cc = carry.as_mut_ptr();
        let (mask, sign, cnt_neg) = normalize_consts_neon(base2k);
        let (mask_lsh, sign_lsh, cnt_neg_lsh) = if lsh == 0 {
            (mask, sign, cnt_neg)
        } else {
            normalize_consts_neon(base2k - lsh)
        };
        let lsh_v = vdupq_n_s64(lsh as i64);
        let has_lsh = lsh != 0;
        for _ in 0..span {
            let x0 = vld1q_s64(xx);
            let x1 = vld1q_s64(xx.add(2));
            let cv0 = vld1q_s64(cc);
            let cv1 = vld1q_s64(cc.add(2));
            let (n0, nc0) = middle_chunk(x0, cv0, mask, sign, cnt_neg, mask_lsh, sign_lsh, cnt_neg_lsh, lsh_v, has_lsh);
            let (n1, nc1) = middle_chunk(x1, cv1, mask, sign, cnt_neg, mask_lsh, sign_lsh, cnt_neg_lsh, lsh_v, has_lsh);
            vst1q_s64(xx, n0);
            vst1q_s64(xx.add(2), n1);
            vst1q_s64(cc, nc0);
            vst1q_s64(cc.add(2), nc1);
            xx = xx.add(4);
            cc = cc.add(4);
        }
    }
    let tail = span << 2;
    if tail < n {
        znx_normalize_middle_step_assign_ref(base2k, lsh, &mut x[tail..], &mut carry[tail..]);
    }
}

/// Middle step (carry-only): two-pass digit/carry chain on `x`, writing only the carry out.
pub(crate) fn znx_normalize_middle_step_carry_only_neon(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
    debug_assert!(x.len() <= carry.len());
    debug_assert!(lsh < base2k);
    let n = x.len();
    let span = n >> 2;
    unsafe {
        let mut xx = x.as_ptr();
        let mut cc = carry.as_mut_ptr();
        let (mask, sign, cnt_neg) = normalize_consts_neon(base2k);
        let (mask_lsh, sign_lsh, cnt_neg_lsh) = if lsh == 0 {
            (mask, sign, cnt_neg)
        } else {
            normalize_consts_neon(base2k - lsh)
        };
        let lsh_v = vdupq_n_s64(lsh as i64);
        let has_lsh = lsh != 0;
        for _ in 0..span {
            let x0 = vld1q_s64(xx);
            let x1 = vld1q_s64(xx.add(2));
            let cv0 = vld1q_s64(cc);
            let cv1 = vld1q_s64(cc.add(2));
            let (_, nc0) = middle_chunk(x0, cv0, mask, sign, cnt_neg, mask_lsh, sign_lsh, cnt_neg_lsh, lsh_v, has_lsh);
            let (_, nc1) = middle_chunk(x1, cv1, mask, sign, cnt_neg, mask_lsh, sign_lsh, cnt_neg_lsh, lsh_v, has_lsh);
            vst1q_s64(cc, nc0);
            vst1q_s64(cc.add(2), nc1);
            xx = xx.add(4);
            cc = cc.add(4);
        }
    }
    let tail = span << 2;
    if tail < n {
        znx_normalize_middle_step_carry_only_ref(base2k, lsh, &x[tail..], &mut carry[tail..]);
    }
}

/// Middle step (generic): two-pass digit/carry chain reading `a`, optionally adding into `x`.
pub(crate) fn znx_normalize_middle_step_neon<const OVERWRITE: bool>(
    base2k: usize,
    lsh: usize,
    x: &mut [i64],
    a: &[i64],
    carry: &mut [i64],
) {
    debug_assert_eq!(x.len(), a.len());
    debug_assert!(x.len() <= carry.len());
    debug_assert!(lsh < base2k);
    let n = x.len();
    let span = n >> 2;
    unsafe {
        let mut xx = x.as_mut_ptr();
        let mut aa = a.as_ptr();
        let mut cc = carry.as_mut_ptr();
        let (mask, sign, cnt_neg) = normalize_consts_neon(base2k);
        let (mask_lsh, sign_lsh, cnt_neg_lsh) = if lsh == 0 {
            (mask, sign, cnt_neg)
        } else {
            normalize_consts_neon(base2k - lsh)
        };
        let lsh_v = vdupq_n_s64(lsh as i64);
        let has_lsh = lsh != 0;
        for _ in 0..span {
            let a0 = vld1q_s64(aa);
            let a1 = vld1q_s64(aa.add(2));
            let cv0 = vld1q_s64(cc);
            let cv1 = vld1q_s64(cc.add(2));
            let (n0, nc0) = middle_chunk(a0, cv0, mask, sign, cnt_neg, mask_lsh, sign_lsh, cnt_neg_lsh, lsh_v, has_lsh);
            let (n1, nc1) = middle_chunk(a1, cv1, mask, sign, cnt_neg, mask_lsh, sign_lsh, cnt_neg_lsh, lsh_v, has_lsh);
            if OVERWRITE {
                vst1q_s64(xx, n0);
                vst1q_s64(xx.add(2), n1);
            } else {
                vst1q_s64(xx, vaddq_s64(vld1q_s64(xx), n0));
                vst1q_s64(xx.add(2), vaddq_s64(vld1q_s64(xx.add(2)), n1));
            }
            vst1q_s64(cc, nc0);
            vst1q_s64(cc.add(2), nc1);
            xx = xx.add(4);
            aa = aa.add(4);
            cc = cc.add(4);
        }
    }
    let tail = span << 2;
    if tail < n {
        znx_normalize_middle_step_ref::<OVERWRITE>(base2k, lsh, &mut x[tail..], &a[tail..], &mut carry[tail..]);
    }
}

/// Middle step (subtract): `x -= digit_chain(a)` ; carry accumulates.
pub(crate) fn znx_normalize_middle_step_sub_neon(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
    debug_assert_eq!(x.len(), a.len());
    debug_assert!(x.len() <= carry.len());
    debug_assert!(lsh < base2k);
    let n = x.len();
    let span = n >> 2;
    unsafe {
        let mut xx = x.as_mut_ptr();
        let mut aa = a.as_ptr();
        let mut cc = carry.as_mut_ptr();
        let (mask, sign, cnt_neg) = normalize_consts_neon(base2k);
        let (mask_lsh, sign_lsh, cnt_neg_lsh) = if lsh == 0 {
            (mask, sign, cnt_neg)
        } else {
            normalize_consts_neon(base2k - lsh)
        };
        let lsh_v = vdupq_n_s64(lsh as i64);
        let has_lsh = lsh != 0;
        for _ in 0..span {
            let a0 = vld1q_s64(aa);
            let a1 = vld1q_s64(aa.add(2));
            let cv0 = vld1q_s64(cc);
            let cv1 = vld1q_s64(cc.add(2));
            let (n0, nc0) = middle_chunk(a0, cv0, mask, sign, cnt_neg, mask_lsh, sign_lsh, cnt_neg_lsh, lsh_v, has_lsh);
            let (n1, nc1) = middle_chunk(a1, cv1, mask, sign, cnt_neg, mask_lsh, sign_lsh, cnt_neg_lsh, lsh_v, has_lsh);
            vst1q_s64(xx, vsubq_s64(vld1q_s64(xx), n0));
            vst1q_s64(xx.add(2), vsubq_s64(vld1q_s64(xx.add(2)), n1));
            vst1q_s64(cc, nc0);
            vst1q_s64(cc.add(2), nc1);
            xx = xx.add(4);
            aa = aa.add(4);
            cc = cc.add(4);
        }
    }
    let tail = span << 2;
    if tail < n {
        znx_normalize_middle_step_sub_ref(base2k, lsh, &mut x[tail..], &a[tail..], &mut carry[tail..]);
    }
}

/// Final-step body (no carry-out): `digit(digit(x [<< lsh]) + carry)`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn final_chunk(
    x: int64x2_t,
    cv: int64x2_t,
    mask: int64x2_t,
    sign: int64x2_t,
    mask_lsh: int64x2_t,
    sign_lsh: int64x2_t,
    lsh_v: int64x2_t,
    has_lsh: bool,
) -> int64x2_t {
    unsafe {
        let d0 = if has_lsh {
            vshlq_s64(get_digit_neon(x, mask_lsh, sign_lsh), lsh_v)
        } else {
            get_digit_neon(x, mask, sign)
        };
        let s = vaddq_s64(d0, cv);
        get_digit_neon(s, mask, sign)
    }
}

/// Final step (in-place): flush `carry` into `x`, no carry-out.
pub(crate) fn znx_normalize_final_step_assign_neon(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
    debug_assert!(x.len() <= carry.len());
    debug_assert!(lsh < base2k);
    let n = x.len();
    let span = n >> 2;
    unsafe {
        let mut xx = x.as_mut_ptr();
        let mut cc = carry.as_mut_ptr();
        let (mask, sign, _) = normalize_consts_neon(base2k);
        let (mask_lsh, sign_lsh) = if lsh == 0 {
            (mask, sign)
        } else {
            let (m, s, _) = normalize_consts_neon(base2k - lsh);
            (m, s)
        };
        let lsh_v = vdupq_n_s64(lsh as i64);
        let has_lsh = lsh != 0;
        for _ in 0..span {
            let x0 = vld1q_s64(xx);
            let x1 = vld1q_s64(xx.add(2));
            let cv0 = vld1q_s64(cc);
            let cv1 = vld1q_s64(cc.add(2));
            vst1q_s64(xx, final_chunk(x0, cv0, mask, sign, mask_lsh, sign_lsh, lsh_v, has_lsh));
            vst1q_s64(
                xx.add(2),
                final_chunk(x1, cv1, mask, sign, mask_lsh, sign_lsh, lsh_v, has_lsh),
            );
            xx = xx.add(4);
            cc = cc.add(4);
        }
    }
    let tail = span << 2;
    if tail < n {
        znx_normalize_final_step_assign_ref(base2k, lsh, &mut x[tail..], &mut carry[tail..]);
    }
}

/// Final step (generic): same as assign but with optional add into `x` (when `!OVERWRITE`).
pub(crate) fn znx_normalize_final_step_neon<const OVERWRITE: bool>(
    base2k: usize,
    lsh: usize,
    x: &mut [i64],
    a: &[i64],
    carry: &mut [i64],
) {
    debug_assert_eq!(x.len(), a.len());
    debug_assert!(x.len() <= carry.len());
    debug_assert!(lsh < base2k);
    let n = x.len();
    let span = n >> 2;
    unsafe {
        let mut xx = x.as_mut_ptr();
        let mut aa = a.as_ptr();
        let mut cc = carry.as_mut_ptr();
        let (mask, sign, _) = normalize_consts_neon(base2k);
        let (mask_lsh, sign_lsh) = if lsh == 0 {
            (mask, sign)
        } else {
            let (m, s, _) = normalize_consts_neon(base2k - lsh);
            (m, s)
        };
        let lsh_v = vdupq_n_s64(lsh as i64);
        let has_lsh = lsh != 0;
        for _ in 0..span {
            let a0 = vld1q_s64(aa);
            let a1 = vld1q_s64(aa.add(2));
            let cv0 = vld1q_s64(cc);
            let cv1 = vld1q_s64(cc.add(2));
            let r0 = final_chunk(a0, cv0, mask, sign, mask_lsh, sign_lsh, lsh_v, has_lsh);
            let r1 = final_chunk(a1, cv1, mask, sign, mask_lsh, sign_lsh, lsh_v, has_lsh);
            if OVERWRITE {
                vst1q_s64(xx, r0);
                vst1q_s64(xx.add(2), r1);
            } else {
                vst1q_s64(xx, vaddq_s64(vld1q_s64(xx), r0));
                vst1q_s64(xx.add(2), vaddq_s64(vld1q_s64(xx.add(2)), r1));
            }
            xx = xx.add(4);
            aa = aa.add(4);
            cc = cc.add(4);
        }
    }
    let tail = span << 2;
    if tail < n {
        znx_normalize_final_step_ref::<OVERWRITE>(base2k, lsh, &mut x[tail..], &a[tail..], &mut carry[tail..]);
    }
}

/// Final step (subtract): `x -= final_chunk(a, carry)`.
pub(crate) fn znx_normalize_final_step_sub_neon(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]) {
    debug_assert_eq!(x.len(), a.len());
    debug_assert!(x.len() <= carry.len());
    debug_assert!(lsh < base2k);
    let n = x.len();
    let span = n >> 2;
    unsafe {
        let mut xx = x.as_mut_ptr();
        let mut aa = a.as_ptr();
        let mut cc = carry.as_mut_ptr();
        let (mask, sign, _) = normalize_consts_neon(base2k);
        let (mask_lsh, sign_lsh) = if lsh == 0 {
            (mask, sign)
        } else {
            let (m, s, _) = normalize_consts_neon(base2k - lsh);
            (m, s)
        };
        let lsh_v = vdupq_n_s64(lsh as i64);
        let has_lsh = lsh != 0;
        for _ in 0..span {
            let a0 = vld1q_s64(aa);
            let a1 = vld1q_s64(aa.add(2));
            let cv0 = vld1q_s64(cc);
            let cv1 = vld1q_s64(cc.add(2));
            let r0 = final_chunk(a0, cv0, mask, sign, mask_lsh, sign_lsh, lsh_v, has_lsh);
            let r1 = final_chunk(a1, cv1, mask, sign, mask_lsh, sign_lsh, lsh_v, has_lsh);
            vst1q_s64(xx, vsubq_s64(vld1q_s64(xx), r0));
            vst1q_s64(xx.add(2), vsubq_s64(vld1q_s64(xx.add(2)), r1));
            xx = xx.add(4);
            aa = aa.add(4);
            cc = cc.add(4);
        }
    }
    let tail = span << 2;
    if tail < n {
        znx_normalize_final_step_sub_ref(base2k, lsh, &mut x[tail..], &a[tail..], &mut carry[tail..]);
    }
}
