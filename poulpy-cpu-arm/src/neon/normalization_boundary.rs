//! NEON floor carry propagation and rounding at a precision boundary.

use core::arch::aarch64::{
    int64x2_t, uint64x2_t, vaddq_s64, vaddq_u64, vandq_u64, vdupq_n_s64, vdupq_n_u64, vld1q_s64, vorrq_u64,
    vreinterpretq_s64_u64, vreinterpretq_u64_s64, vshlq_s64, vshlq_u64, vst1q_s64,
};
use poulpy_cpu_ref::reference::normalization::{
    nfc_normalize_floor_ref, nfc_normalize_round_ref, znx_normalize_floor_ref, znx_normalize_round_assign_ref,
    znx_normalize_round_ref,
};

use super::vec_znx_big::{add2_i128, load2_i128, store2_i128};

struct Boundary {
    mask: uint64x2_t,
    source_mask: uint64x2_t,
    source_right: int64x2_t,
    source_left: int64x2_t,
    base_right: int64x2_t,
    base_left: int64x2_t,
    lsh: int64x2_t,
    guard_right: int64x2_t,
    padding_left: int64x2_t,
    padding_right: int64x2_t,
    half: uint64x2_t,
    center_half: uint64x2_t,
    center_right: int64x2_t,
    digit_left: int64x2_t,
    digit_right: int64x2_t,
}

impl Boundary {
    #[inline(always)]
    fn new(base2k: usize, lsh: usize, padding: usize) -> Self {
        assert!((1..=63).contains(&base2k));
        assert!(lsh < base2k && padding < base2k);
        let source_bits = base2k - lsh;
        let width = base2k - padding;
        unsafe {
            Self {
                mask: vdupq_n_u64((1u64 << base2k) - 1),
                source_mask: vdupq_n_u64((1u64 << source_bits) - 1),
                source_right: vdupq_n_s64(-(source_bits as i64)),
                source_left: vdupq_n_s64((64 - source_bits) as i64),
                base_right: vdupq_n_s64(-(base2k as i64)),
                base_left: vdupq_n_s64((64 - base2k) as i64),
                lsh: vdupq_n_s64(lsh as i64),
                guard_right: vdupq_n_s64(1 - base2k as i64),
                padding_left: vdupq_n_s64(padding as i64),
                padding_right: vdupq_n_s64(-(padding as i64)),
                half: vdupq_n_u64(if padding == 0 { 0 } else { 1u64 << (padding - 1) }),
                center_half: vdupq_n_u64(1u64 << (width - 1)),
                center_right: vdupq_n_s64(-(width as i64)),
                digit_left: vdupq_n_s64((64 - width) as i64),
                digit_right: vdupq_n_s64(-((64 - width) as i64)),
            }
        }
    }

    #[inline(always)]
    unsafe fn floor_i64<const CARRY_IN: bool>(&self, a: int64x2_t, carry: int64x2_t) -> (uint64x2_t, int64x2_t) {
        unsafe {
            let mut low = vshlq_u64(vandq_u64(vreinterpretq_u64_s64(a), self.source_mask), self.lsh);
            let mut high = vshlq_s64(a, self.source_right);
            if CARRY_IN {
                low = vaddq_u64(low, vandq_u64(vreinterpretq_u64_s64(carry), self.mask));
                high = vaddq_s64(high, vshlq_s64(carry, self.base_right));
                high = vaddq_s64(high, vreinterpretq_s64_u64(vshlq_u64(low, self.base_right)));
            }
            (vandq_u64(low, self.mask), high)
        }
    }

    #[inline(always)]
    unsafe fn floor_i128<const CARRY_IN: bool>(
        &self,
        a: (uint64x2_t, int64x2_t),
        carry: (uint64x2_t, int64x2_t),
    ) -> (uint64x2_t, (uint64x2_t, int64x2_t)) {
        unsafe {
            let mut low = vshlq_u64(vandq_u64(a.0, self.source_mask), self.lsh);
            let mut high = shift_i128(a, self.source_right, self.source_left);
            if CARRY_IN {
                low = vaddq_u64(low, vandq_u64(carry.0, self.mask));
                let c_high = shift_i128(carry, self.base_right, self.base_left);
                high = add2_i128(high.0, high.1, c_high.0, c_high.1);
                high = add2_i128(high.0, high.1, vshlq_u64(low, self.base_right), vdupq_n_s64(0));
            }
            (vandq_u64(low, self.mask), high)
        }
    }

    #[inline(always)]
    unsafe fn round<const PAD: bool>(&self, low: uint64x2_t) -> (int64x2_t, uint64x2_t) {
        unsafe {
            let rounded = vshlq_u64(vaddq_u64(low, self.half), self.padding_right);
            let carry = vshlq_u64(vaddq_u64(rounded, self.center_half), self.center_right);
            let digit = vshlq_s64(vshlq_s64(vreinterpretq_s64_u64(rounded), self.digit_left), self.digit_right);
            (if PAD { vshlq_s64(digit, self.padding_left) } else { digit }, carry)
        }
    }
}

#[inline(always)]
unsafe fn shift_i128(a: (uint64x2_t, int64x2_t), right: int64x2_t, left: int64x2_t) -> (uint64x2_t, int64x2_t) {
    unsafe {
        (
            vorrq_u64(vshlq_u64(a.0, right), vshlq_u64(vreinterpretq_u64_s64(a.1), left)),
            vshlq_s64(a.1, right),
        )
    }
}

#[inline]
pub(crate) fn znx_normalize_floor_neon<const CARRY_IN: bool, const ROUND: bool>(
    base2k: usize,
    lsh: usize,
    a: &[i64],
    carry: &mut [i64],
) {
    assert!(a.len() >= carry.len());
    let boundary = Boundary::new(base2k, lsh, 0);
    let tail = carry.len() & !1;
    unsafe {
        let zero = vdupq_n_s64(0);
        for i in (0..tail).step_by(2) {
            let incoming = if CARRY_IN { vld1q_s64(carry.as_ptr().add(i)) } else { zero };
            let (low, mut high) = boundary.floor_i64::<CARRY_IN>(vld1q_s64(a.as_ptr().add(i)), incoming);
            if ROUND {
                high = vaddq_s64(high, vreinterpretq_s64_u64(vshlq_u64(low, boundary.guard_right)));
            }
            vst1q_s64(carry.as_mut_ptr().add(i), high);
        }
    }
    znx_normalize_floor_ref::<CARRY_IN, ROUND>(base2k, lsh, &a[tail..], &mut carry[tail..]);
}

#[inline]
pub(crate) fn znx_normalize_round_neon<const CARRY_IN: bool, const PAD: bool>(
    base2k: usize,
    lsh: usize,
    padding: usize,
    res: &mut [i64],
    a: &[i64],
    carry: &mut [i64],
) {
    assert!(a.len() >= res.len() && carry.len() >= res.len());
    let boundary = Boundary::new(base2k, lsh, padding);
    let tail = res.len() & !1;
    unsafe {
        let zero = vdupq_n_s64(0);
        for i in (0..tail).step_by(2) {
            let incoming = if CARRY_IN { vld1q_s64(carry.as_ptr().add(i)) } else { zero };
            let (low, high) = boundary.floor_i64::<CARRY_IN>(vld1q_s64(a.as_ptr().add(i)), incoming);
            let (digit, next) = boundary.round::<PAD>(low);
            vst1q_s64(res.as_mut_ptr().add(i), digit);
            vst1q_s64(carry.as_mut_ptr().add(i), vaddq_s64(high, vreinterpretq_s64_u64(next)));
        }
    }
    znx_normalize_round_ref::<CARRY_IN, PAD>(base2k, lsh, padding, &mut res[tail..], &a[tail..], &mut carry[tail..]);
}

#[inline]
pub(crate) fn znx_normalize_round_assign_neon<const CARRY_IN: bool>(
    base2k: usize,
    lsh: usize,
    padding: usize,
    res: &mut [i64],
    carry: &mut [i64],
) {
    assert!(carry.len() >= res.len());
    let boundary = Boundary::new(base2k, lsh, padding);
    let tail = res.len() & !1;
    unsafe {
        let zero = vdupq_n_s64(0);
        for i in (0..tail).step_by(2) {
            let incoming = if CARRY_IN { vld1q_s64(carry.as_ptr().add(i)) } else { zero };
            let (low, high) = boundary.floor_i64::<CARRY_IN>(vld1q_s64(res.as_ptr().add(i)), incoming);
            let (digit, next) = boundary.round::<true>(low);
            vst1q_s64(res.as_mut_ptr().add(i), digit);
            vst1q_s64(carry.as_mut_ptr().add(i), vaddq_s64(high, vreinterpretq_s64_u64(next)));
        }
    }
    znx_normalize_round_assign_ref::<CARRY_IN>(base2k, lsh, padding, &mut res[tail..], &mut carry[tail..]);
}

#[inline]
pub(crate) fn nfc_normalize_floor_neon<const CARRY_IN: bool, const ROUND: bool>(
    base2k: usize,
    lsh: usize,
    a: &[i128],
    carry: &mut [i128],
) {
    assert!(a.len() >= carry.len());
    let boundary = Boundary::new(base2k, lsh, 0);
    let tail = carry.len() & !1;
    unsafe {
        let zero = (vdupq_n_u64(0), vdupq_n_s64(0));
        for i in (0..tail).step_by(2) {
            let incoming = if CARRY_IN { load2_i128(carry.as_ptr().add(i)) } else { zero };
            let (low, mut high) = boundary.floor_i128::<CARRY_IN>(load2_i128(a.as_ptr().add(i)), incoming);
            if ROUND {
                high = add2_i128(high.0, high.1, vshlq_u64(low, boundary.guard_right), zero.1);
            }
            store2_i128(carry.as_mut_ptr().add(i), high.0, high.1);
        }
    }
    nfc_normalize_floor_ref::<CARRY_IN, ROUND>(base2k, lsh, &a[tail..], &mut carry[tail..]);
}

#[inline]
pub(crate) fn nfc_normalize_round_neon<const CARRY_IN: bool, const PAD: bool>(
    base2k: usize,
    lsh: usize,
    padding: usize,
    res: &mut [i64],
    a: &[i128],
    carry: &mut [i128],
) {
    assert!(a.len() >= res.len() && carry.len() >= res.len());
    let boundary = Boundary::new(base2k, lsh, padding);
    let tail = res.len() & !1;
    unsafe {
        let zero = (vdupq_n_u64(0), vdupq_n_s64(0));
        for i in (0..tail).step_by(2) {
            let incoming = if CARRY_IN { load2_i128(carry.as_ptr().add(i)) } else { zero };
            let (low, high) = boundary.floor_i128::<CARRY_IN>(load2_i128(a.as_ptr().add(i)), incoming);
            let (digit, next) = boundary.round::<PAD>(low);
            let high = add2_i128(high.0, high.1, next, zero.1);
            vst1q_s64(res.as_mut_ptr().add(i), digit);
            store2_i128(carry.as_mut_ptr().add(i), high.0, high.1);
        }
    }
    nfc_normalize_round_ref::<CARRY_IN, PAD>(base2k, lsh, padding, &mut res[tail..], &a[tail..], &mut carry[tail..]);
}
