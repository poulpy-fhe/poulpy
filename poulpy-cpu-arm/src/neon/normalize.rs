//! NEON kernels for `vec_znx_big_normalize`'s i128 carry-propagation.
//!
//! Mirrors the AVX2 kernels in
//! `poulpy-cpu-avx/src/ntt120/vec_znx_big_avx.rs` (`nfc_middle_chunk`,
//! `nfc_final_chunk`, `nfc_middle_step_avx2`, …). Each NEON chunk processes
//! two i128 coefficients via a deinterleaved `(lo, hi)` split: a pair of
//! `int64x2_t` registers per i128. The tail (`n % 2 != 0` or `base2k > 64`)
//! falls back to the scalar reference defaults from
//! `poulpy_cpu_ref::reference::ntt120::I128NormalizeOps`.
//!
//! Algorithm correctness mirrors the AVX kernel exactly; review side-by-side.

use core::arch::aarch64::{
    int64x2_t, vaddq_s64, vaddq_u64, vcgtq_u64, vdupq_n_s64, vld1q_s64, vorrq_u64, vreinterpretq_s64_u64, vreinterpretq_u64_s64,
    vshlq_s64, vshlq_u64, vst1q_s64, vsubq_s64, vsubq_u64, vuzp1q_s64, vuzp2q_s64, vzip1q_s64, vzip2q_s64,
};
use poulpy_cpu_ref::NTT120Ref;
use poulpy_cpu_ref::reference::ntt120::{I128NormalizeOps, vec_znx_big::AssignOp};

/// Precomputed shift-count broadcast vectors used by every chunk.
///
/// Variable shifts on AArch64 use `vshlq_{s,u}64(value, count)` where each
/// lane in `count` is the per-lane shift amount: positive = left, negative
/// = right (arithmetic for `s64`, logical for `u64`).
struct NfcShifts {
    /// `+ (64 - base2k_lsh)` — left count for digit extraction *and* for
    /// the upper-half OR of `co_lo` (both use the same shift amount).
    sll_b2klsh: int64x2_t,
    /// `− (64 - base2k_lsh)` — arithmetic-right count for digit extraction.
    sra_b2klsh: int64x2_t,
    /// `− base2k_lsh` — logical-right count for `co_lo` low half.
    srl_b2klsh: int64x2_t,
    /// `− base2k_lsh` — arithmetic-right count for `co_hi`.
    sra_b2klsh_co_hi: int64x2_t,
    /// `+ lsh` — left count for `digit << lsh`.
    sll_lsh: int64x2_t,
    /// `+ (64 - base2k)` — left count for out extraction *and* for the
    /// upper-half OR of `carry2_lo`.
    sll_b2k: int64x2_t,
    /// `− (64 - base2k)` — arithmetic-right count for out extraction.
    sra_b2k: int64x2_t,
    /// `− base2k` — logical-right count for `carry2_lo` low half.
    srl_b2k: int64x2_t,
    /// `− base2k` — arithmetic-right count for `carry2_hi`.
    sra_b2k_carry: int64x2_t,
}

impl NfcShifts {
    #[inline(always)]
    fn new(base2k: u32, lsh: u32) -> Self {
        let b2klsh = base2k - lsh;
        unsafe {
            Self {
                sll_b2klsh: vdupq_n_s64((64 - b2klsh) as i64),
                sra_b2klsh: vdupq_n_s64(-((64 - b2klsh) as i64)),
                srl_b2klsh: vdupq_n_s64(-(b2klsh as i64)),
                sra_b2klsh_co_hi: vdupq_n_s64(-(b2klsh as i64)),
                sll_lsh: vdupq_n_s64(lsh as i64),
                sll_b2k: vdupq_n_s64((64 - base2k) as i64),
                sra_b2k: vdupq_n_s64(-((64 - base2k) as i64)),
                srl_b2k: vdupq_n_s64(-(base2k as i64)),
                sra_b2k_carry: vdupq_n_s64(-(base2k as i64)),
            }
        }
    }
}

// ─── deinterleaved load/store helpers (2 × i128 per call) ────────────────────

#[inline(always)]
unsafe fn load2_split_i128(p: *const i128) -> (int64x2_t, int64x2_t) {
    unsafe {
        let v0 = vld1q_s64(p as *const i64); // [lo0, hi0]
        let v1 = vld1q_s64((p as *const i64).add(2)); // [lo1, hi1]
        let lo = vuzp1q_s64(v0, v1); // [lo0, lo1]
        let hi = vuzp2q_s64(v0, v1); // [hi0, hi1]
        (lo, hi)
    }
}

#[inline(always)]
unsafe fn store2_split_i128(p: *mut i128, lo: int64x2_t, hi: int64x2_t) {
    unsafe {
        vst1q_s64(p as *mut i64, vzip1q_s64(lo, hi)); // [lo0, hi0]
        vst1q_s64((p as *mut i64).add(2), vzip2q_s64(lo, hi)); // [lo1, hi1]
    }
}

/// Load 2 i64 values from `r_ptr` and return them as `(lo_a, hi_a)` split-i128.
/// `lo_a` is just the i64 values (lane interpretation as `i128.lo`); `hi_a`
/// is the sign extension (each lane = `lo_a[i] >> 63`).
#[inline(always)]
unsafe fn load2_i64_as_split_i128(r_ptr: *const i64) -> (int64x2_t, int64x2_t) {
    unsafe {
        let lo = vld1q_s64(r_ptr); // [r0, r1]
        // Arithmetic right shift by 63 broadcasts the sign bit.
        let hi = vshlq_s64(lo, vdupq_n_s64(-63));
        (lo, hi)
    }
}

/// Store the i64 lanes from `lo` into `r_ptr[0..2]`.
#[inline(always)]
unsafe fn store2_i64(r_ptr: *mut i64, lo: int64x2_t) {
    unsafe { vst1q_s64(r_ptr, lo) }
}

// ─── shared per-chunk helpers ────────────────────────────────────────────────

/// Shared body of `nfc_middle_step` for one 2-lane chunk.
///
/// Mirrors `nfc_middle_chunk` in the AVX file: input is a deinterleaved
/// `(lo_a, hi_a)` and previous carry `(lo_c, hi_c)`; output is `(lo_out,
/// new_lo_c, new_hi_c)`. The math is identical to AVX line-for-line — see
/// `poulpy-cpu-avx/src/ntt120/vec_znx_big_avx.rs:243`.
#[inline(always)]
unsafe fn nfc_middle_chunk(
    s: &NfcShifts,
    lo_a: int64x2_t,
    hi_a: int64x2_t,
    lo_c: int64x2_t,
    hi_c: int64x2_t,
) -> (int64x2_t, int64x2_t, int64x2_t) {
    unsafe {
        // digit = sign_extend_low_b2klsh_bits(lo_a)
        let lo_dig = vshlq_s64(vshlq_s64(lo_a, s.sll_b2klsh), s.sra_b2klsh);
        // hi_dig = lo_dig >> 63 (sign-extend digit i64 → split i128)
        let hi_dig = vshlq_s64(lo_dig, vdupq_n_s64(-63));

        // co (carry-out from digit extraction) = (a − digit) >> base2k_lsh
        let diff_lo_u = vsubq_u64(vreinterpretq_u64_s64(lo_a), vreinterpretq_u64_s64(lo_dig));
        let borrow_mask = vcgtq_u64(vreinterpretq_u64_s64(lo_dig), vreinterpretq_u64_s64(lo_a));
        let borrow_s = vreinterpretq_s64_u64(borrow_mask); // -1 on borrow, 0 otherwise
        // diff_hi = hi_a - hi_dig + borrow_mask (subtract -1 = add 1 only if borrow)
        let diff_hi = vaddq_s64(vsubq_s64(hi_a, hi_dig), borrow_s);

        // co_lo = (diff_lo_u >> b2klsh) | (diff_hi << (64 − b2klsh))
        let co_lo_u = vorrq_u64(
            vshlq_u64(diff_lo_u, s.srl_b2klsh),
            vshlq_u64(vreinterpretq_u64_s64(diff_hi), s.sll_b2klsh),
        );
        let co_lo = vreinterpretq_s64_u64(co_lo_u);
        // co_hi = diff_hi >> base2k_lsh (arithmetic)
        let co_hi = vshlq_s64(diff_hi, s.sra_b2klsh_co_hi);

        // digit_shifted = digit << lsh
        let lo_dig_sh = vshlq_s64(lo_dig, s.sll_lsh);
        let hi_dig_sh = vshlq_s64(lo_dig_sh, vdupq_n_s64(-63));

        // d_plus_c = digit_shifted + carry
        let lo_dpc = vaddq_s64(lo_dig_sh, lo_c);
        let carry1_mask = vcgtq_u64(vreinterpretq_u64_s64(lo_dig_sh), vreinterpretq_u64_s64(lo_dpc));
        // carry1 = 1 if unsigned overflow happened, else 0; mask is -1 → subtract.
        let carry1_s = vreinterpretq_s64_u64(carry1_mask);
        let hi_dpc = vsubq_s64(vaddq_s64(hi_dig_sh, hi_c), carry1_s);

        // out = sign_extend_low_base2k_bits(lo_dpc)
        let lo_out = vshlq_s64(vshlq_s64(lo_dpc, s.sll_b2k), s.sra_b2k);
        let hi_out = vshlq_s64(lo_out, vdupq_n_s64(-63));

        // carry2 = (d_plus_c − out) >> base2k
        let diff2_lo_u = vsubq_u64(vreinterpretq_u64_s64(lo_dpc), vreinterpretq_u64_s64(lo_out));
        let borrow2_mask = vcgtq_u64(vreinterpretq_u64_s64(lo_out), vreinterpretq_u64_s64(lo_dpc));
        let diff2_hi = vaddq_s64(vsubq_s64(hi_dpc, hi_out), vreinterpretq_s64_u64(borrow2_mask));
        // carry2_lo = (diff2_lo_u >> base2k) | (diff2_hi << (64 − base2k))
        let carry2_lo_u = vorrq_u64(
            vshlq_u64(diff2_lo_u, s.srl_b2k),
            vshlq_u64(vreinterpretq_u64_s64(diff2_hi), s.sll_b2k),
        );
        let carry2_lo = vreinterpretq_s64_u64(carry2_lo_u);
        let carry2_hi = vshlq_s64(diff2_hi, s.sra_b2k_carry);

        // new_carry = co + carry2 (i128 add, propagate carry)
        let new_lo_c_u = vaddq_u64(vreinterpretq_u64_s64(co_lo), vreinterpretq_u64_s64(carry2_lo));
        let cmask = vcgtq_u64(vreinterpretq_u64_s64(co_lo), new_lo_c_u);
        let new_lo_c = vreinterpretq_s64_u64(new_lo_c_u);
        let new_hi_c = vsubq_s64(vaddq_s64(co_hi, carry2_hi), vreinterpretq_s64_u64(cmask));

        (lo_out, new_lo_c, new_hi_c)
    }
}

/// Shared body of `nfc_final_step_assign` for one 2-lane chunk.
///
/// Mirrors `nfc_final_chunk` in the AVX file: input is i64 `lo_a`
/// (sign-extended i128 input) and the low half of i128 carry `lo_c`. Returns
/// `lo_out` such that `*r = lo_out`. `hi_dpc` is never needed because
/// `base2k ≤ 64` ⇒ `get_digit(base2k, dpc)` only depends on the low 64 bits.
#[inline(always)]
unsafe fn nfc_final_chunk(s: &NfcShifts, lo_a: int64x2_t, lo_c: int64x2_t) -> int64x2_t {
    unsafe {
        // digit = sign_extend_low_b2klsh_bits(lo_a)
        let lo_dig = vshlq_s64(vshlq_s64(lo_a, s.sll_b2klsh), s.sra_b2klsh);
        // d_plus_c = (digit << lsh) + carry_lo
        let lo_dpc = vaddq_s64(vshlq_s64(lo_dig, s.sll_lsh), lo_c);
        // out = sign_extend_low_base2k_bits(lo_dpc)
        vshlq_s64(vshlq_s64(lo_dpc, s.sll_b2k), s.sra_b2k)
    }
}

// ─── public NEON kernels ─────────────────────────────────────────────────────

/// `nfc_middle_step` — i128 input + i128 carry → i64 output.
///
/// Falls back to the scalar reference default for `n % 2 != 0` or
/// `base2k > 64`. Caller must satisfy `lsh < base2k`.
pub(crate) fn nfc_middle_step_neon(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
    if base2k > 64 || res.len() < 2 {
        <NTT120Ref as I128NormalizeOps>::nfc_middle_step(base2k, lsh, res, a, carry);
        return;
    }
    let n = res.len();
    let chunks = n >> 1;
    unsafe {
        let s = NfcShifts::new(base2k as u32, lsh as u32);
        let mut a_ptr = a.as_ptr();
        let mut c_ptr = carry.as_mut_ptr();
        let mut r_ptr = res.as_mut_ptr();
        for _ in 0..chunks {
            let (lo_a, hi_a) = load2_split_i128(a_ptr);
            let (lo_c, hi_c) = load2_split_i128(c_ptr as *const i128);
            let (lo_out, new_lo_c, new_hi_c) = nfc_middle_chunk(&s, lo_a, hi_a, lo_c, hi_c);
            store2_i64(r_ptr, lo_out);
            store2_split_i128(c_ptr, new_lo_c, new_hi_c);
            a_ptr = a_ptr.add(2);
            c_ptr = c_ptr.add(2);
            r_ptr = r_ptr.add(2);
        }
    }
    let tail = chunks << 1;
    if tail < n {
        <NTT120Ref as I128NormalizeOps>::nfc_middle_step(base2k, lsh, &mut res[tail..], &a[tail..], &mut carry[tail..]);
    }
}

/// `nfc_middle_step_into` — fused middle step for `res ±= normalize(a)`.
pub(crate) fn nfc_middle_step_into_neon<O: AssignOp>(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
    if base2k > 64 || res.len() < 2 {
        <NTT120Ref as I128NormalizeOps>::nfc_middle_step_into::<O>(base2k, lsh, res, a, carry);
        return;
    }
    let n = res.len();
    let chunks = n >> 1;
    unsafe {
        let s = NfcShifts::new(base2k as u32, lsh as u32);
        let mut a_ptr = a.as_ptr();
        let mut c_ptr = carry.as_mut_ptr();
        let mut r_ptr = res.as_mut_ptr();
        for _ in 0..chunks {
            let (lo_a, hi_a) = load2_split_i128(a_ptr);
            let (lo_c, hi_c) = load2_split_i128(c_ptr as *const i128);
            let (lo_out, new_lo_c, new_hi_c) = nfc_middle_chunk(&s, lo_a, hi_a, lo_c, hi_c);

            let lo_res = vld1q_s64(r_ptr);
            let combined = if O::SUB {
                vsubq_s64(lo_res, lo_out)
            } else {
                vaddq_s64(lo_res, lo_out)
            };
            vst1q_s64(r_ptr, combined);
            store2_split_i128(c_ptr, new_lo_c, new_hi_c);
            a_ptr = a_ptr.add(2);
            c_ptr = c_ptr.add(2);
            r_ptr = r_ptr.add(2);
        }
    }
    let tail = chunks << 1;
    if tail < n {
        <NTT120Ref as I128NormalizeOps>::nfc_middle_step_into::<O>(base2k, lsh, &mut res[tail..], &a[tail..], &mut carry[tail..]);
    }
}

/// `nfc_middle_step_assign` — in-place `i64` `res` update with `i128` carry.
pub(crate) fn nfc_middle_step_assign_neon(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
    if base2k > 64 || res.len() < 2 {
        <NTT120Ref as I128NormalizeOps>::nfc_middle_step_assign(base2k, lsh, res, carry);
        return;
    }
    let n = res.len();
    let chunks = n >> 1;
    unsafe {
        let s = NfcShifts::new(base2k as u32, lsh as u32);
        let mut c_ptr = carry.as_mut_ptr();
        let mut r_ptr = res.as_mut_ptr();
        for _ in 0..chunks {
            let (lo_a, hi_a) = load2_i64_as_split_i128(r_ptr);
            let (lo_c, hi_c) = load2_split_i128(c_ptr as *const i128);
            let (lo_out, new_lo_c, new_hi_c) = nfc_middle_chunk(&s, lo_a, hi_a, lo_c, hi_c);
            store2_i64(r_ptr, lo_out);
            store2_split_i128(c_ptr, new_lo_c, new_hi_c);
            c_ptr = c_ptr.add(2);
            r_ptr = r_ptr.add(2);
        }
    }
    let tail = chunks << 1;
    if tail < n {
        <NTT120Ref as I128NormalizeOps>::nfc_middle_step_assign(base2k, lsh, &mut res[tail..], &mut carry[tail..]);
    }
}

/// `nfc_final_step_assign` — flush i128 carry into the last i64 limb.
pub(crate) fn nfc_final_step_assign_neon(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
    if base2k > 64 || res.len() < 2 {
        <NTT120Ref as I128NormalizeOps>::nfc_final_step_assign(base2k, lsh, res, carry);
        return;
    }
    let n = res.len();
    let chunks = n >> 1;
    unsafe {
        let s = NfcShifts::new(base2k as u32, lsh as u32);
        let mut c_ptr = carry.as_ptr();
        let mut r_ptr = res.as_mut_ptr();
        for _ in 0..chunks {
            let (lo_a, _hi_a) = load2_i64_as_split_i128(r_ptr);
            // We only need lo_c (low 64 bits of i128 carry) — load via uzp1.
            let c0 = vld1q_s64(c_ptr as *const i64);
            let c1 = vld1q_s64((c_ptr as *const i64).add(2));
            let lo_c = vuzp1q_s64(c0, c1);
            let lo_out = nfc_final_chunk(&s, lo_a, lo_c);
            store2_i64(r_ptr, lo_out);
            c_ptr = c_ptr.add(2);
            r_ptr = r_ptr.add(2);
        }
    }
    let tail = chunks << 1;
    if tail < n {
        <NTT120Ref as I128NormalizeOps>::nfc_final_step_assign(base2k, lsh, &mut res[tail..], &mut carry[tail..]);
    }
}

/// `nfc_final_step_into` — fused final step for `res ±= normalize(a)`.
pub(crate) fn nfc_final_step_into_neon<O: AssignOp>(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
    if base2k > 64 || res.len() < 2 {
        <NTT120Ref as I128NormalizeOps>::nfc_final_step_into::<O>(base2k, lsh, res, carry);
        return;
    }
    let n = res.len();
    let chunks = n >> 1;
    unsafe {
        let s = NfcShifts::new(base2k as u32, lsh as u32);
        let mut c_ptr = carry.as_ptr();
        let mut r_ptr = res.as_mut_ptr();
        for _ in 0..chunks {
            let lo_res = vld1q_s64(r_ptr);
            let (lo_a, _hi_a) = load2_i64_as_split_i128(r_ptr);
            let c0 = vld1q_s64(c_ptr as *const i64);
            let c1 = vld1q_s64((c_ptr as *const i64).add(2));
            let lo_c = vuzp1q_s64(c0, c1);
            let lo_out = nfc_final_chunk(&s, lo_a, lo_c);
            let combined = if O::SUB {
                vsubq_s64(lo_res, lo_out)
            } else {
                vaddq_s64(lo_res, lo_out)
            };
            vst1q_s64(r_ptr, combined);
            c_ptr = c_ptr.add(2);
            r_ptr = r_ptr.add(2);
        }
    }
    let tail = chunks << 1;
    if tail < n {
        <NTT120Ref as I128NormalizeOps>::nfc_final_step_into::<O>(base2k, lsh, &mut res[tail..], &mut carry[tail..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    /// Lengths exercise the SIMD body and the scalar tail.
    const LENGTHS: &[usize] = &[0, 1, 2, 3, 4, 5, 7, 8, 16, 17, 64, 65];
    /// Representative `(base2k, lsh)` pairs spanning lsh==0 and lsh!=0.
    const SHIFTS: &[(usize, usize)] = &[(12, 0), (50, 0), (50, 7), (60, 0), (60, 30), (64, 0), (64, 17)];

    fn rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(0xb00b_b00b_b00b_b00b)
    }

    fn random_i128(rng: &mut ChaCha8Rng, n: usize) -> Vec<i128> {
        (0..n)
            .map(|_| {
                let lo: u64 = rng.random();
                let hi: u64 = rng.random();
                (((hi as u128) << 64) | lo as u128) as i128
            })
            .collect()
    }

    fn random_i64(rng: &mut ChaCha8Rng, n: usize) -> Vec<i64> {
        (0..n).map(|_| rng.random::<i64>()).collect()
    }

    /// `AddOp` / `SubOp` re-export for tests.
    use poulpy_cpu_ref::reference::ntt120::vec_znx_big::{AddOp, SubOp};

    #[test]
    fn nfc_middle_step_matches_scalar() {
        let mut rng = rng();
        for &n in LENGTHS {
            for &(b, l) in SHIFTS {
                if l >= b {
                    continue;
                }
                let a = random_i128(&mut rng, n);
                let c0 = random_i128(&mut rng, n);
                let mut got_r = vec![0i64; n];
                let mut got_c = c0.clone();
                let mut want_r = vec![0i64; n];
                let mut want_c = c0;
                nfc_middle_step_neon(b, l, &mut got_r, &a, &mut got_c);
                <NTT120Ref as I128NormalizeOps>::nfc_middle_step(b, l, &mut want_r, &a, &mut want_c);
                assert_eq!(got_r, want_r, "res mismatch n={n} base2k={b} lsh={l}");
                assert_eq!(got_c, want_c, "carry mismatch n={n} base2k={b} lsh={l}");
            }
        }
    }

    #[test]
    fn nfc_middle_step_assign_matches_scalar() {
        let mut rng = rng();
        for &n in LENGTHS {
            for &(b, l) in SHIFTS {
                if l >= b {
                    continue;
                }
                let r0 = random_i64(&mut rng, n);
                let c0 = random_i128(&mut rng, n);
                let mut got_r = r0.clone();
                let mut got_c = c0.clone();
                let mut want_r = r0;
                let mut want_c = c0;
                nfc_middle_step_assign_neon(b, l, &mut got_r, &mut got_c);
                <NTT120Ref as I128NormalizeOps>::nfc_middle_step_assign(b, l, &mut want_r, &mut want_c);
                assert_eq!(got_r, want_r, "res mismatch n={n} base2k={b} lsh={l}");
                assert_eq!(got_c, want_c, "carry mismatch n={n} base2k={b} lsh={l}");
            }
        }
    }

    #[test]
    fn nfc_middle_step_into_add_matches_scalar() {
        let mut rng = rng();
        for &n in LENGTHS {
            for &(b, l) in SHIFTS {
                if l >= b {
                    continue;
                }
                let r0 = random_i64(&mut rng, n);
                let a = random_i128(&mut rng, n);
                let c0 = random_i128(&mut rng, n);
                let mut got_r = r0.clone();
                let mut got_c = c0.clone();
                let mut want_r = r0;
                let mut want_c = c0;
                nfc_middle_step_into_neon::<AddOp>(b, l, &mut got_r, &a, &mut got_c);
                <NTT120Ref as I128NormalizeOps>::nfc_middle_step_into::<AddOp>(b, l, &mut want_r, &a, &mut want_c);
                assert_eq!(got_r, want_r, "res mismatch n={n} base2k={b} lsh={l}");
                assert_eq!(got_c, want_c, "carry mismatch n={n} base2k={b} lsh={l}");
            }
        }
    }

    #[test]
    fn nfc_middle_step_into_sub_matches_scalar() {
        let mut rng = rng();
        for &n in LENGTHS {
            for &(b, l) in SHIFTS {
                if l >= b {
                    continue;
                }
                let r0 = random_i64(&mut rng, n);
                let a = random_i128(&mut rng, n);
                let c0 = random_i128(&mut rng, n);
                let mut got_r = r0.clone();
                let mut got_c = c0.clone();
                let mut want_r = r0;
                let mut want_c = c0;
                nfc_middle_step_into_neon::<SubOp>(b, l, &mut got_r, &a, &mut got_c);
                <NTT120Ref as I128NormalizeOps>::nfc_middle_step_into::<SubOp>(b, l, &mut want_r, &a, &mut want_c);
                assert_eq!(got_r, want_r, "res mismatch n={n} base2k={b} lsh={l}");
                assert_eq!(got_c, want_c, "carry mismatch n={n} base2k={b} lsh={l}");
            }
        }
    }

    #[test]
    fn nfc_final_step_assign_matches_scalar() {
        let mut rng = rng();
        for &n in LENGTHS {
            for &(b, l) in SHIFTS {
                if l >= b {
                    continue;
                }
                let r0 = random_i64(&mut rng, n);
                let c0 = random_i128(&mut rng, n);
                let mut got_r = r0.clone();
                let mut got_c = c0.clone();
                let mut want_r = r0;
                let mut want_c = c0;
                nfc_final_step_assign_neon(b, l, &mut got_r, &mut got_c);
                <NTT120Ref as I128NormalizeOps>::nfc_final_step_assign(b, l, &mut want_r, &mut want_c);
                assert_eq!(got_r, want_r, "res mismatch n={n} base2k={b} lsh={l}");
            }
        }
    }

    #[test]
    fn nfc_final_step_into_add_matches_scalar() {
        let mut rng = rng();
        for &n in LENGTHS {
            for &(b, l) in SHIFTS {
                if l >= b {
                    continue;
                }
                let r0 = random_i64(&mut rng, n);
                let c0 = random_i128(&mut rng, n);
                let mut got_r = r0.clone();
                let mut got_c = c0.clone();
                let mut want_r = r0;
                let mut want_c = c0;
                nfc_final_step_into_neon::<AddOp>(b, l, &mut got_r, &mut got_c);
                <NTT120Ref as I128NormalizeOps>::nfc_final_step_into::<AddOp>(b, l, &mut want_r, &mut want_c);
                assert_eq!(got_r, want_r, "res mismatch n={n} base2k={b} lsh={l}");
            }
        }
    }

    #[test]
    fn nfc_final_step_into_sub_matches_scalar() {
        let mut rng = rng();
        for &n in LENGTHS {
            for &(b, l) in SHIFTS {
                if l >= b {
                    continue;
                }
                let r0 = random_i64(&mut rng, n);
                let c0 = random_i128(&mut rng, n);
                let mut got_r = r0.clone();
                let mut got_c = c0.clone();
                let mut want_r = r0;
                let mut want_c = c0;
                nfc_final_step_into_neon::<SubOp>(b, l, &mut got_r, &mut got_c);
                <NTT120Ref as I128NormalizeOps>::nfc_final_step_into::<SubOp>(b, l, &mut want_r, &mut want_c);
                assert_eq!(got_r, want_r, "res mismatch n={n} base2k={b} lsh={l}");
            }
        }
    }
}
