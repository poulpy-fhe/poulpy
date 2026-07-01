//! NEON kernels for q120b ↔ {i64, i128, q120c} domain conversions.

use core::arch::aarch64::{
    uint32x4_t, vaddq_u32, vaddvq_u64, vcgtq_s64, vdupq_n_s64, vdupq_n_u64, vld1q_u32, vorrq_u64, vreinterpretq_u64_s64,
    vshlq_n_u64, vshrq_n_u64, vst1q_u32, vst1q_u64,
};
use poulpy_cpu_ref::reference::ntt4x30::primes::{PrimeSet, Primes30};

#[allow(unused_imports)]
use super::q120::{
    Q120, add_q120, and_q120, cond_sub_q120, load_const, load_q120, mla_epu32_q120, mul_epu32_q120, shr_q120, store_q120,
    sub_q120,
};

const Q_VEC: [u64; 4] = [
    Primes30::Q[0] as u64,
    Primes30::Q[1] as u64,
    Primes30::Q[2] as u64,
    Primes30::Q[3] as u64,
];

const OQ: [u64; 4] = {
    let mut oq = [0u64; 4];
    let mut k = 0usize;
    while k < 4 {
        let q = Q_VEC[k];
        oq[k] = q - (i64::MIN as u64 % q);
        k += 1;
    }
    oq
};

const BARRETT_MU: [u64; 4] = {
    let mut mu = [0u64; 4];
    let mut k = 0usize;
    while k < 4 {
        mu[k] = (1u64 << 61) / Q_VEC[k];
        k += 1;
    }
    mu
};

const POW32: [u64; 4] = {
    let mut p = [0u64; 4];
    let mut k = 0usize;
    while k < 4 {
        p[k] = ((1u128 << 32) % Q_VEC[k] as u128) as u64;
        k += 1;
    }
    p
};

const CRT_VEC: [u64; 4] = [
    Primes30::CRT_CST[0] as u64,
    Primes30::CRT_CST[1] as u64,
    Primes30::CRT_CST[2] as u64,
    Primes30::CRT_CST[3] as u64,
];

const POW32_CRT: [u64; 4] = {
    let mut r = [0u64; 4];
    let mut k = 0usize;
    while k < 4 {
        r[k] = (POW32[k] * CRT_VEC[k]) % Q_VEC[k];
        k += 1;
    }
    r
};

const POW16_CRT: [u64; 4] = {
    let mut r = [0u64; 4];
    let mut k = 0usize;
    while k < 4 {
        r[k] = ((1u64 << 16) * CRT_VEC[k]) % Q_VEC[k];
        k += 1;
    }
    r
};

const QM: [u128; 4] = {
    let q0 = Primes30::Q[0] as u128;
    let q1 = Primes30::Q[1] as u128;
    let q2 = Primes30::Q[2] as u128;
    let q3 = Primes30::Q[3] as u128;
    [q1 * q2 * q3, q0 * q2 * q3, q0 * q1 * q3, q0 * q1 * q2]
};

const QM_HI: [u64; 4] = [
    (QM[0] >> 64) as u64,
    (QM[1] >> 64) as u64,
    (QM[2] >> 64) as u64,
    (QM[3] >> 64) as u64,
];

const QM_MID: [u64; 4] = [
    ((QM[0] >> 32) & 0xFFFF_FFFF) as u64,
    ((QM[1] >> 32) & 0xFFFF_FFFF) as u64,
    ((QM[2] >> 32) & 0xFFFF_FFFF) as u64,
    ((QM[3] >> 32) & 0xFFFF_FFFF) as u64,
];

const QM_LO: [u64; 4] = [
    (QM[0] & 0xFFFF_FFFF) as u64,
    (QM[1] & 0xFFFF_FFFF) as u64,
    (QM[2] & 0xFFFF_FFFF) as u64,
    (QM[3] & 0xFFFF_FFFF) as u64,
];

const TOTAL_Q: u128 = {
    let q0 = Primes30::Q[0] as u128;
    let q1 = Primes30::Q[1] as u128;
    let q2 = Primes30::Q[2] as u128;
    let q3 = Primes30::Q[3] as u128;
    q0 * q1 * q2 * q3
};

const TOTAL_Q_MULT: [u128; 4] = [0, TOTAL_Q, TOTAL_Q * 2, TOTAL_Q * 3];

/// Barrett reduction: reduce `tmp < 2^61` to `[0, Q[k])` per lane.
#[inline(always)]
unsafe fn barrett_reduce_q120(tmp: Q120, q: Q120, mu: Q120) -> Q120 {
    unsafe {
        let mask32_v = vdupq_n_u64(u32::MAX as u64);
        let mask32 = Q120 {
            lo: mask32_v,
            hi: mask32_v,
        };
        // tmp_hi = tmp >> 32, tmp_lo = tmp & 0xFFFFFFFF
        let tmp_hi = Q120 {
            lo: vshrq_n_u64::<32>(tmp.lo),
            hi: vshrq_n_u64::<32>(tmp.hi),
        };
        let tmp_lo = and_q120(tmp, mask32);
        // q_approx_hi = (tmp_hi * mu) >> 29
        let p_hi = mul_epu32_q120(tmp_hi, mu);
        let q_hi = Q120 {
            lo: vshrq_n_u64::<29>(p_hi.lo),
            hi: vshrq_n_u64::<29>(p_hi.hi),
        };
        // q_approx_lo = (tmp_lo * mu) >> 61
        let p_lo = mul_epu32_q120(tmp_lo, mu);
        let q_lo = Q120 {
            lo: vshrq_n_u64::<61>(p_lo.lo),
            hi: vshrq_n_u64::<61>(p_lo.hi),
        };
        let q_approx = add_q120(q_hi, q_lo);
        // r = tmp − q_approx * Q
        let prod = mul_epu32_q120(q_approx, q);
        let r = sub_q120(tmp, prod);
        // Two corrective subtracts bring r into [0, Q)
        let r = cond_sub_q120(r, q);
        cond_sub_q120(r, q)
    }
}

/// Reduce a q120b value `x ∈ [0, Q << 33)` to its canonical residue per lane.
#[inline(always)]
unsafe fn reduce_b_to_canonical_q120(x: Q120, q: Q120, mu: Q120, pow32: Q120) -> Q120 {
    unsafe {
        let mask32_v = vdupq_n_u64(u32::MAX as u64);
        let mask32 = Q120 {
            lo: mask32_v,
            hi: mask32_v,
        };
        // x_hi = x >> 32 (∈ [0, 2Q))
        let x_hi = Q120 {
            lo: vshrq_n_u64::<32>(x.lo),
            hi: vshrq_n_u64::<32>(x.hi),
        };
        let x_lo = and_q120(x, mask32);
        // x_hi_r ∈ [0, Q) after one cond_sub
        let x_hi_r = cond_sub_q120(x_hi, q);
        // tmp = x_hi_r * pow32 + x_lo  (< 2^61)
        let tmp = mla_epu32_q120(x_lo, x_hi_r, pow32);
        barrett_reduce_q120(tmp, q, mu)
    }
}

/// Fused q120b reduce + CRT multiply: `t[k] = (x[k] * CRT_CST[k]) mod Q[k]`.
#[inline(always)]
unsafe fn reduce_b_and_apply_crt_q120(x: Q120, q: Q120, mu: Q120, pow32_crt: Q120, pow16_crt: Q120, crt: Q120) -> Q120 {
    unsafe {
        let mask32_v = vdupq_n_u64(u32::MAX as u64);
        let mask16_v = vdupq_n_u64(0xFFFF);
        let mask32 = Q120 {
            lo: mask32_v,
            hi: mask32_v,
        };
        let mask16 = Q120 {
            lo: mask16_v,
            hi: mask16_v,
        };
        let x_hi = Q120 {
            lo: vshrq_n_u64::<32>(x.lo),
            hi: vshrq_n_u64::<32>(x.hi),
        };
        let x_hi_r = cond_sub_q120(x_hi, q);
        let x_lo = and_q120(x, mask32);
        let x_lo_hi = Q120 {
            lo: vshrq_n_u64::<16>(x_lo.lo),
            hi: vshrq_n_u64::<16>(x_lo.hi),
        };
        let x_lo_lo = and_q120(x_lo, mask16);
        let p1 = mul_epu32_q120(x_hi_r, pow32_crt);
        let tmp = mla_epu32_q120(p1, x_lo_hi, pow16_crt);
        let tmp = mla_epu32_q120(tmp, x_lo_lo, crt);
        barrett_reduce_q120(tmp, q, mu)
    }
}

/// Vectorized horizontal CRT accumulation: `v = Σ_k t[k] * qm[k]` as `u128`.
/// Uses NEON's
/// `vaddvq_u64` for horizontal sum (no AVX-style hadd shuffle needed).
#[inline(always)]
unsafe fn crt_accumulate_q120(t: Q120, qm_hi: Q120, qm_mid: Q120, qm_lo: Q120) -> u128 {
    unsafe {
        let p_hi = mul_epu32_q120(t, qm_hi);
        let p_mid = mul_epu32_q120(t, qm_mid);
        let p_lo = mul_epu32_q120(t, qm_lo);
        // hadd64 across all 4 lanes
        let s_hi = vaddvq_u64(p_hi.lo).wrapping_add(vaddvq_u64(p_hi.hi));
        let s_mid = vaddvq_u64(p_mid.lo).wrapping_add(vaddvq_u64(p_mid.hi));
        let s_lo = vaddvq_u64(p_lo.lo).wrapping_add(vaddvq_u64(p_lo.hi));
        ((s_hi as u128) << 64) + ((s_mid as u128) << 32) + (s_lo as u128)
    }
}

/// `i64 → q120b` conversion. One coefficient per loop iteration; writes 4 × u64.
pub(crate) fn b_from_znx64_neon(nn: usize, res: &mut [u64], x: &[i64]) {
    assert!(res.len() >= 4 * nn);
    assert!(x.len() >= nn);
    unsafe {
        let oq = load_const(&OQ);
        let i64_max_v = vdupq_n_u64(i64::MAX as u64);
        let i64_max = Q120 {
            lo: i64_max_v,
            hi: i64_max_v,
        };
        let zero_s = vdupq_n_s64(0);
        let mut r_ptr = res.as_mut_ptr();

        for &xval in &x[..nn] {
            // Broadcast xval; we keep the signed view for the negative-mask
            // compare and the bit-pattern view for the AND with i64::MAX.
            let xv_s = vdupq_n_s64(xval);
            let xv_v = vreinterpretq_u64_s64(xv_s);
            let xv = Q120 { lo: xv_v, hi: xv_v };
            // xl = xval as u64 & i64::MAX (strip sign bit)
            let xl = and_q120(xv, i64_max);
            // sign mask: all-ones where xval < 0  (SIGNED compare: 0 > xval)
            let sign_lo = vcgtq_s64(zero_s, xv_s);
            let sign = Q120 {
                lo: sign_lo,
                hi: sign_lo,
            };
            let add = and_q120(sign, oq);
            let out = add_q120(xl, add);
            store_q120(r_ptr, out);
            r_ptr = r_ptr.add(4);
        }
    }
}

/// Masked variant: `(x & mask) → q120b`.
pub(crate) fn b_from_znx64_masked_neon(nn: usize, res: &mut [u64], x: &[i64], mask: i64) {
    assert!(res.len() >= 4 * nn);
    assert!(x.len() >= nn);
    unsafe {
        let oq = load_const(&OQ);
        let i64_max_v = vdupq_n_u64(i64::MAX as u64);
        let i64_max = Q120 {
            lo: i64_max_v,
            hi: i64_max_v,
        };
        let zero_s = vdupq_n_s64(0);
        let mut r_ptr = res.as_mut_ptr();

        for &xval in &x[..nn] {
            let masked = xval & mask;
            let xv_s = vdupq_n_s64(masked);
            let xv_v = vreinterpretq_u64_s64(xv_s);
            let xv = Q120 { lo: xv_v, hi: xv_v };
            let xl = and_q120(xv, i64_max);
            let sign_lo = vcgtq_s64(zero_s, xv_s);
            let sign = Q120 {
                lo: sign_lo,
                hi: sign_lo,
            };
            let add = and_q120(sign, oq);
            let out = add_q120(xl, add);
            store_q120(r_ptr, out);
            r_ptr = r_ptr.add(4);
        }
    }
}

/// `q120b → q120c` (Barrett reduce + pack `[r, r·2^32 mod Q]` per lane as u32 pairs).
pub(crate) fn c_from_b_neon(nn: usize, res: &mut [u32], a: &[u64]) {
    assert!(res.len() >= 8 * nn);
    assert!(a.len() >= 4 * nn);
    unsafe {
        let q = load_const(&Q_VEC);
        let mu = load_const(&BARRETT_MU);
        let pow32 = load_const(&POW32);

        let mut a_ptr = a.as_ptr();
        let mut r_ptr = res.as_mut_ptr() as *mut u64;

        for _ in 0..nn {
            let xv = load_q120(a_ptr);
            // r[k] = xv[k] mod Q[k] in lower 32 bits of each u64 lane.
            let r = reduce_b_to_canonical_q120(xv, q, mu, pow32);
            // r_shift[k] = (r * pow32) mod Q  (one Barrett pass).
            let r_shift = barrett_reduce_q120(mul_epu32_q120(r, pow32), q, mu);
            // Pack: lane = r | (r_shift << 32).
            let packed_lo = vorrq_u64(r.lo, vshlq_n_u64::<32>(r_shift.lo));
            let packed_hi = vorrq_u64(r.hi, vshlq_n_u64::<32>(r_shift.hi));
            vst1q_u64(r_ptr, packed_lo);
            vst1q_u64(r_ptr.add(2), packed_hi);
            a_ptr = a_ptr.add(4);
            r_ptr = r_ptr.add(4);
        }
    }
}

/// `q120b → i128` via fused CRT reconstruction.
pub(crate) fn b_to_znx128_neon(nn: usize, res: &mut [i128], a: &[u64]) {
    assert!(res.len() >= nn);
    assert!(a.len() >= 4 * nn);
    let half_q: u128 = TOTAL_Q.div_ceil(2);

    unsafe {
        let q = load_const(&Q_VEC);
        let mu = load_const(&BARRETT_MU);
        let pow32_crt = load_const(&POW32_CRT);
        let pow16_crt = load_const(&POW16_CRT);
        let crt = load_const(&CRT_VEC);
        let qm_hi = load_const(&QM_HI);
        let qm_mid = load_const(&QM_MID);
        let qm_lo = load_const(&QM_LO);

        let mut a_ptr = a.as_ptr();

        for r in &mut res[..nn] {
            let xv = load_q120(a_ptr);
            // Fused: t[k] = (x[k] * CRT_CST[k]) mod Q[k]
            let t = reduce_b_and_apply_crt_q120(xv, q, mu, pow32_crt, pow16_crt, crt);
            // CRT accumulate: v = Σ t[k] * qm[k]
            let mut v = crt_accumulate_q120(t, qm_hi, qm_mid, qm_lo);
            // Table-based modular reduction
            let q_approx = (v >> 120) as usize;
            v -= TOTAL_Q_MULT[q_approx];
            if v >= TOTAL_Q {
                v -= TOTAL_Q;
            }
            *r = if v >= half_q { v as i128 - TOTAL_Q as i128 } else { v as i128 };
            a_ptr = a_ptr.add(4);
        }
    }
}

/// Per-row q120b → q120c packing (canonical reduce + zero-pad upper 32 bits).
pub(crate) fn pack_left_1blk_x2_neon(dst: &mut [u32], a: &[u64], row_count: usize, row_stride: usize, blk: usize) {
    debug_assert!(dst.len() >= 16 * row_count);
    debug_assert!(a.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 8 * blk + 8);
    unsafe {
        let q = load_const(&Q_VEC);
        let mu = load_const(&BARRETT_MU);
        let pow32 = load_const(&POW32);
        let mut dst_ptr = dst.as_mut_ptr() as *mut u64;
        let mut a_ptr = a.as_ptr().add(8 * blk);

        for _ in 0..row_count {
            // First q120 coefficient (4 u64).
            let r0 = reduce_b_to_canonical_q120(load_q120(a_ptr), q, mu, pow32);
            store_q120(dst_ptr, r0);
            // Second q120 coefficient.
            let r1 = reduce_b_to_canonical_q120(load_q120(a_ptr.add(4)), q, mu, pow32);
            store_q120(dst_ptr.add(4), r1);

            a_ptr = a_ptr.add(row_stride);
            dst_ptr = dst_ptr.add(8);
        }
    }
}

/// Per-row q120c copy in reversed row order.
pub(crate) fn pack_right_1blk_x2_neon(dst: &mut [u32], a: &[u32], row_count: usize, row_stride: usize, blk: usize) {
    debug_assert!(dst.len() >= 16 * row_count);
    debug_assert!(a.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 16 * blk + 16);
    unsafe {
        let mut dst_ptr = dst.as_mut_ptr();
        let mut a_ptr = a.as_ptr().add(row_stride * row_count.saturating_sub(1) + 16 * blk);

        for _ in 0..row_count {
            // 16 u32 per row = 4 × uint32x4_t.
            let v0: uint32x4_t = vld1q_u32(a_ptr);
            let v1: uint32x4_t = vld1q_u32(a_ptr.add(4));
            let v2: uint32x4_t = vld1q_u32(a_ptr.add(8));
            let v3: uint32x4_t = vld1q_u32(a_ptr.add(12));
            vst1q_u32(dst_ptr, v0);
            vst1q_u32(dst_ptr.add(4), v1);
            vst1q_u32(dst_ptr.add(8), v2);
            vst1q_u32(dst_ptr.add(12), v3);

            a_ptr = a_ptr.sub(row_stride);
            dst_ptr = dst_ptr.add(16);
        }
    }
}

/// Per-row pairwise q120b sum → q120c packing.
pub(crate) fn pairwise_pack_left_1blk_x2_neon(
    dst: &mut [u32],
    a: &[u64],
    b: &[u64],
    row_count: usize,
    row_stride: usize,
    blk: usize,
) {
    debug_assert!(dst.len() >= 16 * row_count);
    debug_assert!(a.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 8 * blk + 8);
    debug_assert!(b.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 8 * blk + 8);
    unsafe {
        let q = load_const(&Q_VEC);
        let mu = load_const(&BARRETT_MU);
        let pow32 = load_const(&POW32);
        let mut dst_ptr = dst.as_mut_ptr() as *mut u64;
        let mut a_ptr = a.as_ptr().add(8 * blk);
        let mut b_ptr = b.as_ptr().add(8 * blk);

        for _ in 0..row_count {
            let r0 = reduce_b_to_canonical_q120(load_q120(a_ptr), q, mu, pow32);
            let s0 = reduce_b_to_canonical_q120(load_q120(b_ptr), q, mu, pow32);
            store_q120(dst_ptr, cond_sub_q120(add_q120(r0, s0), q));

            let r1 = reduce_b_to_canonical_q120(load_q120(a_ptr.add(4)), q, mu, pow32);
            let s1 = reduce_b_to_canonical_q120(load_q120(b_ptr.add(4)), q, mu, pow32);
            store_q120(dst_ptr.add(4), cond_sub_q120(add_q120(r1, s1), q));

            a_ptr = a_ptr.add(row_stride);
            b_ptr = b_ptr.add(row_stride);
            dst_ptr = dst_ptr.add(8);
        }
    }
}

/// Per-row pairwise q120c sum (lane-wise u32 add, reversed row order).
pub(crate) fn pairwise_pack_right_1blk_x2_neon(
    dst: &mut [u32],
    a: &[u32],
    b: &[u32],
    row_count: usize,
    row_stride: usize,
    blk: usize,
) {
    debug_assert!(dst.len() >= 16 * row_count);
    debug_assert!(a.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 16 * blk + 16);
    debug_assert!(b.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 16 * blk + 16);
    unsafe {
        let mut dst_ptr = dst.as_mut_ptr();
        let mut a_ptr = a.as_ptr().add(row_stride * row_count.saturating_sub(1) + 16 * blk);
        let mut b_ptr = b.as_ptr().add(row_stride * row_count.saturating_sub(1) + 16 * blk);

        for _ in 0..row_count {
            for off in [0usize, 4, 8, 12] {
                let av: uint32x4_t = vld1q_u32(a_ptr.add(off));
                let bv: uint32x4_t = vld1q_u32(b_ptr.add(off));
                vst1q_u32(dst_ptr.add(off), vaddq_u32(av, bv));
            }
            a_ptr = a_ptr.sub(row_stride);
            b_ptr = b_ptr.sub(row_stride);
            dst_ptr = dst_ptr.add(16);
        }
    }
}

use bytemuck::cast_slice_mut;
use poulpy_cpu_ref::reference::ntt4x30::{ntt::NttTableInv, vec_znx_dft::NttModuleHandle};
use poulpy_hal::layouts::{Data, Module, VecZnxBig, VecZnxDft, VecZnxDftToBackendMut, ZnxViewMut};

use super::ntt4x30_ntt::intt_neon;
use crate::NTT4x30Neon;

/// In-place intt + q120b → i128 CRT compaction over `n_blocks` consecutive
/// blocks of `n` q120b coefficients. Mirrors `compact_all_blocks_avx2` at
/// `vec_znx_dft_consume.rs:34`.
/// Each block reads `4 * n` u64 (q120b) at offset `4 * n * k` and writes
/// `2 * n` u64 (= `n` i128) starting at offset `2 * n * k`. The write window
/// always precedes the next read window in memory, so the in-place compaction
/// is safe.
/// # Safety
/// `u64_ptr` must cover at least `4 * n * n_blocks` u64 values. No live
/// reference may alias the buffer for the duration of the call.
unsafe fn compact_all_blocks_neon(n: usize, n_blocks: usize, u64_ptr: *mut u64, table: &NttTableInv<Primes30>) {
    use core::slice;

    let half_q: u128 = TOTAL_Q.div_ceil(2);

    unsafe {
        let q = load_const(&Q_VEC);
        let mu = load_const(&BARRETT_MU);
        let pow32_crt = load_const(&POW32_CRT);
        let pow16_crt = load_const(&POW16_CRT);
        let crt = load_const(&CRT_VEC);
        let qm_hi = load_const(&QM_HI);
        let qm_mid = load_const(&QM_MID);
        let qm_lo = load_const(&QM_LO);

        for k in 0..n_blocks {
            let src_start = 4 * n * k;
            let dst_start = 2 * n * k;

            // Apply inverse NTT in place over the block's 4*n u64 values.
            let blk = slice::from_raw_parts_mut(u64_ptr.add(src_start), 4 * n);
            intt_neon::<Primes30>(table, blk);

            // Per-coefficient: reduce → CRT accumulate → table reduce → write i128.
            for c in 0..n {
                let xv = load_q120(u64_ptr.add(src_start + 4 * c));
                let t = reduce_b_and_apply_crt_q120(xv, q, mu, pow32_crt, pow16_crt, crt);
                let mut v = crt_accumulate_q120(t, qm_hi, qm_mid, qm_lo);

                let q_approx = (v >> 120) as usize;
                v -= TOTAL_Q_MULT[q_approx];
                if v >= TOTAL_Q {
                    v -= TOTAL_Q;
                }

                let val: i128 = if v >= half_q { v as i128 - TOTAL_Q as i128 } else { v as i128 };
                (u64_ptr.add(dst_start + 2 * c) as *mut i128).write_unaligned(val);
            }
        }
    }
}

/// Public entry for the `vec_znx_idft_apply_consume` macro override.
/// Mirrors the AVX `vec_znx_idft_apply_consume` at `vec_znx_dft_consume.rs:74`.
#[allow(dead_code)]
pub(crate) fn vec_znx_idft_apply_consume<D: Data>(
    module: &Module<NTT4x30Neon>,
    mut a: VecZnxDft<D, NTT4x30Neon>,
) -> VecZnxBig<D, NTT4x30Neon>
where
    VecZnxDft<D, NTT4x30Neon>: VecZnxDftToBackendMut<NTT4x30Neon>,
{
    let table = module.get_intt_table();
    let (n, n_blocks, u64_ptr) = {
        let mut a_mut: VecZnxDft<&mut [u8], NTT4x30Neon> = a.to_backend_mut();
        let n = a_mut.n();
        let n_blocks = a_mut.cols() * a_mut.size();
        let ptr: *mut u64 = {
            let s = a_mut.raw_mut();
            cast_slice_mut::<_, u64>(s).as_mut_ptr()
        };
        (n, n_blocks, ptr)
    };
    unsafe { compact_all_blocks_neon(n, n_blocks, u64_ptr, table) };
    a.into_big()
}

#[cfg(test)]
mod tests {
    use super::*;
    use poulpy_cpu_ref::reference::ntt4x30::{
        arithmetic::{b_from_znx64_ref, b_to_znx128_ref, c_from_b_ref},
        primes::Primes30,
    };

    #[test]
    fn b_from_znx64_neon_matches_ref() {
        let n = 64usize;
        let coeffs: Vec<i64> = (0..n as i64).map(|i| i.wrapping_mul(17).wrapping_sub(500)).collect();
        let mut got = vec![0u64; 4 * n];
        let mut want = vec![0u64; 4 * n];
        b_from_znx64_neon(n, &mut got, &coeffs);
        b_from_znx64_ref::<Primes30>(n, &mut want, &coeffs);
        assert_eq!(got, want);
    }

    #[test]
    fn c_from_b_neon_matches_ref() {
        let n = 64usize;
        let coeffs: Vec<i64> = (0..n as i64).map(|i| i.wrapping_mul(11).wrapping_add(3)).collect();
        let mut b = vec![0u64; 4 * n];
        b_from_znx64_ref::<Primes30>(n, &mut b, &coeffs);
        let mut got = vec![0u32; 8 * n];
        let mut want = vec![0u32; 8 * n];
        c_from_b_neon(n, &mut got, &b);
        c_from_b_ref::<Primes30>(n, &mut want, &b);
        assert_eq!(got, want);
    }

    #[test]
    fn b_to_znx128_neon_matches_ref() {
        let n = 64usize;
        let coeffs: Vec<i64> = (0..n as i64).map(|i| i.wrapping_mul(7).wrapping_sub(20)).collect();
        let mut b = vec![0u64; 4 * n];
        b_from_znx64_ref::<Primes30>(n, &mut b, &coeffs);
        let mut got = vec![0i128; n];
        let mut want = vec![0i128; n];
        b_to_znx128_neon(n, &mut got, &b);
        b_to_znx128_ref::<Primes30>(n, &mut want, &b);
        assert_eq!(got, want);
    }
}
