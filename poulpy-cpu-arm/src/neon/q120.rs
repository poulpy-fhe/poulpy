//! Shared NEON helpers for the Q120 layout.
//!
//! A q120 vector packs four `u64` lanes (one per Primes30 prime) across two
//! `uint64x2_t` registers: `lo` holds primes 0/1, `hi` holds primes 2/3.

use core::arch::aarch64::{
    uint32x2_t, uint64x2_t, vaddq_u64, vandq_u64, vbicq_u64, vcgtq_u64, vdupq_n_u64, vld1q_u64, vmlal_u32, vmovn_u64, vmull_u32,
    vshrq_n_u64, vsraq_n_u64, vst1q_u64, vsubq_u64,
};

/// One q120 coefficient packed across two NEON registers.
/// `lo` holds primes 0/1 in u64 lanes 0/1; `hi` holds primes 2/3.
#[derive(Copy, Clone)]
pub(crate) struct Q120 {
    pub lo: uint64x2_t,
    pub hi: uint64x2_t,
}

#[inline(always)]
pub(crate) unsafe fn load_q120(p: *const u64) -> Q120 {
    unsafe {
        Q120 {
            lo: vld1q_u64(p),
            hi: vld1q_u64(p.add(2)),
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn store_q120(p: *mut u64, v: Q120) {
    unsafe {
        vst1q_u64(p, v.lo);
        vst1q_u64(p.add(2), v.hi);
    }
}

#[inline(always)]
pub(crate) unsafe fn load_const(arr: &[u64; 4]) -> Q120 {
    unsafe { load_q120(arr.as_ptr()) }
}

#[inline(always)]
pub(crate) unsafe fn zero_q120() -> Q120 {
    unsafe {
        let z = vdupq_n_u64(0);
        Q120 { lo: z, hi: z }
    }
}

#[inline(always)]
pub(crate) unsafe fn add_q120(a: Q120, b: Q120) -> Q120 {
    unsafe {
        Q120 {
            lo: vaddq_u64(a.lo, b.lo),
            hi: vaddq_u64(a.hi, b.hi),
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn sub_q120(a: Q120, b: Q120) -> Q120 {
    unsafe {
        Q120 {
            lo: vsubq_u64(a.lo, b.lo),
            hi: vsubq_u64(a.hi, b.hi),
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn and_q120(a: Q120, b: Q120) -> Q120 {
    unsafe {
        Q120 {
            lo: vandq_u64(a.lo, b.lo),
            hi: vandq_u64(a.hi, b.hi),
        }
    }
}

/// Lane-wise unsigned mul: low 32 bits of each u64 lane of `a` and `b` multiplied to a u64 result.
#[inline(always)]
pub(crate) unsafe fn mul_epu32_q120(a: Q120, b: Q120) -> Q120 {
    unsafe {
        let a_lo32: uint32x2_t = vmovn_u64(a.lo);
        let b_lo32: uint32x2_t = vmovn_u64(b.lo);
        let a_hi32: uint32x2_t = vmovn_u64(a.hi);
        let b_hi32: uint32x2_t = vmovn_u64(b.hi);
        Q120 {
            lo: vmull_u32(a_lo32, b_lo32),
            hi: vmull_u32(a_hi32, b_hi32),
        }
    }
}

/// Fused `acc + mul_epu32_q120(a, b)` via `vmlal_u32`. Drops the trailing
/// `vaddq_u64` pair vs `add_q120(acc, mul_epu32_q120(a, b))`.
#[inline(always)]
pub(crate) unsafe fn mla_epu32_q120(acc: Q120, a: Q120, b: Q120) -> Q120 {
    unsafe {
        let a_lo32: uint32x2_t = vmovn_u64(a.lo);
        let b_lo32: uint32x2_t = vmovn_u64(b.lo);
        let a_hi32: uint32x2_t = vmovn_u64(a.hi);
        let b_hi32: uint32x2_t = vmovn_u64(b.hi);
        Q120 {
            lo: vmlal_u32(acc.lo, a_lo32, b_lo32),
            hi: vmlal_u32(acc.hi, a_hi32, b_hi32),
        }
    }
}

/// Conditional subtract per lane: `x − q` if `x >= q` (unsigned), else `x`.
#[inline(always)]
pub(crate) unsafe fn cond_sub_q120(x: Q120, q: Q120) -> Q120 {
    unsafe {
        let lt_lo = vcgtq_u64(q.lo, x.lo);
        let lt_hi = vcgtq_u64(q.hi, x.hi);
        Q120 {
            lo: vsubq_u64(x.lo, vbicq_u64(q.lo, lt_lo)),
            hi: vsubq_u64(x.hi, vbicq_u64(q.hi, lt_hi)),
        }
    }
}

/// Right shift by a compile-time const that is the same for both halves.
#[inline(always)]
pub(crate) unsafe fn shr_q120<const N: i32>(x: Q120) -> Q120 {
    unsafe {
        Q120 {
            lo: vshrq_n_u64::<N>(x.lo),
            hi: vshrq_n_u64::<N>(x.hi),
        }
    }
}

/// Fused `acc + (x >> N)` via `vsraq_n_u64`. Drops the trailing `vaddq_u64`
/// pair vs `add_q120(acc, shr_q120::<N>(x))`.
#[inline(always)]
pub(crate) unsafe fn acc_shr_q120<const N: i32>(acc: Q120, x: Q120) -> Q120 {
    unsafe {
        Q120 {
            lo: vsraq_n_u64::<N>(acc.lo, x.lo),
            hi: vsraq_n_u64::<N>(acc.hi, x.hi),
        }
    }
}
