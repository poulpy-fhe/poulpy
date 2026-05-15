//! Shared NEON helpers for the Q120 layout used by NTT120 kernels.
//!
//! A q120 vector packs four `u64` lanes (one per Primes30 prime). NEON has
//! 128-bit registers (2 × u64 lanes), so each q120 is two `uint64x2_t`s —
//! `lo` for primes 0/1 and `hi` for primes 2/3. Constants follow the same
//! split.
//!
//! All routines here are wrappers around stable NEON intrinsics; none of
//! them take per-call shift immediates outside the function (they would
//! force a const-generic interface). For variable shifts the kernels build
//! signed count vectors via `vdupq_n_s64` and pass them to `vshlq_*64`.

use core::arch::aarch64::{
    uint32x2_t, uint64x2_t, vaddq_u64, vandq_u64, vbicq_u64, vcgtq_u64, vdupq_n_u64, vld1q_u64, vmovn_u64, vmull_u32,
    vshrq_n_u64, vst1q_u64, vsubq_u64,
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

/// Lane-wise unsigned `_mm256_mul_epu32` equivalent: low 32 bits of each
/// u64 lane of `a` and `b` are multiplied to produce a u64 result per lane.
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
