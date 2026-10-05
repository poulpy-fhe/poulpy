//! NEON kernels for the packed NTT4x30 transform domain.
//!
//! A transformed limb is four planes of `n` canonical `u32` residues, plane `p` holding the residues modulo `Q[p]`.
//! Every lane of a register therefore belongs to the same prime.
//!
//! Prepared operands (`SvpPPol`, `VmpPMat` and the right convolution operand) store their residues multiplied by `2^32`.
//! A product against a prepared operand then reduces with a single Montgomery step and lands back in the plain domain.

use core::arch::aarch64::{
    uint32x2_t, uint32x4_t, uint64x2_t, vadd_u32, vaddq_u32, vcombine_u32, vdup_n_u32, vdupq_n_u32, vdupq_n_u64, vget_low_u32,
    vhsubq_s32, vld1_u32, vld1q_u32, vld1q_u64, vmin_u32, vminq_u32, vmlal_high_u32, vmlal_u32, vmovl_high_u32, vmovl_u32,
    vmovn_u64, vmul_u32, vmull_u32, vmulq_u32, vqdmulhq_s32, vreinterpretq_s32_u32, vreinterpretq_u32_s32, vreinterpretq_u32_u64,
    vshrn_n_u64, vst1q_u32, vst1q_u64, vsub_u32, vsubq_u32, vuzp1q_u32, vuzp2q_u32, vzip1q_u32, vzip2q_u32,
};

use poulpy_cpu_portable::kernels::ntt4x30::primes::{PrimeSet, Primes30};

/// The four primes.
pub(crate) const Q: [u32; 4] = Primes30::Q;

const fn inv_mod_2_32(q: u32) -> u32 {
    // Newton iteration, an odd q is its own inverse modulo 8.
    let mut x = q;
    let mut i = 0;
    while i < 5 {
        x = x.wrapping_mul(2u32.wrapping_sub(q.wrapping_mul(x)));
        i += 1;
    }
    x
}

const fn pow2_mod(exp: u32, q: u32) -> u32 {
    let mut r = 1u64;
    let mut i = 0;
    while i < exp {
        r = (r << 1) % q as u64;
        i += 1;
    }
    r as u32
}

const fn build_qinv() -> [u32; 4] {
    [inv_mod_2_32(Q[0]), inv_mod_2_32(Q[1]), inv_mod_2_32(Q[2]), inv_mod_2_32(Q[3])]
}

const fn build_pow(exp: u32) -> [u32; 4] {
    [
        pow2_mod(exp, Q[0]),
        pow2_mod(exp, Q[1]),
        pow2_mod(exp, Q[2]),
        pow2_mod(exp, Q[3]),
    ]
}

/// `Q[p]^-1 mod 2^32`.
pub(crate) const QINV: [u32; 4] = build_qinv();
/// `2^32 mod Q[p]`.
pub(crate) const R1: [u32; 4] = build_pow(32);
/// `2^64 mod Q[p]`.
pub(crate) const R2: [u32; 4] = build_pow(64);
/// `2^96 mod Q[p]`.
pub(crate) const R3: [u32; 4] = build_pow(96);

/// Largest number of products of two canonical residues that one `u64` accumulator holds before its Montgomery step.
///
/// The step computes `S + m * q` with `m < 2^32`, which stays below `2^64` for `S <= 12 * q^2`.
/// Its result is below `4 * q`.
pub(crate) const DOT_CHUNK: usize = 12;

/// Per-plane constants.
#[derive(Clone, Copy)]
pub(crate) struct Plane {
    pub q: uint32x4_t,
    pub q2: uint32x4_t,
    pub qinv: uint32x4_t,
    pub nqinv: uint32x4_t,
}

#[inline(always)]
pub(crate) unsafe fn plane(p: usize) -> Plane {
    unsafe {
        Plane {
            q: vdupq_n_u32(Q[p]),
            q2: vdupq_n_u32(2 * Q[p]),
            qinv: vdupq_n_u32(QINV[p]),
            nqinv: vdupq_n_u32(QINV[p].wrapping_neg()),
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn planes() -> [Plane; 4] {
    unsafe { [plane(0), plane(1), plane(2), plane(3)] }
}

/// Maps `[0, 2q)` to `[0, q)`.
#[inline(always)]
pub(crate) unsafe fn cond_sub(x: uint32x4_t, q: uint32x4_t) -> uint32x4_t {
    unsafe { vminq_u32(x, vsubq_u32(x, q)) }
}

/// `a + b mod q` for canonical inputs.
#[inline(always)]
pub(crate) unsafe fn add_mod(a: uint32x4_t, b: uint32x4_t, q: uint32x4_t) -> uint32x4_t {
    unsafe { cond_sub(vaddq_u32(a, b), q) }
}

/// `a - b mod q` for canonical inputs.
#[inline(always)]
pub(crate) unsafe fn sub_mod(a: uint32x4_t, b: uint32x4_t, q: uint32x4_t) -> uint32x4_t {
    unsafe {
        let d = vsubq_u32(a, b);
        vminq_u32(d, vaddq_u32(d, q))
    }
}

/// `a * b * 2^-32 mod q`, canonical, for `a * b < q * 2^31` and `a, b < 2^31`.
///
/// Signed Montgomery step on four lanes.
/// The two doubling high multiplies share their low 32 bits, so the halving subtract is exact.
#[inline(always)]
pub(crate) unsafe fn mont_mul(a: uint32x4_t, b: uint32x4_t, c: &Plane) -> uint32x4_t {
    unsafe {
        let hi = vqdmulhq_s32(vreinterpretq_s32_u32(a), vreinterpretq_s32_u32(b));
        let m = vmulq_u32(vmulq_u32(a, b), c.qinv);
        let t = vqdmulhq_s32(vreinterpretq_s32_u32(m), vreinterpretq_s32_u32(c.q));
        let r = vreinterpretq_u32_s32(vhsubq_s32(hi, t));
        vminq_u32(r, vaddq_u32(r, c.q))
    }
}

/// Largest number of products for which the Montgomery step of an accumulator lands below `2 * q`.
pub(crate) const DOT_SHORT: usize = 4;

/// Montgomery step of four `u64` sums held as lanes `0, 1` in `lo` and lanes `2, 3` in `hi`.
///
/// Each sum must be at most `DOT_CHUNK * q^2`, or `DOT_SHORT * q^2` when `short` is set.
/// Returns `S * 2^-32 mod q`, canonical.
#[inline(always)]
pub(crate) unsafe fn redc_acc(lo: uint64x2_t, hi: uint64x2_t, c: &Plane, short: bool) -> uint32x4_t {
    unsafe {
        let low = vuzp1q_u32(vreinterpretq_u32_u64(lo), vreinterpretq_u32_u64(hi));
        let m = vmulq_u32(low, c.nqinv);
        let lo = vmlal_u32(lo, vget_low_u32(m), vget_low_u32(c.q));
        let hi = vmlal_high_u32(hi, m, c.q);
        let r = vuzp2q_u32(vreinterpretq_u32_u64(lo), vreinterpretq_u32_u64(hi));
        let r = if short { r } else { vminq_u32(r, vsubq_u32(r, c.q2)) };
        vminq_u32(r, vsubq_u32(r, c.q))
    }
}

/// Running inner product against prepared operands, fed one run of rows at a time.
///
/// A row is 16 `u32`: four lanes for each of the four primes.
pub(crate) struct DotState {
    lo: [uint64x2_t; 4],
    hi: [uint64x2_t; 4],
    out: [uint32x4_t; 4],
    terms: usize,
    flushed: bool,
}

impl DotState {
    #[inline(always)]
    pub(crate) unsafe fn new() -> Self {
        unsafe {
            let zero = vdupq_n_u64(0);
            Self {
                lo: [zero; 4],
                hi: [zero; 4],
                out: [vdupq_n_u32(0); 4],
                terms: 0,
                flushed: false,
            }
        }
    }

    #[inline(always)]
    unsafe fn flush(&mut self, c: &[Plane; 4]) {
        unsafe {
            let short = self.terms <= DOT_SHORT;
            for (p, c) in c.iter().enumerate() {
                let r = redc_acc(self.lo[p], self.hi[p], c, short);
                self.out[p] = if self.flushed { add_mod(self.out[p], r, c.q) } else { r };
            }
            let zero = vdupq_n_u64(0);
            self.lo = [zero; 4];
            self.hi = [zero; 4];
            self.terms = 0;
            self.flushed = true;
        }
    }

    /// Adds `sum_row x[row] * m[row]` over `rows` rows.
    ///
    /// `x` holds canonical residues and `m` prepared residues.
    #[inline(always)]
    pub(crate) unsafe fn push_rows(&mut self, x: *const u32, m: *const u32, rows: usize, c: &[Plane; 4]) {
        unsafe {
            let mut row = 0;
            while row < rows {
                if self.terms == DOT_CHUNK {
                    self.flush(c);
                }
                let end = row + (rows - row).min(DOT_CHUNK - self.terms);
                self.terms += end - row;
                while row < end {
                    let xr = x.add(16 * row);
                    let mr = m.add(16 * row);
                    for p in 0..4 {
                        let xv = vld1q_u32(xr.add(4 * p));
                        let mv = vld1q_u32(mr.add(4 * p));
                        self.lo[p] = vmlal_u32(self.lo[p], vget_low_u32(xv), vget_low_u32(mv));
                        self.hi[p] = vmlal_high_u32(self.hi[p], xv, mv);
                    }
                    row += 1;
                }
            }
        }
    }

    /// One canonical vector per prime.
    #[inline(always)]
    pub(crate) unsafe fn finish(mut self, c: &[Plane; 4]) -> [uint32x4_t; 4] {
        unsafe {
            if self.terms != 0 {
                self.flush(c);
            }
            self.out
        }
    }
}

/// Inner product of `rows` rows against a prepared operand.
///
/// A row is 16 `u32`: four lanes for each of the four primes.
/// `x` holds canonical residues, `m` holds prepared residues, and both advance by one row per term.
/// Returns one canonical vector per prime.
#[inline(always)]
pub(crate) unsafe fn dot_rows(x: *const u32, m: *const u32, rows: usize, c: &[Plane; 4]) -> [uint32x4_t; 4] {
    unsafe {
        let mut state = DotState::new();
        state.push_rows(x, m, rows, c);
        state.finish(c)
    }
}

/// Elementwise operation selector of [`limb_op`].
pub(crate) const OP_ADD: u8 = 0;
/// `a - b`.
pub(crate) const OP_SUB: u8 = 1;
/// `-a`, `b` is not read.
pub(crate) const OP_NEG: u8 = 2;
/// `a * b * 2^-32`, a product against a prepared operand.
pub(crate) const OP_MONT_MUL: u8 = 3;

#[inline(always)]
fn scalar_op<const OP: u8>(p: usize, a: u32, b: u32) -> u32 {
    let q = Q[p] as u64;
    let (a, b) = (a as u64, b as u64);
    (match OP {
        OP_ADD => (a + b) % q,
        OP_SUB => (a + q - b) % q,
        OP_NEG => (q - a) % q,
        _ => {
            // Multiplies by 2^-32 through the inverse of 2^32.
            let r1_inv = pow_mod(R1[p] as u64, q - 2, q);
            (a * b % q) * r1_inv % q
        }
    }) as u32
}

fn pow_mod(mut base: u64, mut exp: u64, q: u64) -> u64 {
    let mut acc = 1u64;
    while exp != 0 {
        if exp & 1 == 1 {
            acc = acc * base % q;
        }
        base = base * base % q;
        exp >>= 1;
    }
    acc
}

/// Applies `OP` to one packed limb of `n` coefficients.
///
/// # Safety
/// `dst`, `a` and `b` each address `4 * n` `u32`.
/// `dst` may alias `a` or `b` exactly.
#[inline(always)]
pub(crate) unsafe fn limb_op<const OP: u8>(n: usize, dst: *mut u32, a: *const u32, b: *const u32) {
    unsafe {
        let vecs = n / 4;
        for p in 0..4 {
            let c = plane(p);
            let (d, a, b) = (dst.add(p * n), a.add(p * n), b.add(p * n));
            for i in 0..vecs {
                let o = 4 * i;
                let av = vld1q_u32(a.add(o));
                let r = match OP {
                    OP_ADD => add_mod(av, vld1q_u32(b.add(o)), c.q),
                    OP_SUB => sub_mod(av, vld1q_u32(b.add(o)), c.q),
                    OP_NEG => cond_sub(vsubq_u32(c.q, av), c.q),
                    _ => mont_mul(av, vld1q_u32(b.add(o)), &c),
                };
                vst1q_u32(d.add(o), r);
            }
            for i in 4 * vecs..n {
                let bv = if OP == OP_NEG { 0 } else { *b.add(i) };
                *d.add(i) = scalar_op::<OP>(p, *a.add(i), bv);
            }
        }
    }
}

/// Multiplies one packed limb of canonical residues by `2^32`, in place.
pub(crate) fn limb_to_prepared(n: usize, data: &mut [u32]) {
    assert!(data.len() >= 4 * n);
    assert!(n.is_multiple_of(4));
    let d = data.as_mut_ptr();
    unsafe {
        for (p, &r2) in R2.iter().enumerate() {
            let c = plane(p);
            let r2 = vdupq_n_u32(r2);
            for i in (0..n).step_by(4) {
                let o = d.add(p * n + i);
                vst1q_u32(o, mont_mul(vld1q_u32(o), r2, &c));
            }
        }
    }
}

#[inline(always)]
unsafe fn reduce_pair(x: uint64x2_t, ch: uint32x2_t, cl: uint32x2_t, q: uint32x2_t, nqinv: uint32x2_t) -> uint32x2_t {
    unsafe {
        // y = hi * ch + lo * cl < 2^63 is congruent to x times the scale carried by (ch, cl).
        let y = vmlal_u32(vmull_u32(vshrn_n_u64::<32>(x), ch), vmovn_u64(x), cl);
        let m = vmul_u32(vmovn_u64(y), nqinv);
        // (y + m * q) / 2^32 < 2^31 + q < 4q.
        let r = vshrn_n_u64::<32>(vmlal_u32(y, m, q));
        let r = vmin_u32(r, vsub_u32(r, vadd_u32(q, q)));
        vmin_u32(r, vsub_u32(r, q))
    }
}

/// Reduces `n` lazy q120b coefficients into one packed limb.
///
/// `src` holds four `u64` per coefficient, one per prime, with any 64-bit value.
/// `dst` receives the canonical residues, multiplied by `2^32` when `prepared` is set.
pub(crate) fn pack_limb(n: usize, dst: &mut [u32], src: &[u64], prepared: bool) {
    assert!(dst.len() >= 4 * n);
    assert!(src.len() >= 4 * n);
    // The Montgomery step divides by 2^32, so the constants carry one more factor than the target scale.
    let (ch, cl) = if prepared { (&R3, &R2) } else { (&R2, &R1) };
    let nqinv = [
        QINV[0].wrapping_neg(),
        QINV[1].wrapping_neg(),
        QINV[2].wrapping_neg(),
        QINV[3].wrapping_neg(),
    ];
    unsafe {
        let (ch01, ch23) = (vld1_u32(ch.as_ptr()), vld1_u32(ch.as_ptr().add(2)));
        let (cl01, cl23) = (vld1_u32(cl.as_ptr()), vld1_u32(cl.as_ptr().add(2)));
        let (q01, q23) = (vld1_u32(Q.as_ptr()), vld1_u32(Q.as_ptr().add(2)));
        let (ni01, ni23) = (vld1_u32(nqinv.as_ptr()), vld1_u32(nqinv.as_ptr().add(2)));
        let s = src.as_ptr();
        let d = dst.as_mut_ptr();
        let vecs = n / 4;
        for i in 0..vecs {
            let base = s.add(16 * i);
            let mut r01 = [vdup_n_u32(0); 4];
            let mut r23 = [vdup_n_u32(0); 4];
            for k in 0..4 {
                r01[k] = reduce_pair(vld1q_u64(base.add(4 * k)), ch01, cl01, q01, ni01);
                r23[k] = reduce_pair(vld1q_u64(base.add(4 * k + 2)), ch23, cl23, q23, ni23);
            }
            let a01 = vcombine_u32(r01[0], r01[1]);
            let b01 = vcombine_u32(r01[2], r01[3]);
            let a23 = vcombine_u32(r23[0], r23[1]);
            let b23 = vcombine_u32(r23[2], r23[3]);
            vst1q_u32(d.add(4 * i), vuzp1q_u32(a01, b01));
            vst1q_u32(d.add(n + 4 * i), vuzp2q_u32(a01, b01));
            vst1q_u32(d.add(2 * n + 4 * i), vuzp1q_u32(a23, b23));
            vst1q_u32(d.add(3 * n + 4 * i), vuzp2q_u32(a23, b23));
        }
        for i in 4 * vecs..n {
            for (p, &q) in Q.iter().enumerate() {
                let q = q as u128;
                let scale = if prepared { 1u128 << 32 } else { 1 };
                *d.add(p * n + i) = ((*s.add(4 * i + p) as u128 % q) * scale % q) as u32;
            }
        }
    }
}

/// Widens one packed limb into `n` q120b coefficients.
pub(crate) fn unpack_limb(n: usize, dst: &mut [u64], src: &[u32]) {
    assert!(dst.len() >= 4 * n);
    assert!(src.len() >= 4 * n);
    unsafe {
        let s = src.as_ptr();
        let d = dst.as_mut_ptr();
        let vecs = n / 4;
        for i in 0..vecs {
            let p0 = vld1q_u32(s.add(4 * i));
            let p1 = vld1q_u32(s.add(n + 4 * i));
            let p2 = vld1q_u32(s.add(2 * n + 4 * i));
            let p3 = vld1q_u32(s.add(3 * n + 4 * i));
            let (a01, b01) = (vzip1q_u32(p0, p1), vzip2q_u32(p0, p1));
            let (a23, b23) = (vzip1q_u32(p2, p3), vzip2q_u32(p2, p3));
            let out = d.add(16 * i);
            vst1q_u64(out, vmovl_u32(vget_low_u32(a01)));
            vst1q_u64(out.add(2), vmovl_u32(vget_low_u32(a23)));
            vst1q_u64(out.add(4), vmovl_high_u32(a01));
            vst1q_u64(out.add(6), vmovl_high_u32(a23));
            vst1q_u64(out.add(8), vmovl_u32(vget_low_u32(b01)));
            vst1q_u64(out.add(10), vmovl_u32(vget_low_u32(b23)));
            vst1q_u64(out.add(12), vmovl_high_u32(b01));
            vst1q_u64(out.add(14), vmovl_high_u32(b23));
        }
        for i in 4 * vecs..n {
            for p in 0..4 {
                *d.add(4 * i + p) = *s.add(p * n + i) as u64;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *state ^ (*state >> 29)
    }

    fn canonical_limb(n: usize, state: &mut u64) -> Vec<u32> {
        let mut v = vec![0u32; 4 * n];
        for p in 0..4 {
            for i in 0..n {
                // Mix in the extreme residues.
                v[p * n + i] = match lcg(state) % 8 {
                    0 => 0,
                    1 => Q[p] - 1,
                    _ => (lcg(state) % Q[p] as u64) as u32,
                };
            }
        }
        v
    }

    #[test]
    fn constants() {
        for p in 0..4 {
            assert_eq!(Q[p].wrapping_mul(QINV[p]), 1);
            assert_eq!(R1[p] as u128, (1u128 << 32) % Q[p] as u128);
            assert_eq!(R2[p] as u128, (1u128 << 64) % Q[p] as u128);
            assert_eq!(R3[p] as u128, (1u128 << 96) % Q[p] as u128);
            // Bound of the accumulator Montgomery step.
            let q = Q[p] as u128;
            assert!(DOT_CHUNK as u128 * q * q + ((1u128 << 32) - 1) * q < 1u128 << 64);
            assert!((DOT_CHUNK as u128 * q * q + ((1u128 << 32) - 1) * q) >> 32 < 4 * q);
            assert!((DOT_SHORT as u128 * q * q + ((1u128 << 32) - 1) * q) >> 32 < 2 * q);
        }
    }

    #[test]
    fn limb_ops_match_scalar() {
        let mut state = 1u64;
        for n in [1usize, 2, 4, 8, 20, 64] {
            let a = canonical_limb(n, &mut state);
            let b = canonical_limb(n, &mut state);
            let mut got = vec![0u32; 4 * n];
            macro_rules! check {
                ($op:expr) => {
                    unsafe { limb_op::<{ $op }>(n, got.as_mut_ptr(), a.as_ptr(), b.as_ptr()) };
                    for p in 0..4 {
                        for i in 0..n {
                            assert_eq!(
                                got[p * n + i],
                                scalar_op::<{ $op }>(p, a[p * n + i], b[p * n + i]),
                                "op {} n {n}",
                                $op
                            );
                        }
                    }
                };
            }
            check!(OP_ADD);
            check!(OP_SUB);
            check!(OP_NEG);
            check!(OP_MONT_MUL);
        }
    }

    #[test]
    fn pack_unpack() {
        let mut state = 7u64;
        for n in [1usize, 2, 4, 12, 64] {
            let mut src = vec![0u64; 4 * n];
            for (i, x) in src.iter_mut().enumerate() {
                *x = match i % 5 {
                    0 => u64::MAX,
                    1 => 0,
                    _ => lcg(&mut state),
                };
            }
            for prepared in [false, true] {
                let mut dst = vec![0u32; 4 * n];
                pack_limb(n, &mut dst, &src, prepared);
                for p in 0..4 {
                    let q = Q[p] as u128;
                    let scale = if prepared { 1u128 << 32 } else { 1 };
                    for i in 0..n {
                        assert_eq!(dst[p * n + i] as u128, (src[4 * i + p] as u128 % q) * scale % q);
                    }
                }
            }
            let limb = canonical_limb(n, &mut state);
            let mut wide = vec![0u64; 4 * n];
            unpack_limb(n, &mut wide, &limb);
            for p in 0..4 {
                for i in 0..n {
                    assert_eq!(wide[4 * i + p], limb[p * n + i] as u64);
                }
            }
        }
    }

    #[test]
    fn dot_rows_match_scalar() {
        let mut state = 3u64;
        let c = unsafe { planes() };
        for rows in [1usize, 2, 4, 5, 11, 12, 13, 24, 25, 40] {
            // Rows of 16 u32, four lanes per prime.
            let gen_rows = |state: &mut u64, max: bool| -> Vec<u32> {
                let mut v = vec![0u32; 16 * rows];
                for row in 0..rows {
                    for p in 0..4 {
                        for lane in 0..4 {
                            v[16 * row + 4 * p + lane] = if max { Q[p] - 1 } else { (lcg(state) % Q[p] as u64) as u32 };
                        }
                    }
                }
                v
            };
            for max in [false, true] {
                let x0 = gen_rows(&mut state, max);
                let m0 = gen_rows(&mut state, max);
                let got = unsafe { dot_rows(x0.as_ptr(), m0.as_ptr(), rows, &c) };
                for p in 0..4 {
                    let q = Q[p] as u128;
                    let r_inv = pow_mod(R1[p] as u64, Q[p] as u64 - 2, Q[p] as u64) as u128;
                    let mut out = [0u32; 4];
                    unsafe { vst1q_u32(out.as_mut_ptr(), got[p]) };
                    for (lane, &got) in out.iter().enumerate() {
                        let mut want = 0u128;
                        for row in 0..rows {
                            let i = 16 * row + 4 * p + lane;
                            want = (want + x0[i] as u128 * m0[i] as u128) % q;
                        }
                        assert_eq!(got as u128, want * r_inv % q, "rows {rows}");
                    }
                }
            }
        }
    }
}
