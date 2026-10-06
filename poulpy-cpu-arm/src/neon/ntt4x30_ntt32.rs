//! Negacyclic NTT on packed limbs, four lanes of one prime per register.
//!
//! The transform works in place on one plane of `n` residues at a time.
//! Residues are kept as signed 32-bit values, congruent to the true residue and bounded in magnitude.
//! A product by a twiddle `w` uses the precomputed quotient `w' = round(w * 2^31 / q)`:
//! `y * w - round(y * w' / 2^31) * q` lies in `[-q, q]` for every `|y| <= 2^31`.
//!
//! The forward transform is a Cooley-Tukey network with bit-reversed twiddles and no separate twist pass.
//! Its output order is the one of the portable q120 kernels, so automorphism plans are shared with them.
//! The inverse transform is the matching Gentleman-Sande network.
//! Its last level folds in `1/n` and the CRT constant, so its output feeds the reconstruction directly.

use core::arch::aarch64::{
    int32x4_t, uint64x2_t, vaddq_s32, vaddq_u32, vandq_s32, vcltzq_s32, vdupq_n_s32, vdupq_n_u32, vdupq_n_u64, vextq_s32,
    vget_low_u32, vld1q_dup_s32, vld1q_s32, vld1q_s64, vld1q_u32, vminq_u32, vmlal_high_u32, vmlal_u32, vmlsq_s32, vmulq_s32,
    vqrdmulhq_s32, vreinterpretq_s32_s64, vreinterpretq_s32_u32, vreinterpretq_s32_u64, vreinterpretq_u32_s32,
    vreinterpretq_u64_s32, vrev64q_s32, vrshrq_n_s32, vst1q_s32, vst1q_u64, vsubq_s32, vsubq_u32, vtrn1q_s32, vtrn2q_s32,
    vuzp1q_s32, vuzp1q_u64, vuzp2q_s32, vuzp2q_u64, vzip1q_s32, vzip1q_u64, vzip2q_s32, vzip2q_u64,
};

use poulpy_cpu_portable::kernels::ntt4x30::{
    conjugate_invariant::BasisChange,
    ntt::modq_pow_portable,
    primes::{PrimeSet, PrimeSetCrt4, Primes30},
};

use super::ntt4x30_packed::{Q, R1, add_mod, mont_mul, plane};

/// Smallest degree the kernels accept: the two last levels work on pairs of registers.
pub(crate) const MIN_N: usize = 8;

/// Elements of a block that is finished level by level before moving to the next one.
const BLOCK: usize = 1 << 13;

/// `round(w * 2^31 / q)`.
const fn quotient(w: u32, q: u32) -> i32 {
    ((((w as u64) << 31) + (q as u64) / 2) / q as u64) as i32
}

fn mul_mod(a: u32, b: u32, q: u32) -> u32 {
    ((a as u64 * b as u64) % q as u64) as u32
}

/// Twiddles of one prime and one direction.
struct PlaneTable {
    /// `w[m + i]` is the twiddle of block `i` of the level that has `m` blocks.
    w: Vec<i32>,
    wp: Vec<i32>,
    /// Twiddles of the level whose blocks hold four elements, each one stored twice.
    w2: Vec<i32>,
    wp2: Vec<i32>,
}

impl PlaneTable {
    fn new(n: usize, psi: u32, q: u32) -> Self {
        let log_n = n.trailing_zeros();
        let mut pows = vec![1u32; n];
        for e in 1..n {
            pows[e] = mul_mod(pows[e - 1], psi, q);
        }
        let mut w = vec![0i32; n];
        let mut wp = vec![0i32; n];
        for k in 1..n {
            let x = pows[k.reverse_bits() >> (usize::BITS - log_n)];
            w[k] = x as i32;
            wp[k] = quotient(x, q);
        }
        let mut w2 = vec![0i32; n / 2];
        let mut wp2 = vec![0i32; n / 2];
        for i in 0..n / 4 {
            w2[2 * i] = w[n / 4 + i];
            w2[2 * i + 1] = w[n / 4 + i];
            wp2[2 * i] = wp[n / 4 + i];
            wp2[2 * i + 1] = wp[n / 4 + i];
        }
        Self { w, wp, w2, wp2 }
    }
}

/// Constants of the conversion from `i64` for one prime.
#[derive(Clone, Copy, Default)]
struct ConvConst {
    /// Scale times `2^32`, with its quotient and its centered representative.
    k: i32,
    kp: i32,
    kc: i32,
    /// Scale, with its quotient.
    s: i32,
    sp: i32,
}

/// Constants of the last inverse level for one prime.
#[derive(Clone, Copy, Default)]
struct FinalConst {
    /// `crt / n`, with its quotient.
    c: i32,
    cp: i32,
    /// `w * crt / n` for the twiddle of the level, with its quotient.
    cw: i32,
    cwp: i32,
}

/// Tables of the packed NTT of one degree.
pub(crate) struct Ntt32Table {
    n: usize,
    fwd: [PlaneTable; 4],
    inv: [PlaneTable; 4],
    /// Indexed by `prepared`, then by prime.
    conv: [[ConvConst; 4]; 2],
    fin: [FinalConst; 4],
    /// Basis changes of the conjugate-invariant ring: into the negacyclic basis, and back.
    ci: Option<[[CiPlane; 4]; 2]>,
}

impl Ntt32Table {
    /// Tables of degree `n`, with the basis changes of the conjugate-invariant ring when `conjugate_invariant` is set.
    pub(crate) fn new(n: usize, conjugate_invariant: bool) -> Self {
        assert!(n.is_power_of_two() && (MIN_N..=1 << Primes30::MAX_LOG_N).contains(&n));
        let psi: [u32; 4] =
            std::array::from_fn(|p| modq_pow_portable(Primes30::OMEGA[p], (1i64 << Primes30::MAX_LOG_N) / n as i64, Q[p]));
        let fwd = std::array::from_fn(|p| PlaneTable::new(n, psi[p], Q[p]));
        let inv = std::array::from_fn(|p| PlaneTable::new(n, modq_pow_portable(psi[p], -1, Q[p]), Q[p]));
        let conv = std::array::from_fn(|prepared| {
            std::array::from_fn(|p| {
                let q = Q[p];
                let s = if prepared == 1 { R1[p] } else { 1 };
                let k = mul_mod(s, R1[p], q);
                ConvConst {
                    k: k as i32,
                    kp: quotient(k, q),
                    kc: if k > q / 2 { k as i32 - q as i32 } else { k as i32 },
                    s: s as i32,
                    sp: quotient(s, q),
                }
            })
        });
        let fin = std::array::from_fn(|p| {
            let q = Q[p];
            let c = mul_mod(modq_pow_portable(n as u32, -1, q), Primes30::CRT_CST[p], q);
            let cw = mul_mod(c, inv[p].w[1] as u32, q);
            FinalConst {
                c: c as i32,
                cp: quotient(c, q),
                cw: cw as i32,
                cwp: quotient(cw, q),
            }
        });
        let ci =
            conjugate_invariant.then(|| std::array::from_fn(|inverse| std::array::from_fn(|p| CiPlane::new(n, p, inverse == 1))));
        Self {
            n,
            fwd,
            inv,
            conv,
            fin,
            ci,
        }
    }
}

/// Basis change of the conjugate-invariant ring for one prime and one direction.
///
/// Coefficient `j` becomes `d[j] * a[j] + c[j] * a[n - j]`, and coefficient `0` is kept.
/// The factors are stored multiplied by `2^32`, and a second time in mirrored order for the coefficients `n - j`.
struct CiPlane {
    d: Vec<u32>,
    c: Vec<u32>,
    /// `d[n - j]` and `c[n - j]` at index `j`.
    dm: Vec<u32>,
    cm: Vec<u32>,
    q: u32,
    /// `2^-32 mod q`.
    r_inv: u32,
}

impl CiPlane {
    fn new(n: usize, p: usize, inverse: bool) -> Self {
        let q = Q[p];
        let basis = BasisChange::new(n, q as u64, Primes30::OMEGA[p] as u64, Primes30::MAX_LOG_N, inverse);
        let prepared = |x: u64| mul_mod(x as u32, R1[p], q);
        let d: Vec<u32> = basis.factors().iter().map(|f| prepared(f[0])).collect();
        let c: Vec<u32> = basis.factors().iter().map(|f| prepared(f[1])).collect();
        let mirror = |v: &[u32]| (0..n).map(|j| v[(n - j) % n]).collect();
        Self {
            dm: mirror(&d),
            cm: mirror(&c),
            d,
            c,
            q,
            r_inv: modq_pow_portable(R1[p], -1, q),
        }
    }

    /// One coefficient: `d * a + c * b`, canonical, for prepared factors and any signed `a`, `b`.
    fn scalar(&self, d: u32, c: u32, a: i32, b: i32) -> i32 {
        let q = self.q as i64;
        let term = |x: i32, w: u32| (x as i64).rem_euclid(q) * w as i64 % q * self.r_inv as i64 % q;
        ((term(a, d) + term(b, c)) % q) as i32
    }

    /// Applies the basis change to one plane, in place.
    ///
    /// Inputs are signed residues of magnitude below `2 * q`, outputs are canonical, coefficient `0` excepted, which is kept.
    unsafe fn apply(&self, a: *mut i32, n: usize) {
        unsafe {
            let c = plane(Q.iter().position(|&q| q == self.q).unwrap());
            let mul = |x: int32x4_t, w: *const u32| vreinterpretq_s32_u32(mont_mul(vreinterpretq_u32_s32(x), vld1q_u32(w), &c));
            let reverse = |x: int32x4_t| {
                let x = vrev64q_s32(x);
                vextq_s32::<2>(x, x)
            };
            let half = n / 2;
            // Coefficients `j..j + 4` against their mirrors `n - j - 3..=n - j`, read before either is written.
            let mut j = 1;
            while j + 4 <= half {
                let lo = a.add(j);
                let hi = a.add(n - j - 3);
                let x = vld1q_s32(lo);
                let y = reverse(vld1q_s32(hi));
                let q = vreinterpretq_u32_s32(vdupq_n_s32(self.q as i32));
                let sum = |u: int32x4_t, v: int32x4_t| {
                    vreinterpretq_s32_u32(add_mod(vreinterpretq_u32_s32(u), vreinterpretq_u32_s32(v), q))
                };
                let new_x = sum(mul(x, self.d.as_ptr().add(j)), mul(y, self.c.as_ptr().add(j)));
                let new_y = sum(mul(y, self.dm.as_ptr().add(j)), mul(x, self.cm.as_ptr().add(j)));
                vst1q_s32(lo, new_x);
                vst1q_s32(hi, reverse(new_y));
                j += 4;
            }
            while j <= half {
                let (x, y) = (*a.add(j), *a.add(n - j));
                *a.add(j) = self.scalar(self.d[j], self.c[j], x, y);
                *a.add(n - j) = self.scalar(self.dm[j], self.cm[j], y, x);
                j += 1;
            }
        }
    }
}

/// `y * w mod q` in `[-q, q]`, for any `y`.
#[inline(always)]
unsafe fn mul_w(y: int32x4_t, w: int32x4_t, wp: int32x4_t, q: int32x4_t) -> int32x4_t {
    unsafe { vmlsq_s32(vmulq_s32(y, w), vqrdmulhq_s32(y, wp), q) }
}

/// Maps any value to a congruent one of magnitude below `0.56 * q`.
#[inline(always)]
unsafe fn reduce(x: int32x4_t, q: int32x4_t) -> int32x4_t {
    unsafe { vmlsq_s32(x, vrshrq_n_s32::<30>(x), q) }
}

/// Maps `[-q, q]` to `[0, q)`.
#[inline(always)]
unsafe fn canonical(x: int32x4_t, q: int32x4_t) -> int32x4_t {
    unsafe {
        let (x, q) = (vreinterpretq_u32_s32(x), vreinterpretq_u32_s32(q));
        let x = vminq_u32(x, vaddq_u32(x, q));
        vreinterpretq_s32_u32(vminq_u32(x, vsubq_u32(x, q)))
    }
}

/// One Cooley-Tukey block: `t` butterflies between `a[..t]` and `a[t..2t]`, `t >= 4`.
#[inline(always)]
unsafe fn fwd_block(a: *mut i32, t: usize, w: *const i32, wp: *const i32, q: int32x4_t) {
    unsafe {
        let (w, wp) = (vld1q_dup_s32(w), vld1q_dup_s32(wp));
        let b = a.add(t);
        let mut j = 0;
        while j < t {
            let x = reduce(vld1q_s32(a.add(j)), q);
            let u = mul_w(vld1q_s32(b.add(j)), w, wp, q);
            vst1q_s32(a.add(j), vaddq_s32(x, u));
            vst1q_s32(b.add(j), vsubq_s32(x, u));
            j += 4;
        }
    }
}

/// The two last forward levels of eight consecutive elements, with canonical outputs.
///
/// `e` is the offset of the elements in the plane.
#[inline(always)]
unsafe fn fwd_tail(a: *mut i32, e: usize, n: usize, tab: &PlaneTable, q: int32x4_t) {
    unsafe {
        let v0 = vreinterpretq_u64_s32(vld1q_s32(a));
        let v1 = vreinterpretq_u64_s32(vld1q_s32(a.add(4)));
        // Blocks of four elements: the two low lanes of each register against the two high ones.
        let x = reduce(vreinterpretq_s32_u64(vuzp1q_u64(v0, v1)), q);
        let y = vreinterpretq_s32_u64(vuzp2q_u64(v0, v1));
        let u = mul_w(
            y,
            vld1q_s32(tab.w2.as_ptr().add(e / 2)),
            vld1q_s32(tab.wp2.as_ptr().add(e / 2)),
            q,
        );
        let (x1, y1) = (vaddq_s32(x, u), vsubq_s32(x, u));
        // Blocks of two elements.
        let x = reduce(vtrn1q_s32(x1, y1), q);
        let y = vtrn2q_s32(x1, y1);
        let k = n / 2 + e / 2;
        let u = mul_w(y, vld1q_s32(tab.w.as_ptr().add(k)), vld1q_s32(tab.wp.as_ptr().add(k)), q);
        // |x| < 0.56 q and |u| <= q: one more reduction brings the outputs within q.
        let x2 = canonical(reduce(vaddq_s32(x, u), q), q);
        let y2 = canonical(reduce(vsubq_s32(x, u), q), q);
        vst1q_s32(a, vzip1q_s32(x2, y2));
        vst1q_s32(a.add(4), vzip2q_s32(x2, y2));
    }
}

/// Forward transform of one plane.
///
/// Inputs are any signed values, outputs are canonical.
unsafe fn fwd_plane(a: *mut i32, n: usize, tab: &PlaneTable, q: i32) {
    unsafe {
        let q = vdupq_n_s32(q);
        let (w, wp) = (tab.w.as_ptr(), tab.wp.as_ptr());
        // Levels that span more than one block run over the whole plane.
        let mut m = 1;
        let mut t = n / 2;
        while t >= 4 && 2 * t > BLOCK {
            for i in 0..m {
                fwd_block(a.add(2 * i * t), t, w.add(m + i), wp.add(m + i), q);
            }
            m *= 2;
            t /= 2;
        }
        // Each block then runs all of its remaining levels.
        let (m0, size) = (m, 2 * t);
        for b in 0..m0 {
            let base = a.add(b * size);
            let (mut m, mut t, mut count) = (m0, size / 2, 1);
            while t >= 4 {
                for i in 0..count {
                    let k = m + b * count + i;
                    fwd_block(base.add(2 * i * t), t, w.add(k), wp.add(k), q);
                }
                m *= 2;
                t /= 2;
                count *= 2;
            }
            let mut e = 0;
            while e < size {
                fwd_tail(base.add(e), b * size + e, n, tab, q);
                e += 8;
            }
        }
    }
}

/// One Gentleman-Sande block: `t` butterflies between `a[..t]` and `a[t..2t]`, `t >= 4`.
///
/// Inputs and outputs have magnitude at most `q`.
#[inline(always)]
unsafe fn inv_block(a: *mut i32, t: usize, w: *const i32, wp: *const i32, q: int32x4_t) {
    unsafe {
        let (w, wp) = (vld1q_dup_s32(w), vld1q_dup_s32(wp));
        let b = a.add(t);
        let mut j = 0;
        while j < t {
            let x = vld1q_s32(a.add(j));
            let y = vld1q_s32(b.add(j));
            vst1q_s32(a.add(j), reduce(vaddq_s32(x, y), q));
            vst1q_s32(b.add(j), mul_w(vsubq_s32(x, y), w, wp, q));
            j += 4;
        }
    }
}

/// The two first inverse levels of eight consecutive elements, read from `src` and written to `dst`.
#[inline(always)]
unsafe fn inv_head(dst: *mut i32, src: *const i32, e: usize, n: usize, tab: &PlaneTable, q: int32x4_t) {
    unsafe {
        let v0 = vld1q_s32(src);
        let v1 = vld1q_s32(src.add(4));
        // Blocks of two elements.
        let (x, y) = (vuzp1q_s32(v0, v1), vuzp2q_s32(v0, v1));
        let k = n / 2 + e / 2;
        let x1 = reduce(vaddq_s32(x, y), q);
        let y1 = mul_w(
            vsubq_s32(x, y),
            vld1q_s32(tab.w.as_ptr().add(k)),
            vld1q_s32(tab.wp.as_ptr().add(k)),
            q,
        );
        // Blocks of four elements.
        let (x, y) = (vtrn1q_s32(x1, y1), vtrn2q_s32(x1, y1));
        let x2 = reduce(vaddq_s32(x, y), q);
        let y2 = mul_w(
            vsubq_s32(x, y),
            vld1q_s32(tab.w2.as_ptr().add(e / 2)),
            vld1q_s32(tab.wp2.as_ptr().add(e / 2)),
            q,
        );
        let (x2, y2) = (vreinterpretq_u64_s32(x2), vreinterpretq_u64_s32(y2));
        vst1q_s32(dst, vreinterpretq_s32_u64(vzip1q_u64(x2, y2)));
        vst1q_s32(dst.add(4), vreinterpretq_s32_u64(vzip2q_u64(x2, y2)));
    }
}

/// Inverse transform of one plane, scaled by `crt / n`.
///
/// Reads canonical residues from `src` and leaves canonical residues in `a`.
/// `src` may be `a` itself.
unsafe fn inv_plane(a: *mut i32, src: *const i32, n: usize, tab: &PlaneTable, fin: &FinalConst, q: i32) {
    unsafe {
        let q = vdupq_n_s32(q);
        let (w, wp) = (tab.w.as_ptr(), tab.wp.as_ptr());
        let half = n / 2;
        // Each block runs its levels up to its own size, the last level of the plane excepted.
        let size = n.min(BLOCK);
        for b in 0..n / size {
            let base = a.add(b * size);
            let mut e = 0;
            while e < size {
                inv_head(base.add(e), src.add(b * size + e), b * size + e, n, tab, q);
                e += 8;
            }
            let mut t = 4;
            while 2 * t <= size && t < half {
                let m = n / (2 * t);
                let count = size / (2 * t);
                for i in 0..count {
                    let k = m + b * count + i;
                    inv_block(base.add(2 * i * t), t, w.add(k), wp.add(k), q);
                }
                t *= 2;
            }
        }
        // Levels that span more than one block run over the whole plane.
        let mut t = size;
        while t < half {
            let m = n / (2 * t);
            for i in 0..m {
                inv_block(a.add(2 * i * t), t, w.add(m + i), wp.add(m + i), q);
            }
            t *= 2;
        }
        // Last level, with the scaling folded into both outputs.
        let (c, cp) = (vdupq_n_s32(fin.c), vdupq_n_s32(fin.cp));
        let (cw, cwp) = (vdupq_n_s32(fin.cw), vdupq_n_s32(fin.cwp));
        let b = a.add(half);
        let mut j = 0;
        while j < half {
            let x = vld1q_s32(a.add(j));
            let y = vld1q_s32(b.add(j));
            vst1q_s32(a.add(j), canonical(mul_w(vaddq_s32(x, y), c, cp, q), q));
            vst1q_s32(b.add(j), canonical(mul_w(vsubq_s32(x, y), cw, cwp, q), q));
            j += 4;
        }
    }
}

/// Residues of four signed coefficients modulo one prime, of magnitude below `1.6 * q`.
///
/// The coefficient is `hi * 2^32 + lo`, with `lo` read as a signed word and its sign bit carried in `carry`.
#[inline(always)]
unsafe fn residues(lo: int32x4_t, hi: int32x4_t, carry: int32x4_t, c: &ConvConst, q: int32x4_t) -> int32x4_t {
    unsafe {
        let u = mul_w(hi, vdupq_n_s32(c.k), vdupq_n_s32(c.kp), q);
        let v = mul_w(lo, vdupq_n_s32(c.s), vdupq_n_s32(c.sp), q);
        let v = reduce(vaddq_s32(v, vandq_s32(carry, vdupq_n_s32(c.kc))), q);
        vaddq_s32(u, v)
    }
}

/// Low words, high words and carries of four signed coefficients.
#[inline(always)]
unsafe fn split_i64(src: *const i64) -> (int32x4_t, int32x4_t, int32x4_t) {
    unsafe {
        let a0 = vreinterpretq_s32_s64(vld1q_s64(src));
        let a1 = vreinterpretq_s32_s64(vld1q_s64(src.add(2)));
        let lo = vuzp1q_s32(a0, a1);
        (lo, vuzp2q_s32(a0, a1), vreinterpretq_s32_u32(vcltzq_s32(lo)))
    }
}

/// Reduces `n` signed coefficients into four planes of signed residues, multiplied by `2^32` when `prepared` is set.
unsafe fn from_i64(dst: *mut i32, src: *const i64, n: usize, conv: &[ConvConst; 4]) {
    unsafe {
        let mut i = 0;
        while i < n {
            let (lo, hi, carry) = split_i64(src.add(i));
            for (p, c) in conv.iter().enumerate() {
                vst1q_s32(dst.add(p * n + i), residues(lo, hi, carry, c, vdupq_n_s32(Q[p] as i32)));
            }
            i += 4;
        }
    }
}

/// Forward transform of `n` coefficients into one packed limb.
///
/// The residues are canonical, multiplied by `2^32` when `prepared` is set.
pub(crate) fn ntt32(table: &Ntt32Table, dst: &mut [u32], src: &[i64], prepared: bool) {
    let n = table.n;
    assert!(dst.len() >= 4 * n);
    assert!(src.len() >= n);
    let d = dst.as_mut_ptr() as *mut i32;
    unsafe {
        from_i64(d, src.as_ptr(), n, &table.conv[prepared as usize]);
        for (p, &q) in Q.iter().enumerate() {
            if let Some(ci) = &table.ci {
                ci[0][p].apply(d.add(p * n), n);
            }
            fwd_plane(d.add(p * n), n, &table.fwd[p], q as i32);
        }
    }
}

const QM: [u128; 4] = {
    let q = [Q[0] as u128, Q[1] as u128, Q[2] as u128, Q[3] as u128];
    [q[1] * q[2] * q[3], q[0] * q[2] * q[3], q[0] * q[1] * q[3], q[0] * q[1] * q[2]]
};
const TOTAL_Q: u128 = QM[0] * Q[0] as u128;
const TOTAL_Q_MULT: [u128; 5] = [0, TOTAL_Q, TOTAL_Q * 2, TOTAL_Q * 3, TOTAL_Q * 4];

/// The complementary products `Q / Q[p]` in three limbs of 30 bits.
const QM_LIMBS: [[u32; 3]; 4] = {
    let mut out = [[0u32; 3]; 4];
    let mut p = 0;
    while p < 4 {
        out[p] = [
            (QM[p] & 0x3FFF_FFFF) as u32,
            ((QM[p] >> 30) & 0x3FFF_FFFF) as u32,
            (QM[p] >> 60) as u32,
        ];
        p += 1;
    }
    out
};

/// `floor(Q / 2)`, the offset that turns the centered reduction into a plain one.
const HALF_Q: u128 = TOTAL_Q / 2;
const HALF_Q_LIMBS: [u64; 3] = [
    (HALF_Q & 0x3FFF_FFFF) as u64,
    ((HALF_Q >> 30) & 0x3FFF_FFFF) as u64,
    (HALF_Q >> 60) as u64,
];

/// CRT reconstruction of `n` coefficients from four planes of canonical residues already multiplied by the CRT constants.
///
/// The sum `floor(Q / 2) + sum_p t[p] * (Q / Q[p])` is accumulated on four coefficients at a time, limb by limb.
/// Each limb sum stays below `2^63`.
/// Reducing it modulo `Q` and removing the offset gives the representative in `[-floor(Q / 2), ceil(Q / 2))`.
unsafe fn crt(dst: *mut i128, t: *const u32, n: usize) {
    unsafe {
        let init: [uint64x2_t; 3] = std::array::from_fn(|j| vdupq_n_u64(HALF_Q_LIMBS[j]));
        let mut sums = [0u64; 12];
        let mut i = 0;
        while i < n {
            let mut lo = init;
            let mut hi = init;
            for (p, limbs) in QM_LIMBS.iter().enumerate() {
                let x = vld1q_u32(t.add(p * n + i));
                for (j, &limb) in limbs.iter().enumerate() {
                    let c = vdupq_n_u32(limb);
                    lo[j] = vmlal_u32(lo[j], vget_low_u32(x), vget_low_u32(c));
                    hi[j] = vmlal_high_u32(hi[j], x, c);
                }
            }
            for j in 0..3 {
                vst1q_u64(sums.as_mut_ptr().add(4 * j), lo[j]);
                vst1q_u64(sums.as_mut_ptr().add(4 * j + 2), hi[j]);
            }
            for k in 0..4 {
                // The sum is below 4.5 Q and Q is between 2^119 and 2^120: at most one multiple of Q remains after the table.
                let mut v = sums[k] as u128 + ((sums[4 + k] as u128) << 30) + ((sums[8 + k] as u128) << 60);
                v -= TOTAL_Q_MULT[(v >> 120) as usize];
                let w = v.wrapping_sub(TOTAL_Q);
                v = if v >= TOTAL_Q { w } else { v };
                *dst.add(i + k) = v as i128 - HALF_Q as i128;
            }
            i += 4;
        }
    }
}

/// Inverse transform of one packed limb into `n` coefficients.
///
/// `work` receives the intermediate planes and may be `src` itself.
///
/// # Safety
/// `src` and `work` address `4 * n` `u32`, `dst` addresses `n` `i128` and does not overlap `work`.
pub(crate) unsafe fn intt32(table: &Ntt32Table, dst: *mut i128, src: *const u32, work: *mut u32) {
    let n = table.n;
    unsafe {
        for (p, &q) in Q.iter().enumerate() {
            inv_plane(
                (work as *mut i32).add(p * n),
                (src as *const i32).add(p * n),
                n,
                &table.inv[p],
                &table.fin[p],
                q as i32,
            );
            if let Some(ci) = &table.ci {
                ci[1][p].apply((work as *mut i32).add(p * n), n);
            }
        }
        crt(dst, work, n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poulpy_cpu_portable::NTT4x30Portable;
    use poulpy_cpu_portable::kernels::ntt4x30::{
        NttDFTExecute, NttFromZnx64, NttToZnx128,
        ntt::{NttTable, NttTableInv},
    };
    use poulpy_hal::layouts::{ConjugateInvariant, Ring, Standard};

    fn lcg(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *state ^ (*state >> 29)
    }

    /// Checks both transforms against the portable q120 kernels of ring `R`, bit for bit.
    fn matches_portable<R: Ring>(log_ns: &[usize], new_table: impl Fn(usize) -> (NttTable<Primes30, R>, NttTableInv<Primes30, R>))
    where
        NTT4x30Portable<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
    {
        let mut state = 11u64;
        for &log_n in log_ns {
            let n = 1 << log_n;
            let (fwd, inv) = new_table(n);
            let table = Ntt32Table::new(n, R::CYCLOTOMIC_ORDER_FACTOR == 4);
            for round in 0..3 {
                let src: Vec<i64> = (0..n)
                    .map(|i| match (round, i % 7) {
                        (0, 0) => i64::MAX,
                        (0, 1) => i64::MIN,
                        (0, 2) => -1,
                        (0, 3) => (1i64 << 32) - 1,
                        (0, 4) => -(1i64 << 31),
                        (1, _) => (lcg(&mut state) as i64) >> 11,
                        _ => lcg(&mut state) as i64,
                    })
                    .collect();
                let mut wide = vec![0u64; 4 * n];
                <NTT4x30Portable<R> as NttFromZnx64>::ntt_from_znx64(&mut wide, &src);
                <NTT4x30Portable<R> as NttDFTExecute<_>>::ntt_dft_execute(&fwd, &mut wide);
                for prepared in [false, true] {
                    let mut got = vec![0u32; 4 * n];
                    ntt32(&table, &mut got, &src, prepared);
                    for p in 0..4 {
                        let q = Q[p] as u128;
                        let scale = if prepared { 1u128 << 32 } else { 1 };
                        for i in 0..n {
                            let want = (wide[4 * i + p] as u128 % q) * scale % q;
                            assert_eq!(
                                got[p * n + i] as u128,
                                want,
                                "forward n {n} round {round} prepared {prepared}"
                            );
                        }
                    }
                }

                // Inverse of arbitrary canonical planes, extremes included.
                let mut planes = vec![0u32; 4 * n];
                for p in 0..4 {
                    for i in 0..n {
                        planes[p * n + i] = match lcg(&mut state) % 8 {
                            0 => 0,
                            1 => Q[p] - 1,
                            _ => (lcg(&mut state) % Q[p] as u64) as u32,
                        };
                        wide[4 * i + p] = planes[p * n + i] as u64;
                    }
                }
                <NTT4x30Portable<R> as NttDFTExecute<_>>::ntt_dft_execute(&inv, &mut wide);
                let mut want = vec![0i128; n];
                <NTT4x30Portable<R> as NttToZnx128>::ntt_to_znx128(&mut want, n, &wide);
                let mut got = vec![0i128; n];
                let mut work = vec![0u32; 4 * n];
                unsafe { intt32(&table, got.as_mut_ptr(), planes.as_ptr(), work.as_mut_ptr()) };
                assert_eq!(got, want, "inverse n {n} round {round}");
                // In place.
                let mut got = vec![0i128; n];
                let p = planes.as_mut_ptr();
                unsafe { intt32(&table, got.as_mut_ptr(), p, p) };
                assert_eq!(got, want, "inverse in place n {n} round {round}");
            }
        }
    }

    #[test]
    fn matches_portable_standard() {
        matches_portable::<Standard>(&[3, 4, 5, 6, 10, 13, 14, 15, 16], |n| {
            (
                NttTable::<Primes30, Standard>::new(n),
                NttTableInv::<Primes30, Standard>::new(n),
            )
        });
    }

    #[test]
    fn matches_portable_conjugate_invariant() {
        matches_portable::<ConjugateInvariant>(&[3, 4, 5, 6, 10, 13, 14, 15, 16], |n| {
            (
                NttTable::<Primes30, ConjugateInvariant>::new(n),
                NttTableInv::<Primes30, ConjugateInvariant>::new(n),
            )
        });
    }
}
