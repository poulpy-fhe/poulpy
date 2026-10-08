//! Negacyclic NTT on packed limbs, one plane of one prime at a time.
//!
//! The transform works in place on a plane of `n` residues.
//! Residues are kept as signed 32-bit values, congruent to the true residue and bounded in magnitude.
//! A product by a twiddle `w` uses the precomputed quotient `w' = round(w * 2^31 / q)`:
//! `y * w - round(y * w' / 2^31) * q` lies in `[-q, q]` for every `|y| <= 2^31`.
//!
//! The forward transform is a Cooley-Tukey network with bit-reversed twiddles and no separate twist pass.
//! Its output order is the one of the q120 kernels, so automorphism plans are shared with them.
//! The inverse transform is the matching Gentleman-Sande network.
//! Its last level folds in `1/n` and the CRT constant, so its output feeds the reconstruction directly.
//!
//! Every loop is a plain loop over slices with a fixed stride, which compilers turn into vector code.

use bytemuck::{cast_slice, cast_slice_mut};
use poulpy_hal::execution::TaskExecutor;

use crate::kernels::ntt4x30::{
    conjugate_invariant::BasisChange,
    ntt::modq_pow_portable,
    primes::{PrimeSet, PrimeSetCrt4, Primes30},
};

use super::packed::{Q, QINV, R1};

/// Smallest degree the kernels accept: the two first and two last levels have their own loops.
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
        Self { w, wp }
    }

    /// Twiddles of the `count` blocks from block `first` of the level that has `m` blocks.
    #[inline(always)]
    fn level(&self, m: usize, first: usize, count: usize) -> (&[i32], &[i32]) {
        (&self.w[m + first..m + first + count], &self.wp[m + first..m + first + count])
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

    #[cfg(test)]
    pub(crate) fn n(&self) -> usize {
        self.n
    }
}

/// Basis change of the conjugate-invariant ring for one prime and one direction.
///
/// Coefficient `j` becomes `d[j] * a[j] + c[j] * a[n - j]`, and coefficient `0` is kept.
/// The factors are stored multiplied by `2^32`.
struct CiPlane {
    d: Vec<u32>,
    c: Vec<u32>,
    q: u32,
    qinv: u32,
}

impl CiPlane {
    fn new(n: usize, p: usize, inverse: bool) -> Self {
        let q = Q[p];
        let basis = BasisChange::new(n, q as u64, Primes30::OMEGA[p] as u64, Primes30::MAX_LOG_N, inverse);
        let prepared = |x: u64| mul_mod(x as u32, R1[p], q);
        Self {
            d: basis.factors().iter().map(|f| prepared(f[0])).collect(),
            c: basis.factors().iter().map(|f| prepared(f[1])).collect(),
            q,
            qinv: QINV[p],
        }
    }

    /// `x * w * 2^-32 mod q` in `(-q, q)`, for a prepared factor `w` and a signed `x` of magnitude below `2 * q`.
    #[inline(always)]
    fn mul(&self, x: i32, w: u32) -> i32 {
        let s = x as i64 * w as i64;
        let m = (s as i32).wrapping_mul(self.qinv as i32);
        ((s - m as i64 * self.q as i64) >> 32) as i32
    }

    /// `d * a + c * b`, canonical.
    #[inline(always)]
    fn combine(&self, d: u32, c: u32, a: i32, b: i32) -> i32 {
        let q = self.q;
        // The sum lies in (-2 q, 2 q): shifted by 2 q it lies in (0, 4 q).
        let r = (self.mul(a, d).wrapping_add(self.mul(b, c)) as u32).wrapping_add(2 * q);
        let r = r.min(r.wrapping_sub(2 * q));
        r.min(r.wrapping_sub(q)) as i32
    }

    /// Applies the basis change to one plane, in place.
    ///
    /// Inputs are signed residues of magnitude below `2 * q`, outputs are canonical, coefficient `0` excepted, which is kept.
    fn apply(&self, a: &mut [i32]) {
        let n = a.len();
        let half = n / 2;
        let (lo, hi) = a.split_at_mut(half);
        // The middle coefficient is its own mirror.
        let mid = hi[0];
        hi[0] = self.combine(self.d[half], self.c[half], mid, mid);
        // Coefficient `j` against its mirror `n - j`, both read before either is written.
        let (d_lo, d_hi) = self.d.split_at(half);
        let (c_lo, c_hi) = self.c.split_at(half);
        let lo = lo[1..].iter_mut().zip(&d_lo[1..]).zip(&c_lo[1..]);
        let hi = hi[1..].iter_mut().zip(&d_hi[1..]).zip(&c_hi[1..]).rev();
        for (((x, &dx), &cx), ((y, &dy), &cy)) in lo.zip(hi) {
            let (xv, yv) = (*x, *y);
            *x = self.combine(dx, cx, xv, yv);
            *y = self.combine(dy, cy, yv, xv);
        }
    }
}

/// `y * w mod q` for any `y`, of magnitude at most `q / 2 + |y| * q / 2^32`: within `[-q, q]`.
#[inline(always)]
fn mul_w(y: i32, w: i32, wp: i32, q: i32) -> i32 {
    let t = ((y as i64 * wp as i64 + (1 << 30)) >> 31) as i32;
    y.wrapping_mul(w).wrapping_sub(t.wrapping_mul(q))
}

/// Largest value [`reduce`] accepts: its rounding offset must not overflow.
const REDUCE_MAX: i32 = i32::MAX - (1 << 29);

/// Maps a value of at most [`REDUCE_MAX`], about `1.52 * q`, to a congruent one of magnitude below `0.55 * q`.
#[inline(always)]
fn reduce(x: i32, q: i32) -> i32 {
    debug_assert!(x <= REDUCE_MAX);
    x.wrapping_sub((x.wrapping_add(1 << 29) >> 30).wrapping_mul(q))
}

/// Largest value [`reduce_sum`] accepts, about `1.96 * q`.
const REDUCE_SUM_MAX: i32 = i32::MAX - (1 << 26);

/// Maps a value in `[-2^31, REDUCE_SUM_MAX]` to a congruent one in `(-0.1 * q, 0.97 * q)`.
///
/// The quotient is rounded with a small offset, so that sums of two inverse butterfly inputs do not overflow it.
#[inline(always)]
fn reduce_sum(x: i32, q: i32) -> i32 {
    debug_assert!(x <= REDUCE_SUM_MAX);
    x.wrapping_sub((x.wrapping_add(1 << 26) >> 30).wrapping_mul(q))
}

/// Maps `[-q, q]` to `[0, q)`.
#[inline(always)]
fn canonical(x: i32, q: i32) -> i32 {
    let (x, q) = (x as u32, q as u32);
    let x = x.min(x.wrapping_add(q));
    x.min(x.wrapping_sub(q)) as i32
}

/// Cooley-Tukey butterfly: `x` and `y` of magnitude below `1.43 * q`, and so are the outputs.
///
/// The reduced `x` is below `0.55 * q` and the product below `0.5 * q + 1.43 * 0.246 * q`.
#[inline(always)]
fn fwd_butterfly(x: i32, y: i32, w: i32, wp: i32, q: i32) -> (i32, i32) {
    let x = reduce(x, q);
    let u = mul_w(y, w, wp, q);
    (x.wrapping_add(u), x.wrapping_sub(u))
}

/// Gentleman-Sande butterfly: inputs and outputs in `(-0.98 * q, 0.98 * q)`.
///
/// The sum is below `1.96 * q`, and the product below `0.5 * q + 1.96 * 0.246 * q`.
#[inline(always)]
fn inv_butterfly(x: i32, y: i32, w: i32, wp: i32, q: i32) -> (i32, i32) {
    (reduce_sum(x.wrapping_add(y), q), mul_w(x.wrapping_sub(y), w, wp, q))
}

/// Gentleman-Sande butterfly of the first level: canonical inputs, outputs in `(-0.98 * q, 0.98 * q)`.
#[inline(always)]
fn inv_butterfly_first(x: i32, y: i32, w: i32, wp: i32, q: i32) -> (i32, i32) {
    (
        reduce_sum(x.wrapping_add(y).wrapping_sub(q), q),
        mul_w(x.wrapping_sub(y), w, wp, q),
    )
}

/// Forward butterflies between two halves of a block that share one twiddle.
///
/// The halves come in as two exclusive slices of a function that is not inlined: this is what tells a compiler that
/// they do not overlap, so that it emits vector code without a run-time check that fails as soon as a level has two blocks.
#[inline(never)]
fn fwd_block(x: &mut [i32], y: &mut [i32], w: i32, wp: i32, q: i32) {
    for (x, y) in x.iter_mut().zip(y.iter_mut()) {
        (*x, *y) = fwd_butterfly(*x, *y, w, wp, q);
    }
}

/// One forward level over `a` for blocks of `2 * T` elements, `T` small: each block is loaded, combined and stored as a unit.
#[inline(never)]
fn fwd_level_small<const T: usize>(a: &mut [i32], w: &[i32], wp: &[i32], q: i32) {
    for ((block, &w), &wp) in a.chunks_exact_mut(2 * T).zip(w).zip(wp) {
        let (x, y) = block.split_at_mut(T);
        let (mut xv, mut yv) = ([0i32; T], [0i32; T]);
        xv.copy_from_slice(x);
        yv.copy_from_slice(y);
        for l in 0..T {
            (xv[l], yv[l]) = fwd_butterfly(xv[l], yv[l], w, wp, q);
        }
        x.copy_from_slice(&xv);
        y.copy_from_slice(&yv);
    }
}

/// One forward level over `a`: blocks of `2 * t` elements, one twiddle per block, `t >= 4`.
#[inline(always)]
fn fwd_level(a: &mut [i32], t: usize, w: &[i32], wp: &[i32], q: i32) {
    match t {
        4 => fwd_level_small::<4>(a, w, wp, q),
        8 => fwd_level_small::<8>(a, w, wp, q),
        _ => {
            for ((block, &w), &wp) in a.chunks_exact_mut(2 * t).zip(w).zip(wp) {
                let (x, y) = block.split_at_mut(t);
                fwd_block(x, y, w, wp, q);
            }
        }
    }
}

/// The forward level with blocks of four elements.
#[inline(never)]
fn fwd_level_2(a: &mut [i32], w: &[i32], wp: &[i32], q: i32) {
    for ((c, &w), &wp) in a.chunks_exact_mut(4).zip(w).zip(wp) {
        let (x0, y0) = fwd_butterfly(c[0], c[2], w, wp, q);
        let (x1, y1) = fwd_butterfly(c[1], c[3], w, wp, q);
        c[0] = x0;
        c[1] = x1;
        c[2] = y0;
        c[3] = y1;
    }
}

/// The last forward level, blocks of two elements, with canonical outputs.
#[inline(never)]
fn fwd_level_1(a: &mut [i32], w: &[i32], wp: &[i32], q: i32) {
    for ((c, &w), &wp) in a.chunks_exact_mut(2).zip(w).zip(wp) {
        let (x, y) = fwd_butterfly(c[0], c[1], w, wp, q);
        // One more reduction brings the outputs within q.
        c[0] = canonical(reduce(x, q), q);
        c[1] = canonical(reduce(y, q), q);
    }
}

/// Forward transform of one plane.
///
/// Inputs are signed values of magnitude below `1.43 * q`, outputs are canonical.
fn fwd_plane(a: &mut [i32], tab: &PlaneTable, q: i32) {
    let n = a.len();
    // Levels that span more than one block run over the whole plane.
    let mut m = 1;
    let mut t = n / 2;
    while t >= 4 && 2 * t > BLOCK {
        let (w, wp) = tab.level(m, 0, m);
        fwd_level(a, t, w, wp, q);
        m *= 2;
        t /= 2;
    }
    // Each block then runs all of its remaining levels.
    let (m0, size) = (m, 2 * t);
    for (b, block) in a.chunks_exact_mut(size).enumerate() {
        let (mut m, mut t, mut count) = (m0, size / 2, 1);
        while t >= 4 {
            let (w, wp) = tab.level(m, b * count, count);
            fwd_level(block, t, w, wp, q);
            m *= 2;
            t /= 2;
            count *= 2;
        }
        let (w, wp) = tab.level(n / 4, b * size / 4, size / 4);
        fwd_level_2(block, w, wp, q);
        let (w, wp) = tab.level(n / 2, b * size / 2, size / 2);
        fwd_level_1(block, w, wp, q);
    }
}

/// Inverse butterflies between two halves of a block that share one twiddle, see [`fwd_block`].
#[inline(never)]
fn inv_block(x: &mut [i32], y: &mut [i32], w: i32, wp: i32, q: i32) {
    for (x, y) in x.iter_mut().zip(y.iter_mut()) {
        (*x, *y) = inv_butterfly(*x, *y, w, wp, q);
    }
}

/// One inverse level over `a` for blocks of `2 * T` elements, `T` small, see [`fwd_level_small`].
#[inline(never)]
fn inv_level_small<const T: usize>(a: &mut [i32], w: &[i32], wp: &[i32], q: i32) {
    for ((block, &w), &wp) in a.chunks_exact_mut(2 * T).zip(w).zip(wp) {
        let (x, y) = block.split_at_mut(T);
        let (mut xv, mut yv) = ([0i32; T], [0i32; T]);
        xv.copy_from_slice(x);
        yv.copy_from_slice(y);
        for l in 0..T {
            (xv[l], yv[l]) = inv_butterfly(xv[l], yv[l], w, wp, q);
        }
        x.copy_from_slice(&xv);
        y.copy_from_slice(&yv);
    }
}

/// One inverse level over `a`: blocks of `2 * t` elements, one twiddle per block, `t >= 4`.
#[inline(always)]
fn inv_level(a: &mut [i32], t: usize, w: &[i32], wp: &[i32], q: i32) {
    match t {
        4 => inv_level_small::<4>(a, w, wp, q),
        8 => inv_level_small::<8>(a, w, wp, q),
        _ => {
            for ((block, &w), &wp) in a.chunks_exact_mut(2 * t).zip(w).zip(wp) {
                let (x, y) = block.split_at_mut(t);
                inv_block(x, y, w, wp, q);
            }
        }
    }
}

/// The last inverse level between the two halves of a plane, with the scaling folded into both outputs.
#[inline(never)]
fn inv_last(x: &mut [i32], y: &mut [i32], fin: &FinalConst, q: i32) {
    for (x, y) in x.iter_mut().zip(y.iter_mut()) {
        let (u, v) = (x.wrapping_add(*y), x.wrapping_sub(*y));
        *x = canonical(mul_w(u, fin.c, fin.cp, q), q);
        *y = canonical(mul_w(v, fin.cw, fin.cwp, q), q);
    }
}

/// The first inverse level, blocks of two elements, in place.
#[inline(never)]
fn inv_level_1(a: &mut [i32], w: &[i32], wp: &[i32], q: i32) {
    for ((c, &w), &wp) in a.chunks_exact_mut(2).zip(w).zip(wp) {
        (c[0], c[1]) = inv_butterfly_first(c[0], c[1], w, wp, q);
    }
}

/// The first inverse level, read from `src` and written to `dst`.
#[inline(never)]
fn inv_level_1_from(dst: &mut [i32], src: &[i32], w: &[i32], wp: &[i32], q: i32) {
    for (((d, s), &w), &wp) in dst.chunks_exact_mut(2).zip(src.chunks_exact(2)).zip(w).zip(wp) {
        (d[0], d[1]) = inv_butterfly_first(s[0], s[1], w, wp, q);
    }
}

/// The inverse level with blocks of four elements.
#[inline(never)]
fn inv_level_2(a: &mut [i32], w: &[i32], wp: &[i32], q: i32) {
    for ((c, &w), &wp) in a.chunks_exact_mut(4).zip(w).zip(wp) {
        let (x0, y0) = inv_butterfly(c[0], c[2], w, wp, q);
        let (x1, y1) = inv_butterfly(c[1], c[3], w, wp, q);
        c[0] = x0;
        c[1] = x1;
        c[2] = y0;
        c[3] = y1;
    }
}

/// Inverse transform of one plane, scaled by `crt / n`.
///
/// Reads canonical residues from `src`, or from `a` itself when `src` is `None`, and leaves canonical residues in `a`.
fn inv_plane(a: &mut [i32], src: Option<&[i32]>, tab: &PlaneTable, fin: &FinalConst, q: i32) {
    let n = a.len();
    let half = n / 2;
    // Each block runs its levels up to its own size, the last level of the plane excepted.
    let size = n.min(BLOCK);
    for (b, block) in a.chunks_exact_mut(size).enumerate() {
        let (w, wp) = tab.level(n / 2, b * size / 2, size / 2);
        match src {
            Some(src) => inv_level_1_from(block, &src[b * size..(b + 1) * size], w, wp, q),
            None => inv_level_1(block, w, wp, q),
        }
        let (w, wp) = tab.level(n / 4, b * size / 4, size / 4);
        inv_level_2(block, w, wp, q);
        let mut t = 4;
        while 2 * t <= size && t < half {
            let count = size / (2 * t);
            let (w, wp) = tab.level(n / (2 * t), b * count, count);
            inv_level(block, t, w, wp, q);
            t *= 2;
        }
    }
    // Levels that span more than one block run over the whole plane.
    let mut t = size;
    while t < half {
        let m = n / (2 * t);
        let (w, wp) = tab.level(m, 0, m);
        inv_level(a, t, w, wp, q);
        t *= 2;
    }
    let (x, y) = a.split_at_mut(half);
    inv_last(x, y, fin, q);
}

/// Reduces signed coefficients into one plane of signed residues of magnitude below `1.1 * q`.
///
/// The coefficient is `hi * 2^32 + lo`, with `lo` read as a signed word whose sign bit is carried into `hi`.
fn from_i64(dst: &mut [i32], src: &[i64], c: &ConvConst, q: i32) {
    for (d, &x) in dst.iter_mut().zip(src) {
        let (lo, hi) = (x as i32, (x >> 32) as i32);
        let u = mul_w(hi, c.k, c.kp, q);
        let v = mul_w(lo, c.s, c.sp, q).wrapping_add(if lo < 0 { c.kc } else { 0 });
        *d = reduce(u, q).wrapping_add(reduce(v, q));
    }
}

/// Forward transform of plane `p` of a packed limb from `n` coefficients.
///
/// `dst` is the plane itself, `n` words.
/// The residues are canonical, multiplied by `2^32` when `prepared` is set.
fn ntt32_plane(table: &Ntt32Table, p: usize, dst: &mut [u32], src: &[i64], prepared: bool) {
    let n = table.n;
    let dst: &mut [i32] = cast_slice_mut(&mut dst[..n]);
    let q = Q[p] as i32;
    from_i64(dst, &src[..n], &table.conv[prepared as usize][p], q);
    if let Some(ci) = &table.ci {
        ci[0][p].apply(dst);
    }
    fwd_plane(dst, &table.fwd[p], q);
}

/// Ring degree from which the four planes of a limb are transformed as separate tasks of a parallel executor.
///
/// A ciphertext has few limbs at a large `base2k`, fewer than a pool has threads.
const PLANE_TASKS_MIN_N: usize = 1 << 13;
const _: () = assert!((PLANE_TASKS_MIN_N / 4).is_multiple_of(CRT_RUN));

#[inline]
fn plane_tasks<E: TaskExecutor>(n: usize) -> bool {
    E::is_parallel() && n >= PLANE_TASKS_MIN_N && E::max_parallelism() > 1
}

/// Runs `task` on the four items, as four tasks of `E` when `parallel` is set and in order otherwise.
#[inline]
fn for_each_of_four<E: TaskExecutor, T: Send>(parallel: bool, items: [T; 4], task: impl Fn(usize, T) + Sync) {
    let [a, b, c, d] = items;
    if parallel {
        E::join(
            || E::join(|| task(0, a), || task(1, b)),
            || E::join(|| task(2, c), || task(3, d)),
        );
    } else {
        task(0, a);
        task(1, b);
        task(2, c);
        task(3, d);
    }
}

/// The four planes of a packed limb.
#[inline]
fn planes_mut(n: usize, limb: &mut [u32]) -> [&mut [u32]; 4] {
    let (p0, rest) = limb[..4 * n].split_at_mut(n);
    let (p1, rest) = rest.split_at_mut(n);
    let (p2, p3) = rest.split_at_mut(n);
    [p0, p1, p2, p3]
}

/// Forward transform of `n` coefficients into one packed limb.
///
/// The residues are canonical, multiplied by `2^32` when `prepared` is set.
pub(crate) fn ntt32<E: TaskExecutor>(table: &Ntt32Table, dst: &mut [u32], src: &[i64], prepared: bool) {
    let n = table.n;
    for_each_of_four::<E, _>(plane_tasks::<E>(n), planes_mut(n, dst), |p, plane| {
        ntt32_plane(table, p, plane, src, prepared)
    });
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

/// Coefficients reconstructed together.
const CRT_RUN: usize = 8;

/// CRT reconstruction of the coefficients from four planes of canonical residues already multiplied by the CRT constants.
///
/// The sum `floor(Q / 2) + sum_p t[p] * (Q / Q[p])` is accumulated limb by limb, on a run of coefficients at a time.
/// Each limb sum stays below `2^63`.
/// Reducing it modulo `Q` and removing the offset gives the representative in `[-floor(Q / 2), ceil(Q / 2))`.
fn crt(dst: &mut [i128], [t0, t1, t2, t3]: [&[u32]; 4]) {
    let planes = t0
        .chunks_exact(CRT_RUN)
        .zip(t1.chunks_exact(CRT_RUN))
        .zip(t2.chunks_exact(CRT_RUN))
        .zip(t3.chunks_exact(CRT_RUN));
    for (dst, (((t0, t1), t2), t3)) in dst.chunks_exact_mut(CRT_RUN).zip(planes) {
        let mut sums = [[0u64; CRT_RUN]; 3];
        for (j, sum) in sums.iter_mut().enumerate() {
            for i in 0..CRT_RUN {
                sum[i] = HALF_Q_LIMBS[j]
                    + t0[i] as u64 * QM_LIMBS[0][j] as u64
                    + t1[i] as u64 * QM_LIMBS[1][j] as u64
                    + t2[i] as u64 * QM_LIMBS[2][j] as u64
                    + t3[i] as u64 * QM_LIMBS[3][j] as u64;
            }
        }
        for (i, d) in dst.iter_mut().enumerate() {
            // The sum is below 4.5 Q and Q is between 2^119 and 2^120: at most one multiple of Q remains after the table.
            let mut v = sums[0][i] as u128 + ((sums[1][i] as u128) << 30) + ((sums[2][i] as u128) << 60);
            v -= TOTAL_Q_MULT[(v >> 120) as usize];
            let w = v.wrapping_sub(TOTAL_Q);
            v = if v >= TOTAL_Q { w } else { v };
            *d = v as i128 - HALF_Q as i128;
        }
    }
}

/// Inverse transform of plane `p` up to the reconstruction, from `src` when given, in place otherwise.
fn intt32_plane(table: &Ntt32Table, p: usize, work: &mut [u32], src: Option<&[u32]>) {
    let n = table.n;
    let work: &mut [i32] = cast_slice_mut(&mut work[..n]);
    inv_plane(
        work,
        src.map(|src| cast_slice(&src[..n])),
        &table.inv[p],
        &table.fin[p],
        Q[p] as i32,
    );
    if let Some(ci) = &table.ci {
        ci[1][p].apply(work);
    }
}

/// Reconstruction of the coefficients from the four planes of `work`, by quarters when `parallel` is set.
fn crt_planes<E: TaskExecutor>(parallel: bool, n: usize, dst: &mut [i128], work: &[u32]) {
    if !parallel {
        return crt(&mut dst[..n], std::array::from_fn(|p| &work[p * n..(p + 1) * n]));
    }
    // A quarter is a whole number of reconstruction runs at the degrees that split.
    let quarter = n / 4;
    let (d0, rest) = dst[..n].split_at_mut(quarter);
    let (d1, rest) = rest.split_at_mut(quarter);
    let (d2, d3) = rest.split_at_mut(quarter);
    for_each_of_four::<E, _>(parallel, [d0, d1, d2, d3], |k, dst| {
        let range = k * quarter..(k + 1) * quarter;
        crt(dst, std::array::from_fn(|p| &work[p * n..][range.clone()]));
    });
}

/// Inverse transform of the packed limb `src` into `n` coefficients, with the intermediate planes in `work`.
pub(crate) fn intt32<E: TaskExecutor>(table: &Ntt32Table, dst: &mut [i128], src: &[u32], work: &mut [u32]) {
    let n = table.n;
    let parallel = plane_tasks::<E>(n);
    for_each_of_four::<E, _>(parallel, planes_mut(n, work), |p, work| {
        intt32_plane(table, p, work, Some(&src[p * n..(p + 1) * n]))
    });
    crt_planes::<E>(parallel, n, dst, work);
}

/// Inverse transform of the packed limb `work` into `n` coefficients, overwriting the limb with its intermediate planes.
pub(crate) fn intt32_assign<E: TaskExecutor>(table: &Ntt32Table, dst: &mut [i128], work: &mut [u32]) {
    let n = table.n;
    let parallel = plane_tasks::<E>(n);
    for_each_of_four::<E, _>(parallel, planes_mut(n, work), |p, work| intt32_plane(table, p, work, None));
    crt_planes::<E>(parallel, n, dst, work);
}

/// Inverse transform of the packed limb `slot` into `n` coefficients that overwrite it.
///
/// The planes move to `work` during the transform, and the coefficients take the 16 bytes per coefficient of the limb.
pub(crate) fn intt32_compact<E: TaskExecutor>(table: &Ntt32Table, slot: &mut [u32], work: &mut [u32]) {
    let n = table.n;
    let parallel = plane_tasks::<E>(n);
    for_each_of_four::<E, _>(parallel, planes_mut(n, work), |p, work| {
        intt32_plane(table, p, work, Some(&slot[p * n..(p + 1) * n]))
    });
    crt_planes::<E>(parallel, n, cast_slice_mut(&mut slot[..4 * n]), work);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::ntt4x30::{
        NttDFTExecute, NttFromZnx64, NttToZnx128,
        ntt::{NttTable, NttTableInv},
        vec_znx_dft::{NttPlan, NttPlanNew},
    };
    use crate::{NTT4x30CIPortable, NTT4x30Portable};
    use poulpy_hal::layouts::{ConjugateInvariant, Ring, Standard};
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    const LOG_N: [usize; 9] = [3, 4, 5, 6, 10, 13, 14, 15, 16];

    fn source(n: usize, rng: &mut StdRng) -> Vec<i64> {
        let mut src: Vec<i64> = (0..n).map(|_| rng.random()).collect();
        src[0] = i64::MAX;
        src[1] = i64::MIN;
        src[2] = -1;
        src[3] = (1 << 32) - 1;
        src[4] = -(1 << 31);
        src
    }

    /// The packed transforms against the q120 kernels, residue by residue.
    fn matches_q120<R: Ring, BE>(conjugate_invariant: bool)
    where
        BE: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>> + NttFromZnx64 + NttToZnx128,
        NttPlan<Primes30, R>: NttPlanNew,
    {
        let mut rng = StdRng::seed_from_u64(3);
        for log_n in LOG_N {
            let n = 1 << log_n;
            let table = Ntt32Table::new(n, conjugate_invariant);
            assert_eq!(table.n(), n);
            let plan = <NttPlan<Primes30, R> as NttPlanNew>::new(n);
            let (fwd, inv) = (plan.ntt(), plan.intt());
            let src = source(n, &mut rng);
            let mut wide = vec![0u64; 4 * n];
            BE::ntt_from_znx64(&mut wide, &src);
            BE::ntt_dft_execute(fwd, &mut wide);
            for prepared in [false, true] {
                let mut packed = vec![0u32; 4 * n];
                ntt32::<poulpy_hal::execution::SerialTaskExecutor>(&table, &mut packed, &src, prepared);
                for p in 0..4 {
                    let q = Q[p] as u64;
                    let scale = if prepared { R1[p] as u64 } else { 1 };
                    for i in 0..n {
                        assert_eq!(
                            packed[p * n + i] as u64,
                            wide[4 * i + p] % q * scale % q,
                            "log_n={log_n} prepared={prepared} p={p} i={i}"
                        );
                    }
                }
            }
            // Inverse, on canonical planes with the extreme residues mixed in.
            let mut planes = vec![0u32; 4 * n];
            for p in 0..4 {
                for i in 0..n {
                    planes[p * n + i] = match i % 7 {
                        0 => 0,
                        1 => Q[p] - 1,
                        _ => rng.random_range(0..Q[p]),
                    };
                }
            }
            for (i, w) in wide.chunks_exact_mut(4).enumerate() {
                for p in 0..4 {
                    w[p] = planes[p * n + i] as u64;
                }
            }
            BE::ntt_dft_execute(inv, &mut wide);
            let mut want = vec![0i128; n];
            BE::ntt_to_znx128(&mut want, n, &wide);
            let mut got = vec![0i128; n];
            let mut work = vec![0u32; 4 * n];
            intt32::<poulpy_hal::execution::SerialTaskExecutor>(&table, &mut got, &planes, &mut work);
            assert_eq!(got, want, "log_n={log_n} inverse");
            let mut slot = poulpy_hal::alloc_aligned::<u32>(4 * n);
            slot.copy_from_slice(&planes);
            intt32_compact::<poulpy_hal::execution::SerialTaskExecutor>(&table, &mut slot, &mut work);
            assert_eq!(cast_slice::<u32, i128>(&slot), &want[..], "log_n={log_n} inverse compacted");
            let mut got = vec![0i128; n];
            intt32_assign::<poulpy_hal::execution::SerialTaskExecutor>(&table, &mut got, &mut planes);
            assert_eq!(got, want, "log_n={log_n} inverse in place");
        }
    }

    #[test]
    fn matches_q120_standard() {
        matches_q120::<Standard, NTT4x30Portable>(false);
    }

    #[test]
    fn matches_q120_conjugate_invariant() {
        matches_q120::<ConjugateInvariant, NTT4x30CIPortable>(true);
    }
}
