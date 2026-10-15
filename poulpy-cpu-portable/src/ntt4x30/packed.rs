//! Scalar kernels for the packed NTT4x30 transform domain.
//!
//! A transformed limb is four planes of `n` canonical `u32` residues, plane `p` holding the residues modulo `Q[p]`.
//! The loops run over one plane, or over rows of four lanes per prime, so that a compiler can map them to vector instructions.
//!
//! Prepared operands (`SvpPPol`, `VmpPMat` and the right convolution operand) store their residues multiplied by `2^32`.
//! A product against a prepared operand then reduces with a single Montgomery step and lands back in the plain domain.

use crate::kernels::ntt4x30::primes::{PrimeSet, Primes30};

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

const fn build_pow(exp: u32) -> [u32; 4] {
    [
        pow2_mod(exp, Q[0]),
        pow2_mod(exp, Q[1]),
        pow2_mod(exp, Q[2]),
        pow2_mod(exp, Q[3]),
    ]
}

/// `Q[p]^-1 mod 2^32`.
pub(crate) const QINV: [u32; 4] = [inv_mod_2_32(Q[0]), inv_mod_2_32(Q[1]), inv_mod_2_32(Q[2]), inv_mod_2_32(Q[3])];
/// `2^32 mod Q[p]`.
pub(crate) const R1: [u32; 4] = build_pow(32);
/// `2^64 mod Q[p]`.
pub(crate) const R2: [u32; 4] = build_pow(64);

/// `u32` per row of a block: four lanes for each of the four primes.
pub(crate) const ROW: usize = 16;

const fn per_lane(v: [u32; 4]) -> [u32; ROW] {
    let mut out = [0u32; ROW];
    let mut i = 0;
    while i < ROW {
        out[i] = v[i / 4];
        i += 1;
    }
    out
}

/// The prime of each lane of a row.
const ROW_Q: [u32; ROW] = per_lane(Q);
const ROW_QINV: [u32; ROW] = per_lane(QINV);

/// Largest number of products of two centered residues that one `i64` accumulator holds before its Montgomery step.
///
/// A centered residue has magnitude at most `(q - 1) / 2`, so the sum `S` has magnitude at most `6 * (q - 1)^2`.
/// The step returns `(S - m * q) / 2^32` with `|m| <= 2^31`, of magnitude below `2 * q`, which fits a signed word.
pub(crate) const DOT_CHUNK: usize = 24;

/// Largest number of products for which the Montgomery step of an accumulator has magnitude below `q`.
pub(crate) const DOT_SHORT: usize = 8;

/// Maps `[0, 2q)` to `[0, q)`.
#[inline(always)]
pub(crate) fn cond_sub(x: u32, q: u32) -> u32 {
    x.min(x.wrapping_sub(q))
}

/// `a + b mod q` for canonical inputs.
#[inline(always)]
pub(crate) fn add_mod(a: u32, b: u32, q: u32) -> u32 {
    cond_sub(a + b, q)
}

/// `a - b mod q` for canonical inputs.
#[inline(always)]
pub(crate) fn sub_mod(a: u32, b: u32, q: u32) -> u32 {
    let d = a.wrapping_sub(b);
    d.min(d.wrapping_add(q))
}

/// `-a mod q` for a canonical input.
#[inline(always)]
pub(crate) fn neg_mod(a: u32, q: u32) -> u32 {
    cond_sub(q - a, q)
}

/// `a * b * 2^-32 mod q`, canonical, for `a * b < q * 2^32`.
///
/// Montgomery step: `a * b - m * q` is a multiple of `2^32`, so its quotient is the difference of the two high words.
#[inline(always)]
pub(crate) fn mont_mul(a: u32, b: u32, q: u32, qinv: u32) -> u32 {
    let p = a as u64 * b as u64;
    let m = (p as u32).wrapping_mul(qinv);
    let t = ((m as u64 * q as u64) >> 32) as u32;
    let r = ((p >> 32) as u32).wrapping_sub(t);
    r.min(r.wrapping_add(q))
}

/// Maps a canonical residue to its centered representative, of magnitude at most `(q - 1) / 2`, as a two's complement word.
#[inline(always)]
pub(crate) fn center(x: u32, q: u32) -> u32 {
    x.wrapping_sub(if x > (q - 1) / 2 { q } else { 0 })
}

/// Montgomery step of a signed sum of at most [`DOT_CHUNK`] products of centered residues, or [`DOT_SHORT`] when `short` is set.
///
/// Returns `S * 2^-32 mod q`, canonical.
#[inline(always)]
fn redc_acc(s: i64, q: u32, qinv: u32, short: bool) -> u32 {
    let m = (s as i32).wrapping_mul(qinv as i32);
    // S - m * q is a multiple of 2^32.
    let r = ((s - m as i64 * q as i64) >> 32) as u32;
    if short {
        // |r| < q.
        r.min(r.wrapping_add(q))
    } else {
        // |r| < 2 q: shifted by 2 q it lies in [0, 4 q).
        let r = r.wrapping_add(2 * q);
        let r = r.min(r.wrapping_sub(2 * q));
        r.min(r.wrapping_sub(q))
    }
}

/// Running inner product of centered rows, fed one run of rows at a time.
///
/// A row is [`ROW`] words: four lanes for each of the four primes.
/// The state is passed and returned by value so that it stays in registers.
#[derive(Clone, Copy)]
pub(crate) struct DotState {
    acc: [i64; ROW],
    out: [u32; ROW],
    /// Weight held by the accumulators, in products of two centered residues.
    terms: usize,
    flushed: bool,
}

impl DotState {
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Self {
            acc: [0; ROW],
            out: [0; ROW],
            terms: 0,
            flushed: false,
        }
    }

    #[inline(always)]
    fn flush(mut self) -> Self {
        let short = self.terms <= DOT_SHORT;
        for l in 0..ROW {
            let r = redc_acc(self.acc[l], ROW_Q[l], ROW_QINV[l], short);
            self.out[l] = if self.flushed { add_mod(self.out[l], r, ROW_Q[l]) } else { r };
        }
        self.acc = [0; ROW];
        self.terms = 0;
        self.flushed = true;
        self
    }

    /// Room for rows of weight `weight`: how many of at most `rows` rows the accumulators take, after a flush when they are full.
    #[inline(always)]
    fn reserve(mut self, rows: usize, weight: usize) -> (Self, usize) {
        if self.terms + weight > DOT_CHUNK {
            self = self.flush();
        }
        let take = rows.min((DOT_CHUNK - self.terms) / weight);
        self.terms += weight * take;
        (self, take)
    }

    /// Adds `sum_row x[row] * m[row]` over the rows of `x`.
    ///
    /// `x` holds centered residues and `m` centered prepared residues, with at least as many rows.
    #[inline(always)]
    pub(crate) fn push_rows(mut self, mut x: &[u32], mut m: &[u32]) -> Self {
        while !x.is_empty() {
            let (state, take) = self.reserve(x.len() / ROW, 1);
            self = state;
            let (xr, mr) = (&x[..take * ROW], &m[..take * ROW]);
            let mut acc = self.acc;
            for (xr, mr) in xr.chunks_exact(ROW).zip(mr.chunks_exact(ROW)) {
                for l in 0..ROW {
                    acc[l] += (xr[l] as i32 as i64) * (mr[l] as i32 as i64);
                }
            }
            self.acc = acc;
            x = &x[take * ROW..];
            m = &m[take * ROW..];
        }
        self
    }

    /// Adds the products of `rows` rows read through `row`, which returns the two centered operands of row `i`.
    ///
    /// A row of weight `weight` has operands of magnitude at most `sqrt(weight) * (q - 1) / 2`.
    #[inline(always)]
    pub(crate) fn push_with(
        mut self,
        rows: usize,
        weight: usize,
        mut row: impl FnMut(usize) -> ([u32; ROW], [u32; ROW]),
    ) -> Self {
        let mut i = 0;
        while i < rows {
            let (state, take) = self.reserve(rows - i, weight);
            self = state;
            let mut acc = self.acc;
            for i in i..i + take {
                let (x, m) = row(i);
                for l in 0..ROW {
                    acc[l] += (x[l] as i32 as i64) * (m[l] as i32 as i64);
                }
            }
            self.acc = acc;
            i += take;
        }
        self
    }

    /// The canonical residues of the sum, four lanes per prime.
    #[inline(always)]
    pub(crate) fn finish(self) -> [u32; ROW] {
        if self.terms != 0 { self.flush().out } else { self.out }
    }
}

/// The four planes of a packed limb of `n` coefficients, each with its prime.
#[inline(always)]
pub(crate) fn planes(n: usize, limb: &[u32]) -> impl Iterator<Item = (&[u32], u32)> {
    limb[..4 * n].chunks_exact(n).zip(Q)
}

/// As [`planes`], mutable.
#[inline(always)]
pub(crate) fn planes_mut(n: usize, limb: &mut [u32]) -> impl Iterator<Item = (&mut [u32], u32)> {
    limb[..4 * n].chunks_exact_mut(n).zip(Q)
}

/// `dst = a + b` on packed limbs of canonical residues.
pub(crate) fn limb_add(n: usize, dst: &mut [u32], a: &[u32], b: &[u32]) {
    for (((d, q), a), b) in planes_mut(n, dst)
        .zip(a[..4 * n].chunks_exact(n))
        .zip(b[..4 * n].chunks_exact(n))
    {
        for ((d, &a), &b) in d.iter_mut().zip(a).zip(b) {
            *d = add_mod(a, b, q);
        }
    }
}

/// `dst += a`.
pub(crate) fn limb_add_assign(n: usize, dst: &mut [u32], a: &[u32]) {
    for ((d, q), a) in planes_mut(n, dst).zip(a[..4 * n].chunks_exact(n)) {
        for (d, &a) in d.iter_mut().zip(a) {
            *d = add_mod(*d, a, q);
        }
    }
}

/// `dst = a - b`.
pub(crate) fn limb_sub(n: usize, dst: &mut [u32], a: &[u32], b: &[u32]) {
    for (((d, q), a), b) in planes_mut(n, dst)
        .zip(a[..4 * n].chunks_exact(n))
        .zip(b[..4 * n].chunks_exact(n))
    {
        for ((d, &a), &b) in d.iter_mut().zip(a).zip(b) {
            *d = sub_mod(a, b, q);
        }
    }
}

/// `dst -= a`.
pub(crate) fn limb_sub_assign(n: usize, dst: &mut [u32], a: &[u32]) {
    for ((d, q), a) in planes_mut(n, dst).zip(a[..4 * n].chunks_exact(n)) {
        for (d, &a) in d.iter_mut().zip(a) {
            *d = sub_mod(*d, a, q);
        }
    }
}

/// `dst = a - dst`.
pub(crate) fn limb_sub_negate_assign(n: usize, dst: &mut [u32], a: &[u32]) {
    for ((d, q), a) in planes_mut(n, dst).zip(a[..4 * n].chunks_exact(n)) {
        for (d, &a) in d.iter_mut().zip(a) {
            *d = sub_mod(a, *d, q);
        }
    }
}

/// `dst = -dst`.
pub(crate) fn limb_negate_assign(n: usize, dst: &mut [u32]) {
    for (d, q) in planes_mut(n, dst) {
        for d in d.iter_mut() {
            *d = neg_mod(*d, q);
        }
    }
}

/// `dst = a * b * 2^-32`, the product of a packed limb by a prepared one.
pub(crate) fn limb_mont_mul(n: usize, dst: &mut [u32], a: &[u32], b: &[u32]) {
    for ((((d, q), qinv), a), b) in planes_mut(n, dst)
        .zip(QINV)
        .zip(a[..4 * n].chunks_exact(n))
        .zip(b[..4 * n].chunks_exact(n))
    {
        for ((d, &a), &b) in d.iter_mut().zip(a).zip(b) {
            *d = mont_mul(a, b, q, qinv);
        }
    }
}

/// `dst *= b * 2^-32`.
pub(crate) fn limb_mont_mul_assign(n: usize, dst: &mut [u32], b: &[u32]) {
    for (((d, q), qinv), b) in planes_mut(n, dst).zip(QINV).zip(b[..4 * n].chunks_exact(n)) {
        for (d, &b) in d.iter_mut().zip(b) {
            *d = mont_mul(*d, b, q, qinv);
        }
    }
}

/// Multiplies one packed limb of canonical residues by `2^32`, in place.
pub(crate) fn limb_to_prepared(n: usize, data: &mut [u32]) {
    for (((d, q), qinv), r2) in planes_mut(n, data).zip(QINV).zip(R2) {
        for d in d.iter_mut() {
            *d = mont_mul(*d, r2, q, qinv);
        }
    }
}

/// Gathers `len` blocks of one packed limb, from block `blk`, as row `row` of each block, centered.
///
/// Block `b` of the group has its `rows` rows at `x[b * rows * ROW..]`.
#[inline(always)]
pub(crate) fn gather_limb(n: usize, blk: usize, len: usize, limb: &[u32], x: &mut [u32], row: usize, rows: usize) {
    for (p, (plane, q)) in planes(n, limb).enumerate() {
        let src = &plane[4 * blk..4 * (blk + len)];
        for (b, src) in src.chunks_exact(4).enumerate() {
            let dst = &mut x[(b * rows + row) * ROW + 4 * p..][..4];
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = center(s, q);
            }
        }
    }
}

/// Scatters one packed limb of canonical residues into the rows at `offset(blk)` of a block-major buffer, centered.
#[inline(always)]
pub(crate) fn scatter_centered_limb(n: usize, dst: &mut [u32], src: &[u32], offset: impl Fn(usize) -> usize) {
    for (p, (plane, q)) in planes(n, src).enumerate() {
        for (blk, src) in plane.chunks_exact(4).enumerate() {
            let row = &mut dst[offset(blk) + 4 * p..][..4];
            for (d, &s) in row.iter_mut().zip(src) {
                *d = center(s, q);
            }
        }
    }
}

/// Stores row `r`, four lanes per prime, as block `b` of the run of each plane of a stage.
///
/// Plane `p` holds its run of `run` blocks at `stage[p * run * 4..]`.
#[inline(always)]
pub(crate) fn stage_store(stage: &mut [u32], run: usize, b: usize, r: &[u32; ROW]) {
    for p in 0..4 {
        stage[(p * run + b) * 4..][..4].copy_from_slice(&r[4 * p..4 * p + 4]);
    }
}

/// Moves a staged output of `len` blocks to the blocks from `blk` of the packed limb at `dst`.
///
/// The limb receives the sum of its content and the stage when `add` is set, and is overwritten otherwise.
///
/// # Safety
/// `dst` addresses a packed limb of `4 * n` words, and no other access touches its blocks `blk..blk + len` meanwhile.
#[inline(always)]
pub(crate) unsafe fn stage_flush(n: usize, dst: *mut u32, stage: &[u32], run: usize, blk: usize, len: usize, add: bool) {
    for (p, &q) in Q.iter().enumerate() {
        // Tasks write distinct blocks, so each takes only its own range of the plane.
        let d = unsafe { std::slice::from_raw_parts_mut(dst.add(p * n + 4 * blk), 4 * len) };
        let s = &stage[p * run * 4..][..4 * len];
        if add {
            for (d, &s) in d.iter_mut().zip(s) {
                *d = add_mod(*d, s, q);
            }
        } else {
            d.copy_from_slice(s);
        }
    }
}

/// A pointer shared by tasks that write distinct ranges behind it.
#[derive(Clone, Copy)]
pub(crate) struct SendPtr<T>(pub(crate) *mut T);

unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    #[inline(always)]
    pub(crate) fn get(self) -> *mut T {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

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

    fn r1_inv(p: usize) -> u64 {
        pow_mod(R1[p] as u64, Q[p] as u64 - 2, Q[p] as u64)
    }

    #[test]
    fn constants() {
        for p in 0..4 {
            assert_eq!(Q[p].wrapping_mul(QINV[p]), 1);
            assert_eq!(R1[p] as u64, (1u64 << 32) % Q[p] as u64);
            assert_eq!(R2[p] as u128, (1u128 << 64) % Q[p] as u128);
            let half = ((Q[p] - 1) / 2) as i128;
            // A full chunk fits the accumulator, and its Montgomery step fits a signed word below 2 q.
            let s = DOT_CHUNK as i128 * half * half;
            assert!(s < i64::MAX as i128);
            assert!((s >> 32) + (Q[p] as i128) / 2 < 2 * Q[p] as i128);
            let s = DOT_SHORT as i128 * half * half;
            assert!((s >> 32) + (Q[p] as i128) / 2 < Q[p] as i128);
            // The largest operand sum of a pairwise row squares to four units.
            assert!((2 * half) * (2 * half) <= 4 * half * half);
        }
    }

    #[test]
    fn limb_ops_match_modular_arithmetic() {
        let n = 16;
        let mut rng = StdRng::seed_from_u64(1);
        let mut a = vec![0u32; 4 * n];
        let mut b = vec![0u32; 4 * n];
        for p in 0..4 {
            for i in 0..n {
                a[p * n + i] = match i {
                    0 => 0,
                    1 => Q[p] - 1,
                    _ => rng.random_range(0..Q[p]),
                };
                b[p * n + i] = match i {
                    0 | 2 => Q[p] - 1,
                    1 => 0,
                    _ => rng.random_range(0..Q[p]),
                };
            }
        }
        let check = |got: &[u32], want: &dyn Fn(usize, u64, u64) -> u64| {
            for p in 0..4 {
                for i in 0..n {
                    assert_eq!(got[p * n + i] as u64, want(p, a[p * n + i] as u64, b[p * n + i] as u64));
                }
            }
        };
        let mut dst = vec![0u32; 4 * n];
        limb_add(n, &mut dst, &a, &b);
        check(&dst, &|p, a, b| (a + b) % Q[p] as u64);
        limb_sub(n, &mut dst, &a, &b);
        check(&dst, &|p, a, b| (a + Q[p] as u64 - b) % Q[p] as u64);
        limb_mont_mul(n, &mut dst, &a, &b);
        check(&dst, &|p, a, b| a * b % Q[p] as u64 * r1_inv(p) % Q[p] as u64);
        dst.copy_from_slice(&a);
        limb_add_assign(n, &mut dst, &b);
        check(&dst, &|p, a, b| (a + b) % Q[p] as u64);
        dst.copy_from_slice(&a);
        limb_sub_assign(n, &mut dst, &b);
        check(&dst, &|p, a, b| (a + Q[p] as u64 - b) % Q[p] as u64);
        dst.copy_from_slice(&a);
        limb_sub_negate_assign(n, &mut dst, &b);
        check(&dst, &|p, a, b| (b + Q[p] as u64 - a) % Q[p] as u64);
        dst.copy_from_slice(&a);
        limb_negate_assign(n, &mut dst);
        check(&dst, &|p, a, _| (Q[p] as u64 - a) % Q[p] as u64);
        dst.copy_from_slice(&a);
        limb_mont_mul_assign(n, &mut dst, &b);
        check(&dst, &|p, a, b| a * b % Q[p] as u64 * r1_inv(p) % Q[p] as u64);
        dst.copy_from_slice(&a);
        limb_to_prepared(n, &mut dst);
        check(&dst, &|p, a, _| a * R1[p] as u64 % Q[p] as u64);
    }

    /// Inner products against a `u128` modular sum, across the flush boundaries and at the extreme sums.
    #[test]
    fn dot_rows_match_modular_sum() {
        let mut rng = StdRng::seed_from_u64(2);
        for rows in [1usize, 2, 8, 9, 23, 24, 25, 48, 49, 60] {
            for case in 0..4 {
                let mut x = vec![0u32; rows * ROW];
                let mut m = vec![0u32; rows * ROW];
                for i in 0..rows * ROW {
                    let q = ROW_Q[i % ROW];
                    let half = (q - 1) / 2;
                    let (xv, mv) = match case {
                        0 => (rng.random_range(0..q), rng.random_range(0..q)),
                        // Largest positive sum, largest negative sum, and alternating signs.
                        1 => (half, half),
                        2 => (half, half + 1),
                        _ => (half + (i / ROW % 2) as u32, half),
                    };
                    x[i] = center(xv, q);
                    m[i] = center(mv, q);
                }
                let want: Vec<u32> = (0..ROW)
                    .map(|l| {
                        let p = l / 4;
                        let q = Q[p] as i128;
                        let s: i128 = (0..rows)
                            .map(|r| (x[r * ROW + l] as i32 as i128) * (m[r * ROW + l] as i32 as i128))
                            .sum();
                        (s.rem_euclid(q) * r1_inv(p) as i128 % q) as u32
                    })
                    .collect();
                assert_eq!(
                    DotState::new().push_rows(&x, &m).finish().to_vec(),
                    want,
                    "rows={rows} case={case}"
                );
                // The same sum fed in two runs, and through the row closure.
                let cut = rows / 2 * ROW;
                let split = DotState::new()
                    .push_rows(&x[..cut], &m[..cut])
                    .push_rows(&x[cut..], &m[cut..])
                    .finish();
                assert_eq!(split.to_vec(), want, "rows={rows} case={case} split");
                let with = DotState::new()
                    .push_with(rows, 1, |r| {
                        (
                            x[r * ROW..(r + 1) * ROW].try_into().unwrap(),
                            m[r * ROW..(r + 1) * ROW].try_into().unwrap(),
                        )
                    })
                    .finish();
                assert_eq!(with.to_vec(), want, "rows={rows} case={case} closure");
            }
        }
    }
}
