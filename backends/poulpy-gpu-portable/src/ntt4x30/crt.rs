//! From four residues to the centered big value, and the tail of the inverse
//! transform that writes it.
//!
//! Garner's mixed-radix digits `v = x0 + x1 q0 + x2 q0 q1 + x3 q0 q1 q2` are
//! computed with 32-bit Shoup products, accumulated in two 64-bit words, and
//! centered to `(-Q/2, Q/2]` as a two's complement 128-bit value: the
//! representative `NTT4x30Portable` produces.

use cubecl::prelude::*;

use super::{DftColumn, primes::Consts, tables::Tables};
use crate::{
    runtime::{GpuRuntime, Label, launch},
    znx::Column,
};

/// `x * w mod q` in `[0, 2q)` for any 32-bit `x`, `ws = floor(w * 2^32 / q)`.
#[cube]
fn shoup_lazy(x: u32, #[comptime] w: u32, #[comptime] ws: u32, #[comptime] q: u32) -> u32 {
    let qh = x.mul_hi(ws);
    x * w - qh * q
}

#[cube]
fn csub(x: u32, #[comptime] q: u32) -> u32 {
    let mut r = x;
    if r >= q {
        r -= q;
    }
    r
}

/// Mixed-radix digits `x1, x2, x3` (`x0 = r0`). Every left operand is
/// `r + 2q - x` with `r < 2q` and `x < 2q`, in `(0, 4q)`.
#[cube]
fn garner(r0: u32, r1: u32, r2: u32, r3: u32, #[comptime] c: Consts) -> (u32, u32, u32) {
    let x1 = csub(shoup_lazy(r1 + c.q2[1] - r0, c.g[0], c.g_shoup[0], c.q[1]), c.q[1]);
    let u2 = shoup_lazy(r2 + c.q2[2] - r0, c.g[1], c.g_shoup[1], c.q[2]);
    let x2 = csub(shoup_lazy(u2 + c.q2[2] - x1, c.g[2], c.g_shoup[2], c.q[2]), c.q[2]);
    let u3 = shoup_lazy(r3 + c.q2[3] - r0, c.g[3], c.g_shoup[3], c.q[3]);
    let t3 = shoup_lazy(u3 + c.q2[3] - x1, c.g[4], c.g_shoup[4], c.q[3]);
    let x3 = csub(shoup_lazy(t3 + c.q2[3] - x2, c.g[5], c.g_shoup[5], c.q[3]), c.q[3]);
    (x1, x2, x3)
}

/// `x0 + x1 W1 + x2 W2 + x3 W3` in `[0, Q)`, the weights cut in 32-bit pieces
/// and the 62-bit partial products summed by column.
#[cube]
fn accumulate(x0: u32, x1: u32, x2: u32, x3: u32, #[comptime] c: Consts) -> (u64, u64) {
    let y1 = u64::cast_from(x1);
    let y2 = u64::cast_from(x2);
    let y3 = u64::cast_from(x3);
    let c0 = u64::cast_from(x0) + y1 * (c.w1 as u64) + y2 * (c.w2[0] as u64) + y3 * (c.w3[0] as u64);
    let c1 = y2 * (c.w2[1] as u64) + y3 * (c.w3[1] as u64);
    let c2 = y3 * (c.w3[2] as u64);
    let lo = c0 + (c1 << 32u64);
    let k = u64::cast_from(lo < c0);
    let hi = c2 + (c1 >> 32u64) + k;
    (lo, hi)
}

/// Subtracts `Q` from `(lo, hi)` when it is at least `(Q + 1) / 2`.
#[cube]
fn center(lo: u64, hi: u64, #[comptime] c: Consts) -> (u64, u64) {
    let below = u64::cast_from(lo < c.half_lo);
    let ge = u64::cast_from(hi >= below + c.half_hi);
    let m = 0u64 - ge;
    let al = m & c.q_lo;
    let ah = m & c.q_hi;
    let b = u64::cast_from(lo < al);
    (lo - al, hi - ah - b)
}

/// Residue of coefficient `i` in the plane starting at `plane`. Over the
/// conjugate-invariant ring the plane holds the negacyclic basis and the
/// coefficient is `c_0 = b_0`, `c_i = (alpha^i b_i - alpha^-i b_{n-i}) / 2`;
/// `table` is the start of the prime's plane of `alpha^i / 2`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn residue(
    dft: &[u32],
    tw: &[u32],
    plane: u32,
    table: u32,
    i: u32,
    n: u32,
    #[comptime] q: u32,
    #[comptime] q2: u32,
    #[comptime] invariant: bool,
) -> u32 {
    let mut r = dft[(plane + i) as usize];
    if invariant {
        let other = dft[(plane + ((n - i) & (n - 1u32))) as usize];
        let direct = tw[(table + i) as usize];
        let reflected = tw[(table + 4u32 * n + i) as usize];
        let u = r * (direct.mul_hi(q) + 1u32) - r.mul_hi(direct) * q;
        let v = other * (reflected.mul_hi(q) + 1u32) - other.mul_hi(reflected) * q;
        let mut s = u + v;
        s = min(s, s - q2);
        s = min(s, s - q);
        r = select(i == 0u32, r, s);
    }
    r
}

/// The centered value of coefficient `i` of the block starting at `block`,
/// as its low and high words. `tw` is the table of degree `n`.
#[cube]
pub(crate) fn crt_at(
    dft: &[u32],
    tw: &[u32],
    block: u32,
    i: u32,
    n: u32,
    #[comptime] c: Consts,
    #[comptime] invariant: bool,
) -> (u64, u64) {
    let table = 12u32 * n;
    let r0 = residue(dft, tw, block, table, i, n, c.q[0], c.q2[0], invariant);
    let r1 = residue(dft, tw, block + n, table + n, i, n, c.q[1], c.q2[1], invariant);
    let r2 = residue(dft, tw, block + 2u32 * n, table + 2u32 * n, i, n, c.q[2], c.q2[2], invariant);
    let r3 = residue(dft, tw, block + 3u32 * n, table + 3u32 * n, i, n, c.q[3], c.q2[3], invariant);
    let (x1, x2, x3) = garner(r0, r1, r2, r3, c);
    let (lo, hi) = accumulate(r0, x1, x2, x3, c);
    center(lo, hi, c)
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn to_big_kernel(
    res: &mut [u64],
    dft: &[u32],
    tw: &[u32],
    n: u32,
    limbs: u32,
    res_base: u32,
    res_stride: u32,
    dft_base: u32,
    dft_stride: u32,
    dft_size: u32,
    #[comptime] c: Consts,
    #[comptime] invariant: bool,
) {
    let pos = ABSOLUTE_POS as u32;
    if pos < n * limbs {
        let limb = pos / n;
        let i = pos - limb * n;
        let mut lo = 0u64;
        let mut hi = 0u64;
        if limb < dft_size {
            let value = crt_at(dft, tw, dft_base + limb * dft_stride, i, n, c, invariant);
            lo = value.0;
            hi = value.1;
        }
        let index = res_base + limb * res_stride + 2u32 * i;
        res[index as usize] = lo;
        res[(index + 1u32) as usize] = hi;
    }
}

/// Writes every limb of the big column `res` with the centered values of
/// `dft`, which holds the output of the inverse transform; limbs past `dft`'s
/// are zero.
pub(crate) fn to_big<RT: GpuRuntime>(tables: &Tables<RT>, res: &Column, dft: &DftColumn) {
    assert_eq!(res.n, dft.n, "reconstruction between different degrees");
    assert_eq!(res.words, 2, "reconstruction writes `i128` values");
    if res.size == 0 {
        return;
    }
    let table = tables.of(res.n).view().binding().expect("transform tables hold storage");
    let units = res.n * res.size;
    launch!(
        RT, Label::CrtToBig, units * 16 + res.n * res.size.min(dft.size) * 16, to_big_kernel, (units, RT::GROUP, RT::GROUP);
        [
            res.binding => u64,
            dft.binding => u32,
            table => u32,
        ];
        [
            res.n as u32,
            res.size as u32,
            res.base,
            res.stride,
            dft.base,
            dft.stride,
            dft.size as u32,
        ];
        [Consts::get(), tables.invariant()]
    );
}
