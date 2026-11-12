//! From four residues to the centered big value, and the tail of the inverse
//! transform that writes it.
//!
//! Garner's mixed-radix digits `v = x0 + x1 q0 + x2 q0 q1 + x3 q0 q1 q2` are
//! computed with 32-bit Shoup products, accumulated in two 64-bit words, and
//! centered to `(-Q/2, Q/2]` as a two's complement 128-bit value: the
//! representative `NTT4x30Portable` produces.

use cubecl::prelude::*;

use super::{
    DftColumn,
    primes::{Consts, mul_lazy},
    tables::RingTables,
};
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

/// The centered value of the residues `r0..r3` of a coefficient, as its low and high words.
#[cube]
fn reconstruct(r0: u32, r1: u32, r2: u32, r3: u32, #[comptime] c: Consts) -> (u64, u64) {
    let (x1, x2, x3) = garner(r0, r1, r2, r3, c);
    let (lo, hi) = accumulate(r0, x1, x2, x3, c);
    center(lo, hi, c)
}

/// `(i x - y) / 2` modulo prime `k`, in `[0, q)`, for `x` and `y` in `[0, 2q)`
/// and `i = alpha^n`.
#[cube]
fn half_difference(x: u32, y: u32, #[comptime] c: Consts, #[comptime] k: u32) -> u32 {
    let q = comptime!(c.q[k as usize]);
    let q2 = comptime!(c.q2[k as usize]);
    // `i x` and `y` are in `[0, 2q)`, so `i x + 2q - y` is in `(0, 4q)`.
    let mut s = shoup_lazy(x, comptime!(c.i[k as usize]), comptime!(c.i_shoup[k as usize]), q) + q2 - y;
    s = min(s, s - q2);
    s = min(s, s - q);
    // Halved modulo the odd `q`: an odd `s` is first raised by `q`.
    (s + (s & 1u32) * q) >> 1u32
}

/// Residues modulo prime `k` over the conjugate-invariant ring of coefficient
/// `i` and of its reflection `j = n - i` of the block starting at `block`, which
/// holds the negacyclic basis `b`, from the same products: `c_0 = b_0` and, with
/// `f[j] = alpha^-j` word `k n + j` of `basis`, `u = f[j] b_i` and `v = f[i] b_j`,
/// `c_i = (alpha^n u - v) / 2` and `c_j = (alpha^n v - u) / 2`.
#[cube]
fn residue_pair(dft: &[u32], basis: &[u32], block: u32, i: u32, n: u32, #[comptime] c: Consts, #[comptime] k: u32) -> (u32, u32) {
    let q = comptime!(c.q[k as usize]);
    let plane = block + k * n;
    let j = (n - i) & (n - 1u32);
    let x = dft[(plane + i) as usize];
    let y = dft[(plane + j) as usize];
    let u = mul_lazy(x, basis[(k * n + j) as usize], q);
    let v = mul_lazy(y, basis[(k * n + i) as usize], q);
    let first = i == 0u32;
    (
        select(first, x, half_difference(u, v, c, k)),
        select(first, y, half_difference(v, u, c, k)),
    )
}

/// The centered value over the standard ring of coefficient `i` of the block
/// starting at `block`, as its low and high words.
#[cube]
pub(crate) fn crt_at(dft: &[u32], block: u32, i: u32, n: u32, #[comptime] c: Consts) -> (u64, u64) {
    let r0 = dft[(block + i) as usize];
    let r1 = dft[(block + n + i) as usize];
    let r2 = dft[(block + 2u32 * n + i) as usize];
    let r3 = dft[(block + 3u32 * n + i) as usize];
    reconstruct(r0, r1, r2, r3, c)
}

/// The centered values over the conjugate-invariant ring of coefficient `i`
/// and of its reflection `n - i` of the block starting at `block`, in this
/// order. `basis` is the basis change of degree `n`.
#[cube]
pub(crate) fn crt_pair_at(
    dft: &[u32],
    basis: &[u32],
    block: u32,
    i: u32,
    n: u32,
    #[comptime] c: Consts,
) -> ((u64, u64), (u64, u64)) {
    let (r0, s0) = residue_pair(dft, basis, block, i, n, c, 0u32);
    let (r1, s1) = residue_pair(dft, basis, block, i, n, c, 1u32);
    let (r2, s2) = residue_pair(dft, basis, block, i, n, c, 2u32);
    let (r3, s3) = residue_pair(dft, basis, block, i, n, c, 3u32);
    (reconstruct(r0, r1, r2, r3, c), reconstruct(s0, s1, s2, s3, c))
}

/// A unit owns coefficient `i` of a limb and, over the conjugate-invariant
/// ring, its reflection `n - i`, whose basis change shares its products.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn to_big_kernel(
    res: &mut [u64],
    dft: &[u32],
    basis: &[u32],
    n: u32,
    // Units per limb: `n`, or `n / 2 + 1` over the conjugate-invariant ring.
    per_limb: u32,
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
    if pos < per_limb * limbs {
        let limb = pos / per_limb;
        let i = pos - limb * per_limb;
        let j = (n - i) & (n - 1u32);
        let block = dft_base + limb * dft_stride;
        let mut lo = 0u64;
        let mut hi = 0u64;
        let mut lo_j = 0u64;
        let mut hi_j = 0u64;
        if limb < dft_size {
            if invariant {
                let (value, value_j) = crt_pair_at(dft, basis, block, i, n, c);
                lo = value.0;
                hi = value.1;
                lo_j = value_j.0;
                hi_j = value_j.1;
            } else {
                let value = crt_at(dft, block, i, n, c);
                lo = value.0;
                hi = value.1;
            }
        }
        let index = res_base + limb * res_stride + 2u32 * i;
        res[index as usize] = lo;
        res[(index + 1u32) as usize] = hi;
        if invariant {
            let index_j = res_base + limb * res_stride + 2u32 * j;
            res[index_j as usize] = lo_j;
            res[(index_j + 1u32) as usize] = hi_j;
        }
    }
}

/// [`to_big_kernel`] for a temporary `dft` that is zeroed once read: a unit
/// clears the residues of the coefficients it owns, which no other unit reads.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn to_big_clearing_kernel(
    res: &mut [u64],
    // The temporary, bound twice: cleared through this view, read through `dft`.
    clear: &mut [u32],
    basis: &[u32],
    dft: &[u32],
    n: u32,
    // Units per limb: `n`, or `n / 2 + 1` over the conjugate-invariant ring.
    per_limb: u32,
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
    if pos < per_limb * limbs {
        let limb = pos / per_limb;
        let i = pos - limb * per_limb;
        let j = (n - i) & (n - 1u32);
        let block = dft_base + limb * dft_stride;
        let mut lo = 0u64;
        let mut hi = 0u64;
        let mut lo_j = 0u64;
        let mut hi_j = 0u64;
        if limb < dft_size {
            if invariant {
                let (value, value_j) = crt_pair_at(dft, basis, block, i, n, c);
                lo = value.0;
                hi = value.1;
                lo_j = value_j.0;
                hi_j = value_j.1;
            } else {
                let value = crt_at(dft, block, i, n, c);
                lo = value.0;
                hi = value.1;
            }
            for k in 0..4u32 {
                clear[(block + k * n + i) as usize] = 0u32;
                if invariant {
                    clear[(block + k * n + j) as usize] = 0u32;
                }
            }
        }
        let index = res_base + limb * res_stride + 2u32 * i;
        res[index as usize] = lo;
        res[(index + 1u32) as usize] = hi;
        if invariant {
            let index_j = res_base + limb * res_stride + 2u32 * j;
            res[index_j as usize] = lo_j;
            res[(index_j + 1u32) as usize] = hi_j;
        }
    }
}

/// Writes every limb of the big column `res` with the centered values of
/// `dft`, which holds the output of the inverse transform; limbs past `dft`'s
/// are zero.
pub(crate) fn to_big<RT: GpuRuntime>(tables: RingTables<'_, RT>, res: &Column, dft: &DftColumn) {
    assert!(res.n == dft.n, "reconstruction between different degrees");
    assert!(res.words == 2, "reconstruction writes `i128` values");
    if res.size == 0 {
        return;
    }
    let table = tables.of(res.n);
    let basis = table.basis.view().binding().expect("transform tables hold storage");
    let per_limb = if tables.invariant() { res.n / 2 + 1 } else { res.n };
    let units = per_limb * res.size;
    let bytes = res.n * res.size * 16 + res.n * res.size.min(dft.size) * 16;
    launch!(
        RT, Label::CrtToBig, bytes, to_big_kernel, (units, RT::GROUP, RT::GROUP);
        [
            res.binding => u64,
            dft.binding => u32,
            basis => u32,
        ];
        [
            res.n as u32,
            per_limb as u32,
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

/// [`to_big`] from a temporary `dft`, every word of which it reads is zeroed:
/// the temporary leaves the call holding nothing it computed.
pub(crate) fn to_big_clearing<RT: GpuRuntime>(tables: RingTables<'_, RT>, res: &Column, dft: &DftColumn) {
    assert!(res.n == dft.n, "reconstruction between different degrees");
    assert!(res.words == 2, "reconstruction writes `i128` values");
    assert!(dft.size <= res.size, "the temporary has no limb past the result's");
    if res.size == 0 {
        return;
    }
    let table = tables.of(res.n);
    let basis = table.basis.view().binding().expect("transform tables hold storage");
    let per_limb = if tables.invariant() { res.n / 2 + 1 } else { res.n };
    let units = per_limb * res.size;
    launch!(
        RT, Label::CrtToBigClearing, res.n * res.size * 16 + res.n * dft.size * 32, to_big_clearing_kernel, (units, RT::GROUP, RT::GROUP);
        [
            res.binding => u64,
            dft.binding => u32,
            basis => u32,
            dft.binding => u32,
        ];
        [
            res.n as u32,
            per_limb as u32,
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
