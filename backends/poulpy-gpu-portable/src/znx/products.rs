//! Coefficient-domain products and sums into big values: `i64 * i64 -> i128`.

use cubecl::prelude::*;

use super::Column;
use crate::runtime::{Binding, GpuRuntime, Label, launch};

/// High 64 bits of the signed product `x * y`.
#[cube]
fn mul_hi_signed(x: u64, y: u64) -> u64 {
    let mask = 0xFFFF_FFFFu64;
    let x0 = x & mask;
    let x1 = x >> 32u64;
    let y0 = y & mask;
    let y1 = y >> 32u64;
    let p00 = x0 * y0;
    let p01 = x0 * y1;
    let p10 = x1 * y0;
    let p11 = x1 * y1;
    let middle = (p00 >> 32u64) + (p01 & mask) + (p10 & mask);
    let mut hi = p11 + (p01 >> 32u64) + (p10 >> 32u64) + (middle >> 32u64);
    // Signed correction of the unsigned product.
    if (x >> 63u64) != 0u64 {
        hi -= y;
    }
    if (y >> 63u64) != 0u64 {
        hi -= x;
    }
    hi
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn scalar_product_kernel(
    res: &mut [u64],
    a: &[u64],
    b: &[u64],
    n: u32,
    limbs: u32,
    res_base: u32,
    res_stride: u32,
    a_base: u32,
    a_stride: u32,
    b_base: u32,
) {
    let pos = ABSOLUTE_POS as u32;
    if pos < n * limbs {
        let limb = pos / n;
        let i = pos - limb * n;
        let x = a[(a_base + limb * a_stride + i) as usize];
        let y = b[(b_base + i) as usize];
        let index = res_base + limb * res_stride + 2u32 * i;
        res[index as usize] = x * y;
        res[(index + 1u32) as usize] = mul_hi_signed(x, y);
    }
}

/// `res[j][i] = a[j][i] * b[i]` for the first `a.n` coefficients of every limb of `res`.
pub(crate) fn scalar_product<RT: GpuRuntime>(res: &Column, a: &Column, b: &Binding, b_base: u32) {
    assert!(res.words == 2 && a.words == 1);
    assert!(res.n >= a.n && res.size <= a.size, "vec_znx_scalar_product: operand shapes");
    if res.size == 0 || a.n == 0 {
        return;
    }
    let units = a.n * res.size;
    launch!(
        RT, Label::BigScalarProduct, units * 32, scalar_product_kernel, (units, RT::GROUP, RT::GROUP);
        [
            res.binding => u64,
            a.binding => u64,
            b => u64,
        ];
        [
            a.n as u32,
            res.size as u32,
            res.base,
            res.stride,
            a.base,
            a.stride,
            b_base,
        ];
        []
    );
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn col_weighted_sum_kernel(
    res: &mut [u64],
    a: &[u64],
    weights: &[u64],
    res_n: u32,
    limbs: u32,
    res_base: u32,
    res_stride: u32,
    // Column 0 of `a`; column `c` of limb `j` starts `c * a_col_stride` words later.
    a_base: u32,
    a_stride: u32,
    a_col_stride: u32,
    weights_base: u32,
    cols: u32,
    coeffs: u32,
) {
    let pos = ABSOLUTE_POS as u32;
    if pos < res_n * limbs {
        let limb = pos / res_n;
        let i = pos - limb * res_n;
        let mut lo = 0u64;
        let mut hi = 0u64;
        if i < coeffs {
            for c in 0..cols {
                let x = a[(a_base + limb * a_stride + c * a_col_stride + i) as usize];
                let w = weights[(weights_base + c) as usize];
                let p_lo = x * w;
                let sum = lo + p_lo;
                hi += mul_hi_signed(x, w);
                if sum < lo {
                    hi += 1u64;
                }
                lo = sum;
            }
        }
        let index = res_base + limb * res_stride + 2u32 * i;
        res[index as usize] = lo;
        res[(index + 1u32) as usize] = hi;
    }
}

/// `res[j][i] = sum_c a[c][j][i] * weights[c]` for `i < coeffs`, zero beyond.
#[allow(clippy::too_many_arguments)]
pub(crate) fn col_weighted_sum<RT: GpuRuntime>(
    res: &Column,
    a_col0: &Column,
    a_col_stride: u32,
    weights: &Binding,
    weights_base: u32,
    cols: usize,
    coeffs: usize,
) {
    assert!(res.words == 2 && a_col0.words == 1);
    assert!(
        coeffs <= a_col0.n.min(res.n) && res.size <= a_col0.size,
        "vec_znx_big_col_weighted_sum: operand shapes"
    );
    if res.size == 0 || res.n == 0 {
        return;
    }
    let units = res.n * res.size;
    launch!(
        RT, Label::BigColWeightedSum, units * 16 + coeffs * res.size * cols * 8, col_weighted_sum_kernel, (units, RT::GROUP, RT::GROUP);
        [
            res.binding => u64,
            a_col0.binding => u64,
            weights => u64,
        ];
        [
            res.n as u32,
            res.size as u32,
            res.base,
            res.stride,
            a_col0.base,
            a_col0.stride,
            a_col_stride,
            weights_base,
            cols as u32,
            coeffs as u32,
        ];
        []
    );
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn inner_sum_kernel(
    res: &mut [u64],
    a: &[u64],
    limbs: u32,
    res_base: u32,
    res_stride: u32,
    res_coeff: u32,
    a_base: u32,
    a_stride: u32,
    a_n: u32,
) {
    let limb = ABSOLUTE_POS as u32;
    if limb < limbs {
        let mut lo = 0u64;
        let mut hi = 0u64;
        let src = a_base + limb * a_stride;
        for i in 0..a_n {
            let x_lo = a[(src + 2u32 * i) as usize];
            let sum = lo + x_lo;
            hi += a[(src + 2u32 * i + 1u32) as usize];
            if sum < lo {
                hi += 1u64;
            }
            lo = sum;
        }
        let index = res_base + limb * res_stride + 2u32 * res_coeff;
        res[index as usize] = lo;
        res[(index + 1u32) as usize] = hi;
    }
}

/// `res[j][res_coeff] = sum_i a[j][i]` for every limb of `res`.
pub(crate) fn inner_sum<RT: GpuRuntime>(res: &Column, res_coeff: usize, a: &Column) {
    assert!(res.words == 2 && a.words == 2);
    assert!(
        res_coeff < res.n && res.size <= a.size,
        "vec_znx_big_inner_sum: operand shapes"
    );
    if res.size == 0 {
        return;
    }
    launch!(
        RT, Label::BigInnerSum, res.size * (a.n + 1) * 16, inner_sum_kernel, (res.size, RT::GROUP, RT::GROUP);
        [
            res.binding => u64,
            a.binding => u64,
        ];
        [
            res.size as u32,
            res.base,
            res.stride,
            res_coeff as u32,
            a.base,
            a.stride,
            a.n as u32,
        ];
        []
    );
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn by_const_kernel(
    res: &mut [u64],
    a: &[u64],
    b: &[u64],
    n: u32,
    limbs: u32,
    res_base: u32,
    res_stride: u32,
    a_base: u32,
    a_stride: u32,
    a_size: u32,
    // The selected coefficient of limb 0 of `b`.
    b_base: u32,
    b_stride: u32,
    b_size: u32,
    offset: u32,
    #[comptime] accumulate: bool,
) {
    let pos = ABSOLUTE_POS as u32;
    if pos < n * limbs {
        let limb = pos / n;
        let i = pos - limb * n;
        // Terms `a[top - v] * b[v]` for `v` in `first..end`.
        let top = limb + offset;
        let first = top + 1u32 - min(top + 1u32, a_size);
        let end = min(top + 1u32, b_size);
        let mut lo = 0u64;
        let mut hi = 0u64;
        for v in first..end {
            let x = a[(a_base + (top - v) * a_stride + i) as usize];
            let y = b[(b_base + v * b_stride) as usize];
            let sum = lo + x * y;
            hi += mul_hi_signed(x, y);
            if sum < lo {
                hi += 1u64;
            }
            lo = sum;
        }
        let index = res_base + limb * res_stride + 2u32 * i;
        if accumulate {
            let sum = lo + res[index as usize];
            hi += res[(index + 1u32) as usize];
            if sum < lo {
                hi += 1u64;
            }
            lo = sum;
        }
        res[index as usize] = lo;
        res[(index + 1u32) as usize] = hi;
    }
}

/// The 128-bit sum of the terms `a[top - v][i] * b[v]` of one result limb,
/// `top` being that limb plus the offset: one product of [`by_const_kernel`].
#[cube]
#[allow(clippy::too_many_arguments)]
fn const_product(
    a: &[u64],
    b: &[u64],
    i: u32,
    top: u32,
    a_base: u32,
    a_stride: u32,
    a_size: u32,
    b_base: u32,
    b_stride: u32,
    b_size: u32,
) -> (u64, u64) {
    let first = top + 1u32 - min(top + 1u32, a_size);
    let end = min(top + 1u32, b_size);
    let mut lo = 0u64;
    let mut hi = 0u64;
    for v in first..end {
        let x = a[(a_base + (top - v) * a_stride + i) as usize];
        let y = b[(b_base + v * b_stride) as usize];
        let sum = lo + x * y;
        hi += mul_hi_signed(x, y);
        if sum < lo {
            hi += 1u64;
        }
        lo = sum;
    }
    (lo, hi)
}

/// Most terms [`convolution_by_const_sum`] takes.
#[cfg_attr(not(feature = "enable-core"), allow(dead_code))]
pub(crate) const CONST_TERMS: usize = 4;

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, unused_assignments)]
fn by_const_sum_kernel(
    res: &mut [u64],
    a0: &[u64],
    a1: &[u64],
    a2: &[u64],
    a3: &[u64],
    b0: &[u64],
    b1: &[u64],
    b2: &[u64],
    b3: &[u64],
    n: u32,
    limbs: u32,
    res_base: u32,
    res_stride: u32,
    // Per term: `a` base, stride and size, `b` base, stride and size, offset.
    t0_a_base: u32,
    t0_a_stride: u32,
    t0_a_size: u32,
    t0_b_base: u32,
    t0_b_stride: u32,
    t0_b_size: u32,
    t0_offset: u32,
    t1_a_base: u32,
    t1_a_stride: u32,
    t1_a_size: u32,
    t1_b_base: u32,
    t1_b_stride: u32,
    t1_b_size: u32,
    t1_offset: u32,
    t2_a_base: u32,
    t2_a_stride: u32,
    t2_a_size: u32,
    t2_b_base: u32,
    t2_b_stride: u32,
    t2_b_size: u32,
    t2_offset: u32,
    t3_a_base: u32,
    t3_a_stride: u32,
    t3_a_size: u32,
    t3_b_base: u32,
    t3_b_stride: u32,
    t3_b_size: u32,
    t3_offset: u32,
    #[comptime] terms: u32,
) {
    let pos = ABSOLUTE_POS as u32;
    if pos < n * limbs {
        let limb = pos / n;
        let i = pos - limb * n;
        let first = const_product(
            a0,
            b0,
            i,
            limb + t0_offset,
            t0_a_base,
            t0_a_stride,
            t0_a_size,
            t0_b_base,
            t0_b_stride,
            t0_b_size,
        );
        let mut lo = first.0;
        let mut hi = first.1;
        #[unroll]
        for term in 1..terms {
            let mut next_lo = 0u64;
            let mut next_hi = 0u64;
            if comptime!(term == 1) {
                let product = const_product(
                    a1,
                    b1,
                    i,
                    limb + t1_offset,
                    t1_a_base,
                    t1_a_stride,
                    t1_a_size,
                    t1_b_base,
                    t1_b_stride,
                    t1_b_size,
                );
                next_lo = product.0;
                next_hi = product.1;
            } else if comptime!(term == 2) {
                let product = const_product(
                    a2,
                    b2,
                    i,
                    limb + t2_offset,
                    t2_a_base,
                    t2_a_stride,
                    t2_a_size,
                    t2_b_base,
                    t2_b_stride,
                    t2_b_size,
                );
                next_lo = product.0;
                next_hi = product.1;
            } else {
                let product = const_product(
                    a3,
                    b3,
                    i,
                    limb + t3_offset,
                    t3_a_base,
                    t3_a_stride,
                    t3_a_size,
                    t3_b_base,
                    t3_b_stride,
                    t3_b_size,
                );
                next_lo = product.0;
                next_hi = product.1;
            }
            let sum = lo + next_lo;
            hi += next_hi;
            if sum < lo {
                hi += 1u64;
            }
            lo = sum;
        }
        let index = res_base + limb * res_stride + 2u32 * i;
        res[index as usize] = lo;
        res[(index + 1u32) as usize] = hi;
    }
}

/// `res[j][i] = sum_t sum_{u + v = j + offset_t} a_t[u][i] * b_t[v][coeff_t]`
/// for every limb of `res`, over one to [`CONST_TERMS`] terms
/// `(a_t, b_t, coeff_t, offset_t)`: the products of
/// [`convolution_by_const`] summed in one launch.
#[cfg_attr(not(feature = "enable-core"), allow(dead_code))]
pub(crate) fn convolution_by_const_sum<RT: GpuRuntime>(res: &Column, terms: &[(Column, Column, usize, usize)]) {
    assert!((1..=CONST_TERMS).contains(&terms.len()), "one to {CONST_TERMS} terms");
    assert_eq!(res.words, 2, "the sum is written in 128-bit words");
    for (a, b, b_coeff, _) in terms {
        assert!(a.words == 1 && b.words == 1);
        assert_eq!(a.n, res.n, "cnv_by_const_apply: a.n() != res.n()");
        assert!(*b_coeff < b.n || b.size == 0, "cnv_by_const_apply: b_coeff out of range");
    }
    if res.size == 0 || res.n == 0 {
        return;
    }
    // Terms beyond the last repeat it: the kernel does not read them.
    let term = |index: usize| &terms[index.min(terms.len() - 1)];
    let scalars = |index: usize| {
        let (a, b, b_coeff, offset) = term(index);
        [
            a.base,
            a.stride,
            a.size as u32,
            b.base + *b_coeff as u32,
            b.stride,
            b.size as u32,
            (*offset).min(a.size + b.size) as u32,
        ]
    };
    let units = res.n * res.size;
    let read: usize = terms.iter().map(|(a, b, _, _)| 16 * a.size.min(b.size)).sum();
    launch!(
        RT, Label::CnvByConst, units * (16 + read), by_const_sum_kernel, (units, RT::GROUP, RT::GROUP);
        [
            res.binding => u64,
            term(0).0.binding => u64,
            term(1).0.binding => u64,
            term(2).0.binding => u64,
            term(3).0.binding => u64,
            term(0).1.binding => u64,
            term(1).1.binding => u64,
            term(2).1.binding => u64,
            term(3).1.binding => u64,
        ];
        [
            res.n as u32,
            res.size as u32,
            res.base,
            res.stride,
            scalars(0)[0],
            scalars(0)[1],
            scalars(0)[2],
            scalars(0)[3],
            scalars(0)[4],
            scalars(0)[5],
            scalars(0)[6],
            scalars(1)[0],
            scalars(1)[1],
            scalars(1)[2],
            scalars(1)[3],
            scalars(1)[4],
            scalars(1)[5],
            scalars(1)[6],
            scalars(2)[0],
            scalars(2)[1],
            scalars(2)[2],
            scalars(2)[3],
            scalars(2)[4],
            scalars(2)[5],
            scalars(2)[6],
            scalars(3)[0],
            scalars(3)[1],
            scalars(3)[2],
            scalars(3)[3],
            scalars(3)[4],
            scalars(3)[5],
            scalars(3)[6],
        ];
        [terms.len() as u32]
    );
}

/// `res[j][i] = sum_{u + v = j + offset} a[u][i] * b[v][b_coeff]` for every
/// limb of `res`, added to `res` when `accumulate`. Limbs without a term are
/// zero, or unchanged when accumulating. `b` may have any degree.
pub(crate) fn convolution_by_const<RT: GpuRuntime>(
    res: &Column,
    a: &Column,
    b: &Column,
    b_coeff: usize,
    offset: usize,
    accumulate: bool,
) {
    assert!(res.words == 2 && a.words == 1 && b.words == 1);
    assert!(a.n == res.n, "cnv_by_const_apply: a.n() != res.n()");
    assert!(b_coeff < b.n || b.size == 0, "cnv_by_const_apply: b_coeff out of range");
    if res.size == 0 || res.n == 0 {
        return;
    }
    let units = res.n * res.size;
    launch!(
        RT, if accumulate { Label::CnvByConstAdd } else { Label::CnvByConst }, units * (16 + 16 * a.size.min(b.size)), by_const_kernel, (units, RT::GROUP, RT::GROUP);
        [
            res.binding => u64,
            a.binding => u64,
            b.binding => u64,
        ];
        [
            res.n as u32,
            res.size as u32,
            res.base,
            res.stride,
            a.base,
            a.stride,
            a.size as u32,
            b.base + b_coeff as u32,
            b.stride,
            b.size as u32,
            offset.min(a.size + b.size) as u32,
        ];
        [accumulate]
    );
}
