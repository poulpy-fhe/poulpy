//! Coefficient-domain products and sums into big values: `i64 * i64 -> i128`.

use cubecl::prelude::*;

use super::{Column, DIM};
use crate::runtime::{Binding, GpuRuntime, cubes, launched};

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
    let client = RT::client();
    let units = a.n * res.size;
    unsafe {
        scalar_product_kernel::launch_unchecked(
            client,
            cubes(client, units, DIM),
            CubeDim::new_1d(DIM),
            res.binding.arg::<u64>(),
            a.binding.arg::<u64>(),
            b.arg::<u64>(),
            a.n as u32,
            res.size as u32,
            res.base,
            res.stride,
            a.base,
            a.stride,
            b_base,
        );
    }
    launched(units * 32);
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
    let client = RT::client();
    let units = res.n * res.size;
    unsafe {
        col_weighted_sum_kernel::launch_unchecked(
            client,
            cubes(client, units, DIM),
            CubeDim::new_1d(DIM),
            res.binding.arg::<u64>(),
            a_col0.binding.arg::<u64>(),
            weights.arg::<u64>(),
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
        );
    }
    launched(units * 16 + coeffs * res.size * cols * 8);
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
    let client = RT::client();
    unsafe {
        inner_sum_kernel::launch_unchecked(
            client,
            cubes(client, res.size, 1),
            CubeDim::new_1d(1),
            res.binding.arg::<u64>(),
            a.binding.arg::<u64>(),
            res.size as u32,
            res.base,
            res.stride,
            res_coeff as u32,
            a.base,
            a.stride,
            a.n as u32,
        );
    }
    launched(res.size * (a.n + 1) * 16);
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
    assert_eq!(a.n, res.n, "cnv_by_const_apply: a.n() != res.n()");
    assert!(b_coeff < b.n || b.size == 0, "cnv_by_const_apply: b_coeff out of range");
    if res.size == 0 || res.n == 0 {
        return;
    }
    let client = RT::client();
    let units = res.n * res.size;
    unsafe {
        by_const_kernel::launch_unchecked(
            client,
            cubes(client, units, DIM),
            CubeDim::new_1d(DIM),
            res.binding.arg::<u64>(),
            a.binding.arg::<u64>(),
            b.binding.arg::<u64>(),
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
            // Past `a.size + b.size` no limb has a term.
            offset.min(a.size + b.size) as u32,
            accumulate,
        );
    }
    launched(units * (16 + 16 * a.size.min(b.size)));
}
