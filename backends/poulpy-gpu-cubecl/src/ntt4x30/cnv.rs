//! Limb convolution of prepared operands:
//! `res[j] = sum_{u + v = j + offset} a[u] * b[v]`, slot by slot.
//!
//! A prepared operand, left or right, is the transform of its source in the
//! layout of a DFT-domain vector. One thread per result residue walks the
//! terms of its limb, accumulating products of canonical residues in a `u64`
//! folded every [`FOLD_TERMS`] terms.
//!
//! A right operand may have a degree `m` dividing the degree `n` of the
//! product; it then denotes its polynomial in `X^(n/m)`, whose value at the
//! point of slot `s` is slot `s >> log2(n/m)` of the operand's transform.
//! The right operand may also be read through an automorphism `sigma_p`: that
//! slot is then the one the DFT-domain automorphism moves to it (see
//! [`super::automorphism`]), found once per thread.
//!
//! Three forms share the kernel:
//! - one product of a column of `a` by a column of `b`, overwriting or
//!   accumulating;
//! - the pairwise product `(a_i + a_j) * (b_i + b_j)`, the sums being formed
//!   in registers;
//! - the three products of a rank-1 tensor in one walk: `d0 = a_0 b_0` and
//!   `d2 = a_1 b_1` into two columns, and
//!   `(a_0 + a_1)(b_0 + b_1) - d0 - d2` into a third.

use cubecl::prelude::*;

use super::{
    AutomorphismPlan, DIM, DftColumn,
    primes::{Consts, FOLD_TERMS, barrett60, fold64, sel4},
};
use crate::runtime::{GpuRuntime, Label, launch};

/// `res = a * b`.
const SINGLE: u32 = 0;
/// `res = (a + a2) * (b + b2)`.
const PAIRWISE: u32 = 1;
/// `res = a * b`, `res2 = a2 * b2`, `cross = (a + a2) * (b + b2) - res - res2`.
const TENSOR: u32 = 2;

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, unused_assignments)]
fn convolution_kernel(
    res: &mut [u32],
    cross: &mut [u32],
    a: &[u32],
    b: &[u32],
    log_n: u32,
    limbs: u32,
    res_base: u32,
    res2_base: u32,
    res_stride: u32,
    // Limbs of `res` and of `cross` in the tensor form.
    res_size: u32,
    cross_base: u32,
    cross_stride: u32,
    cross_size: u32,
    a_base: u32,
    a2_base: u32,
    a_stride: u32,
    a_size: u32,
    b_base: u32,
    b2_base: u32,
    b_stride: u32,
    b_size: u32,
    // log2 of the ratio between the degree of the product and that of `b`.
    b_gap: u32,
    // The automorphism `b` is read through, as in `automorphism_kernel`.
    b_mul: u32,
    b_shift: u32,
    offset: u32,
    #[comptime] c: Consts,
    #[comptime] form: u32,
    #[comptime] accumulate: bool,
) {
    let log_block = log_n + 2u32;
    let pos = ABSOLUTE_POS as u32;
    if pos < (limbs << log_block) {
        let j = pos >> log_block;
        let word = pos - (j << log_block);
        let plane = word >> log_n;
        let slot = word - (plane << log_n);
        let log_b = log_n - b_gap;
        let up = 32u32 - log_b;
        let point = (slot >> b_gap).reverse_bits() >> up;
        let exponent = (b_mul * ((point << b_shift) + 1u32)) & ((1u32 << (log_b + b_shift)) - 1u32);
        let b_word = (plane << log_b) + ((exponent >> b_shift).reverse_bits() >> up);
        let q = sel4(plane, c.q);
        let r = sel4(plane, c.r);
        let mu = sel4(plane, c.mu60);

        // Terms `a[top - v] * b[v]` for `v` in `v..end`.
        let top = j + offset;
        let mut v = top + 1u32 - min(top + 1u32, a_size);
        let end = min(top + 1u32, b_size);
        let mut acc = 0u64;
        let mut acc2 = 0u64;
        let mut acc_sum = 0u64;
        while v < end {
            let block_end = min(v + FOLD_TERMS, end);
            acc = fold64(acc, r);
            if form == TENSOR {
                acc2 = fold64(acc2, r);
                acc_sum = fold64(acc_sum, r);
            }
            while v < block_end {
                let a_index = (top - v) * a_stride + word;
                let b_index = v * b_stride + b_word;
                let mut x = a[(a_base + a_index) as usize];
                let mut y = b[(b_base + b_index) as usize];
                if form != SINGLE {
                    let x2 = a[(a2_base + a_index) as usize];
                    let y2 = b[(b2_base + b_index) as usize];
                    let x_sum = x + x2;
                    let y_sum = y + y2;
                    let xs = min(x_sum, x_sum - q);
                    let ys = min(y_sum, y_sum - q);
                    if form == TENSOR {
                        acc2 += u64::cast_from(x2) * u64::cast_from(y2);
                        acc_sum += u64::cast_from(xs) * u64::cast_from(ys);
                    } else {
                        x = xs;
                        y = ys;
                    }
                }
                acc += u64::cast_from(x) * u64::cast_from(y);
                v += 1u32;
            }
        }
        let mut value = barrett60(fold64(acc, r), q, mu);
        if form == TENSOR {
            let value2 = barrett60(fold64(acc2, r), q, mu);
            let mut middle = barrett60(fold64(acc_sum, r), q, mu);
            if j < res_size {
                res[(res_base + j * res_stride + word) as usize] = value;
                res[(res2_base + j * res_stride + word) as usize] = value2;
                // `middle - value - value2` on canonical residues.
                let sum = value + value2;
                let both = min(sum, sum - q);
                let difference = middle - both;
                middle = min(difference, difference + q);
            }
            if j < cross_size {
                cross[(cross_base + j * cross_stride + word) as usize] = middle;
            }
        } else {
            let index = res_base + j * res_stride + word;
            if accumulate {
                let sum = value + res[index as usize];
                value = min(sum, sum - q);
            }
            res[index as usize] = value;
        }
    }
}

/// The operands of a convolution; `a2` and `b2` are only read by the pairwise
/// and tensor forms. Both right operands are read through `b_plan`.
struct Operands<'a> {
    a: &'a DftColumn,
    a2: &'a DftColumn,
    b: &'a DftColumn,
    b2: &'a DftColumn,
    b_plan: AutomorphismPlan,
}

#[allow(clippy::too_many_arguments)]
fn launch<RT: GpuRuntime>(
    form: u32,
    accumulate: bool,
    offset: usize,
    limbs: usize,
    res: &DftColumn,
    res2: &DftColumn,
    cross: &DftColumn,
    operands: Operands<'_>,
) {
    let Operands { a, a2, b, b2, b_plan } = operands;
    let n = res.n;
    assert_eq!(a.n, n, "the left operand has the degree of the product");
    assert!(
        b.n.is_power_of_two() && b.n <= n,
        "the right operand has degree {} which does not divide {n}",
        b.n
    );
    assert_eq!(
        b_plan.n, b.n,
        "the right operand is read through an automorphism of its degree"
    );
    if limbs == 0 {
        return;
    }
    let units = 4 * n * limbs;
    let label = match (form, accumulate) {
        (TENSOR, _) => Label::CnvTensor,
        (PAIRWISE, _) => Label::CnvPairwise,
        (_, true) => Label::CnvSingleAdd,
        _ => Label::CnvSingle,
    };
    let terms = a.size.min(b.size) * if form == SINGLE { 2 } else { 4 };
    launch!(
        RT, label, units * 4 * (terms + if form == TENSOR { 3 } else { 1 + accumulate as usize }), convolution_kernel, (units, DIM, DIM);
        [
            res.binding => u32,
            *if form == TENSOR { &cross.binding } else { &a.binding } => u32,
            a.binding => u32,
            b.binding => u32,
        ];
        [
            n.ilog2(),
            limbs as u32,
            res.base,
            res2.base,
            res.stride,
            res.size as u32,
            cross.base,
            cross.stride,
            cross.size as u32,
            a.base,
            a2.base,
            a.stride,
            a.size as u32,
            b.base,
            b2.base,
            b.stride,
            b.size as u32,
            (n / b.n).ilog2(),
            b_plan.mul,
            b_plan.shift,
            // Past `a.size + b.size` no limb has a term.
            offset.min(a.size + b.size) as u32,
        ];
        [Consts::get(), form, accumulate]
    );
}

/// `res[j] = sum_{u + v = j + offset} a[u] * sigma(b[v])` for every limb of
/// `res`, `sigma` being `b_plan`, added to `res` when `accumulate`. Limbs
/// without a term are zero, or unchanged when accumulating.
pub(crate) fn convolution<RT: GpuRuntime>(
    res: &DftColumn,
    a: &DftColumn,
    b: &DftColumn,
    b_plan: &AutomorphismPlan,
    offset: usize,
    accumulate: bool,
) {
    launch::<RT>(
        SINGLE,
        accumulate,
        offset,
        res.size,
        res,
        res,
        res,
        Operands {
            a,
            a2: a,
            b,
            b2: b,
            b_plan: *b_plan,
        },
    );
}

/// `res[j] = sum_{u + v = j + offset} (a[u] + a2[u]) * (b[v] + b2[v])` for
/// every limb of `res`. `a2` and `b2` are columns of the vectors of `a` and `b`.
pub(crate) fn convolution_pairwise<RT: GpuRuntime>(
    res: &DftColumn,
    a: &DftColumn,
    a2: &DftColumn,
    b: &DftColumn,
    b2: &DftColumn,
    offset: usize,
) {
    let b_plan = AutomorphismPlan::new(b.n, 1, false);
    launch::<RT>(
        PAIRWISE,
        false,
        offset,
        res.size,
        res,
        res,
        res,
        Operands { a, a2, b, b2, b_plan },
    );
}

/// The three products of a rank-1 tensor: `diag0 = a0 * b0`, `diag1 = a1 * b1`
/// and `cross = (a0 + a1) * (b0 + b1) - diag0 - diag1`, every limb of each
/// being written and the subtraction covering the limbs `diag` has. `diag0` and `diag1` are columns of one
/// vector, as are `a0` and `a1`, and `b0` and `b1`.
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn convolution_tensor<RT: GpuRuntime>(
    diag0: &DftColumn,
    diag1: &DftColumn,
    cross: &DftColumn,
    a0: &DftColumn,
    a1: &DftColumn,
    b0: &DftColumn,
    b1: &DftColumn,
    offset: usize,
) {
    assert_eq!(cross.n, diag0.n, "tensor columns of different degrees");
    assert_eq!(diag0.size, diag1.size, "the diagonal products are columns of one vector");
    let limbs = cross.size.max(diag0.size);
    launch::<RT>(
        TENSOR,
        false,
        offset,
        limbs,
        diag0,
        diag1,
        cross,
        Operands {
            a: a0,
            a2: a1,
            b: b0,
            b2: b1,
            b_plan: AutomorphismPlan::new(b0.n, 1, false),
        },
    );
}
