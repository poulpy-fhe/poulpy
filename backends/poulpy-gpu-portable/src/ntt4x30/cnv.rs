//! Limb convolution of prepared operands:
//! `res[j] = sum_{u + v = j + offset} a[u] * b[v]`, slot by slot.
//!
//! A prepared operand, left or right, is the transform of its source in the
//! layout of a DFT-domain vector. One thread per slot and prime reads its
//! residue in every operand limb once, into thread-local arrays, then walks
//! the terms of each result limb, accumulating products of canonical
//! residues in a `u64` folded every [`FOLD_TERMS`] terms. Device memory is
//! read and written once per residue, whatever the number of limbs.
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
    AutomorphismPlan, DftColumn,
    primes::{Consts, FOLD_TERMS, barrett60, fold64, sel4},
};
use crate::runtime::{GpuRuntime, Label, launch};

/// Default limbs by which the capacities of a thread's operand copies grow,
/// a module's [`crate::GpuHandle::convolution_capacity_step`], which a module
/// sets for its device.
pub(crate) const CAPACITY_STEP: usize = 8;

/// The slot of a degree-`2^log_b` right operand that `sigma_mul` moves to
/// slot `slot`, as in `automorphism_kernel`.
#[cube]
fn galois_slot(slot: u32, log_b: u32, mul: u32, shift: u32) -> u32 {
    let up = 32u32 - log_b;
    let point = slot.reverse_bits() >> up;
    let exponent = (mul * ((point << shift) + 1u32)) & ((1u32 << (log_b + shift)) - 1u32);
    (exponent >> shift).reverse_bits() >> up
}

/// `res = a * b`.
const SINGLE: u32 = 0;
/// `res = (a + a2) * (b + b2)`.
const PAIRWISE: u32 = 1;

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn convolution_kernel(
    res: &mut [u32],
    a: &[u32],
    b: &[u32],
    log_n: u32,
    limbs: u32,
    res_base: u32,
    res_stride: u32,
    a_base: u32,
    a2_base: u32,
    a_stride: u32,
    a_size: u32,
    // The limbs of `a` some term reads: `a_len` of them from `a_first`.
    a_first: u32,
    a_len: u32,
    b_base: u32,
    b2_base: u32,
    b_stride: u32,
    b_size: u32,
    b_first: u32,
    b_len: u32,
    // log2 of the ratio between the degree of the product and that of `b`.
    b_gap: u32,
    // The automorphism `b` is read through.
    b_mul: u32,
    b_shift: u32,
    offset: u32,
    #[comptime] c: Consts,
    #[comptime] form: u32,
    #[comptime] accumulate: bool,
    // Capacity of the thread's copies of the operands, at least `a_len` and `b_len`.
    #[comptime] a_cap: usize,
    #[comptime] b_cap: usize,
) {
    let word = ABSOLUTE_POS as u32;
    if word < (4u32 << log_n) {
        let plane = word >> log_n;
        let slot = word - (plane << log_n);
        let log_b = log_n - b_gap;
        let b_word = (plane << log_b) + galois_slot(slot >> b_gap, log_b, b_mul, b_shift);
        let q = sel4(plane, c.q);
        let r = sel4(plane, c.r);
        let mu = sel4(plane, c.mu60);

        // The residues of this slot in every limb a term reads, read once.
        // The pairwise form keeps the sums of its two columns.
        let mut x = Array::<u32>::new(a_cap);
        let mut y = Array::<u32>::new(b_cap);
        for t in 0..a_len {
            let index = (a_first + t) * a_stride + word;
            let mut value = a[(a_base + index) as usize];
            if comptime!(form == PAIRWISE) {
                let sum = value + a[(a2_base + index) as usize];
                value = min(sum, sum - q);
            }
            x[t as usize] = value;
        }
        for t in 0..b_len {
            let index = (b_first + t) * b_stride + b_word;
            let mut value = b[(b_base + index) as usize];
            if comptime!(form == PAIRWISE) {
                let sum = value + b[(b2_base + index) as usize];
                value = min(sum, sum - q);
            }
            y[t as usize] = value;
        }

        for j in 0..limbs {
            // Terms `a[top - v] * b[v]` for `v` in `v..end`.
            let top = j + offset;
            let mut v = top + 1u32 - min(top + 1u32, a_size);
            let end = min(top + 1u32, b_size);
            let mut acc = 0u64;
            while v < end {
                let block_end = min(v + FOLD_TERMS, end);
                acc = fold64(acc, r);
                while v < block_end {
                    let u = (top - v - a_first) as usize;
                    let w = (v - b_first) as usize;
                    acc += u64::cast_from(x[u]) * u64::cast_from(y[w]);
                    v += 1u32;
                }
            }
            let mut value = barrett60(fold64(acc, r), q, mu);
            let index = res_base + j * res_stride + word;
            if accumulate {
                let sum = value + res[index as usize];
                value = min(sum, sum - q);
            }
            res[index as usize] = value;
        }
    }
}

/// The shape of a tensor product, fixed at compile time for [`tensor_kernel`].
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct TensorShape {
    limbs: u32,
    offset: u32,
    a_size: u32,
    a_first: u32,
    a_len: u32,
    b_size: u32,
    b_first: u32,
    b_len: u32,
    res_size: u32,
    cross_size: u32,
    /// The right columns are the left ones.
    square: bool,
}

impl TensorShape {
    /// The terms `a[top - v] * b[v]` of result limb `j`: `v` in `start..end`.
    fn terms(&self, j: u32) -> (u32, u32) {
        let top = j + self.offset;
        (top + 1 - (top + 1).min(self.a_size), (top + 1).min(self.b_size))
    }
}

/// The products of a rank-1 tensor: `res = a * b`, `res2 = a2 * b2` and
/// `cross = (a + a2) * (b + b2) - res - res2`. The square form, for `b = a`
/// and `b2 = a2`, reads the operand once, takes each pair of limbs of the
/// squares once, doubled, and the cross column is `2 a a2`. Every loop is
/// unrolled: the operand residues of the slot are held in registers, and each
/// result limb is a straight line of products.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, unused_assignments)]
fn tensor_kernel(
    res: &mut [u32],
    cross: &mut [u32],
    a: &[u32],
    b: &[u32],
    log_n: u32,
    res_base: u32,
    res2_base: u32,
    res_stride: u32,
    cross_base: u32,
    cross_stride: u32,
    a_base: u32,
    a2_base: u32,
    a_stride: u32,
    b_base: u32,
    b2_base: u32,
    b_stride: u32,
    b_gap: u32,
    #[comptime] c: Consts,
    #[comptime] s: TensorShape,
) {
    let word = ABSOLUTE_POS as u32;
    if word < (4u32 << log_n) {
        let plane = word >> log_n;
        let slot = word - (plane << log_n);
        let b_word = (plane << (log_n - b_gap)) + (slot >> b_gap);
        let q = sel4(plane, c.q);
        let r = sel4(plane, c.r);
        let mu = sel4(plane, c.mu60);

        // Each column's residues, and the sums of the two columns of an operand.
        let mut x = Sequence::<u32>::new();
        let mut x2 = Sequence::<u32>::new();
        let mut xs = Sequence::<u32>::new();
        #[unroll]
        for t in 0..s.a_len {
            let index = (s.a_first + t) * a_stride + word;
            let first = a[(a_base + index) as usize];
            let second = a[(a2_base + index) as usize];
            let sum = first + second;
            x.push(first);
            x2.push(second);
            xs.push(min(sum, sum - q));
        }
        let mut y = Sequence::<u32>::new();
        let mut y2 = Sequence::<u32>::new();
        let mut ys = Sequence::<u32>::new();
        if comptime!(!s.square) {
            #[unroll]
            for t in 0..s.b_len {
                let index = (s.b_first + t) * b_stride + b_word;
                let first = b[(b_base + index) as usize];
                let second = b[(b2_base + index) as usize];
                let sum = first + second;
                y.push(first);
                y2.push(second);
                ys.push(min(sum, sum - q));
            }
        }

        #[unroll]
        for j in 0..s.limbs {
            let start = comptime!(s.terms(j).0);
            let end = comptime!(s.terms(j).1);
            let top = comptime!(j + s.offset);
            let mut acc = 0u64;
            let mut acc2 = 0u64;
            let mut acc_sum = 0u64;
            let mut value = 0u32;
            let mut value2 = 0u32;
            // The cross column, before the squares are taken out of it.
            let mut middle = 0u32;
            if comptime!(s.square) {
                // The pair `(u, v)` and its mirror are one product, taken below
                // the middle and doubled; the cross column is `2 a a2`.
                #[unroll]
                for i in 0..comptime!(end.saturating_sub(start)) {
                    let v = comptime!(start + i);
                    let u = comptime!((top - v - s.a_first) as usize);
                    let w = comptime!((v - s.a_first) as usize);
                    if comptime!(i > 0 && i % FOLD_TERMS == 0) {
                        acc = fold64(acc, r);
                        acc2 = fold64(acc2, r);
                        acc_sum = fold64(acc_sum, r);
                    }
                    acc_sum += u64::cast_from(*x.index(u)) * u64::cast_from(*x2.index(w));
                    if comptime!(2 * v < top) {
                        acc += u64::cast_from(*x.index(u)) * u64::cast_from(*x.index(w));
                        acc2 += u64::cast_from(*x2.index(u)) * u64::cast_from(*x2.index(w));
                    }
                }
                let mut centre = 0u64;
                let mut centre2 = 0u64;
                if comptime!(top % 2 == 0 && top / 2 >= start && top / 2 < end) {
                    let m = comptime!((top / 2 - s.a_first) as usize);
                    centre = u64::cast_from(*x.index(m)) * u64::cast_from(*x.index(m));
                    centre2 = u64::cast_from(*x2.index(m)) * u64::cast_from(*x2.index(m));
                }
                value = barrett60(fold64(2u64 * fold64(fold64(acc, r), r) + centre, r), q, mu);
                value2 = barrett60(fold64(2u64 * fold64(fold64(acc2, r), r) + centre2, r), q, mu);
                let doubled = barrett60(fold64(2u64 * fold64(fold64(acc_sum, r), r), r), q, mu);
                let sum = value + value2;
                let both = min(sum, sum - q);
                let whole = doubled + both;
                middle = min(whole, whole - q);
            } else {
                #[unroll]
                for i in 0..comptime!(end.saturating_sub(start)) {
                    let v = comptime!(start + i);
                    let u = comptime!((top - v - s.a_first) as usize);
                    let w = comptime!((v - s.b_first) as usize);
                    if comptime!(i > 0 && i % FOLD_TERMS == 0) {
                        acc = fold64(acc, r);
                        acc2 = fold64(acc2, r);
                        acc_sum = fold64(acc_sum, r);
                    }
                    acc += u64::cast_from(*x.index(u)) * u64::cast_from(*y.index(w));
                    acc2 += u64::cast_from(*x2.index(u)) * u64::cast_from(*y2.index(w));
                    acc_sum += u64::cast_from(*xs.index(u)) * u64::cast_from(*ys.index(w));
                }
                value = barrett60(fold64(acc, r), q, mu);
                value2 = barrett60(fold64(acc2, r), q, mu);
                middle = barrett60(fold64(acc_sum, r), q, mu);
            }
            if comptime!(j < s.res_size) {
                res[(res_base + j * res_stride + word) as usize] = value;
                res[(res2_base + j * res_stride + word) as usize] = value2;
                // `middle - value - value2` on canonical residues.
                let sum = value + value2;
                let both = min(sum, sum - q);
                let difference = middle - both;
                middle = min(difference, difference + q);
            }
            if comptime!(j < s.cross_size) {
                cross[(cross_base + j * cross_stride + word) as usize] = middle;
            }
        }
    }
}

/// Terms one launch of the sum sums: the width of the signatures of
/// `convolution_sum_kernel` and `convolution_scale_kernel`, which bind the
/// operands of each term as buffers of their own (`a0..a3`, `b0..b3`).
pub(crate) const SUM_TERMS: usize = 4;

/// `res[j] = sum_t sum_{u + v = j + offset} a_t[u] * sigma_t(b_t[v])`, the
/// operands of the `terms` terms having the same limb counts and degrees. Like
/// [`convolution_kernel`], with the copies of term `t` at `t * a_cap` and
/// `t * b_cap` of the thread's arrays.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, unused_assignments)]
fn convolution_sum_kernel(
    res: &mut [u32],
    a0: &[u32],
    a1: &[u32],
    a2: &[u32],
    a3: &[u32],
    b0: &[u32],
    b1: &[u32],
    b2: &[u32],
    b3: &[u32],
    log_n: u32,
    limbs: u32,
    res_base: u32,
    res_stride: u32,
    a0_base: u32,
    a0_stride: u32,
    a1_base: u32,
    a1_stride: u32,
    a2_base: u32,
    a2_stride: u32,
    a3_base: u32,
    a3_stride: u32,
    b0_base: u32,
    b0_stride: u32,
    b1_base: u32,
    b1_stride: u32,
    b2_base: u32,
    b2_stride: u32,
    b3_base: u32,
    b3_stride: u32,
    a_size: u32,
    a_first: u32,
    a_len: u32,
    b_size: u32,
    b_first: u32,
    b_len: u32,
    b_gap: u32,
    // The automorphisms the right operands are read through, of one shift.
    b0_mul: u32,
    b1_mul: u32,
    b2_mul: u32,
    b3_mul: u32,
    b_shift: u32,
    offset: u32,
    #[comptime] c: Consts,
    #[comptime] terms: u32,
    #[comptime] accumulate: bool,
    #[comptime] a_cap: usize,
    #[comptime] b_cap: usize,
) {
    let word = ABSOLUTE_POS as u32;
    if word < (4u32 << log_n) {
        let plane = word >> log_n;
        let slot = word - (plane << log_n);
        let log_b = log_n - b_gap;
        let b_slot = slot >> b_gap;
        let b_plane = plane << log_b;
        let b0_word = b_plane + galois_slot(b_slot, log_b, b0_mul, b_shift);
        let b1_word = b_plane + galois_slot(b_slot, log_b, b1_mul, b_shift);
        let b2_word = b_plane + galois_slot(b_slot, log_b, b2_mul, b_shift);
        let b3_word = b_plane + galois_slot(b_slot, log_b, b3_mul, b_shift);
        let q = sel4(plane, c.q);
        let r = sel4(plane, c.r);
        let mu = sel4(plane, c.mu60);

        let mut x = Array::<u32>::new(comptime!(terms as usize * a_cap));
        let mut y = Array::<u32>::new(comptime!(terms as usize * b_cap));
        for t in 0..a_len {
            let limb = a_first + t;
            let at = t as usize;
            x[at] = a0[(a0_base + limb * a0_stride + word) as usize];
            if terms > 1 {
                x[a_cap + at] = a1[(a1_base + limb * a1_stride + word) as usize];
            }
            if terms > 2 {
                x[comptime!(2 * a_cap) + at] = a2[(a2_base + limb * a2_stride + word) as usize];
            }
            if terms > 3 {
                x[comptime!(3 * a_cap) + at] = a3[(a3_base + limb * a3_stride + word) as usize];
            }
        }
        for t in 0..b_len {
            let limb = b_first + t;
            let at = t as usize;
            y[at] = b0[(b0_base + limb * b0_stride + b0_word) as usize];
            if terms > 1 {
                y[b_cap + at] = b1[(b1_base + limb * b1_stride + b1_word) as usize];
            }
            if terms > 2 {
                y[comptime!(2 * b_cap) + at] = b2[(b2_base + limb * b2_stride + b2_word) as usize];
            }
            if terms > 3 {
                y[comptime!(3 * b_cap) + at] = b3[(b3_base + limb * b3_stride + b3_word) as usize];
            }
        }

        for j in 0..limbs {
            let top = j + offset;
            let start = top + 1u32 - min(top + 1u32, a_size);
            let end = min(top + 1u32, b_size);
            let mut acc = 0u64;
            for term in 0..terms {
                let x_at = term as usize * a_cap;
                let y_at = term as usize * b_cap;
                let mut v = start;
                while v < end {
                    let block_end = min(v + FOLD_TERMS, end);
                    acc = fold64(acc, r);
                    while v < block_end {
                        let xv = x[x_at + (top - v - a_first) as usize];
                        let yv = y[y_at + (v - b_first) as usize];
                        acc += u64::cast_from(xv) * u64::cast_from(yv);
                        v += 1u32;
                    }
                }
            }
            let mut value = barrett60(fold64(acc, r), q, mu);
            let index = res_base + j * res_stride + word;
            if accumulate {
                let sum = value + res[index as usize];
                value = min(sum, sum - q);
            }
            res[index as usize] = value;
        }
    }
}

/// [`convolution_sum_kernel`] for right operands of one limb, the diagonals
/// of a linear transformation: `res[j] = sum_t a_t[j + offset] * sigma_t(b_t[0])`. No
/// result limb shares an operand limb with another, so one thread per result
/// residue streams through the operands.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, unused_assignments)]
fn convolution_scale_kernel(
    res: &mut [u32],
    a0: &[u32],
    a1: &[u32],
    a2: &[u32],
    a3: &[u32],
    b0: &[u32],
    b1: &[u32],
    b2: &[u32],
    b3: &[u32],
    log_n: u32,
    limbs: u32,
    res_base: u32,
    res_stride: u32,
    a0_base: u32,
    a0_stride: u32,
    a1_base: u32,
    a1_stride: u32,
    a2_base: u32,
    a2_stride: u32,
    a3_base: u32,
    a3_stride: u32,
    b0_base: u32,
    b1_base: u32,
    b2_base: u32,
    b3_base: u32,
    a_size: u32,
    b_gap: u32,
    // The automorphisms the right operands are read through, of one shift.
    b0_mul: u32,
    b1_mul: u32,
    b2_mul: u32,
    b3_mul: u32,
    b_shift: u32,
    offset: u32,
    #[comptime] c: Consts,
    #[comptime] terms: u32,
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
        let b_slot = slot >> b_gap;
        let b_plane = plane << log_b;
        let q = sel4(plane, c.q);
        let r = sel4(plane, c.r);
        let mu = sel4(plane, c.mu60);
        let u = j + offset;
        let index = res_base + j * res_stride + word;
        let mut value = 0u32;
        if u < a_size {
            // Up to four products of canonical residues: below `2^62`.
            let b0_word = b_plane + galois_slot(b_slot, log_b, b0_mul, b_shift);
            let mut acc =
                u64::cast_from(a0[(a0_base + u * a0_stride + word) as usize]) * u64::cast_from(b0[(b0_base + b0_word) as usize]);
            if terms > 1 {
                let b1_word = b_plane + galois_slot(b_slot, log_b, b1_mul, b_shift);
                acc += u64::cast_from(a1[(a1_base + u * a1_stride + word) as usize])
                    * u64::cast_from(b1[(b1_base + b1_word) as usize]);
            }
            if terms > 2 {
                let b2_word = b_plane + galois_slot(b_slot, log_b, b2_mul, b_shift);
                acc += u64::cast_from(a2[(a2_base + u * a2_stride + word) as usize])
                    * u64::cast_from(b2[(b2_base + b2_word) as usize]);
            }
            if terms > 3 {
                let b3_word = b_plane + galois_slot(b_slot, log_b, b3_mul, b_shift);
                acc += u64::cast_from(a3[(a3_base + u * a3_stride + word) as usize])
                    * u64::cast_from(b3[(b3_base + b3_word) as usize]);
            }
            value = barrett60(fold64(acc, r), q, mu);
        }
        if accumulate {
            let sum = value + res[index as usize];
            value = min(sum, sum - q);
        }
        res[index as usize] = value;
    }
}

fn launch_scale<RT: GpuRuntime>(res: &DftColumn, group: &Group, offset: usize, accumulate: bool) {
    let (a, b, _) = group.term(0);
    let (n, limbs) = (res.n, res.size);
    assert!(
        group.iter().all(|(a, _, _)| a.n == n),
        "the left operands have the degree of the product"
    );
    assert!(
        b.n.is_power_of_two() && b.n <= n,
        "the right operand degree does not divide the product degree"
    );
    assert!(
        group.iter().all(|(_, _, plan)| plan.n == b.n),
        "the right operands are read through automorphisms of their degree"
    );
    if limbs == 0 {
        return;
    }
    let term = |index: usize| group.term(index);
    let units = 4 * n * limbs;
    let read = limbs.min(a.size.saturating_sub(offset));
    let bytes = 4 * (group.len * 4 * (n * read + b.n) + units * (1 + accumulate as usize));
    launch!(
        RT, if accumulate { Label::CnvScaleAdd } else { Label::CnvScale }, bytes, convolution_scale_kernel, (units, RT::GROUP, RT::GROUP);
        [
            res.binding => u32,
            term(0).0.binding => u32,
            term(1).0.binding => u32,
            term(2).0.binding => u32,
            term(3).0.binding => u32,
            term(0).1.binding => u32,
            term(1).1.binding => u32,
            term(2).1.binding => u32,
            term(3).1.binding => u32,
        ];
        [
            n.ilog2(),
            limbs as u32,
            res.base,
            res.stride,
            term(0).0.base,
            term(0).0.stride,
            term(1).0.base,
            term(1).0.stride,
            term(2).0.base,
            term(2).0.stride,
            term(3).0.base,
            term(3).0.stride,
            term(0).1.base,
            term(1).1.base,
            term(2).1.base,
            term(3).1.base,
            a.size as u32,
            (n / b.n).ilog2(),
            term(0).2.mul,
            term(1).2.mul,
            term(2).2.mul,
            term(3).2.mul,
            term(0).2.shift,
            offset.min(a.size) as u32,
        ];
        [Consts::get(), group.len as u32, accumulate]
    );
}

/// Most products [`scale_many`] writes per launch: the length of the compile-time
/// [`Schedule`] the kernel is unrolled from. A larger batch of giant steps takes the
/// reference, one sum per step.
#[cfg_attr(not(feature = "enable-core"), allow(dead_code))]
pub(crate) const MANY_PRODUCTS: usize = 8;
/// Most left operands [`scale_many`] reads per launch: the kernel binds six left buffers.
#[cfg_attr(not(feature = "enable-core"), allow(dead_code))]
pub(crate) const MANY_LEFTS: usize = 6;
/// Most terms of a product of [`scale_many`], summed without a fold: products of
/// canonical residues are below `2^60` ([`Primes30`](poulpy_cpu_portable::kernels::ntt4x30::primes::Primes30)
/// are below `2^30`), and eight of them below `2^63`, inside the `u64` `fold64` reduces.
#[cfg_attr(not(feature = "enable-core"), allow(dead_code))]
pub(crate) const MANY_TERMS: usize = 8;

/// Which left operand and which right operand each term of each product of
/// [`scale_many`] multiplies, fixed at compile time.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(not(feature = "enable-core"), allow(dead_code))]
pub(crate) struct Schedule {
    pub(crate) products: u32,
    pub(crate) lefts: u32,
    pub(crate) counts: [u32; MANY_PRODUCTS],
    pub(crate) left: [[u32; MANY_TERMS]; MANY_PRODUCTS],
    pub(crate) right: [[u32; MANY_TERMS]; MANY_PRODUCTS],
}

/// [`convolution_scale_kernel`] for several products that share their left
/// operands: one thread per result residue reads each left operand once and
/// writes every product.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, unused_assignments)]
fn scale_many_kernel(
    res: &mut [u32],
    a0: &[u32],
    a1: &[u32],
    a2: &[u32],
    a3: &[u32],
    a4: &[u32],
    a5: &[u32],
    b: &[u32],
    log_n: u32,
    limbs: u32,
    res_base: u32,
    res_stride: u32,
    // Words between two products.
    res_gap: u32,
    a0_base: u32,
    a1_base: u32,
    a2_base: u32,
    a3_base: u32,
    a4_base: u32,
    a5_base: u32,
    a_stride: u32,
    a_size: u32,
    b_base: u32,
    // Words of one right operand.
    b_block: u32,
    b_gap: u32,
    offset: u32,
    #[comptime] c: Consts,
    #[comptime] s: Schedule,
) {
    let log_block = log_n + 2u32;
    let pos = ABSOLUTE_POS as u32;
    if pos < (limbs << log_block) {
        let j = pos >> log_block;
        let word = pos - (j << log_block);
        let plane = word >> log_n;
        let slot = word - (plane << log_n);
        let b_word = (plane << (log_n - b_gap)) + (slot >> b_gap);
        let q = sel4(plane, c.q);
        let r = sel4(plane, c.r);
        let mu = sel4(plane, c.mu60);
        let u = j + offset;
        let from = u * a_stride + word;

        let mut xs = Sequence::<u32>::new();
        #[unroll]
        for left in 0..s.lefts {
            let mut x = 0u32;
            if u < a_size {
                if comptime!(left == 0) {
                    x = a0[(a0_base + from) as usize];
                } else if comptime!(left == 1) {
                    x = a1[(a1_base + from) as usize];
                } else if comptime!(left == 2) {
                    x = a2[(a2_base + from) as usize];
                } else if comptime!(left == 3) {
                    x = a3[(a3_base + from) as usize];
                } else if comptime!(left == 4) {
                    x = a4[(a4_base + from) as usize];
                } else {
                    x = a5[(a5_base + from) as usize];
                }
            }
            xs.push(x);
        }

        #[unroll]
        for product in 0..s.products {
            // At most eight products of canonical residues: below `2^63`.
            let mut acc = 0u64;
            #[unroll]
            for term in 0..comptime!(s.counts[product as usize]) {
                let x = *xs.index(comptime!(s.left[product as usize][term as usize] as usize));
                let right = comptime!(s.right[product as usize][term as usize]);
                acc += u64::cast_from(x) * u64::cast_from(b[(b_base + right * b_block + b_word) as usize]);
            }
            res[(res_base + product * res_gap + j * res_stride + word) as usize] = barrett60(fold64(acc, r), q, mu);
        }
    }
}

/// `res_g[j] = sum_t lefts[left(g, t)][j + offset] * rights[right(g, t)][0]` for
/// every limb of every product `g` of `schedule`: the sums of
/// [`convolution_sum`] over right operands of one limb, for products that
/// share their left operands, each of which is read once.
///
/// Product `g` is the column at `res.base + g * res_gap`. The left operands
/// have one size; right operand `d` is the block at `rights.base + d * rights.stride`.
#[cfg_attr(not(feature = "enable-core"), allow(dead_code))]
pub(crate) fn scale_many<RT: GpuRuntime>(
    res: &DftColumn,
    res_gap: u32,
    lefts: &[DftColumn],
    rights: &DftColumn,
    schedule: Schedule,
    offset: usize,
) {
    let (n, limbs) = (res.n, res.size);
    let a = &lefts[0];
    assert_eq!(lefts.len(), schedule.lefts as usize);
    assert!(
        (1..=MANY_LEFTS).contains(&lefts.len()) && (1..=MANY_PRODUCTS as u32).contains(&schedule.products),
        "more operands or products than one launch takes"
    );
    assert!(
        lefts
            .iter()
            .all(|left| left.n == n && left.size == a.size && left.stride == a.stride),
        "the left operands have the shape of the first"
    );
    assert!(
        rights.n.is_power_of_two() && rights.n <= n,
        "the right operands have degree {} which does not divide {n}",
        rights.n
    );
    if limbs == 0 {
        return;
    }
    // Operands beyond the last repeat it: the kernel does not read them.
    let left = |index: usize| &lefts[index.min(lefts.len() - 1)];
    let units = 4 * n * limbs;
    let read = limbs.min(a.size.saturating_sub(offset));
    let terms: u32 = schedule.counts.iter().sum();
    let bytes = 4 * (4 * n * read * lefts.len() + 4 * rights.n * terms as usize + units * schedule.products as usize);
    launch!(
        RT, Label::CnvScaleMany, bytes, scale_many_kernel, (units, RT::GROUP, RT::GROUP);
        [
            res.binding => u32,
            left(0).binding => u32,
            left(1).binding => u32,
            left(2).binding => u32,
            left(3).binding => u32,
            left(4).binding => u32,
            left(5).binding => u32,
            rights.binding => u32,
        ];
        [
            n.ilog2(),
            limbs as u32,
            res.base,
            res.stride,
            res_gap,
            left(0).base,
            left(1).base,
            left(2).base,
            left(3).base,
            left(4).base,
            left(5).base,
            a.stride,
            a.size as u32,
            rights.base,
            rights.stride,
            (n / rights.n).ilog2(),
            offset.min(a.size) as u32,
        ];
        [Consts::get(), schedule]
    );
}

/// `res[j] = sum_t sum_{u + v = j + offset} a_t[u] * sigma_t(b_t[v])` for
/// every limb of `res`, over the `(a_t, b_t, sigma_t)` of `terms`, added to
/// `res` when `accumulate`. Terms of the same shape are summed [`SUM_TERMS`]
/// per launch.
///
/// Returns whether some term was launched: without one, `res` is untouched.
pub(crate) fn convolution_sum<RT: GpuRuntime>(
    res: &DftColumn,
    terms: impl IntoIterator<Item = SumTerm>,
    offset: usize,
    accumulate: bool,
    step: usize,
) -> bool {
    let mut accumulate = accumulate;
    let mut launched = false;
    let mut group = Group::default();
    for term in terms {
        if group.len == SUM_TERMS || (group.len != 0 && !group.alike(&term)) {
            group.launch::<RT>(res, offset, accumulate, step);
            (accumulate, launched) = (true, true);
            group = Group::default();
        }
        group.terms[group.len] = Some(term);
        group.len += 1;
    }
    if group.len != 0 {
        group.launch::<RT>(res, offset, accumulate, step);
        launched = true;
    }
    launched
}

/// One term `a * sigma(b)` of [`convolution_sum`].
pub(crate) type SumTerm = (DftColumn, DftColumn, AutomorphismPlan);

/// Consecutive terms of one shape, at most [`SUM_TERMS`], launched together.
#[derive(Default)]
struct Group {
    terms: [Option<SumTerm>; SUM_TERMS],
    len: usize,
}

impl Group {
    /// Term `index`, terms beyond the last repeating it, which the kernel does not read.
    fn term(&self, index: usize) -> &SumTerm {
        self.terms[index.min(self.len - 1)]
            .as_ref()
            .expect("a group holds its first terms")
    }

    fn iter(&self) -> impl Iterator<Item = &SumTerm> {
        self.terms[..self.len].iter().flatten()
    }

    /// Whether `term` has the shape of the group's terms.
    fn alike(&self, (a_t, b_t, plan_t): &SumTerm) -> bool {
        let (a, b, b_plan) = self.term(0);
        a_t.size == a.size && b_t.size == b.size && b_t.n == b.n && plan_t.shift == b_plan.shift
    }

    fn launch<RT: GpuRuntime>(&self, res: &DftColumn, offset: usize, accumulate: bool, step: usize) {
        let (a, b, b_plan) = self.term(0);
        if b.size == 1 {
            launch_scale::<RT>(res, self, offset, accumulate);
        } else if self.len == 1 {
            convolution::<RT>(res, a, b, b_plan, offset, accumulate, step);
        } else {
            launch_sum::<RT>(res, self, offset, accumulate, step);
        }
    }
}

fn launch_sum<RT: GpuRuntime>(res: &DftColumn, group: &Group, offset: usize, accumulate: bool, step: usize) {
    let (a, b, _) = group.term(0);
    let (n, limbs) = (res.n, res.size);
    assert!(
        group.iter().all(|(a, _, _)| a.n == n),
        "the left operands have the degree of the product"
    );
    assert!(
        b.n.is_power_of_two() && b.n <= n,
        "the right operand degree does not divide the product degree"
    );
    assert!(
        group.iter().all(|(_, _, plan)| plan.n == b.n),
        "the right operands are read through automorphisms of their degree"
    );
    if limbs == 0 {
        return;
    }
    let offset = offset.min(a.size + b.size);
    let ranges = Ranges::new(limbs, offset, a.size, b.size);
    let term = |index: usize| group.term(index);
    let units = 4 * n;
    let bytes = 4 * (group.len * (units * ranges.a_len + 4 * b.n * ranges.b_len) + units * limbs * (1 + accumulate as usize));
    launch!(
        RT, if accumulate { Label::CnvSumAdd } else { Label::CnvSum }, bytes, convolution_sum_kernel, (units, RT::GROUP, RT::GROUP);
        [
            res.binding => u32,
            term(0).0.binding => u32,
            term(1).0.binding => u32,
            term(2).0.binding => u32,
            term(3).0.binding => u32,
            term(0).1.binding => u32,
            term(1).1.binding => u32,
            term(2).1.binding => u32,
            term(3).1.binding => u32,
        ];
        [
            n.ilog2(),
            limbs as u32,
            res.base,
            res.stride,
            term(0).0.base,
            term(0).0.stride,
            term(1).0.base,
            term(1).0.stride,
            term(2).0.base,
            term(2).0.stride,
            term(3).0.base,
            term(3).0.stride,
            term(0).1.base,
            term(0).1.stride,
            term(1).1.base,
            term(1).1.stride,
            term(2).1.base,
            term(2).1.stride,
            term(3).1.base,
            term(3).1.stride,
            a.size as u32,
            ranges.a_first as u32,
            ranges.a_len as u32,
            b.size as u32,
            ranges.b_first as u32,
            ranges.b_len as u32,
            (n / b.n).ilog2(),
            term(0).2.mul,
            term(1).2.mul,
            term(2).2.mul,
            term(3).2.mul,
            term(0).2.shift,
            offset as u32,
        ];
        [Consts::get(), group.len as u32, accumulate, ranges.a_cap(step), ranges.b_cap(step)]
    );
}

/// The operand limbs some term of a convolution reads: `u + v = j + offset`
/// over the `limbs` limbs `j` written.
struct Ranges {
    a_first: usize,
    a_len: usize,
    b_first: usize,
    b_len: usize,
}

impl Ranges {
    fn new(limbs: usize, offset: usize, a_size: usize, b_size: usize) -> Self {
        let top = limbs + offset;
        let a = (offset + 1).saturating_sub(b_size).min(a_size)..top.min(a_size);
        let b = (offset + 1).saturating_sub(a_size).min(b_size)..top.min(b_size);
        Self {
            a_first: a.start,
            a_len: a.len(),
            b_first: b.start,
            b_len: b.len(),
        }
    }

    /// Capacity of a thread's copy of `a`: in steps of `step` limbs, which
    /// keeps the number of compiled kernels small.
    fn a_cap(&self, step: usize) -> usize {
        self.a_len.max(1).next_multiple_of(step)
    }

    fn b_cap(&self, step: usize) -> usize {
        self.b_len.max(1).next_multiple_of(step)
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

impl Operands<'_> {
    /// Checks the operands of a product of degree `n`.
    fn check(&self, n: usize) {
        assert!(self.a.n == n, "the left operand has the degree of the product");
        assert!(
            self.b.n.is_power_of_two() && self.b.n <= n,
            "the right operand degree does not divide the product degree"
        );
        assert!(
            self.b_plan.n == self.b.n,
            "the right operand is read through an automorphism of its degree"
        );
    }
}

/// The single and pairwise forms, over the `limbs` limbs of `res` from the
/// first, the capacities of a thread's operand copies growing by `step` limbs.
#[allow(clippy::too_many_arguments)]
fn launch<RT: GpuRuntime>(
    form: u32,
    accumulate: bool,
    offset: usize,
    limbs: usize,
    res: &DftColumn,
    operands: Operands<'_>,
    step: usize,
) {
    operands.check(res.n);
    let Operands { a, a2, b, b2, b_plan } = operands;
    if limbs == 0 {
        return;
    }
    let n = res.n;
    let offset = offset.min(a.size + b.size);
    let ranges = Ranges::new(limbs, offset, a.size, b.size);
    let (a_len, b_len) = (ranges.a_len, ranges.b_len);
    let columns = if form == SINGLE { 1 } else { 2 };
    let written = limbs * (1 + accumulate as usize);
    let units = 4 * n;
    let label = match (form, accumulate) {
        (PAIRWISE, _) => Label::CnvPairwise,
        (_, true) => Label::CnvSingleAdd,
        _ => Label::CnvSingle,
    };
    launch!(
        RT, label, 4 * (units * (a_len * columns + written) + 4 * b.n * b_len * columns), convolution_kernel, (units, RT::GROUP, RT::GROUP);
        [
            res.binding => u32,
            a.binding => u32,
            b.binding => u32,
        ];
        [
            n.ilog2(),
            limbs as u32,
            res.base,
            res.stride,
            a.base,
            a2.base,
            a.stride,
            a.size as u32,
            ranges.a_first as u32,
            a_len as u32,
            b.base,
            b2.base,
            b.stride,
            b.size as u32,
            ranges.b_first as u32,
            b_len as u32,
            (n / b.n).ilog2(),
            b_plan.mul,
            b_plan.shift,
            offset as u32,
        ];
        [Consts::get(), form, accumulate, ranges.a_cap(step), ranges.b_cap(step)]
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
    step: usize,
) {
    launch::<RT>(
        SINGLE,
        accumulate,
        offset,
        res.size,
        res,
        Operands {
            a,
            a2: a,
            b,
            b2: b,
            b_plan: *b_plan,
        },
        step,
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
    step: usize,
) {
    let b_plan = AutomorphismPlan::new(b.n, 1, false);
    launch::<RT>(
        PAIRWISE,
        false,
        offset,
        res.size,
        res,
        Operands { a, a2, b, b2, b_plan },
        step,
    );
}

/// The three products of a rank-1 tensor: `diag0 = a0 * b0`, `diag1 = a1 * b1`
/// and `cross = (a0 + a1) * (b0 + b1) - diag0 - diag1`, every limb of each
/// being written and the subtraction covering the limbs `diag` has. `diag0` and `diag1` are columns of one
/// vector, as are `a0` and `a1`, and `b0` and `b1`. Right columns that are the
/// left ones make a square, which reads them once.
#[allow(clippy::too_many_arguments)]
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
    assert!(cross.n == diag0.n, "tensor columns of different degrees");
    assert!(diag0.size == diag1.size, "the diagonal products are columns of one vector");
    let operands = Operands {
        a: a0,
        a2: a1,
        b: b0,
        b2: b1,
        b_plan: AutomorphismPlan::new(b0.n, 1, false),
    };
    operands.check(diag0.n);
    let limbs = cross.size.max(diag0.size);
    if limbs == 0 {
        return;
    }
    let same = |x: &DftColumn, y: &DftColumn| {
        x.binding.id == y.binding.id && x.base == y.base && x.stride == y.stride && x.size == y.size && x.n == y.n
    };
    let square = same(a0, b0) && same(a1, b1);
    let n = diag0.n;
    let offset = offset.min(a0.size + b0.size);
    let ranges = Ranges::new(limbs, offset, a0.size, b0.size);
    let narrow = |value: usize| value as u32;
    let shape = TensorShape {
        limbs: narrow(limbs),
        offset: narrow(offset),
        a_size: narrow(a0.size),
        a_first: narrow(ranges.a_first),
        a_len: narrow(ranges.a_len),
        b_size: narrow(b0.size),
        b_first: narrow(ranges.b_first),
        b_len: narrow(ranges.b_len),
        res_size: narrow(diag0.size),
        cross_size: narrow(cross.size),
        square,
    };
    let units = 4 * n;
    let written = 2 * diag0.size + cross.size;
    let b_read = if square { 0 } else { 4 * b0.n * ranges.b_len * 2 };
    let group = RT::GROUP;
    launch!(
        RT, Label::CnvTensor, 4 * (units * (ranges.a_len * 2 + written) + b_read), tensor_kernel, (units, group, group);
        [
            diag0.binding => u32,
            cross.binding => u32,
            a0.binding => u32,
            b0.binding => u32,
        ];
        [
            n.ilog2(),
            diag0.base,
            diag1.base,
            diag0.stride,
            cross.base,
            cross.stride,
            a0.base,
            a1.base,
            a0.stride,
            b0.base,
            b1.base,
            b0.stride,
            (n / b0.n).ilog2(),
        ];
        [Consts::get(), shape]
    );
}
