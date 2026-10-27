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

/// Step of the capacity of a thread's operand copies, in limbs.
const CAPACITY_STEP: usize = 8;

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
        // Only the tensor form keeps its second columns apart.
        let mut x2 = Array::<u32>::new(comptime!(if form == TENSOR { a_cap } else { 1 }));
        let mut y2 = Array::<u32>::new(comptime!(if form == TENSOR { b_cap } else { 1 }));
        for t in 0..a_len {
            let index = (a_first + t) * a_stride + word;
            let mut value = a[(a_base + index) as usize];
            if form != SINGLE {
                let second = a[(a2_base + index) as usize];
                if form == TENSOR {
                    x2[t as usize] = second;
                } else {
                    let sum = value + second;
                    value = min(sum, sum - q);
                }
            }
            x[t as usize] = value;
        }
        for t in 0..b_len {
            let index = (b_first + t) * b_stride + b_word;
            let mut value = b[(b_base + index) as usize];
            if form != SINGLE {
                let second = b[(b2_base + index) as usize];
                if form == TENSOR {
                    y2[t as usize] = second;
                } else {
                    let sum = value + second;
                    value = min(sum, sum - q);
                }
            }
            y[t as usize] = value;
        }

        for j in 0..limbs {
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
                    let u = (top - v - a_first) as usize;
                    let w = (v - b_first) as usize;
                    let xv = x[u];
                    let yv = y[w];
                    if form == TENSOR {
                        let xv2 = x2[u];
                        let yv2 = y2[w];
                        let x_sum = xv + xv2;
                        let y_sum = yv + yv2;
                        let xs = min(x_sum, x_sum - q);
                        let ys = min(y_sum, y_sum - q);
                        acc2 += u64::cast_from(xv2) * u64::cast_from(yv2);
                        acc_sum += u64::cast_from(xs) * u64::cast_from(ys);
                    }
                    acc += u64::cast_from(xv) * u64::cast_from(yv);
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
) -> bool {
    let mut accumulate = accumulate;
    let mut launched = false;
    let mut group = Group::default();
    for term in terms {
        if group.len == SUM_TERMS || (group.len != 0 && !group.alike(&term)) {
            group.launch::<RT>(res, offset, accumulate);
            (accumulate, launched) = (true, true);
            group = Group::default();
        }
        group.terms[group.len] = Some(term);
        group.len += 1;
    }
    if group.len != 0 {
        group.launch::<RT>(res, offset, accumulate);
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

    fn launch<RT: GpuRuntime>(&self, res: &DftColumn, offset: usize, accumulate: bool) {
        let (a, b, b_plan) = self.term(0);
        if b.size == 1 {
            launch_scale::<RT>(res, self, offset, accumulate);
        } else if self.len == 1 {
            convolution::<RT>(res, a, b, b_plan, offset, accumulate);
        } else {
            launch_sum::<RT>(res, self, offset, accumulate);
        }
    }
}

fn launch_sum<RT: GpuRuntime>(res: &DftColumn, group: &Group, offset: usize, accumulate: bool) {
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
        [Consts::get(), group.len as u32, accumulate, ranges.a_cap(), ranges.b_cap()]
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

    /// Capacity of a thread's copy of `a`: in steps of [`CAPACITY_STEP`]
    /// limbs, which keeps the number of compiled kernels small.
    fn a_cap(&self) -> usize {
        self.a_len.max(1).next_multiple_of(CAPACITY_STEP)
    }

    fn b_cap(&self) -> usize {
        self.b_len.max(1).next_multiple_of(CAPACITY_STEP)
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
    assert!(a.n == n, "the left operand has the degree of the product");
    assert!(
        b.n.is_power_of_two() && b.n <= n,
        "the right operand degree does not divide the product degree"
    );
    assert!(
        b_plan.n == b.n,
        "the right operand is read through an automorphism of its degree"
    );
    if limbs == 0 {
        return;
    }
    let offset = offset.min(a.size + b.size);
    let ranges = Ranges::new(limbs, offset, a.size, b.size);
    let (a_len, b_len) = (ranges.a_len, ranges.b_len);
    let columns = if form == SINGLE { 1 } else { 2 };
    let written = if form == TENSOR {
        2 * res.size + cross.size
    } else {
        limbs * (1 + accumulate as usize)
    };
    let units = 4 * n;
    let label = match (form, accumulate) {
        (TENSOR, _) => Label::CnvTensor,
        (PAIRWISE, _) => Label::CnvPairwise,
        (_, true) => Label::CnvSingleAdd,
        _ => Label::CnvSingle,
    };
    launch!(
        RT, label, 4 * (units * (a_len * columns + written) + 4 * b.n * b_len * columns), convolution_kernel, (units, RT::GROUP, RT::GROUP);
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
        [Consts::get(), form, accumulate, ranges.a_cap(), ranges.b_cap()]
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
    assert!(cross.n == diag0.n, "tensor columns of different degrees");
    assert!(diag0.size == diag1.size, "the diagonal products are columns of one vector");
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
