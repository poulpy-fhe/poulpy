//! Vector-matrix product over interleaved gadget digits.
//!
//! A prepared matrix holds, for every `(row, input column)` entry, a
//! DFT-domain vector of `cols_out` columns and `size` limbs in the layout of
//! such a vector: block `((row * cols_in + col_in) * size + limb) * cols_out + col_out`,
//! four planes of canonical residues each. Preparing is one batched forward
//! transform of the coefficient-domain matrix, which has the same block order.
//!
//! The product: digit `di` of the source `a` gathers its limbs congruent to
//! `dsize - 1 - di` modulo `dsize`, row `r` of the digit being limb
//! `dsize - 1 - di + r * dsize`, and
//!
//! `res[d][j] = sum_di sum_{r, c} a[c][dsize - 1 - di + r dsize] * K[r][c][d][j + di + limb_offset]`
//!
//! where digit `di` contributes to the result limbs `j < w_di` only (see
//! [`digit_product`]). One thread per (prime, coefficient) loads the source
//! limbs once into registers, then walks the matrix limbs once for every
//! output column: matrix limb `h` contributes through digit `di` to result limb
//! `h - limb_offset - di`, so the thread keeps a window of `dsize`
//! accumulators per output column, stores the oldest after each matrix limb
//! and shifts the window. Each matrix element is read exactly once. Accumulators are `u64`
//! sums of products of canonical residues, folded every [`FOLD_TERMS`] terms.
//!
//! The kernel is specialized for the terms of one result limb: the number of
//! source limbs it reads, the input and output columns and the digit size.

use cubecl::prelude::*;

use super::{
    AutomorphismPlan, DftColumn,
    ntt::gs,
    primes::{Consts, FOLD_TERMS, barrett60, fold64, sel4},
};
use crate::runtime::{Binding, GpuRuntime, Label, launch};

/// A prepared matrix as a kernel addresses it.
pub(crate) struct Pmat {
    pub(crate) binding: Binding,
    /// `u32` word of block 0.
    pub(crate) base: u32,
    pub(crate) n: usize,
    pub(crate) rows: usize,
    pub(crate) cols_in: usize,
    pub(crate) cols_out: usize,
    pub(crate) size: usize,
}

impl Pmat {
    /// Every block of the matrix as the limbs of one column.
    pub(crate) fn blocks(&self) -> DftColumn {
        DftColumn {
            binding: self.binding.clone(),
            base: self.base,
            stride: (4 * self.n) as u32,
            n: self.n,
            size: self.rows * self.cols_in * self.cols_out * self.size,
        }
    }
}

/// The terms of one result limb, fixed at compile time.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct Terms {
    cols_in: u32,
    cols_out: u32,
    dsize: u32,
    /// Source limbs the product reads: `min(a.size, rows * dsize)`.
    limbs: u32,
    /// Output columns 0 and 1 that receive an addend, one bit each.
    addend: u32,
    /// Whether the result is stored through an automorphism.
    permute: bool,
}

/// What a product does with its result besides storing it.
#[derive(Default)]
pub(crate) struct Tail<'a> {
    /// Columns added to output columns 0 and 1 of the product, each over the
    /// limbs both have.
    pub(crate) addends: [Option<&'a DftColumn>; 2],
    /// The result is `sigma(product + addend)`.
    pub(crate) automorphism: Option<&'a AutomorphismPlan>,
}

impl Terms {
    /// Matrix rows the product reads.
    fn rows(&self) -> u32 {
        self.limbs.div_ceil(self.dsize)
    }

    /// Rows of digit `di`.
    fn digit_rows(&self, di: u32) -> u32 {
        (self.limbs + di) / self.dsize
    }

    /// Whether the accumulator is folded before term `(di, row, col)`.
    ///
    /// An accumulator meets the digits in order, but not all of them: a
    /// digit past its width contributes nothing, and the folds scheduled
    /// inside it do not happen. The schedule therefore tracks the most
    /// products an accumulator can hold unfolded whichever digits were
    /// skipped, and folds before the term that would be the sixteenth.
    fn fold_before(&self, di: u32, row: u32, col: u32) -> bool {
        let mut pending = 0;
        for digit in 0..self.dsize {
            let mut inside = pending;
            for term in 0..self.digit_rows(digit) * self.cols_in {
                let fold = inside == FOLD_TERMS;
                if fold {
                    inside = 0;
                }
                if digit == di && term == row * self.cols_in + col {
                    return fold;
                }
                inside += 1;
            }
            // A skipped digit leaves what was pending before it.
            pending = pending.max(inside);
        }
        false
    }
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
fn product_kernel(
    res: &mut [u32],
    a: &[u32],
    key: &[u32],
    add0: &[u32],
    add1: &[u32],
    log_n: u32,
    res_base: u32,
    res_size: u32,
    a_base: u32,
    key_base: u32,
    key_size: u32,
    limb_offset: u32,
    product_limbs: u32,
    add0_base: u32,
    add0_stride: u32,
    add0_size: u32,
    add1_base: u32,
    add1_stride: u32,
    add1_size: u32,
    mul: u32,
    shift: u32,
    #[comptime] c: Consts,
    #[comptime] t: Terms,
    #[comptime] accumulate: bool,
) {
    let log_block = log_n + 2u32;
    let word = ABSOLUTE_POS as u32;
    if word < (1u32 << log_block) {
        let plane = word >> log_n;
        let q = sel4(plane, c.q);
        let r = sel4(plane, c.r);
        let mu = sel4(plane, c.mu60);
        let cols_in = t.cols_in;
        let cols_out = t.cols_out;
        let dsize = t.dsize;
        let rows = comptime!(t.rows());

        // Where the thread stores: its own word, or the word whose slot the
        // automorphism reads this one from.
        let mut out = word;
        if comptime!(t.permute) {
            let slot = word - (plane << log_n);
            let up = 32u32 - log_n;
            let point = slot.reverse_bits() >> up;
            let exponent = (mul * ((point << shift) + 1u32)) & ((1u32 << (log_n + shift)) - 1u32);
            out = (plane << log_n) + ((exponent >> shift).reverse_bits() >> up);
        }

        // Source limb `l` of input column `col` is register `l * cols_in + col`.
        let mut xs = Sequence::<u32>::new();
        #[unroll]
        for l in 0..t.limbs {
            #[unroll]
            for col in 0..cols_in {
                xs.push(a[(a_base + ((l * cols_in + col) << log_block) + word) as usize]);
            }
        }

        // Result limbs each digit contributes to.
        let reach = key_size - min(key_size, limb_offset);
        let first = min(res_size, reach);
        let mut widths = Sequence::<u32>::new();
        widths.push(first);
        #[unroll]
        for di in 1..dsize {
            let span = dsize - di;
            let omitted = span - min(span, product_limbs);
            let kept = key_size - min(key_size, omitted);
            widths.push(min(min(res_size, kept), reach - min(reach, di)));
        }

        // While matrix limb `step + limb_offset` is walked, window slot `k`
        // of output column `d` (cell `d * dsize + k`) holds result limb
        // `step - (dsize - 1) + k` of that column.
        let mut window = Sequence::<RuntimeCell<u64>>::new();
        #[unroll]
        for _ in 0..comptime!(dsize * cols_out) {
            window.push(RuntimeCell::<u64>::new(0u64));
        }

        for step in 0..first + dsize - 1u32 {
            let mut live = Sequence::<bool>::new();
            let mut any = false;
            #[unroll]
            for di in 0..dsize {
                // A wrapped difference is above every width.
                let reached = step - di < *widths.index(di as usize);
                live.push(reached);
                any = any || reached;
            }
            if any {
                let h = step + limb_offset;
                #[unroll]
                for d in 0..cols_out {
                    let mut ks = Sequence::<u32>::new();
                    #[unroll]
                    for row in 0..rows {
                        #[unroll]
                        for col in 0..cols_in {
                            let block = ((row * cols_in + col) * key_size + h) * cols_out + d;
                            ks.push(key[(key_base + (block << log_block) + word) as usize]);
                        }
                    }
                    #[unroll]
                    for di in 0..dsize {
                        if *live.index(di as usize) {
                            let slot = window.index((d * dsize + dsize - 1 - di) as usize);
                            let mut acc = slot.read();
                            #[unroll]
                            for row in 0..comptime!(t.digit_rows(di)) {
                                #[unroll]
                                for col in 0..cols_in {
                                    if comptime!(t.fold_before(di, row, col)) {
                                        acc = fold64(acc, r);
                                    }
                                    let x = *xs.index(((dsize - 1 - di + row * dsize) * cols_in + col) as usize);
                                    let k = *ks.index((row * cols_in + col) as usize);
                                    acc += u64::cast_from(x) * u64::cast_from(k);
                                }
                            }
                            slot.store(acc);
                        }
                    }
                }
            }
            #[unroll]
            for d in 0..cols_out {
                if step >= dsize - 1u32 {
                    let j = step - (dsize - 1u32);
                    let index = res_base + ((j * cols_out + d) << log_block) + out;
                    let mut value = barrett60(fold64(window.index((d * dsize) as usize).read(), r), q, mu);
                    if comptime!(t.addend >> d & 1 == 1) {
                        let mut y = 0u32;
                        if comptime!(d == 0) {
                            if j < add0_size {
                                y = add0[(add0_base + j * add0_stride + word) as usize];
                            }
                        } else if j < add1_size {
                            y = add1[(add1_base + j * add1_stride + word) as usize];
                        }
                        let sum = value + y;
                        value = min(sum, sum - q);
                    }
                    if accumulate {
                        let sum = value + res[index as usize];
                        value = min(sum, sum - q);
                    }
                    res[index as usize] = value;
                }
                #[unroll]
                for k in 1..dsize {
                    window
                        .index((d * dsize + k - 1) as usize)
                        .store(window.index((d * dsize + k) as usize).read());
                }
                window.index((d * dsize + dsize - 1) as usize).store(0u64);
            }
        }
        // Past the product: the addend alone, or zero.
        #[unroll]
        for d in 0..cols_out {
            let added = comptime!(t.addend >> d & 1 == 1);
            if comptime!(added || !accumulate) {
                for j in first..res_size {
                    let index = res_base + ((j * cols_out + d) << log_block) + out;
                    let mut value = 0u32;
                    if added {
                        if comptime!(d == 0) {
                            if j < add0_size {
                                value = add0[(add0_base + j * add0_stride + word) as usize];
                            }
                        } else if j < add1_size {
                            value = add1[(add1_base + j * add1_stride + word) as usize];
                        }
                    }
                    if accumulate {
                        let sum = value + res[index as usize];
                        value = min(sum, sum - q);
                    }
                    res[index as usize] = value;
                }
            }
        }
    }
}

/// The product of `a` by `pmat` over `dsize` interleaved digits, into every
/// column of `res`. `res` and `a` are column 0 of dense vectors of
/// `pmat.cols_out` and `pmat.cols_in` columns.
///
/// Digit 0 contributes to the result limbs below
/// `w_0 = min(res.size, pmat.size - limb_offset)`; digit `di >= 1` to those below
/// `min(res.size, pmat.size - max(dsize - di - product_limbs, 0), pmat.size - limb_offset - di)`.
/// Without `accumulate` every limb of `res` is written, those from `w_0` on
/// with zero. With `accumulate` the product is added to `res`, whose limbs
/// from `w_0` on are left untouched.
///
/// With a [`Tail`] the value stored or added is `sigma(product + addend)`,
/// the addend reaching past `w_0` over the limbs it has.
#[allow(clippy::too_many_arguments)]
pub(crate) fn digit_product<RT: GpuRuntime>(
    res: &DftColumn,
    a: &DftColumn,
    pmat: &Pmat,
    dsize: usize,
    product_limbs: usize,
    limb_offset: usize,
    accumulate: bool,
    tail: Tail<'_>,
) {
    let n = pmat.n;
    let block = 4 * n;
    assert!(dsize >= 1, "digit size must be positive");
    assert!(res.n == n && a.n == n, "vector-matrix product between different degrees");
    assert!(a.stride as usize == pmat.cols_in * block, "a.cols() != pmat.cols_in()");
    assert!(res.stride as usize == pmat.cols_out * block, "res.cols() != pmat.cols_out()");
    let limbs = a.size.min(pmat.rows * dsize);
    if res.size == 0 {
        return;
    }
    for (col, addend) in tail.addends.iter().enumerate() {
        if let Some(addend) = addend {
            assert_eq!(addend.n, n, "vector-matrix product: addend of another degree");
            assert!(col < pmat.cols_out, "vector-matrix product: addend column out of range");
        }
    }
    if let Some(plan) = tail.automorphism {
        assert_eq!(plan.n(), n, "vector-matrix product: automorphism plan of another degree");
    }
    let plain = tail.addends.iter().all(Option::is_none) && tail.automorphism.is_none();
    assert!(
        plain || (limbs != 0 && limb_offset < pmat.size),
        "vector-matrix product: a tail needs a product"
    );
    if limbs == 0 || limb_offset >= pmat.size {
        // No term: the product is zero.
        if !accumulate {
            let all = DftColumn {
                binding: res.binding.clone(),
                base: res.base,
                stride: block as u32,
                n,
                size: res.size * pmat.cols_out,
            };
            super::combine::<RT>(&all, all.size, super::DftTerm::zero(&all), super::DftTerm::zero(&all));
        }
        return;
    }
    let narrow = |value: usize| u32::try_from(value).expect("vector-matrix product beyond the 32-bit range");
    narrow(pmat.base as usize + pmat.blocks().size * block);
    let terms = Terms {
        cols_in: narrow(pmat.cols_in),
        cols_out: narrow(pmat.cols_out),
        dsize: narrow(dsize),
        limbs: narrow(limbs),
        addend: tail.addends[0].is_some() as u32 | (tail.addends[1].is_some() as u32) << 1,
        permute: tail.automorphism.is_some(),
    };
    // An absent addend is bound to a buffer of the launch and never read.
    let [(add0, add0_size), (add1, add1_size)] = tail
        .addends
        .map(|addend| addend.map_or((a, 0), |addend| (addend, addend.size.min(res.size))));
    let (mul, shift) = tail.automorphism.map_or((1, 1), AutomorphismPlan::scatter);
    let units = block;
    let key_limbs = pmat.size.min(res.size + dsize - 1);
    let words = limbs * pmat.cols_in
        + pmat.cols_out * (terms.rows() as usize * pmat.cols_in * key_limbs + res.size * (1 + accumulate as usize))
        + add0_size
        + add1_size;
    launch!(
        RT, if accumulate { Label::VmpDigitProductAdd } else { Label::VmpDigitProduct }, units * 4 * words, product_kernel, (units, RT::GROUP, RT::GROUP);
        [
            res.binding => u32,
            a.binding => u32,
            pmat.binding => u32,
            add0.binding => u32,
            add1.binding => u32,
        ];
        [
            n.ilog2(),
            res.base,
            narrow(res.size),
            a.base,
            pmat.base,
            narrow(pmat.size),
            narrow(limb_offset),
            narrow(product_limbs.min(dsize)),
            add0.base,
            add0.stride,
            narrow(add0_size),
            add1.base,
            add1.stride,
            narrow(add1_size),
            mul,
            shift,
        ];
        [Consts::get(), terms, accumulate]
    );
}

/// Most sources [`digit_products`] takes per launch.
pub(crate) const MANY_SOURCES: usize = 4;

/// Base word `s` of four, chosen at compile time.
#[cube]
fn pick4(b0: u32, b1: u32, b2: u32, b3: u32, #[comptime] s: u32) -> u32 {
    let mut base = b3;
    if comptime!(s == 0) {
        base = b0;
    } else if comptime!(s == 1) {
        base = b1;
    } else if comptime!(s == 2) {
        base = b2;
    }
    base
}

/// [`product_kernel`] for `m` sources against one key, at most [`MANY_SOURCES`]: source `s`
/// reads `a` from word `a_base{s}`, an addend on output column 0 from `add_base{s}` and stores
/// from `res_base{s}`, every other argument shared. Each matrix word is loaded once for all
/// the sources.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
fn product_many_kernel(
    res: &mut [u32],
    a: &[u32],
    key: &[u32],
    add: &[u32],
    log_n: u32,
    res_size: u32,
    key_base: u32,
    key_size: u32,
    product_limbs: u32,
    add_stride: u32,
    add_size: u32,
    mul: u32,
    shift: u32,
    res_base0: u32,
    res_base1: u32,
    res_base2: u32,
    res_base3: u32,
    a_base0: u32,
    a_base1: u32,
    a_base2: u32,
    a_base3: u32,
    add_base0: u32,
    add_base1: u32,
    add_base2: u32,
    add_base3: u32,
    #[comptime] c: Consts,
    #[comptime] t: Terms,
    #[comptime] m: u32,
    #[comptime] accumulate: bool,
) {
    let log_block = log_n + 2u32;
    let word = ABSOLUTE_POS as u32;
    if word < (1u32 << log_block) {
        let plane = word >> log_n;
        let q = sel4(plane, c.q);
        let r = sel4(plane, c.r);
        let mu = sel4(plane, c.mu60);
        let cols_in = t.cols_in;
        let cols_out = t.cols_out;
        let dsize = t.dsize;
        let rows = comptime!(t.rows());

        let mut out = word;
        if comptime!(t.permute) {
            let slot = word - (plane << log_n);
            let up = 32u32 - log_n;
            let point = slot.reverse_bits() >> up;
            let exponent = (mul * ((point << shift) + 1u32)) & ((1u32 << (log_n + shift)) - 1u32);
            out = (plane << log_n) + ((exponent >> shift).reverse_bits() >> up);
        }

        // Source limb `l` of input column `col` of source `s` is register `(s * limbs + l) * cols_in + col`.
        let mut xs = Sequence::<u32>::new();
        #[unroll]
        for s in 0..m {
            let a_base = pick4(a_base0, a_base1, a_base2, a_base3, s);
            #[unroll]
            for l in 0..t.limbs {
                #[unroll]
                for col in 0..cols_in {
                    xs.push(a[(a_base + ((l * cols_in + col) << log_block) + word) as usize]);
                }
            }
        }

        let first = min(res_size, key_size);
        let mut widths = Sequence::<u32>::new();
        widths.push(first);
        #[unroll]
        for di in 1..dsize {
            let span = dsize - di;
            let omitted = span - min(span, product_limbs);
            let kept = key_size - min(key_size, omitted);
            widths.push(min(min(res_size, kept), key_size - min(key_size, di)));
        }

        // Window slot `k` of output column `d` of source `s` is cell `(s * cols_out + d) * dsize + k`.
        let mut window = Sequence::<RuntimeCell<u64>>::new();
        #[unroll]
        for _ in 0..comptime!(m * t.dsize * t.cols_out) {
            window.push(RuntimeCell::<u64>::new(0u64));
        }

        for step in 0..first + dsize - 1u32 {
            let mut live = Sequence::<bool>::new();
            let mut any = false;
            #[unroll]
            for di in 0..dsize {
                let reached = step - di < *widths.index(di as usize);
                live.push(reached);
                any = any || reached;
            }
            if any {
                #[unroll]
                for d in 0..cols_out {
                    let mut ks = Sequence::<u32>::new();
                    #[unroll]
                    for row in 0..rows {
                        #[unroll]
                        for col in 0..cols_in {
                            let block = ((row * cols_in + col) * key_size + step) * cols_out + d;
                            ks.push(key[(key_base + (block << log_block) + word) as usize]);
                        }
                    }
                    #[unroll]
                    for s in 0..m {
                        #[unroll]
                        for di in 0..dsize {
                            if *live.index(di as usize) {
                                let slot = window.index(((s * cols_out + d) * dsize + dsize - 1 - di) as usize);
                                let mut acc = slot.read();
                                #[unroll]
                                for row in 0..comptime!(t.digit_rows(di)) {
                                    #[unroll]
                                    for col in 0..cols_in {
                                        if comptime!(t.fold_before(di, row, col)) {
                                            acc = fold64(acc, r);
                                        }
                                        let x =
                                            *xs.index(((s * t.limbs + dsize - 1 - di + row * dsize) * cols_in + col) as usize);
                                        let k = *ks.index((row * cols_in + col) as usize);
                                        acc += u64::cast_from(x) * u64::cast_from(k);
                                    }
                                }
                                slot.store(acc);
                            }
                        }
                    }
                }
            }
            #[unroll]
            for s in 0..m {
                let res_base = pick4(res_base0, res_base1, res_base2, res_base3, s);
                let add_base = pick4(add_base0, add_base1, add_base2, add_base3, s);
                #[unroll]
                for d in 0..cols_out {
                    let w0 = (s * cols_out + d) * dsize;
                    if step >= dsize - 1u32 {
                        let j = step - (dsize - 1u32);
                        let index = res_base + ((j * cols_out + d) << log_block) + out;
                        let mut value = barrett60(fold64(window.index(w0 as usize).read(), r), q, mu);
                        if comptime!(d == 0 && t.addend & 1 == 1) {
                            if j < add_size {
                                let sum = value + add[(add_base + j * add_stride + word) as usize];
                                value = min(sum, sum - q);
                            }
                        }
                        if accumulate {
                            let sum = value + res[index as usize];
                            value = min(sum, sum - q);
                        }
                        res[index as usize] = value;
                    }
                    #[unroll]
                    for k in 1..dsize {
                        window
                            .index((w0 + k - 1) as usize)
                            .store(window.index((w0 + k) as usize).read());
                    }
                    window.index((w0 + dsize - 1) as usize).store(0u64);
                }
            }
        }
        // Past the product: the addend alone, or zero.
        #[unroll]
        for s in 0..m {
            let res_base = pick4(res_base0, res_base1, res_base2, res_base3, s);
            let add_base = pick4(add_base0, add_base1, add_base2, add_base3, s);
            #[unroll]
            for d in 0..cols_out {
                let added = comptime!(d == 0 && t.addend & 1 == 1);
                if comptime!(added || !accumulate) {
                    for j in first..res_size {
                        let index = res_base + ((j * cols_out + d) << log_block) + out;
                        let mut value = 0u32;
                        if added {
                            if j < add_size {
                                value = add[(add_base + j * add_stride + word) as usize];
                            }
                        }
                        if accumulate {
                            let sum = value + res[index as usize];
                            value = min(sum, sum - q);
                        }
                        res[index as usize] = value;
                    }
                }
            }
        }
    }
}

/// [`digit_product`] with no limb offset for the sources `a` against one key, at most
/// [`MANY_SOURCES`], into `res`, with `addends` on output column 0 when given and the
/// automorphism `automorphism` shared: one launch that reads `pmat` once. The views of each
/// role share an allocation, a layout and a size.
#[allow(clippy::too_many_arguments)]
pub(crate) fn digit_products<RT: GpuRuntime>(
    res: &[DftColumn],
    a: &[DftColumn],
    pmat: &Pmat,
    dsize: usize,
    product_limbs: usize,
    accumulate: bool,
    addends: Option<&[DftColumn]>,
    automorphism: Option<&AutomorphismPlan>,
) {
    let m = res.len();
    assert!((1..=MANY_SOURCES).contains(&m), "vector-matrix products: {m} sources");
    assert_eq!(
        a.len(),
        m,
        "vector-matrix products: sources and destinations differ in number"
    );
    let n = pmat.n;
    let block = 4 * n;
    assert!(dsize >= 1, "digit size must be positive");
    let alike = |views: &[DftColumn], what: &str| {
        let first = &views[0];
        assert!(
            views
                .iter()
                .all(|v| v.binding.id == first.binding.id && v.stride == first.stride && v.size == first.size && v.n == n),
            "vector-matrix products: {what} of different allocations or layouts"
        );
    };
    alike(res, "destinations");
    alike(a, "sources");
    assert_eq!(a[0].stride as usize, pmat.cols_in * block, "a.cols() != pmat.cols_in()");
    assert_eq!(res[0].stride as usize, pmat.cols_out * block, "res.cols() != pmat.cols_out()");
    if let Some(addends) = addends {
        assert_eq!(addends.len(), m, "vector-matrix products: one addend per source");
        alike(addends, "addends");
    }
    if let Some(plan) = automorphism {
        assert_eq!(plan.n(), n, "vector-matrix products: automorphism plan of another degree");
    }
    let limbs = a[0].size.min(pmat.rows * dsize);
    assert!(limbs != 0 && res[0].size != 0, "vector-matrix products: no product");
    let narrow = |value: usize| u32::try_from(value).expect("vector-matrix product beyond the 32-bit range");
    narrow(pmat.base as usize + pmat.blocks().size * block);
    let terms = Terms {
        cols_in: narrow(pmat.cols_in),
        cols_out: narrow(pmat.cols_out),
        dsize: narrow(dsize),
        limbs: narrow(limbs),
        addend: addends.is_some() as u32,
        permute: automorphism.is_some(),
    };
    let add = addends.map_or(&a[0], |addends| &addends[0]);
    let add_size = addends.map_or(0, |addends| addends[0].size.min(res[0].size));
    let base = |views: &[DftColumn], s: usize| views.get(s).map_or(views[0].base, |v| v.base);
    let add_base = |s: usize| addends.map_or(0, |addends| base(addends, s));
    let (mul, shift) = automorphism.map_or((1, 1), AutomorphismPlan::scatter);
    let key_limbs = pmat.size.min(res[0].size + dsize - 1);
    let words = pmat.cols_out * terms.rows() as usize * pmat.cols_in * key_limbs
        + m * (limbs * pmat.cols_in + pmat.cols_out * res[0].size * (1 + accumulate as usize) + add_size);
    launch!(
        RT, if accumulate { Label::VmpDigitProductsAdd } else { Label::VmpDigitProducts }, block * 4 * words, product_many_kernel, (block, RT::GROUP, RT::GROUP);
        [
            res[0].binding => u32,
            a[0].binding => u32,
            pmat.binding => u32,
            add.binding => u32,
        ];
        [
            n.ilog2(),
            narrow(res[0].size),
            pmat.base,
            narrow(pmat.size),
            narrow(product_limbs.min(dsize)),
            add.stride,
            narrow(add_size),
            mul,
            shift,
            base(res, 0),
            base(res, 1),
            base(res, 2),
            base(res, 3),
            base(a, 0),
            base(a, 1),
            base(a, 2),
            base(a, 3),
            add_base(0),
            add_base(1),
            add_base(2),
            add_base(3),
        ];
        [Consts::get(), terms, narrow(m), accumulate]
    );
}

/// [`product_kernel`] of one source without a tail, followed by the first
/// launch of the inverse transform of every result limb: its last `mt`
/// stages, on the contiguous block of `2^mt` lanes a workgroup owns in one
/// plane. A thread computes the product at its own point, as
/// `product_kernel` does; every time a result limb is complete the workgroup
/// passes it through shared memory, one barrier per stage, and stores the
/// lanes the inverse launch would have stored. The stages run while the
/// matrix streams.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
fn product_inverse_kernel(
    res: &mut [u32],
    a: &[u32],
    key: &[u32],
    tw: &[u32],
    add0: &[u32],
    add1: &[u32],
    log_n: u32,
    res_base: u32,
    res_size: u32,
    a_base: u32,
    key_base: u32,
    key_size: u32,
    limb_offset: u32,
    product_limbs: u32,
    add0_base: u32,
    add0_stride: u32,
    add0_size: u32,
    add1_base: u32,
    add1_stride: u32,
    add1_size: u32,
    #[comptime] c: Consts,
    #[comptime] t: Terms,
    #[comptime] mt: u32,
) {
    let log_block = log_n + 2u32;
    let cols_in = t.cols_in;
    let cols_out = t.cols_out;
    let dsize = t.dsize;
    let rows = comptime!(t.rows());
    let mut sh = Shared::<[u32]>::new_slice(comptime!((t.cols_out as usize) << mt));
    let u = UNIT_POS;
    let tile = CUBE_POS as u32;
    let word = (tile << mt) + u;
    // Threads past the planes take the barriers and touch no memory.
    let active = word < (1u32 << log_block);
    let plane = min(word >> log_n, 3u32);
    let q = sel4(plane, c.q);
    let r = sel4(plane, c.r);
    let mu = sel4(plane, c.mu60);
    // The block of the workgroup in its plane, and its twiddle entry at the
    // first of the stages: see `tile_pass`.
    let block = tile - (plane << (log_n - mt));
    let tile_tb = (1u32 << (log_n - mt)) + block;
    let toff = (plane << log_n) + (4u32 << log_n);

    let mut xs = Sequence::<u32>::new();
    #[unroll]
    for l in 0..t.limbs {
        #[unroll]
        for col in 0..cols_in {
            let mut x = 0u32;
            if active {
                x = a[(a_base + ((l * cols_in + col) << log_block) + word) as usize];
            }
            xs.push(x);
        }
    }

    let reach = key_size - min(key_size, limb_offset);
    let first = min(res_size, reach);
    let mut widths = Sequence::<u32>::new();
    widths.push(first);
    #[unroll]
    for di in 1..dsize {
        let span = dsize - di;
        let omitted = span - min(span, product_limbs);
        let kept = key_size - min(key_size, omitted);
        widths.push(min(min(res_size, kept), reach - min(reach, di)));
    }

    let mut window = Sequence::<RuntimeCell<u64>>::new();
    #[unroll]
    for _ in 0..comptime!(dsize * cols_out) {
        window.push(RuntimeCell::<u64>::new(0u64));
    }

    // Result limbs that are not zero: those of the product and of the addends.
    let filled = max(first, max(add0_size, add1_size));
    for step in 0..filled + dsize - 1u32 {
        let mut live = Sequence::<bool>::new();
        let mut any = false;
        #[unroll]
        for di in 0..dsize {
            let reached = step - di < *widths.index(di as usize);
            live.push(reached);
            any = any || reached;
        }
        if any && active {
            let h = step + limb_offset;
            #[unroll]
            for d in 0..cols_out {
                let mut ks = Sequence::<u32>::new();
                #[unroll]
                for row in 0..rows {
                    #[unroll]
                    for col in 0..cols_in {
                        let block = ((row * cols_in + col) * key_size + h) * cols_out + d;
                        ks.push(key[(key_base + (block << log_block) + word) as usize]);
                    }
                }
                #[unroll]
                for di in 0..dsize {
                    if *live.index(di as usize) {
                        let slot = window.index((d * dsize + dsize - 1 - di) as usize);
                        let mut acc = slot.read();
                        #[unroll]
                        for row in 0..comptime!(t.digit_rows(di)) {
                            #[unroll]
                            for col in 0..cols_in {
                                if comptime!(t.fold_before(di, row, col)) {
                                    acc = fold64(acc, r);
                                }
                                let x = *xs.index(((dsize - 1 - di + row * dsize) * cols_in + col) as usize);
                                let k = *ks.index((row * cols_in + col) as usize);
                                acc += u64::cast_from(x) * u64::cast_from(k);
                            }
                        }
                        slot.store(acc);
                    }
                }
            }
        }
        // No limb is complete before step `dsize - 1`: the same for every thread.
        if step >= dsize - 1u32 {
            // The complete limb of every output column, to the workgroup.
            let j = step - (dsize - 1u32);
            #[unroll]
            for d in 0..cols_out {
                let mut value = barrett60(fold64(window.index((d * dsize) as usize).read(), r), q, mu);
                if comptime!(t.addend >> d & 1 == 1) {
                    let mut y = 0u32;
                    if comptime!(d == 0) {
                        if j < add0_size && active {
                            y = add0[(add0_base + j * add0_stride + word) as usize];
                        }
                    } else if j < add1_size && active {
                        y = add1[(add1_base + j * add1_stride + word) as usize];
                    }
                    let sum = value + y;
                    value = min(sum, sum - q);
                }
                sh[((d << mt) + u) as usize] = value;
            }
            sync_cube();
            // Stage `mt - 1 - e` of the block pairs lanes `2^e` apart. With two
            // output columns every thread runs one butterfly of a stage: thread
            // `u` that of index `u mod 2^(mt-1)` in column `u >> (mt - 1)`.
            // Otherwise the thread of the lower lane of a pair runs it in every
            // column.
            #[unroll]
            for e in 0..mt {
                let half = comptime!(1u32 << e);
                if comptime!(cols_out == 2) {
                    if active {
                        let pair_index = u & comptime!((1u32 << (mt - 1)) - 1);
                        let low = ((pair_index >> e) << comptime!(e + 1)) | (pair_index & comptime!(half - 1));
                        let wq = tw[(toff + tile_tb * comptime!(1u32 << (mt - 1 - e)) + (low >> comptime!(e + 1))) as usize];
                        let w = wq.mul_hi(q) + 1u32;
                        let i0 = (((u >> comptime!(mt - 1)) << mt) + low) as usize;
                        let i1 = i0 + half as usize;
                        let pair = gs(sh[i0], sh[i1], wq, w, q);
                        sh[i0] = pair.0;
                        sh[i1] = pair.1;
                    }
                } else if (u & half) == 0u32 && active {
                    let wq = tw[(toff + tile_tb * comptime!(1u32 << (mt - 1 - e)) + (u >> comptime!(e + 1))) as usize];
                    let w = wq.mul_hi(q) + 1u32;
                    #[unroll]
                    for d in 0..cols_out {
                        let i0 = ((d << mt) + u) as usize;
                        let i1 = i0 + half as usize;
                        let pair = gs(sh[i0], sh[i1], wq, w, q);
                        sh[i0] = pair.0;
                        sh[i1] = pair.1;
                    }
                }
                sync_cube();
            }
        }
        #[unroll]
        for d in 0..cols_out {
            if step >= dsize - 1u32 && active {
                let j = step - (dsize - 1u32);
                res[(res_base + ((j * cols_out + d) << log_block) + word) as usize] = sh[((d << mt) + u) as usize];
            }
            #[unroll]
            for k in 1..dsize {
                window
                    .index((d * dsize + k - 1) as usize)
                    .store(window.index((d * dsize + k) as usize).read());
            }
            window.index((d * dsize + dsize - 1) as usize).store(0u64);
        }
    }
    // Past the product and the addends the result is zero, and so is its transform.
    if active {
        #[unroll]
        for d in 0..cols_out {
            for j in filled..res_size {
                res[(res_base + ((j * cols_out + d) << log_block) + word) as usize] = 0u32;
            }
        }
    }
}

/// [`digit_product`] of one source into a fresh result, with `addends` on its
/// first two columns and no automorphism, and with the first launch of the
/// inverse transform of the result fused in:
/// `res` receives the lanes [`super::inverse_tail`] completes. `mt` is the
/// stage count of that launch ([`super::inverse_head`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn digit_product_inverse<RT: GpuRuntime>(
    table: &crate::runtime::GpuBuf<RT>,
    res: &DftColumn,
    a: &DftColumn,
    pmat: &Pmat,
    dsize: usize,
    product_limbs: usize,
    addends: [Option<&DftColumn>; 2],
    mt: u32,
) {
    let n = pmat.n;
    let block = 4 * n;
    assert!(dsize >= 1, "digit size must be positive");
    assert!(res.n == n && a.n == n, "vector-matrix product between different degrees");
    assert_eq!(a.stride as usize, pmat.cols_in * block, "a.cols() != pmat.cols_in()");
    assert_eq!(res.stride as usize, pmat.cols_out * block, "res.cols() != pmat.cols_out()");
    assert!(
        (1..=10).contains(&mt) && mt < n.ilog2(),
        "fused product: no such inverse launch"
    );
    let limbs = a.size.min(pmat.rows * dsize);
    assert!(limbs != 0 && res.size != 0, "fused product: nothing to multiply");
    let narrow = |value: usize| u32::try_from(value).expect("vector-matrix product beyond the 32-bit range");
    narrow(pmat.base as usize + pmat.blocks().size * block);
    let terms = Terms {
        cols_in: narrow(pmat.cols_in),
        cols_out: narrow(pmat.cols_out),
        dsize: narrow(dsize),
        limbs: narrow(limbs),
        addend: addends[0].is_some() as u32 | (addends[1].is_some() as u32) << 1,
        permute: false,
    };
    for (col, addend) in addends.iter().enumerate() {
        if let Some(addend) = addend {
            assert_eq!(addend.n, n, "fused product: addend of another degree");
            assert!(col < pmat.cols_out, "fused product: addend column out of range");
        }
    }
    // An absent addend is bound to a buffer of the launch and never read.
    let [(add0, add0_size), (add1, add1_size)] =
        addends.map(|addend| addend.map_or((a, 0), |addend| (addend, addend.size.min(res.size))));
    let table = table.view().binding().expect("transform tables hold storage");
    let key_limbs = pmat.size.min(res.size + dsize - 1);
    let words = limbs * pmat.cols_in
        + pmat.cols_out * (terms.rows() as usize * pmat.cols_in * key_limbs + res.size)
        + add0_size
        + add1_size;
    launch!(
        RT, Label::VmpDigitProductInverse, block * 4 * words, product_inverse_kernel, (block, 1 << mt, 1 << mt);
        [
            res.binding => u32,
            a.binding => u32,
            pmat.binding => u32,
            table => u32,
            add0.binding => u32,
            add1.binding => u32,
        ];
        [
            n.ilog2(),
            res.base,
            narrow(res.size),
            a.base,
            pmat.base,
            narrow(pmat.size),
            0u32,
            narrow(product_limbs.min(dsize)),
            add0.base,
            add0.stride,
            narrow(add0_size),
            add1.base,
            add1.stride,
            narrow(add1_size),
        ];
        [Consts::get(), terms, mt]
    );
}

/// [`product_inverse_kernel`] without addends for `m` sources against one key, at most
/// [`MANY_SOURCES`]: source `s` reads `a` from word `a_base{s}` and stores from `res_base{s}`,
/// each with its own block of shared memory, the barriers shared. Each matrix word is loaded
/// once for all the sources.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
fn product_inverse_many_kernel(
    res: &mut [u32],
    a: &[u32],
    key: &[u32],
    tw: &[u32],
    log_n: u32,
    res_size: u32,
    key_base: u32,
    key_size: u32,
    product_limbs: u32,
    res_base0: u32,
    res_base1: u32,
    res_base2: u32,
    res_base3: u32,
    a_base0: u32,
    a_base1: u32,
    a_base2: u32,
    a_base3: u32,
    #[comptime] c: Consts,
    #[comptime] t: Terms,
    #[comptime] mt: u32,
    #[comptime] m: u32,
) {
    let log_block = log_n + 2u32;
    let cols_in = t.cols_in;
    let cols_out = t.cols_out;
    let dsize = t.dsize;
    let rows = comptime!(t.rows());
    // Block `s * cols_out + d` of `2^mt` lanes holds output column `d` of source `s`.
    let mut sh = Shared::<[u32]>::new_slice(comptime!(((m * t.cols_out) as usize) << mt));
    let u = UNIT_POS;
    let tile = CUBE_POS as u32;
    let word = (tile << mt) + u;
    let active = word < (1u32 << log_block);
    let plane = min(word >> log_n, 3u32);
    let q = sel4(plane, c.q);
    let r = sel4(plane, c.r);
    let mu = sel4(plane, c.mu60);
    let block = tile - (plane << (log_n - mt));
    let tile_tb = (1u32 << (log_n - mt)) + block;
    let toff = (plane << log_n) + (4u32 << log_n);

    let mut xs = Sequence::<u32>::new();
    #[unroll]
    for s in 0..m {
        let a_base = pick4(a_base0, a_base1, a_base2, a_base3, s);
        #[unroll]
        for l in 0..t.limbs {
            #[unroll]
            for col in 0..cols_in {
                let mut x = 0u32;
                if active {
                    x = a[(a_base + ((l * cols_in + col) << log_block) + word) as usize];
                }
                xs.push(x);
            }
        }
    }

    let first = min(res_size, key_size);
    let mut widths = Sequence::<u32>::new();
    widths.push(first);
    #[unroll]
    for di in 1..dsize {
        let span = dsize - di;
        let omitted = span - min(span, product_limbs);
        let kept = key_size - min(key_size, omitted);
        widths.push(min(min(res_size, kept), key_size - min(key_size, di)));
    }

    let mut window = Sequence::<RuntimeCell<u64>>::new();
    #[unroll]
    for _ in 0..comptime!(m * t.dsize * t.cols_out) {
        window.push(RuntimeCell::<u64>::new(0u64));
    }

    for step in 0..first + dsize - 1u32 {
        let mut live = Sequence::<bool>::new();
        let mut any = false;
        #[unroll]
        for di in 0..dsize {
            let reached = step - di < *widths.index(di as usize);
            live.push(reached);
            any = any || reached;
        }
        if any && active {
            #[unroll]
            for d in 0..cols_out {
                let mut ks = Sequence::<u32>::new();
                #[unroll]
                for row in 0..rows {
                    #[unroll]
                    for col in 0..cols_in {
                        let block = ((row * cols_in + col) * key_size + step) * cols_out + d;
                        ks.push(key[(key_base + (block << log_block) + word) as usize]);
                    }
                }
                #[unroll]
                for s in 0..m {
                    #[unroll]
                    for di in 0..dsize {
                        if *live.index(di as usize) {
                            let slot = window.index(((s * cols_out + d) * dsize + dsize - 1 - di) as usize);
                            let mut acc = slot.read();
                            #[unroll]
                            for row in 0..comptime!(t.digit_rows(di)) {
                                #[unroll]
                                for col in 0..cols_in {
                                    if comptime!(t.fold_before(di, row, col)) {
                                        acc = fold64(acc, r);
                                    }
                                    let x = *xs.index(((s * t.limbs + dsize - 1 - di + row * dsize) * cols_in + col) as usize);
                                    let k = *ks.index((row * cols_in + col) as usize);
                                    acc += u64::cast_from(x) * u64::cast_from(k);
                                }
                            }
                            slot.store(acc);
                        }
                    }
                }
            }
        }
        if step >= dsize - 1u32 {
            #[unroll]
            for s in 0..m {
                #[unroll]
                for d in 0..cols_out {
                    let value = barrett60(fold64(window.index(((s * cols_out + d) * dsize) as usize).read(), r), q, mu);
                    sh[(((s * cols_out + d) << mt) + u) as usize] = value;
                }
            }
            sync_cube();
            #[unroll]
            for e in 0..mt {
                let half = comptime!(1u32 << e);
                if comptime!(cols_out == 2) {
                    if active {
                        let pair_index = u & comptime!((1u32 << (mt - 1)) - 1);
                        let low = ((pair_index >> e) << comptime!(e + 1)) | (pair_index & comptime!(half - 1));
                        let wq = tw[(toff + tile_tb * comptime!(1u32 << (mt - 1 - e)) + (low >> comptime!(e + 1))) as usize];
                        let w = wq.mul_hi(q) + 1u32;
                        #[unroll]
                        for s in 0..m {
                            let i0 = (((s * 2u32 + (u >> comptime!(mt - 1))) << mt) + low) as usize;
                            let i1 = i0 + half as usize;
                            let pair = gs(sh[i0], sh[i1], wq, w, q);
                            sh[i0] = pair.0;
                            sh[i1] = pair.1;
                        }
                    }
                } else if (u & half) == 0u32 && active {
                    let wq = tw[(toff + tile_tb * comptime!(1u32 << (mt - 1 - e)) + (u >> comptime!(e + 1))) as usize];
                    let w = wq.mul_hi(q) + 1u32;
                    #[unroll]
                    for s in 0..m {
                        #[unroll]
                        for d in 0..cols_out {
                            let i0 = (((s * cols_out + d) << mt) + u) as usize;
                            let i1 = i0 + half as usize;
                            let pair = gs(sh[i0], sh[i1], wq, w, q);
                            sh[i0] = pair.0;
                            sh[i1] = pair.1;
                        }
                    }
                }
                sync_cube();
            }
        }
        #[unroll]
        for s in 0..m {
            let res_base = pick4(res_base0, res_base1, res_base2, res_base3, s);
            #[unroll]
            for d in 0..cols_out {
                let w0 = (s * cols_out + d) * dsize;
                if step >= dsize - 1u32 && active {
                    let j = step - (dsize - 1u32);
                    res[(res_base + ((j * cols_out + d) << log_block) + word) as usize] =
                        sh[(((s * cols_out + d) << mt) + u) as usize];
                }
                #[unroll]
                for k in 1..dsize {
                    window
                        .index((w0 + k - 1) as usize)
                        .store(window.index((w0 + k) as usize).read());
                }
                window.index((w0 + dsize - 1) as usize).store(0u64);
            }
        }
    }
    // Past the product the result is zero, and so is its transform.
    if active {
        #[unroll]
        for s in 0..m {
            let res_base = pick4(res_base0, res_base1, res_base2, res_base3, s);
            #[unroll]
            for d in 0..cols_out {
                for j in first..res_size {
                    res[(res_base + ((j * cols_out + d) << log_block) + word) as usize] = 0u32;
                }
            }
        }
    }
}

/// [`digit_product_inverse`] without addends for the sources `a` against one key, at most
/// [`MANY_SOURCES`], into fresh results `res`: one launch that reads `pmat` once. The views of
/// each role share an allocation, a layout and a size.
#[allow(clippy::too_many_arguments)]
pub(crate) fn digit_products_inverse<RT: GpuRuntime>(
    table: &crate::runtime::GpuBuf<RT>,
    res: &[DftColumn],
    a: &[DftColumn],
    pmat: &Pmat,
    dsize: usize,
    product_limbs: usize,
    mt: u32,
) {
    let m = res.len();
    assert!((1..=MANY_SOURCES).contains(&m), "fused products: {m} sources");
    assert_eq!(a.len(), m, "fused products: sources and destinations differ in number");
    let n = pmat.n;
    let block = 4 * n;
    assert!(dsize >= 1, "digit size must be positive");
    let alike = |views: &[DftColumn], what: &str| {
        let first = &views[0];
        assert!(
            views
                .iter()
                .all(|v| v.binding.id == first.binding.id && v.stride == first.stride && v.size == first.size && v.n == n),
            "fused products: {what} of different allocations or layouts"
        );
    };
    alike(res, "destinations");
    alike(a, "sources");
    assert_eq!(a[0].stride as usize, pmat.cols_in * block, "a.cols() != pmat.cols_in()");
    assert_eq!(res[0].stride as usize, pmat.cols_out * block, "res.cols() != pmat.cols_out()");
    assert!(
        (1..=10).contains(&mt) && mt < n.ilog2(),
        "fused products: no such inverse launch"
    );
    let limbs = a[0].size.min(pmat.rows * dsize);
    assert!(limbs != 0 && res[0].size != 0, "fused products: nothing to multiply");
    let narrow = |value: usize| u32::try_from(value).expect("vector-matrix product beyond the 32-bit range");
    narrow(pmat.base as usize + pmat.blocks().size * block);
    let terms = Terms {
        cols_in: narrow(pmat.cols_in),
        cols_out: narrow(pmat.cols_out),
        dsize: narrow(dsize),
        limbs: narrow(limbs),
        addend: 0,
        permute: false,
    };
    let base = |views: &[DftColumn], s: usize| views.get(s).map_or(views[0].base, |v| v.base);
    let table = table.view().binding().expect("transform tables hold storage");
    let key_limbs = pmat.size.min(res[0].size + dsize - 1);
    let words = pmat.cols_out * terms.rows() as usize * pmat.cols_in * key_limbs
        + m * (limbs * pmat.cols_in + pmat.cols_out * res[0].size);
    launch!(
        RT, Label::VmpDigitProductsInverse, block * 4 * words, product_inverse_many_kernel, (block, 1 << mt, 1 << mt);
        [
            res[0].binding => u32,
            a[0].binding => u32,
            pmat.binding => u32,
            table => u32,
        ];
        [
            n.ilog2(),
            narrow(res[0].size),
            pmat.base,
            narrow(pmat.size),
            narrow(product_limbs.min(dsize)),
            base(res, 0),
            base(res, 1),
            base(res, 2),
            base(res, 3),
            base(a, 0),
            base(a, 1),
            base(a, 2),
            base(a, 3),
        ];
        [Consts::get(), terms, mt, narrow(m)]
    );
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn extract_rows_kernel(
    res: &mut [u32],
    a: &[u32],
    // Words of one `(row, input column)` entry of `res`.
    entry: u32,
    entries: u32,
    cols_in: u32,
    res_base: u32,
    a_base: u32,
    // Words of one entry of `a`.
    a_entry: u32,
    first_row: u32,
    row_step: u32,
) {
    let pos = ABSOLUTE_POS as u32;
    if pos < entries * entry {
        let index = pos / entry;
        let word = pos - index * entry;
        let row = index / cols_in;
        let col = index - row * cols_in;
        let from = (first_row + row * row_step) * cols_in + col;
        res[(res_base + pos) as usize] = a[(a_base + from * a_entry + word) as usize];
    }
}

/// `res[r] = a[first_row + r * row_step]` for every row of `res`, over the
/// leading `res.size` limbs.
pub(crate) fn extract_rows<RT: GpuRuntime>(res: &Pmat, a: &Pmat, first_row: usize, row_step: usize) {
    assert!(
        res.n == a.n && res.cols_in == a.cols_in && res.cols_out == a.cols_out,
        "row extraction between matrices of different shapes"
    );
    assert!(res.size <= a.size, "row extraction: res.size() > a.size()");
    assert!(row_step > 0, "row extraction: row_step must be positive");
    if res.rows == 0 || res.size == 0 {
        return;
    }
    assert!(
        first_row + (res.rows - 1) * row_step < a.rows,
        "row extraction: selected rows out of range"
    );
    let block = 4 * res.n;
    let entry = res.size * res.cols_out * block;
    let entries = res.rows * res.cols_in;
    let narrow = |value: usize| u32::try_from(value).expect("prepared matrix beyond the 32-bit word range");
    narrow(a.base as usize + a.blocks().size * block);
    narrow(res.base as usize + entries * entry);
    launch!(
        RT, Label::VmpExtractRows, entries * entry * 8, extract_rows_kernel, (entries * entry, RT::GROUP, RT::GROUP);
        [
            res.binding => u32,
            a.binding => u32,
        ];
        [
            narrow(entry),
            narrow(entries),
            narrow(res.cols_in),
            res.base,
            a.base,
            narrow(a.size * a.cols_out * block),
            narrow(first_row),
            narrow(row_step),
        ];
        []
    );
}
