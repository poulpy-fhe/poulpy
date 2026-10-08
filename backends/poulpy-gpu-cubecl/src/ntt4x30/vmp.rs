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
//! [`digit_product`]). One thread per (output column, prime, coefficient)
//! loads the source limbs once into registers, then walks the matrix limbs
//! once: matrix limb `h` contributes through digit `di` to result limb
//! `h - limb_offset - di`, so the thread keeps a window of `dsize`
//! accumulators, stores the oldest after each matrix limb and shifts the
//! window. Each matrix element is read exactly once. Accumulators are `u64`
//! sums of products of canonical residues, folded every [`FOLD_TERMS`] terms.
//!
//! The kernel is specialized for the terms of one result limb: the number of
//! source limbs it reads, the input columns and the digit size.

use cubecl::prelude::*;

use super::{
    DIM, DftColumn,
    primes::{Consts, FOLD_TERMS, barrett60, fold64, sel4},
};
use crate::runtime::{Binding, GpuRuntime, cubes, launched};

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
    dsize: u32,
    /// Source limbs the product reads: `min(a.size, rows * dsize)`.
    limbs: u32,
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

    /// Whether the accumulator is folded before term `(di, row, col)`, terms
    /// being counted over the digits in the order an accumulator meets them.
    fn fold_before(&self, di: u32, row: u32, col: u32) -> bool {
        let before: u32 = (0..di).map(|d| self.digit_rows(d) * self.cols_in).sum::<u32>() + row * self.cols_in + col;
        before != 0 && before.is_multiple_of(FOLD_TERMS)
    }
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn product_kernel(
    res: &mut [u32],
    a: &[u32],
    key: &[u32],
    log_n: u32,
    cols_out: u32,
    res_base: u32,
    res_size: u32,
    a_base: u32,
    key_base: u32,
    key_size: u32,
    limb_offset: u32,
    product_limbs: u32,
    #[comptime] c: Consts,
    #[comptime] t: Terms,
    #[comptime] accumulate: bool,
) {
    let log_block = log_n + 2u32;
    let pos = ABSOLUTE_POS as u32;
    if pos < (cols_out << log_block) {
        let d = pos >> log_block;
        let word = pos - (d << log_block);
        let plane = word >> log_n;
        let q = sel4(plane, c.q);
        let r = sel4(plane, c.r);
        let mu = sel4(plane, c.mu60);
        let cols_in = t.cols_in;
        let dsize = t.dsize;
        let rows = comptime!(t.rows());

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
        // holds result limb `step - (dsize - 1) + k`.
        let mut window = Sequence::<RuntimeCell<u64>>::new();
        #[unroll]
        for _ in 0..dsize {
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
                        let slot = window.index((dsize - 1 - di) as usize);
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
            if step >= dsize - 1u32 {
                let j = step - (dsize - 1u32);
                let index = res_base + ((j * cols_out + d) << log_block) + word;
                let mut value = barrett60(fold64(window.index(0usize).read(), r), q, mu);
                if accumulate {
                    let sum = value + res[index as usize];
                    value = min(sum, sum - q);
                }
                res[index as usize] = value;
            }
            #[unroll]
            for k in 1..dsize {
                window.index((k - 1) as usize).store(window.index(k as usize).read());
            }
            window.index((dsize - 1) as usize).store(0u64);
        }
        if !accumulate {
            for j in first..res_size {
                res[(res_base + ((j * cols_out + d) << log_block) + word) as usize] = 0u32;
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn digit_product<RT: GpuRuntime>(
    res: &DftColumn,
    a: &DftColumn,
    pmat: &Pmat,
    dsize: usize,
    product_limbs: usize,
    limb_offset: usize,
    accumulate: bool,
) {
    let n = pmat.n;
    let block = 4 * n;
    assert!(dsize >= 1, "digit size must be positive");
    assert!(res.n == n && a.n == n, "vector-matrix product between different degrees");
    assert_eq!(a.stride as usize, pmat.cols_in * block, "a.cols() != pmat.cols_in()");
    assert_eq!(res.stride as usize, pmat.cols_out * block, "res.cols() != pmat.cols_out()");
    let limbs = a.size.min(pmat.rows * dsize);
    if res.size == 0 {
        return;
    }
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
        dsize: narrow(dsize),
        limbs: narrow(limbs),
    };
    let client = RT::client();
    let units = pmat.cols_out * block;
    unsafe {
        product_kernel::launch_unchecked(
            client,
            cubes(client, units, DIM),
            CubeDim::new_1d(DIM),
            res.binding.arg::<u32>(),
            a.binding.arg::<u32>(),
            pmat.binding.arg::<u32>(),
            n.ilog2(),
            narrow(pmat.cols_out),
            res.base,
            narrow(res.size),
            a.base,
            pmat.base,
            narrow(pmat.size),
            narrow(limb_offset),
            narrow(product_limbs.min(dsize)),
            Consts::get(),
            terms,
            accumulate,
        );
    }
    let key_limbs = pmat.size.min(res.size + dsize - 1);
    let words = limbs * pmat.cols_in + terms.rows() as usize * pmat.cols_in * key_limbs + res.size * (1 + accumulate as usize);
    launched(units * 4 * words);
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
    let client = RT::client();
    unsafe {
        extract_rows_kernel::launch_unchecked(
            client,
            cubes(client, entries * entry, DIM),
            CubeDim::new_1d(DIM),
            res.binding.arg::<u32>(),
            a.binding.arg::<u32>(),
            narrow(entry),
            narrow(entries),
            narrow(res.cols_in),
            res.base,
            a.base,
            narrow(a.size * a.cols_out * block),
            narrow(first_row),
            narrow(row_step),
        );
    }
    launched(entries * entry * 8);
}
