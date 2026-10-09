//! `res = ±a ± b`, limb by limb, with zero extension past an operand's limbs,
//! ring embedding of an operand of another degree, and sign extension of an
//! `i64` operand into an `i128` destination.

use cubecl::prelude::*;

use super::{Column, DIM};
use crate::runtime::{GpuRuntime, Label, launch};

/// How an operand enters the combination.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Sign {
    Zero = 0,
    Plus = 1,
    Minus = 2,
}

/// One operand of [`lincomb`].
pub(crate) struct Term<'a> {
    pub(crate) column: &'a Column,
    pub(crate) sign: Sign,
}

impl<'a> Term<'a> {
    pub(crate) fn plus(column: &'a Column) -> Self {
        Self {
            column,
            sign: Sign::Plus,
        }
    }

    pub(crate) fn minus(column: &'a Column) -> Self {
        Self {
            column,
            sign: Sign::Minus,
        }
    }

    /// An operand that contributes nothing; `column` only fills the binding.
    pub(crate) fn zero(column: &'a Column) -> Self {
        Self {
            column,
            sign: Sign::Zero,
        }
    }
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn lincomb_kernel(
    res: &mut [u64],
    a: &[u64],
    b: &[u64],
    n: u32,
    limbs: u32,
    res_base: u32,
    res_stride: u32,
    res_words: u32,
    a_base: u32,
    a_stride: u32,
    a_size: u32,
    a_words: u32,
    // Log of the embedding gap of a smaller operand: its coefficient `k`
    // sits at `k << down`, the rest are zero.
    a_down: u32,
    // Log of the decimation of a larger operand: coefficient `i` is its `i << up`.
    a_up: u32,
    a_sign: u32,
    b_base: u32,
    b_stride: u32,
    b_size: u32,
    b_words: u32,
    b_down: u32,
    b_up: u32,
    b_sign: u32,
) {
    let pos = ABSOLUTE_POS as u32;
    if pos < n * limbs {
        let limb = pos / n;
        let i = pos - limb * n;

        // Each operand as a 128-bit two's complement value.
        let mut x_lo = 0u64;
        let mut x_hi = 0u64;
        if a_sign != 0u32 && limb < a_size && (i & ((1u32 << a_down) - 1u32)) == 0u32 {
            let index = a_base + limb * a_stride + ((i >> a_down) << a_up) * a_words;
            x_lo = a[index as usize];
            if a_words == 2u32 {
                x_hi = a[(index + 1u32) as usize];
            } else {
                let sign = (x_lo as i64) >> 63i64;
                x_hi = sign as u64;
            }
            if a_sign == 2u32 {
                x_hi = 0u64 - x_hi;
                if x_lo != 0u64 {
                    x_hi -= 1u64;
                }
                x_lo = 0u64 - x_lo;
            }
        }
        let mut y_lo = 0u64;
        let mut y_hi = 0u64;
        if b_sign != 0u32 && limb < b_size && (i & ((1u32 << b_down) - 1u32)) == 0u32 {
            let index = b_base + limb * b_stride + ((i >> b_down) << b_up) * b_words;
            y_lo = b[index as usize];
            if b_words == 2u32 {
                y_hi = b[(index + 1u32) as usize];
            } else {
                let sign = (y_lo as i64) >> 63i64;
                y_hi = sign as u64;
            }
            if b_sign == 2u32 {
                y_hi = 0u64 - y_hi;
                if y_lo != 0u64 {
                    y_hi -= 1u64;
                }
                y_lo = 0u64 - y_lo;
            }
        }

        let lo = x_lo + y_lo;
        let mut hi = x_hi + y_hi;
        if lo < x_lo {
            hi += 1u64;
        }
        let index = res_base + limb * res_stride + i * res_words;
        res[index as usize] = lo;
        if res_words == 2u32 {
            res[(index + 1u32) as usize] = hi;
        }
    }
}

/// Embedding gap of an operand of degree `n` in a destination of degree
/// `res_n`, as `(down, up)` logs. A larger operand is only read when
/// `decimate` is set (ring switching); the other operations reject it.
fn gaps(res_n: usize, n: usize, decimate: bool) -> (u32, u32) {
    if n == res_n {
        return (0, 0);
    }
    if n < res_n {
        assert!(
            res_n.is_multiple_of(n) && (res_n / n).is_power_of_two(),
            "operand of degree {n} does not embed into degree {res_n}"
        );
        return ((res_n / n).ilog2(), 0);
    }
    assert!(
        decimate && n.is_multiple_of(res_n) && (n / res_n).is_power_of_two(),
        "operand of degree {n} does not embed into degree {res_n}"
    );
    (0, (n / res_n).ilog2())
}

/// Writes limbs `0..limbs` of `res` with `±a ± b`. An operand may be `res`
/// itself, read at the same coefficient it is written.
pub(crate) fn lincomb<RT: GpuRuntime>(res: &Column, limbs: usize, a: Term<'_>, b: Term<'_>) {
    lincomb_with::<RT>(res, limbs, a, b, false)
}

/// [`lincomb`], reading an operand of larger degree at every `n / res.n`-th coefficient.
pub(crate) fn lincomb_decimating<RT: GpuRuntime>(res: &Column, limbs: usize, a: Term<'_>, b: Term<'_>) {
    lincomb_with::<RT>(res, limbs, a, b, true)
}

fn lincomb_with<RT: GpuRuntime>(res: &Column, limbs: usize, a: Term<'_>, b: Term<'_>, decimate: bool) {
    assert!(limbs <= res.size);
    if limbs == 0 || res.n == 0 {
        return;
    }
    let (a_down, a_up) = gaps(res.n, a.column.n, decimate);
    let (b_down, b_up) = gaps(res.n, b.column.n, decimate);
    let units = res.n * limbs;
    let label = match ((a.sign != Sign::Zero) as u32 + (b.sign != Sign::Zero) as u32, res.words) {
        (0, 1) => Label::ZnxZero,
        (0, _) => Label::BigZero,
        (1, 1) if a.sign != Sign::Minus && b.sign != Sign::Minus => Label::ZnxCopy,
        (1, 2) if a.sign != Sign::Minus && b.sign != Sign::Minus => Label::BigCopy,
        (_, 1) => Label::ZnxCombine,
        _ => Label::BigCombine,
    };
    let mut bytes = units * 8 * res.words as usize;
    for term in [&a, &b] {
        if term.sign != Sign::Zero {
            bytes += units * 8 * term.column.words as usize;
        }
    }
    launch!(
        RT, label, bytes, lincomb_kernel, (units, DIM, DIM);
        [
            res.binding => u64,
            a.column.binding => u64,
            b.column.binding => u64,
        ];
        [
            res.n as u32,
            limbs as u32,
            res.base,
            res.stride,
            res.words,
            a.column.base,
            a.column.stride,
            a.column.size as u32,
            a.column.words,
            a_down,
            a_up,
            a.sign as u32,
            b.column.base,
            b.column.stride,
            b.column.size as u32,
            b.column.words,
            b_down,
            b_up,
            b.sign as u32,
        ];
        []
    );
}
