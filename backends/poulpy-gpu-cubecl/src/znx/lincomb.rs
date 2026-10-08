//! `res = ±a ± b`, limb by limb, with zero extension past an operand's limbs
//! and ring embedding of an operand of another degree.

use cubecl::prelude::*;

use super::{Column, DIM};
use crate::runtime::{GpuRuntime, cubes, launched};

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

/// Reads coefficient `i` of limb `limb` of an operand embedded in degree `n`.
///
/// `down` is the log of the embedding gap of a smaller operand (its coefficient
/// `k` sits at `k << down`, the rest are zero); `up` is the log of the
/// decimation of a larger one (coefficient `i` is its `i << up`).
#[cube]
fn term(x: &[i64], limb: u32, i: u32, base: u32, stride: u32, size: u32, down: u32, up: u32, sign: u32) -> i64 {
    let mut value = 0i64;
    if sign != 0u32 && limb < size && (i & ((1u32 << down) - 1u32)) == 0u32 {
        let index = base + limb * stride + ((i >> down) << up);
        value = x[index as usize];
        if sign == 2u32 {
            value = 0i64 - value;
        }
    }
    value
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn lincomb_kernel(
    res: &mut [i64],
    a: &[i64],
    b: &[i64],
    n: u32,
    limbs: u32,
    res_base: u32,
    res_stride: u32,
    a_base: u32,
    a_stride: u32,
    a_size: u32,
    a_down: u32,
    a_up: u32,
    a_sign: u32,
    b_base: u32,
    b_stride: u32,
    b_size: u32,
    b_down: u32,
    b_up: u32,
    b_sign: u32,
) {
    let pos = ABSOLUTE_POS as u32;
    if pos < n * limbs {
        let limb = pos / n;
        let i = pos - limb * n;
        let x = term(a, limb, i, a_base, a_stride, a_size, a_down, a_up, a_sign);
        let y = term(b, limb, i, b_base, b_stride, b_size, b_down, b_up, b_sign);
        res[(res_base + limb * res_stride + i) as usize] = x + y;
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
            n.is_power_of_two() && res_n.is_multiple_of(n),
            "operand of degree {n} does not embed into degree {res_n}"
        );
        return ((res_n / n).ilog2(), 0);
    }
    assert!(
        decimate && res_n.is_power_of_two() && n.is_multiple_of(res_n),
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
    let client = RT::client();
    let units = res.n * limbs;
    unsafe {
        lincomb_kernel::launch_unchecked(
            client,
            cubes(client, units, DIM),
            CubeDim::new_1d(DIM),
            res.binding.arg::<i64>(),
            a.column.binding.arg::<i64>(),
            b.column.binding.arg::<i64>(),
            res.n as u32,
            limbs as u32,
            res.base,
            res.stride,
            a.column.base,
            a.column.stride,
            a.column.size as u32,
            a_down,
            a_up,
            a.sign as u32,
            b.column.base,
            b.column.stride,
            b.column.size as u32,
            b_down,
            b_up,
            b.sign as u32,
        );
    }
    let read = (a.sign != Sign::Zero) as usize + (b.sign != Sign::Zero) as usize;
    launched(units * 8 * (1 + read));
}
