//! `res = ±a ± b` on canonical residues, limb by limb, with zero extension
//! past an operand's limbs. One thread per residue.

use cubecl::prelude::*;

use super::{
    DIM, DftColumn,
    primes::{Q, sel4},
};
use crate::runtime::{GpuRuntime, cubes, launched};

const ZERO: u32 = 0;
const PLUS: u32 = 1;
const MINUS: u32 = 2;

/// One operand of [`combine`].
pub(crate) struct DftTerm<'a> {
    column: &'a DftColumn,
    sign: u32,
}

impl<'a> DftTerm<'a> {
    pub(crate) fn plus(column: &'a DftColumn) -> Self {
        Self { column, sign: PLUS }
    }

    pub(crate) fn minus(column: &'a DftColumn) -> Self {
        Self { column, sign: MINUS }
    }

    /// An operand that contributes nothing; `column` only fills the binding.
    pub(crate) fn zero(column: &'a DftColumn) -> Self {
        Self { column, sign: ZERO }
    }
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn combine_kernel(
    res: &mut [u32],
    a: &[u32],
    b: &[u32],
    // log2 of the words of a limb, `4n`.
    log_block: u32,
    limbs: u32,
    res_base: u32,
    res_stride: u32,
    a_base: u32,
    a_stride: u32,
    a_size: u32,
    a_sign: u32,
    b_base: u32,
    b_stride: u32,
    b_size: u32,
    b_sign: u32,
    #[comptime] q: [u32; 4],
) {
    let pos = ABSOLUTE_POS as u32;
    if pos < (limbs << log_block) {
        let limb = pos >> log_block;
        let word = pos - (limb << log_block);
        let q = sel4(word >> (log_block - 2u32), q);
        let mut x = 0u32;
        if a_sign != 0u32 && limb < a_size {
            x = a[(a_base + limb * a_stride + word) as usize];
            if a_sign == 2u32 {
                x = select(x == 0u32, 0u32, q - x);
            }
        }
        let mut y = 0u32;
        if b_sign != 0u32 && limb < b_size {
            y = b[(b_base + limb * b_stride + word) as usize];
            if b_sign == 2u32 {
                y = select(y == 0u32, 0u32, q - y);
            }
        }
        let sum = x + y;
        res[(res_base + limb * res_stride + word) as usize] = min(sum, sum - q);
    }
}

/// Writes limbs `0..limbs` of `res` with `±a ± b`. An operand may be `res`
/// itself, read at the same word it is written.
pub(crate) fn combine<RT: GpuRuntime>(res: &DftColumn, limbs: usize, a: DftTerm<'_>, b: DftTerm<'_>) {
    assert!(limbs <= res.size);
    assert!(
        a.column.n == res.n && b.column.n == res.n,
        "DFT-domain operands of different degrees"
    );
    if limbs == 0 {
        return;
    }
    let client = RT::client();
    let units = 4 * res.n * limbs;
    unsafe {
        combine_kernel::launch_unchecked(
            client,
            cubes(client, units, DIM),
            CubeDim::new_1d(DIM),
            res.binding.arg::<u32>(),
            a.column.binding.arg::<u32>(),
            b.column.binding.arg::<u32>(),
            res.n.ilog2() + 2,
            limbs as u32,
            res.base,
            res.stride,
            a.column.base,
            a.column.stride,
            a.column.size as u32,
            a.sign,
            b.column.base,
            b.column.stride,
            b.column.size as u32,
            b.sign,
            Q,
        );
    }
    let operands = 1 + (a.sign != ZERO) as usize + (b.sign != ZERO) as usize;
    launched(units * 4 * operands);
}
