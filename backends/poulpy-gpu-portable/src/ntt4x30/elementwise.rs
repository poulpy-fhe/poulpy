//! `res = ±a ± b` on canonical residues, limb by limb, with zero extension
//! past an operand's limbs. One thread per residue.

use cubecl::prelude::*;

use super::{
    DftColumn,
    primes::{Q, sel4},
};
use crate::runtime::{GpuRuntime, Label, launch};

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
    let units = 4 * res.n * limbs;
    let label = match (a.sign, b.sign) {
        (ZERO, ZERO) => Label::DftZero,
        (PLUS, ZERO) | (ZERO, PLUS) => Label::DftCopy,
        _ => Label::DftCombine,
    };
    let operands = 1 + (a.sign != ZERO) as usize + (b.sign != ZERO) as usize;
    launch!(
        RT, label, units * 4 * operands, combine_kernel, (units, RT::GROUP, RT::GROUP);
        [
            res.binding => u32,
            a.column.binding => u32,
            b.column.binding => u32,
        ];
        [
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
        ];
        [Q]
    );
}

/// Blocks [`gather`] copies per launch: the kernel binds eight source buffers.
#[cfg_attr(not(feature = "enable-core"), allow(dead_code))]
pub(crate) const GATHER_BLOCKS: usize = 8;

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, unused_assignments)]
fn gather_kernel(
    dst: &mut [u32],
    s0: &[u32],
    s1: &[u32],
    s2: &[u32],
    s3: &[u32],
    s4: &[u32],
    s5: &[u32],
    s6: &[u32],
    s7: &[u32],
    // Words of one block.
    block: u32,
    dst_base: u32,
    b0: u32,
    b1: u32,
    b2: u32,
    b3: u32,
    b4: u32,
    b5: u32,
    b6: u32,
    b7: u32,
    #[comptime] count: u32,
) {
    let pos = ABSOLUTE_POS as u32;
    if pos < count * block {
        let index = pos / block;
        let word = pos - index * block;
        let mut value = s0[(b0 + word) as usize];
        #[unroll]
        for source in 1..count {
            if index == source {
                if comptime!(source == 1) {
                    value = s1[(b1 + word) as usize];
                } else if comptime!(source == 2) {
                    value = s2[(b2 + word) as usize];
                } else if comptime!(source == 3) {
                    value = s3[(b3 + word) as usize];
                } else if comptime!(source == 4) {
                    value = s4[(b4 + word) as usize];
                } else if comptime!(source == 5) {
                    value = s5[(b5 + word) as usize];
                } else if comptime!(source == 6) {
                    value = s6[(b6 + word) as usize];
                } else {
                    value = s7[(b7 + word) as usize];
                }
            }
        }
        dst[(dst_base + pos) as usize] = value;
    }
}

/// Copies limb 0 of each of `sources` into consecutive limbs of `dst`, limb
/// `first + i` receiving `sources[i]`, [`GATHER_BLOCKS`] per launch. The
/// limbs of `dst` are contiguous.
#[cfg_attr(not(feature = "enable-core"), allow(dead_code))]
pub(crate) fn gather<RT: GpuRuntime>(dst: &DftColumn, sources: &[DftColumn]) {
    let block = 4 * dst.n;
    assert!(dst.stride as usize == block && sources.len() <= dst.size);
    assert!(sources.iter().all(|source| source.n == dst.n && source.size >= 1));
    for (group, chunk) in sources.chunks(GATHER_BLOCKS).enumerate() {
        // Sources beyond the last repeat it: the kernel does not read them.
        let source = |index: usize| &chunk[index.min(chunk.len() - 1)];
        let units = chunk.len() * block;
        launch!(
            RT, Label::DftGather, units * 8, gather_kernel, (units, RT::GROUP, RT::GROUP);
            [
                dst.binding => u32,
                source(0).binding => u32,
                source(1).binding => u32,
                source(2).binding => u32,
                source(3).binding => u32,
                source(4).binding => u32,
                source(5).binding => u32,
                source(6).binding => u32,
                source(7).binding => u32,
            ];
            [
                block as u32,
                dst.base + (group * GATHER_BLOCKS * block) as u32,
                source(0).base,
                source(1).base,
                source(2).base,
                source(3).base,
                source(4).base,
                source(5).base,
                source(6).base,
                source(7).base,
            ];
            [chunk.len() as u32]
        );
    }
}
