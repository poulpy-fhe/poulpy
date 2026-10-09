//! Product of a DFT-domain vector by a prepared scalar polynomial: one modular
//! product per residue, one thread per residue.
//!
//! A prepared scalar polynomial is the transform of the polynomial, one block
//! of four planes of canonical residues per column: the layout of one limb of
//! a DFT-domain vector.

use cubecl::prelude::*;

use super::{
    DIM, DftColumn,
    primes::{Consts, mul_mod_canonical, sel4},
};
use crate::runtime::{GpuRuntime, Label, launch};

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn scale_kernel(
    res: &mut [u32],
    ppol: &[u32],
    b: &[u32],
    // log2 of the words of a limb, `4n`.
    log_block: u32,
    limbs: u32,
    res_base: u32,
    res_stride: u32,
    ppol_base: u32,
    b_base: u32,
    b_stride: u32,
    b_size: u32,
    #[comptime] c: Consts,
) {
    let pos = ABSOLUTE_POS as u32;
    if pos < (limbs << log_block) {
        let limb = pos >> log_block;
        let word = pos - (limb << log_block);
        let plane = word >> (log_block - 2u32);
        let mut x = 0u32;
        if limb < b_size {
            x = mul_mod_canonical(
                b[(b_base + limb * b_stride + word) as usize],
                ppol[(ppol_base + word) as usize],
                sel4(plane, c.q),
                sel4(plane, c.mu60),
            );
        }
        res[(res_base + limb * res_stride + word) as usize] = x;
    }
}

/// `res[j] = ppol * b[j]` for every limb of `res`; limbs past `b`'s are zero.
/// `ppol` is a one-limb column. `res` may be `b` itself.
pub(crate) fn scale<RT: GpuRuntime>(res: &DftColumn, ppol: &DftColumn, b: &DftColumn) {
    assert!(
        ppol.n == res.n && b.n == res.n,
        "scalar product between operands of different degrees"
    );
    assert_eq!(ppol.size, 1, "a prepared scalar polynomial has one limb");
    if res.size == 0 {
        return;
    }
    let units = 4 * res.n * res.size;
    launch!(
        RT, Label::SvpScale, units * 4 + 4 * res.n * res.size.min(b.size) * 8, scale_kernel, (units, DIM, DIM);
        [
            res.binding => u32,
            ppol.binding => u32,
            b.binding => u32,
        ];
        [
            res.n.ilog2() + 2,
            res.size as u32,
            res.base,
            res.stride,
            ppol.base,
            b.base,
            b.stride,
            b.size as u32,
        ];
        [Consts::get()]
    );
}
