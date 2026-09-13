//! CPU override of the `normalize(offset = +k)` default for `vec_znx_lsh_assign`.
//!
//! The OEP default shifts into a full-width `VecZnx` temporary and copies back;
//! the kernel below moves the limbs in place through a single carry buffer, one
//! memory pass instead of two. It is bit-exact with the default — the parity
//! test in `poulpy_hal::test_suite::derived` pins that — and the rest of the
//! shift family (`lsh`, `lsh_add`, `lsh_sub`, `rsh*`) stays on its default.

use std::mem::size_of;

use crate::{
    layouts::{Backend, HostDataMut, VecZnxBackendMut, ZnxView, ZnxViewMut},
    reference::znx::{ZnxCopy, ZnxNormalizeFinalStepAssign, ZnxNormalizeFirstStepAssign, ZnxNormalizeMiddleStepAssign, ZnxZero},
};

/// Scratch the [`vec_znx_lsh_assign`] kernel carves: one normalization carry.
///
/// The api-level `vec_znx_lsh_tmp_bytes` keeps its OEP default, which also has
/// to cover the still-derived `lsh_add` / `lsh_sub`; it is strictly larger than
/// this.
pub fn vec_znx_lsh_assign_carry_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_lsh_assign<'r, BE>(base2k: usize, k: usize, res: &mut VecZnxBackendMut<'r, BE>, res_col: usize, carry: &mut [i64])
where
    BE: Backend<ZnxWord = i64>,
    BE::BufMut<'r>: HostDataMut,
    BE: ZnxZero + ZnxCopy + ZnxNormalizeFirstStepAssign + ZnxNormalizeMiddleStepAssign + ZnxNormalizeFinalStepAssign,
{
    let n: usize = res.n();
    let size: usize = res.size();
    let (steps, k_rem) = (k / base2k, k % base2k);

    // Everything is shifted past the top limb: the column is zero, as in the
    // `normalize(offset = +k)` default.
    if steps >= size {
        for j in 0..size {
            BE::znx_zero(res.at_mut(res_col, j));
        }
        return;
    }

    // Assign shift of limbs by a k/base2k. The limbs are moved one at a time
    // through `carry`: the normalization below starts with a first step, which
    // overwrites the carry instead of reading it.
    if steps > 0 {
        let bounce = &mut carry[..n];
        for j in 0..size - steps {
            BE::znx_copy(bounce, res.at(res_col, j + steps));
            BE::znx_copy(res.at_mut(res_col, j), bounce);
        }

        for j in size - steps..size {
            BE::znx_zero(res.at_mut(res_col, j));
        }
    }

    for j in (0..size - steps).rev() {
        if j == size - steps - 1 {
            BE::znx_normalize_first_step_assign(base2k, k_rem, res.at_mut(res_col, j), carry);
        } else if j == 0 {
            BE::znx_normalize_final_step_assign(base2k, k_rem, res.at_mut(res_col, j), carry);
        } else {
            BE::znx_normalize_middle_step_assign(base2k, k_rem, res.at_mut(res_col, j), carry);
        }
    }
}
