//! CPU override of the `normalize(offset = +k)` default for `vec_znx_lsh_assign`.
//!
//! The OEP default shifts into a full-width `VecZnx` temporary and copies back;
//! the kernel below moves the limbs within the column, then normalizes them in
//! place, with no temporary. It is bit-exact with the default; the parity test in
//! `poulpy_hal::test_suite::derived` pins that, and the rest of the shift family
//! (`lsh`, `lsh_add`, `lsh_sub`, `rsh*`) stays on its default.

use crate::{
    kernels::znx_normalize::{Limbs, LimbsMut, ZnxNormalize},
    layouts::{HostDataMut, VecZnxBackendMut, ZnxView, ZnxViewMut},
};

pub fn vec_znx_lsh_assign_portable<'r, BE>(base2k: usize, k: usize, res: &mut VecZnxBackendMut<'r, BE>, res_col: usize)
where
    BE: ZnxNormalize,
    BE::BufMut<'r>: HostDataMut,
{
    let (n, size, shape) = (res.n(), res.size(), res.shape());
    assert!(res_col < res.cols(), "lsh_assign: column out of range");
    if size > 0 {
        // Bounds-checks the last limb, the farthest one the kernel writes.
        let _ = res.at_ptr(res_col, size - 1);
    }
    let (steps, k_rem) = (k / base2k, k % base2k);
    let base = res.base_mut_ptr();
    let limb = |j: usize| unsafe { base.add(shape.scalar_offset(res_col, j)) };
    let kept = size.saturating_sub(steps);
    if steps > 0 {
        for j in 0..kept {
            // Distinct limbs of one column never overlap.
            unsafe { std::ptr::copy_nonoverlapping(limb(j + steps), limb(j), n) };
        }
    }
    for j in kept..size {
        unsafe { std::ptr::write_bytes(limb(j), 0, n) };
    }
    if kept == 0 {
        return;
    }
    let res = LimbsMut {
        size: kept,
        ..LimbsMut::new(base, shape, res_col, 0)
    };
    let a = Limbs {
        ptr: res.ptr.cast_const(),
        stride: res.stride,
        size: kept,
    };
    unsafe { BE::znx_normalize(res, base2k, kept * base2k, k_rem as i64, a, base2k, n) }
}
