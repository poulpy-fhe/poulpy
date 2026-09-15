//! CPU override of the rotate-into-a-full-width-temporary default for
//! `vec_znx_mul_xp_minus_one_assign`.
//!
//! The OEP default carves a whole `res.size()`-limb `VecZnx`; the kernel below
//! rotates limb by limb through a one-limb temporary, so the backend also
//! overrides `vec_znx_mul_xp_minus_one_assign_tmp_bytes` down to that single
//! limb. The out-of-place `vec_znx_mul_xp_minus_one` stays on its default.

use std::mem::size_of;

use crate::{
    layouts::{Backend, HostDataMut, VecZnxBackendMut, ZnxView, ZnxViewMut},
    reference::znx::{ZnxNegate, ZnxRotate, ZnxSubNegateAssign},
};

pub fn vec_znx_mul_xp_minus_one_assign_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_mul_xp_minus_one_assign<'r, BE>(p: i64, res: &mut VecZnxBackendMut<'r, BE>, res_col: usize, tmp: &mut [i64])
where
    BE: Backend<ZnxWord = i64> + ZnxRotate + ZnxNegate + ZnxSubNegateAssign,
    BE::BufMut<'r>: HostDataMut,
{
    poulpy_hal::layouts::assert_dense(res, "vec_znx_mul_xp_minus_one_assign");
    {
        assert_eq!(res.n(), tmp.len());
    }
    for j in 0..res.size() {
        BE::znx_rotate(p, tmp, res.at(res_col, j));
        BE::znx_sub_negate_assign(res.at_mut(res_col, j), tmp);
    }
}
