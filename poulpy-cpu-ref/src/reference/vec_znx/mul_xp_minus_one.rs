use std::mem::size_of;

use crate::{
    layouts::{Backend, HostDataMut, HostDataRef, VecZnxBackendMut, VecZnxBackendRef, ZnxView, ZnxViewMut},
    reference::{
        vec_znx::{vec_znx_rotate, vec_znx_sub_assign},
        znx::{ZnxNegate, ZnxRotate, ZnxSubAssign, ZnxSubNegateAssign, ZnxZero},
    },
};

pub fn vec_znx_mul_xp_minus_one_assign_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_mul_xp_minus_one<'r, 'a, BE>(
    p: i64,
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend + ZnxRotate + ZnxZero + ZnxSubAssign,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    vec_znx_rotate::<BE>(p, res, res_col, a, a_col);
    vec_znx_sub_assign::<BE>(res, res_col, a, a_col);
}

pub fn vec_znx_mul_xp_minus_one_assign<'r, BE>(p: i64, res: &mut VecZnxBackendMut<'r, BE>, res_col: usize, tmp: &mut [i64])
where
    BE: Backend + ZnxRotate + ZnxNegate + ZnxSubNegateAssign,
    BE::BufMut<'r>: HostDataMut,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), tmp.len());
    }
    for j in 0..res.size() {
        BE::znx_rotate(p, tmp, res.at(res_col, j));
        BE::znx_sub_negate_assign(res.at_mut(res_col, j), tmp);
    }
}
