use std::mem::size_of;

use crate::{
    kernels::znx::{ZnxAutomorphism, ZnxCopy, ZnxZero},
    layouts::{Backend, HostDataMut, HostDataRef, Ring, VecZnxBackendMut, VecZnxBackendRef, ZnxView, ZnxViewMut},
};

pub fn vec_znx_automorphism_assign_tmp_bytes_portable(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_automorphism_portable<'r, 'a, BE, Rg: Ring>(
    p: i64,
    res: &mut VecZnxBackendMut<'r, BE, Rg>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE, Rg>,
    a_col: usize,
) where
    BE: Backend<ZnxWord = i64> + ZnxAutomorphism<Rg> + ZnxZero,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    poulpy_hal::layouts::assert_dense(res, "vec_znx_automorphism");
    poulpy_hal::layouts::assert_dense(a, "vec_znx_automorphism");
    {
        assert_eq!(a.n(), res.n());
    }

    let min_size: usize = res.size().min(a.size());

    for j in 0..min_size {
        BE::znx_automorphism(p, res.at_mut(res_col, j), a.at(a_col, j));
    }

    for j in min_size..res.size() {
        BE::znx_zero(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_automorphism_assign_portable<'r, BE, Rg: Ring>(
    p: i64,
    res: &mut VecZnxBackendMut<'r, BE, Rg>,
    res_col: usize,
    tmp: &mut [i64],
) where
    BE: Backend<ZnxWord = i64> + ZnxAutomorphism<Rg> + ZnxCopy,
    BE::BufMut<'r>: HostDataMut,
{
    poulpy_hal::layouts::assert_dense(res, "vec_znx_automorphism_assign");
    {
        assert_eq!(res.n(), tmp.len());
    }

    for j in 0..res.size() {
        BE::znx_automorphism(p, tmp, res.at(res_col, j));
        BE::znx_copy(res.at_mut(res_col, j), tmp);
    }
}
