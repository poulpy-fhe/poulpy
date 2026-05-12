use crate::{
    layouts::{Backend, HostDataMut, HostDataRef, ScalarZnxBackendRef, VecZnxBackendMut, VecZnxBackendRef, ZnxView, ZnxViewMut},
    reference::znx::{ZnxSub, ZnxSubAssign, ZnxZero},
};

pub fn vec_znx_sub_scalar<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &ScalarZnxBackendRef<'a, BE>,
    a_col: usize,
    b: &VecZnxBackendRef<'a, BE>,
    b_col: usize,
    b_limb: usize,
) where
    BE: Backend + ZnxSub + ZnxZero,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    let min_size: usize = b.size().min(res.size());

    #[cfg(debug_assertions)]
    {
        assert!(b_limb < min_size, "b_limb: {b_limb} > min_size: {min_size}");
    }

    for j in 0..min_size {
        if j == b_limb {
            BE::znx_sub(res.at_mut(res_col, j), b.at(b_col, j), a.at(a_col, 0));
        } else {
            res.at_mut(res_col, j).copy_from_slice(b.at(b_col, j));
        }
    }

    for j in min_size..res.size() {
        BE::znx_zero(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_sub_scalar_assign<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    res_limb: usize,
    a: &ScalarZnxBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend + ZnxSubAssign,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert!(res_limb < res.size());
    }

    BE::znx_sub_assign(res.at_mut(res_col, res_limb), a.at(a_col, 0));
}
