use crate::{
    layouts::{Backend, HostDataMut, HostDataRef, VecZnxBackendMut, VecZnxBackendRef, ZnxView, ZnxViewMut},
    reference::znx::{ZnxCopy, ZnxNegate, ZnxNegateAssign, ZnxSub, ZnxSubAssign, ZnxSubNegateAssign, ZnxZero},
};

pub fn vec_znx_sub<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
    b: &VecZnxBackendRef<'a, BE>,
    b_col: usize,
) where
    BE: Backend + ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
        assert_eq!(b.n(), res.n());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();
    let b_size: usize = b.size();

    if a_size <= b_size {
        let sum_size: usize = a_size.min(res_size);
        let cpy_size: usize = b_size.min(res_size);

        for j in 0..sum_size {
            BE::znx_sub(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
        }

        for j in sum_size..cpy_size {
            BE::znx_negate(res.at_mut(res_col, j), b.at(b_col, j));
        }

        for j in cpy_size..res_size {
            BE::znx_zero(res.at_mut(res_col, j));
        }
    } else {
        let sum_size: usize = b_size.min(res_size);
        let cpy_size: usize = a_size.min(res_size);

        for j in 0..sum_size {
            BE::znx_sub(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
        }

        for j in sum_size..cpy_size {
            BE::znx_copy(res.at_mut(res_col, j), a.at(a_col, j));
        }

        for j in cpy_size..res_size {
            BE::znx_zero(res.at_mut(res_col, j));
        }
    }
}

pub fn vec_znx_sub_assign<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend + ZnxSubAssign,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let sum_size: usize = a_size.min(res_size);

    for j in 0..sum_size {
        BE::znx_sub_assign(res.at_mut(res_col, j), a.at(a_col, j));
    }
}

pub fn vec_znx_sub_negate_assign<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend + ZnxSubNegateAssign + ZnxNegateAssign,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let sum_size: usize = a_size.min(res_size);

    for j in 0..sum_size {
        BE::znx_sub_negate_assign(res.at_mut(res_col, j), a.at(a_col, j));
    }

    for j in sum_size..res_size {
        BE::znx_negate_assign(res.at_mut(res_col, j));
    }
}
