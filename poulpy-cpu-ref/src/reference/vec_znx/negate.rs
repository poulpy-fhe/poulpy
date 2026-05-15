use crate::{
    layouts::{Backend, HostDataMut, HostDataRef, VecZnxBackendMut, VecZnxBackendRef, ZnxView, ZnxViewMut},
    reference::znx::{ZnxNegate, ZnxNegateAssign, ZnxZero},
};

pub fn vec_znx_negate<'r, 'a, BE>(res: &mut VecZnxBackendMut<'r, BE>, res_col: usize, a: &VecZnxBackendRef<'a, BE>, a_col: usize)
where
    BE: Backend + ZnxNegate + ZnxZero,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
    }

    let min_size: usize = res.size().min(a.size());

    for j in 0..min_size {
        BE::znx_negate(res.at_mut(res_col, j), a.at(a_col, j));
    }

    for j in min_size..res.size() {
        BE::znx_zero(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_negate_assign<'r, BE>(res: &mut VecZnxBackendMut<'r, BE>, res_col: usize)
where
    BE: Backend + ZnxNegateAssign,
    BE::BufMut<'r>: HostDataMut,
{
    for j in 0..res.size() {
        BE::znx_negate_assign(res.at_mut(res_col, j));
    }
}
