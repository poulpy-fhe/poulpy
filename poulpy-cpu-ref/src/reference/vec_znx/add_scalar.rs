use crate::{
    layouts::{Backend, HostDataMut, HostDataRef, ScalarZnxBackendRef, VecZnxBackendMut, ZnxView, ZnxViewMut},
    reference::znx::ZnxAddAssign,
};

pub fn vec_znx_add_scalar_assign<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    res_limb: usize,
    a: &ScalarZnxBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend<ZnxWord = i64> + ZnxAddAssign,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    {
        assert!(res_limb < res.size());
    }

    BE::znx_add_assign(res.at_mut(res_col, res_limb), a.at(a_col, 0));
}
