use crate::{
    layouts::{Backend, HostDataMut, HostDataRef, VecZnxBackendMut, VecZnxBackendRef, ZnxView, ZnxViewMut},
    reference::znx::{ZnxCopy, ZnxZero},
};

pub fn vec_znx_copy<'r, 'a, BE>(res: &mut VecZnxBackendMut<'r, BE>, res_col: usize, a: &VecZnxBackendRef<'a, BE>, a_col: usize)
where
    BE: Backend<ZnxWord = i64> + ZnxCopy + ZnxZero,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    {
        assert_eq!(res.n(), a.n())
    }

    let res_size = res.size();
    let a_size = a.size();

    let min_size = res_size.min(a_size);

    for j in 0..min_size {
        BE::znx_copy(res.at_mut(res_col, j), a.at(a_col, j));
    }

    for j in min_size..res_size {
        BE::znx_zero(res.at_mut(res_col, j));
    }
}
