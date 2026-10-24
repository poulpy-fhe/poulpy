use crate::{
    kernels::znx::ZnxZero,
    layouts::{Backend, HostDataMut, VecZnxBackendMut, ZnxViewMut},
};

pub fn vec_znx_zero_portable<'r, BE>(res: &mut VecZnxBackendMut<'r, BE>, res_col: usize)
where
    BE: Backend<ZnxWord = i64> + ZnxZero,
    BE::BufMut<'r>: HostDataMut,
{
    let res_size = res.size();
    for j in 0..res_size {
        BE::znx_zero(res.at_mut(res_col, j));
    }
}
