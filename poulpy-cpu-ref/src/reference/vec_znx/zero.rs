use crate::{
    layouts::{Backend, HostDataMut, NormalizationState, VecZnxBackendMut, ZnxViewMut},
    reference::znx::ZnxZero,
};

pub fn vec_znx_zero<'r, BE>(res: &mut VecZnxBackendMut<'r, BE, impl NormalizationState>, res_col: usize)
where
    BE: Backend<ZnxWord = i64> + ZnxZero,
    BE::BufMut<'r>: HostDataMut,
{
    let res_size = res.size();
    for j in 0..res_size {
        BE::znx_zero(res.at_mut(res_col, j));
    }
}
