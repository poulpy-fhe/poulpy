use crate::{
    layouts::{
        ArithmeticState, Backend, CoeffFitsIn, HostDataMut, HostDataRef, VecZnxBackendMut, VecZnxBackendRef, ZnxView, ZnxViewMut,
    },
    reference::znx::{ZnxNegate, ZnxNegateAssign, ZnxZero},
};

pub fn vec_znx_negate<'r, 'a, BE, S: ArithmeticState>(
    res: &mut VecZnxBackendMut<'r, BE, S>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE, impl CoeffFitsIn<S>>,
    a_col: usize,
) where
    BE: Backend<ZnxWord = i64> + ZnxNegate + ZnxZero,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
    }

    let min_size: usize = res.size().min(a.size());

    for j in 0..min_size {
        BE::znx_negate(crate::reference::kernel_words_mut(res).at_mut(res_col, j), a.at(a_col, j));
    }

    for j in min_size..res.size() {
        BE::znx_zero(crate::reference::kernel_words_mut(res).at_mut(res_col, j));
    }
}

pub fn vec_znx_negate_assign<'r, BE>(res: &mut VecZnxBackendMut<'r, BE, impl ArithmeticState>, res_col: usize)
where
    BE: Backend<ZnxWord = i64> + ZnxNegateAssign,
    BE::BufMut<'r>: HostDataMut,
{
    for j in 0..res.size() {
        BE::znx_negate_assign(crate::reference::kernel_words_mut(res).at_mut(res_col, j));
    }
}
