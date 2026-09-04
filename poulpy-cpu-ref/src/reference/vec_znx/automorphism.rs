use std::mem::size_of;

use crate::{
    layouts::{
        ArithmeticState, Backend, CoeffFitsIn, HostDataMut, HostDataRef, VecZnxBackendMut, VecZnxBackendRef, ZnxView, ZnxViewMut,
    },
    reference::znx::{ZnxAutomorphism, ZnxCopy, ZnxZero},
};

pub fn vec_znx_automorphism_assign_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_automorphism<'r, 'a, BE, S: ArithmeticState>(
    p: i64,
    res: &mut VecZnxBackendMut<'r, BE, S>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE, impl CoeffFitsIn<S>>,
    a_col: usize,
) where
    BE: Backend<ZnxWord = i64> + ZnxAutomorphism + ZnxZero,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
    }

    let min_size: usize = res.size().min(a.size());

    for j in 0..min_size {
        BE::znx_automorphism(p, crate::reference::kernel_words_mut(res).at_mut(res_col, j), a.at(a_col, j));
    }

    for j in min_size..res.size() {
        BE::znx_zero(crate::reference::kernel_words_mut(res).at_mut(res_col, j));
    }
}

pub fn vec_znx_automorphism_assign<'r, BE>(
    p: i64,
    res: &mut VecZnxBackendMut<'r, BE, impl ArithmeticState>,
    res_col: usize,
    tmp: &mut [i64],
) where
    BE: Backend<ZnxWord = i64> + ZnxAutomorphism + ZnxCopy,
    BE::BufMut<'r>: HostDataMut,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), tmp.len());
    }

    for j in 0..res.size() {
        BE::znx_automorphism(p, tmp, res.at(res_col, j));
        BE::znx_copy(crate::reference::kernel_words_mut(res).at_mut(res_col, j), tmp);
    }
}
