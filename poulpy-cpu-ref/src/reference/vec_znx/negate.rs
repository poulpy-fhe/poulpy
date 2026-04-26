use crate::{
    layouts::{VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut},
    reference::znx::{ZnxNegate, ZnxNegateAssign, ZnxZero},
};

pub fn vec_znx_negate<R, A, ZNXARI>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
    ZNXARI: ZnxNegate + ZnxZero,
{
    let a: VecZnx<&[u8]> = a.to_ref();
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
    }

    let min_size: usize = res.size().min(a.size());

    for j in 0..min_size {
        ZNXARI::znx_negate(res.at_mut(res_col, j), a.at(a_col, j));
    }

    for j in min_size..res.size() {
        ZNXARI::znx_zero(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_negate_assign<R, ZNXARI>(res: &mut R, res_col: usize)
where
    R: VecZnxToMut,
    ZNXARI: ZnxNegateAssign,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    for j in 0..res.size() {
        ZNXARI::znx_negate_assign(res.at_mut(res_col, j));
    }
}
