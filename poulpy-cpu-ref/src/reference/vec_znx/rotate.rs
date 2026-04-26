use crate::{
    layouts::{VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut},
    reference::znx::{ZnxCopy, ZnxRotate, ZnxZero},
};

pub fn vec_znx_rotate_assign_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_rotate<R, A, ZNXARI>(p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
    ZNXARI: ZnxRotate + ZnxZero,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), a.n())
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let min_size: usize = res_size.min(a_size);

    for j in 0..min_size {
        ZNXARI::znx_rotate(p, res.at_mut(res_col, j), a.at(a_col, j))
    }

    for j in min_size..res_size {
        ZNXARI::znx_zero(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_rotate_assign<R, ZNXARI>(p: i64, res: &mut R, res_col: usize, tmp: &mut [i64])
where
    R: VecZnxToMut,
    ZNXARI: ZnxRotate + ZnxCopy,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), tmp.len());
    }
    for j in 0..res.size() {
        ZNXARI::znx_rotate(p, tmp, res.at(res_col, j));
        ZNXARI::znx_copy(res.at_mut(res_col, j), tmp);
    }
}
