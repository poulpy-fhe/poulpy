use crate::{
    layouts::{VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut},
    reference::{
        vec_znx::{vec_znx_rotate, vec_znx_sub_assign},
        znx::{ZnxNegate, ZnxRotate, ZnxSubAssign, ZnxSubNegateAssign, ZnxZero},
    },
};

pub fn vec_znx_mul_xp_minus_one_assign_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_mul_xp_minus_one<R, A, ZNXARI>(p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
    ZNXARI: ZnxRotate + ZnxZero + ZnxSubAssign,
{
    vec_znx_rotate::<_, _, ZNXARI>(p, res, res_col, a, a_col);
    vec_znx_sub_assign::<_, _, ZNXARI>(res, res_col, a, a_col);
}

pub fn vec_znx_mul_xp_minus_one_assign<R, ZNXARI>(p: i64, res: &mut R, res_col: usize, tmp: &mut [i64])
where
    R: VecZnxToMut,
    ZNXARI: ZnxRotate + ZnxNegate + ZnxSubNegateAssign,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), tmp.len());
    }
    for j in 0..res.size() {
        ZNXARI::znx_rotate(p, tmp, res.at(res_col, j));
        ZNXARI::znx_sub_negate_assign(res.at_mut(res_col, j), tmp);
    }
}
