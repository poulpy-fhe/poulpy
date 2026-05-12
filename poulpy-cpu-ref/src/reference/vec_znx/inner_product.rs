use crate::layouts::{ScalarZnx, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef, ZnxView, ZnxViewMut};

#[allow(clippy::too_many_arguments)]
pub fn vec_znx_sub_inner_product_assign<R, A>(
    res: &mut R,
    res_col: usize,
    res_limb: usize,
    res_offset: usize,
    a: &A,
    a_col: usize,
    a_limb: usize,
    a_offset: usize,
    b: &ScalarZnx<&[u8]>,
    b_col: usize,
    b_offset: usize,
    len: usize,
) where
    R: VecZnxToBackendMut,
    A: VecZnxToBackendRef,
{
    let mut res: VecZnx<&mut [u8]> = res.to_backend_mut();
    let a: VecZnx<&[u8]> = a.to_backend_ref();

    assert!(res_limb < res.size());
    assert!(res_offset < res.n());
    assert!(a_offset + len <= a.n());
    assert!(b_offset + len <= b.n());

    let sum: i64 = a.at(a_col, a_limb)[a_offset..a_offset + len]
        .iter()
        .zip(&b.at(b_col, 0)[b_offset..b_offset + len])
        .map(|(x, y)| x * y)
        .sum();

    res.at_mut(res_col, res_limb)[res_offset] -= sum;
}
