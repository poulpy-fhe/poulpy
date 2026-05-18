use std::{
    num::Wrapping,
    ops::{Add, Mul},
};

#[inline]
fn packed_offset(nrows: usize, ncols: usize, col: usize, coeff: usize) -> usize {
    let block = coeff / 8;
    let lane = coeff % 8;
    let block_stride = nrows * ncols * 8;

    if col == ncols - 1 && !ncols.is_multiple_of(2) {
        col * nrows * 8 + block * block_stride + lane
    } else {
        (col / 2) * (nrows * 16) + (col % 2) * 8 + block * block_stride + lane
    }
}

#[inline]
pub fn coeff_mat1col_product<T>(rows_in: usize, nrows: usize, ncols: usize, col: usize, res: &mut T, a: &[i64], pmat: &[i64])
where
    T: Copy + From<i64>,
    Wrapping<T>: Add<Output = Wrapping<T>> + Mul<Output = Wrapping<T>>,
{
    let mut acc = Wrapping(T::from(0));
    for (coeff, &a_value) in a.iter().enumerate().take(rows_in) {
        let pmat_value = pmat[packed_offset(nrows, ncols, col, coeff)];
        acc = acc + Wrapping(T::from(pmat_value)) * Wrapping(T::from(a_value));
    }
    *res = acc.0;
}

#[inline]
pub fn coeff_mat2cols_product<T>(
    rows_in: usize,
    nrows: usize,
    ncols: usize,
    col: usize,
    res: &mut [T; 2],
    a: &[i64],
    pmat: &[i64],
) where
    T: Copy + From<i64>,
    Wrapping<T>: Add<Output = Wrapping<T>> + Mul<Output = Wrapping<T>>,
{
    debug_assert!(col.is_multiple_of(2));
    debug_assert!(col + 1 < ncols);

    let mut acc0 = Wrapping(T::from(0));
    let mut acc1 = Wrapping(T::from(0));
    for (coeff, &a_value) in a.iter().enumerate().take(rows_in) {
        let block = coeff / 8;
        let lane = coeff % 8;
        let block_stride = nrows * ncols * 8;
        let offset = (col / 2) * (nrows * 16) + block * block_stride + lane;
        let a_value = Wrapping(T::from(a_value));
        acc0 = acc0 + Wrapping(T::from(pmat[offset])) * a_value;
        acc1 = acc1 + Wrapping(T::from(pmat[offset + 8])) * a_value;
    }

    res[0] = acc0.0;
    res[1] = acc1.0;
}
