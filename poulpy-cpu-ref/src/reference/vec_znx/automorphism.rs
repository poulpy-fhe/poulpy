use rand_distr::num_traits;
use std::mem::size_of;

use crate::{
    layouts::{Backend, HostDataMut, HostDataRef, VecZnxBackendMut, VecZnxBackendRef, ZnxView, ZnxViewMut},
    reference::znx::{ZnxAutomorphism, ZnxCopy, ZnxZero},
};

pub fn vec_znx_automorphism_assign_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_automorphism<'r, 'a, BE>(
    p: i64,
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend<ZnxWord = i64> + ZnxAutomorphism + ZnxZero,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    poulpy_hal::layouts::assert_dense(res, "vec_znx_automorphism");
    poulpy_hal::layouts::assert_dense(a, "vec_znx_automorphism");
    {
        assert_eq!(a.n(), res.n());
    }

    let min_size: usize = res.size().min(a.size());

    for j in 0..min_size {
        BE::znx_automorphism(p, res.at_mut(res_col, j), a.at(a_col, j));
    }

    for j in min_size..res.size() {
        BE::znx_zero(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_automorphism_assign<'r, BE>(p: i64, res: &mut VecZnxBackendMut<'r, BE>, res_col: usize, tmp: &mut [i64])
where
    BE: Backend<ZnxWord = i64> + ZnxAutomorphism + ZnxCopy,
    BE::BufMut<'r>: HostDataMut,
{
    poulpy_hal::layouts::assert_dense(res, "vec_znx_automorphism_assign");
    {
        assert_eq!(res.n(), tmp.len());
    }

    for j in 0..res.size() {
        BE::znx_automorphism(p, tmp, res.at(res_col, j));
        BE::znx_copy(res.at_mut(res_col, j), tmp);
    }
}

pub fn vec_znx_ci_automorphism<'r, 'a, BE>(
    p: i64,
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend<ZnxWord = i64> + ZnxZero,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    poulpy_hal::layouts::assert_dense(res, "vec_znx_ci_automorphism");
    poulpy_hal::layouts::assert_dense(a, "vec_znx_ci_automorphism");
    assert_eq!(res.n(), a.n());
    let min_size = res.size().min(a.size());
    for limb in 0..min_size {
        znx_ci_automorphism(p, res.at_mut(res_col, limb), a.at(a_col, limb));
    }
    for limb in min_size..res.size() {
        BE::znx_zero(res.at_mut(res_col, limb));
    }
}

pub fn vec_znx_ci_automorphism_assign<'r, BE>(p: i64, res: &mut VecZnxBackendMut<'r, BE>, res_col: usize, tmp: &mut [i64])
where
    BE: Backend<ZnxWord = i64> + ZnxCopy,
    BE::BufMut<'r>: HostDataMut,
{
    poulpy_hal::layouts::assert_dense(res, "vec_znx_ci_automorphism_assign");
    assert_eq!(res.n(), tmp.len());
    for limb in 0..res.size() {
        znx_ci_automorphism(p, tmp, res.at(res_col, limb));
        BE::znx_copy(res.at_mut(res_col, limb), tmp);
    }
}

pub fn znx_ci_automorphism<T: Copy + num_traits::Zero + num_traits::ops::wrapping::WrappingNeg>(p: i64, res: &mut [T], a: &[T]) {
    let n = a.len();
    assert_eq!(res.len(), n);
    assert!(n.is_power_of_two());
    assert!(p & 1 == 1, "p must be odd, got {p}");
    res.fill(T::zero());
    res[0] = a[0];
    let order = (4 * n) as i64;
    let p = p.rem_euclid(order);
    for (j, &value) in a.iter().enumerate().skip(1) {
        let mut exponent = (p * j as i64).rem_euclid(order);
        if exponent > 2 * n as i64 {
            exponent = order - exponent;
        }
        let sign = if exponent > n as i64 {
            exponent = 2 * n as i64 - exponent;
            -1
        } else {
            1
        };
        if exponent != n as i64 {
            res[exponent as usize] = if sign < 0 { value.wrapping_neg() } else { value };
        }
    }
}

pub fn vec_znx_big_ci_automorphism<'r, 'a, BE>(
    p: i64,
    res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'r, BE>,
    res_col: usize,
    a: &poulpy_hal::layouts::VecZnxBigBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend,
    BE::BigWord: Copy + num_traits::Zero + num_traits::ops::wrapping::WrappingNeg,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    poulpy_hal::layouts::assert_dense(res, "vec_znx_big_ci_automorphism");
    poulpy_hal::layouts::assert_dense(a, "vec_znx_big_ci_automorphism");
    assert_eq!(res.n(), a.n());
    let size = res.size().min(a.size());
    for limb in 0..size {
        znx_ci_automorphism(p, res.at_mut(res_col, limb), a.at(a_col, limb));
    }
    for limb in size..res.size() {
        res.at_mut(res_col, limb).fill(num_traits::Zero::zero());
    }
}

pub fn vec_znx_big_ci_automorphism_assign<'r, BE>(
    p: i64,
    res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'r, BE>,
    res_col: usize,
    tmp: &mut [BE::BigWord],
) where
    BE: Backend,
    BE::BigWord: Copy + num_traits::Zero + num_traits::ops::wrapping::WrappingNeg,
    BE::BufMut<'r>: HostDataMut,
{
    poulpy_hal::layouts::assert_dense(res, "vec_znx_big_ci_automorphism_assign");
    assert_eq!(res.n(), tmp.len());
    for limb in 0..res.size() {
        znx_ci_automorphism(p, tmp, res.at(res_col, limb));
        res.at_mut(res_col, limb).copy_from_slice(tmp);
    }
}
