use super::{HostBufMut, take_host_typed};

use std::{
    mem::size_of,
    num::Wrapping,
    ops::{Add, Mul},
};

use crate::reference::{
    fft64::vec_znx_big::{
        vec_znx_big_add as fft64_vec_znx_big_add, vec_znx_big_add_assign as fft64_vec_znx_big_add_assign,
        vec_znx_big_add_small_assign as fft64_vec_znx_big_add_small_assign,
        vec_znx_big_automorphism as fft64_vec_znx_big_automorphism,
        vec_znx_big_automorphism_assign as fft64_vec_znx_big_automorphism_assign,
        vec_znx_big_automorphism_assign_tmp_bytes as fft64_vec_znx_big_automorphism_assign_tmp_bytes,
        vec_znx_big_negate as fft64_vec_znx_big_negate, vec_znx_big_negate_assign as fft64_vec_znx_big_negate_assign,
        vec_znx_big_sub as fft64_vec_znx_big_sub, vec_znx_big_sub_assign as fft64_vec_znx_big_sub_assign,
        vec_znx_big_sub_negate_assign as fft64_vec_znx_big_sub_negate_assign,
        vec_znx_big_sub_small_a_assign as fft64_vec_znx_big_sub_small_a_assign,
        vec_znx_big_sub_small_b_assign as fft64_vec_znx_big_sub_small_b_assign,
    },
    ntt4x30::vec_znx_big::{
        I128BigOps, ntt4x30_vec_znx_big_add, ntt4x30_vec_znx_big_add_assign, ntt4x30_vec_znx_big_add_small_assign,
        ntt4x30_vec_znx_big_automorphism, ntt4x30_vec_znx_big_automorphism_assign,
        ntt4x30_vec_znx_big_automorphism_assign_tmp_bytes, ntt4x30_vec_znx_big_from_small, ntt4x30_vec_znx_big_negate,
        ntt4x30_vec_znx_big_negate_assign, ntt4x30_vec_znx_big_sub, ntt4x30_vec_znx_big_sub_assign,
        ntt4x30_vec_znx_big_sub_negate_assign, ntt4x30_vec_znx_big_sub_small_assign, ntt4x30_vec_znx_big_sub_small_negate_assign,
    },
    znx::{
        ZnxAdd, ZnxAddAssign, ZnxAutomorphism, ZnxCopy, ZnxNegate, ZnxNegateAssign, ZnxSub, ZnxSubAssign, ZnxSubNegateAssign,
        ZnxZero, znx_copy_ref, znx_zero_ref,
    },
};
use poulpy_hal::layouts::{
    Backend, HostDataMut, HostDataRef, Module, ScalarZnxBackendRef, ScratchArena, VecZnx, VecZnxBackendRef,
    VecZnxBigToBackendMut, VecZnxBigToBackendRef, ZnxView, ZnxViewMut,
};

fn vec_znx_backend_ref_as_host_ref<'a, 'b, BE>(a: &'a VecZnx<BE::BufRef<'b>, BE::ZnxWord>) -> VecZnx<&'a [u8], i64>
where
    BE: Backend<ZnxWord = i64> + 'b,
    for<'x> BE::BufRef<'x>: AsRef<[u8]>,
{
    VecZnx::from_shape(a.data().as_ref(), a.shape())
}

fn vec_znx_big_inner_sum_default_impl<R, A, BE>(res: &mut R, res_col: usize, res_coeff: usize, a: &A, a_col: usize)
where
    BE: Backend<ZnxWord = i64>,
    BE::BigWord: Copy + From<i64>,
    Wrapping<BE::BigWord>: Add<Output = Wrapping<BE::BigWord>>,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();

    assert!(res_coeff < res.n());
    assert!(res.size() <= a.size());
    for limb in 0..res.size() {
        let sum = a
            .at(a_col, limb)
            .iter()
            .fold(Wrapping(BE::BigWord::from(0)), |acc, &x| acc + Wrapping(x));

        res.at_mut(res_col, limb)[res_coeff] = sum.0;
    }
}

fn vec_znx_scalar_product_default_impl<R, BE>(
    res: &mut R,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    b: &ScalarZnxBackendRef<'_, BE>,
    b_col: usize,
) where
    BE: Backend<ZnxWord = i64>,
    BE::BigWord: Copy + From<i64>,
    Wrapping<BE::BigWord>: Mul<Output = Wrapping<BE::BigWord>>,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
    R: VecZnxBigToBackendMut<BE>,
{
    let mut res = res.to_backend_mut();

    let n = a.n();
    assert_eq!(n, b.n());
    assert!(res.n() >= n);
    assert!(res.size() <= a.size());

    let b_slice = b.at(b_col, 0);
    for limb in 0..res.size() {
        let a_slice = a.at(a_col, limb);
        let res_slice = res.at_mut(res_col, limb);
        for k in 0..n {
            res_slice[k] = (Wrapping(BE::BigWord::from(a_slice[k])) * Wrapping(BE::BigWord::from(b_slice[k]))).0;
        }
    }
}

fn vec_znx_big_col_weighted_sum_default_impl<R, BE>(
    res: &mut R,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    weights: &ScalarZnxBackendRef<'_, BE>,
    weights_col: usize,
    cols: usize,
    coeffs: usize,
) where
    BE: Backend<ZnxWord = i64>,
    BE::BigWord: Copy + From<i64>,
    Wrapping<BE::BigWord>: Add<Output = Wrapping<BE::BigWord>> + Mul<Output = Wrapping<BE::BigWord>>,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
    R: VecZnxBigToBackendMut<BE>,
{
    let mut res = res.to_backend_mut();

    assert!(cols <= a.cols());
    assert!(cols <= weights.n());
    assert!(weights_col < weights.cols());
    assert!(coeffs <= a.n());
    assert!(coeffs <= res.n());
    assert!(res.size() <= a.size());

    let weights_slice = weights.at(weights_col, 0);
    let zero = BE::BigWord::from(0);

    for limb in 0..res.size() {
        let res_slice = res.at_mut(res_col, limb);
        res_slice.fill(zero);

        for (col, slice) in weights_slice.iter().enumerate().take(cols) {
            let weight = BE::BigWord::from(*slice);
            let a_slice = a.at(col, limb);
            for k in 0..coeffs {
                res_slice[k] = (Wrapping(res_slice[k]) + Wrapping(BE::BigWord::from(a_slice[k])) * Wrapping(weight)).0;
            }
        }
    }
}

pub trait FFT64VecZnxBigDefault: Backend<ZnxWord = i64>
where
    Self::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'x> Self::BufMut<'x>: HostDataMut,
    for<'x> Self::BufRef<'x>: HostDataRef,
{
    fn vec_znx_big_from_small_default<R>(res: &mut R, res_col: usize, a: &VecZnxBackendRef<'_, Self>, a_col: usize)
    where
        Self: Backend<BigWord = i64, ZnxWord = i64>,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<Self>,
    {
        let mut res = res.to_backend_mut();
        let a: VecZnx<&[u8], i64> = vec_znx_backend_ref_as_host_ref::<Self>(a);

        let res_size = res.size();
        let a_size = a.size();
        let min_size = res_size.min(a_size);

        for j in 0..min_size {
            znx_copy_ref(res.at_mut(res_col, j), a.at(a_col, j));
        }

        for j in min_size..res_size {
            znx_zero_ref(res.at_mut(res_col, j));
        }
    }

    fn vec_znx_big_add_default<R, A, C>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        Self: Backend<BigWord = i64, ZnxWord = i64> + ZnxAdd + ZnxCopy + ZnxZero,
        for<'x> Self::BufRef<'x>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<Self>,
        A: VecZnxBigToBackendRef<Self>,
        C: VecZnxBigToBackendRef<Self>,
    {
        fft64_vec_znx_big_add::<_, _, _, Self>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_big_add_assign_default<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        Self: Backend<BigWord = i64, ZnxWord = i64> + ZnxAddAssign,
        R: VecZnxBigToBackendMut<Self>,
        A: VecZnxBigToBackendRef<Self>,
    {
        fft64_vec_znx_big_add_assign::<_, _, Self>(res, res_col, a, a_col);
    }

    fn vec_znx_big_add_small_assign_default<R>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: Backend<BigWord = i64, ZnxWord = i64> + ZnxAddAssign,
        for<'x> Self::BufRef<'x>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<Self>,
    {
        fft64_vec_znx_big_add_small_assign::<_, _, Self>(res, res_col, &a, a_col);
    }

    fn vec_znx_big_sub_default<R, A, C>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        Self: Backend<BigWord = i64, ZnxWord = i64> + ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
        R: VecZnxBigToBackendMut<Self>,
        A: VecZnxBigToBackendRef<Self>,
        C: VecZnxBigToBackendRef<Self>,
    {
        fft64_vec_znx_big_sub::<_, _, _, Self>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_big_sub_assign_default<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        Self: Backend<BigWord = i64, ZnxWord = i64> + ZnxSubAssign,
        R: VecZnxBigToBackendMut<Self>,
        A: VecZnxBigToBackendRef<Self>,
    {
        fft64_vec_znx_big_sub_assign::<_, _, Self>(res, res_col, a, a_col);
    }

    fn vec_znx_big_sub_negate_assign_default<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        Self: Backend<BigWord = i64, ZnxWord = i64> + ZnxSubNegateAssign + ZnxNegateAssign,
        R: VecZnxBigToBackendMut<Self>,
        A: VecZnxBigToBackendRef<Self>,
    {
        fft64_vec_znx_big_sub_negate_assign::<_, _, Self>(res, res_col, a, a_col);
    }

    fn vec_znx_big_sub_small_assign_default<R>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: Backend<BigWord = i64, ZnxWord = i64> + ZnxSubAssign,
        for<'x> Self::BufRef<'x>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<Self>,
    {
        fft64_vec_znx_big_sub_small_a_assign::<_, _, Self>(res, res_col, &a, a_col);
    }

    fn vec_znx_big_sub_small_negate_assign_default<R>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: Backend<BigWord = i64, ZnxWord = i64> + ZnxSubNegateAssign + ZnxNegateAssign,
        for<'x> Self::BufRef<'x>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<Self>,
    {
        fft64_vec_znx_big_sub_small_b_assign::<_, _, Self>(res, res_col, &a, a_col);
    }

    fn vec_znx_big_inner_sum_default<R, A>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        res_coeff: usize,
        a: &A,
        a_col: usize,
    ) where
        Self: Backend<BigWord = i64, ZnxWord = i64>,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: HostDataRef,
        R: VecZnxBigToBackendMut<Self>,
        A: VecZnxBigToBackendRef<Self>,
    {
        vec_znx_big_inner_sum_default_impl::<R, A, Self>(res, res_col, res_coeff, a, a_col);
    }

    fn vec_znx_scalar_product_default<R>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &ScalarZnxBackendRef<'_, Self>,
        b_col: usize,
    ) where
        Self: Backend<BigWord = i64, ZnxWord = i64>,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: HostDataRef,
        R: VecZnxBigToBackendMut<Self>,
    {
        vec_znx_scalar_product_default_impl::<R, Self>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_big_col_weighted_sum_default<R>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        weights: &ScalarZnxBackendRef<'_, Self>,
        weights_col: usize,
        cols: usize,
        coeffs: usize,
    ) where
        Self: Backend<BigWord = i64, ZnxWord = i64>,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: HostDataRef,
        R: VecZnxBigToBackendMut<Self>,
    {
        vec_znx_big_col_weighted_sum_default_impl::<R, Self>(res, res_col, a, weights, weights_col, cols, coeffs);
    }

    fn vec_znx_big_negate_default<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        Self: Backend<BigWord = i64, ZnxWord = i64> + ZnxNegate + ZnxZero,
        R: VecZnxBigToBackendMut<Self>,
        A: VecZnxBigToBackendRef<Self>,
    {
        fft64_vec_znx_big_negate::<_, _, Self>(res, res_col, a, a_col);
    }

    fn vec_znx_big_negate_assign_default<R>(_module: &Module<Self>, res: &mut R, res_col: usize)
    where
        Self: Backend<BigWord = i64, ZnxWord = i64> + ZnxNegateAssign,
        R: VecZnxBigToBackendMut<Self>,
    {
        fft64_vec_znx_big_negate_assign::<_, Self>(res, res_col);
    }

    fn vec_znx_big_automorphism_default<R, A>(_module: &Module<Self>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        Self: Backend<BigWord = i64, ZnxWord = i64> + ZnxAutomorphism + ZnxZero,
        R: VecZnxBigToBackendMut<Self>,
        A: VecZnxBigToBackendRef<Self>,
    {
        fft64_vec_znx_big_automorphism::<_, _, Self>(k, res, res_col, a, a_col);
    }

    fn vec_znx_big_automorphism_assign_tmp_bytes_default(module: &Module<Self>) -> usize
    where
        Self: Backend<BigWord = i64, ZnxWord = i64>,
    {
        fft64_vec_znx_big_automorphism_assign_tmp_bytes(module.n())
    }

    fn vec_znx_big_automorphism_assign_default<R>(
        module: &Module<Self>,
        k: i64,
        res: &mut R,
        res_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Self: Backend<BigWord = i64, ZnxWord = i64> + ZnxAutomorphism + ZnxCopy,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
        R: VecZnxBigToBackendMut<Self>,
    {
        let (tmp, _) = take_host_typed::<Self, i64>(
            scratch.borrow(),
            fft64_vec_znx_big_automorphism_assign_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        fft64_vec_znx_big_automorphism_assign::<_, Self>(k, res, res_col, tmp);
    }
}

impl<BE: Backend<ZnxWord = i64>> FFT64VecZnxBigDefault for BE
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
}

pub trait NTT4x30VecZnxBigDefault: Backend<ZnxWord = i64>
where
    Self::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'x> Self::BufMut<'x>: HostDataMut,
    for<'x> Self::BufRef<'x>: HostDataRef,
{
    fn vec_znx_big_from_small_default<R>(res: &mut R, res_col: usize, a: &VecZnxBackendRef<'_, Self>, a_col: usize)
    where
        Self: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        for<'x> Self::BufRef<'x>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<Self>,
    {
        let a = vec_znx_backend_ref_as_host_ref::<Self>(a);
        ntt4x30_vec_znx_big_from_small::<_, _, Self>(res, res_col, &a, a_col);
    }

    fn vec_znx_big_add_default<R, A, C>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        Self: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        for<'x> Self::BufRef<'x>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<Self>,
        A: VecZnxBigToBackendRef<Self>,
        C: VecZnxBigToBackendRef<Self>,
    {
        ntt4x30_vec_znx_big_add::<_, _, _, Self>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_big_add_assign_default<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        Self: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        R: VecZnxBigToBackendMut<Self>,
        A: VecZnxBigToBackendRef<Self>,
    {
        ntt4x30_vec_znx_big_add_assign::<_, _, Self>(res, res_col, a, a_col);
    }

    fn vec_znx_big_add_small_assign_default<R>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        for<'x> Self::BufRef<'x>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<Self>,
    {
        let a = vec_znx_backend_ref_as_host_ref::<Self>(a);
        ntt4x30_vec_znx_big_add_small_assign::<_, _, Self>(res, res_col, &a, a_col);
    }

    fn vec_znx_big_sub_default<R, A, C>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        Self: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        R: VecZnxBigToBackendMut<Self>,
        A: VecZnxBigToBackendRef<Self>,
        C: VecZnxBigToBackendRef<Self>,
    {
        ntt4x30_vec_znx_big_sub::<_, _, _, Self>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_big_sub_assign_default<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        Self: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        R: VecZnxBigToBackendMut<Self>,
        A: VecZnxBigToBackendRef<Self>,
    {
        ntt4x30_vec_znx_big_sub_assign::<_, _, Self>(res, res_col, a, a_col);
    }

    fn vec_znx_big_sub_negate_assign_default<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        Self: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        R: VecZnxBigToBackendMut<Self>,
        A: VecZnxBigToBackendRef<Self>,
    {
        ntt4x30_vec_znx_big_sub_negate_assign::<_, _, Self>(res, res_col, a, a_col);
    }

    fn vec_znx_big_sub_small_assign_default<R>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        for<'x> Self::BufRef<'x>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<Self>,
    {
        let a = vec_znx_backend_ref_as_host_ref::<Self>(a);
        ntt4x30_vec_znx_big_sub_small_assign::<_, _, Self>(res, res_col, &a, a_col);
    }

    fn vec_znx_big_sub_small_negate_assign_default<R>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        for<'x> Self::BufRef<'x>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<Self>,
    {
        let a = vec_znx_backend_ref_as_host_ref::<Self>(a);
        ntt4x30_vec_znx_big_sub_small_negate_assign::<_, _, Self>(res, res_col, &a, a_col);
    }

    fn vec_znx_big_inner_sum_default<R, A>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        res_coeff: usize,
        a: &A,
        a_col: usize,
    ) where
        Self: Backend<BigWord = i128, ZnxWord = i64>,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: HostDataRef,
        R: VecZnxBigToBackendMut<Self>,
        A: VecZnxBigToBackendRef<Self>,
    {
        vec_znx_big_inner_sum_default_impl::<R, A, Self>(res, res_col, res_coeff, a, a_col);
    }

    fn vec_znx_scalar_product_default<R>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &ScalarZnxBackendRef<'_, Self>,
        b_col: usize,
    ) where
        Self: Backend<BigWord = i128, ZnxWord = i64>,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: HostDataRef,
        R: VecZnxBigToBackendMut<Self>,
    {
        vec_znx_scalar_product_default_impl::<R, Self>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_big_col_weighted_sum_default<R>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        weights: &ScalarZnxBackendRef<'_, Self>,
        weights_col: usize,
        cols: usize,
        coeffs: usize,
    ) where
        Self: Backend<BigWord = i128, ZnxWord = i64>,
        for<'x> Self::BufMut<'x>: HostDataMut,
        for<'x> Self::BufRef<'x>: HostDataRef,
        R: VecZnxBigToBackendMut<Self>,
    {
        vec_znx_big_col_weighted_sum_default_impl::<R, Self>(res, res_col, a, weights, weights_col, cols, coeffs);
    }

    fn vec_znx_big_negate_default<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        Self: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        R: VecZnxBigToBackendMut<Self>,
        A: VecZnxBigToBackendRef<Self>,
    {
        ntt4x30_vec_znx_big_negate::<_, _, Self>(res, res_col, a, a_col);
    }

    fn vec_znx_big_negate_assign_default<R>(_module: &Module<Self>, res: &mut R, res_col: usize)
    where
        Self: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        R: VecZnxBigToBackendMut<Self>,
    {
        ntt4x30_vec_znx_big_negate_assign::<_, Self>(res, res_col);
    }

    fn vec_znx_big_automorphism_default<R, A>(_module: &Module<Self>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        Self: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        R: VecZnxBigToBackendMut<Self>,
        A: VecZnxBigToBackendRef<Self>,
    {
        ntt4x30_vec_znx_big_automorphism::<_, _, Self>(k, res, res_col, a, a_col);
    }

    fn vec_znx_big_automorphism_assign_tmp_bytes_default(module: &Module<Self>) -> usize
    where
        Self: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
    {
        ntt4x30_vec_znx_big_automorphism_assign_tmp_bytes(module.n())
    }

    fn vec_znx_big_automorphism_assign_default<R>(
        module: &Module<Self>,
        k: i64,
        res: &mut R,
        res_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Self: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
        R: VecZnxBigToBackendMut<Self>,
    {
        let (tmp, _) = take_host_typed::<Self, i128>(
            scratch.borrow(),
            ntt4x30_vec_znx_big_automorphism_assign_tmp_bytes(module.n()) / size_of::<i128>(),
        );
        ntt4x30_vec_znx_big_automorphism_assign::<_, Self>(k, res, res_col, tmp);
    }
}

impl<BE: Backend<ZnxWord = i64>> NTT4x30VecZnxBigDefault for BE
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
}
