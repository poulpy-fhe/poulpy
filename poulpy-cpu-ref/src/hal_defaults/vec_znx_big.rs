//! Backend extension points for extended-precision [`poulpy_hal::layouts::VecZnxBig`] operations.

#![allow(clippy::too_many_arguments)]

use std::{
    mem::size_of,
    num::Wrapping,
    ops::{Add, Mul},
};

use crate::reference::{
    fft64::vec_znx_big::{
        vec_znx_big_add_assign as fft64_vec_znx_big_add_assign, vec_znx_big_add_into as fft64_vec_znx_big_add_into,
        vec_znx_big_add_normal_ref as fft64_vec_znx_big_add_normal_ref,
        vec_znx_big_add_small_assign as fft64_vec_znx_big_add_small_assign,
        vec_znx_big_add_small_into as fft64_vec_znx_big_add_small_into,
        vec_znx_big_automorphism as fft64_vec_znx_big_automorphism,
        vec_znx_big_automorphism_assign as fft64_vec_znx_big_automorphism_assign,
        vec_znx_big_automorphism_assign_tmp_bytes as fft64_vec_znx_big_automorphism_assign_tmp_bytes,
        vec_znx_big_negate as fft64_vec_znx_big_negate, vec_znx_big_negate_assign as fft64_vec_znx_big_negate_assign,
        vec_znx_big_normalize as fft64_vec_znx_big_normalize,
        vec_znx_big_normalize_tmp_bytes as fft64_vec_znx_big_normalize_tmp_bytes, vec_znx_big_sub as fft64_vec_znx_big_sub,
        vec_znx_big_sub_assign as fft64_vec_znx_big_sub_assign,
        vec_znx_big_sub_negate_assign as fft64_vec_znx_big_sub_negate_assign,
        vec_znx_big_sub_small_a as fft64_vec_znx_big_sub_small_a,
        vec_znx_big_sub_small_a_assign as fft64_vec_znx_big_sub_small_a_assign,
        vec_znx_big_sub_small_b as fft64_vec_znx_big_sub_small_b,
        vec_znx_big_sub_small_b_assign as fft64_vec_znx_big_sub_small_b_assign,
    },
    ntt4x30::vec_znx_big::{
        I128BigOps, I128NormalizeOps, ntt4x30_vec_znx_big_add_assign, ntt4x30_vec_znx_big_add_into,
        ntt4x30_vec_znx_big_add_normal_ref, ntt4x30_vec_znx_big_add_small_assign, ntt4x30_vec_znx_big_add_small_into,
        ntt4x30_vec_znx_big_automorphism, ntt4x30_vec_znx_big_automorphism_assign,
        ntt4x30_vec_znx_big_automorphism_assign_tmp_bytes, ntt4x30_vec_znx_big_from_small, ntt4x30_vec_znx_big_negate,
        ntt4x30_vec_znx_big_negate_assign, ntt4x30_vec_znx_big_normalize, ntt4x30_vec_znx_big_normalize_add_assign,
        ntt4x30_vec_znx_big_normalize_sub_assign, ntt4x30_vec_znx_big_normalize_tmp_bytes, ntt4x30_vec_znx_big_sub,
        ntt4x30_vec_znx_big_sub_assign, ntt4x30_vec_znx_big_sub_negate_assign, ntt4x30_vec_znx_big_sub_small_a,
        ntt4x30_vec_znx_big_sub_small_assign, ntt4x30_vec_znx_big_sub_small_b, ntt4x30_vec_znx_big_sub_small_negate_assign,
    },
    znx::{
        ZnxAdd, ZnxAddAssign, ZnxAutomorphism, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulPowerOfTwoAssign, ZnxNegate,
        ZnxNegateAssign, ZnxNormalizeDigit, ZnxNormalizeFinalStep, ZnxNormalizeFinalStepAssign, ZnxNormalizeFirstStep,
        ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepAssign, ZnxNormalizeMiddleStepCarryOnly,
        ZnxSub, ZnxSubAssign, ZnxSubNegateAssign, ZnxZero, znx_copy_ref, znx_zero_ref,
    },
};
use poulpy_hal::{
    api::HostBufMut,
    layouts::{
        ArithmeticState, Backend, HostDataMut, HostDataRef, Module, NoiseInfos, ScalarZnxBackendRef, ScratchArena, VecZnx,
        VecZnxBackendRef, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxToBackendMut, ZnxView, ZnxViewMut,
    },
    source::Source,
};

#[inline]
fn take_host_typed<'a, BE, T>(arena: ScratchArena<'a, BE>, len: usize) -> (&'a mut [T], ScratchArena<'a, BE>)
where
    BE: Backend<ZnxWord = i64> + 'a,
    BE::BufMut<'a>: HostBufMut<'a>,
    T: Copy,
{
    assert!(
        BE::SCRATCH_ALIGN.is_multiple_of(std::mem::align_of::<T>()),
        "B::SCRATCH_ALIGN ({}) must be a multiple of align_of::<T>() ({})",
        BE::SCRATCH_ALIGN,
        std::mem::align_of::<T>()
    );
    let byte_len = len
        .checked_mul(std::mem::size_of::<T>())
        .expect("typed scratch byte size overflows usize");
    let (buf, arena) = arena.take_region(byte_len);
    let bytes: &'a mut [u8] = buf.into_bytes();
    assert!(
        (bytes.as_mut_ptr() as usize).is_multiple_of(std::mem::align_of::<T>()),
        "scratch region is not aligned to align_of::<T>() = {}",
        std::mem::align_of::<T>()
    );
    let slice = unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, len) };
    (slice, arena)
}

#[inline]
fn vec_znx_backend_ref_as_host_ref<'a, 'b, BE, S: ArithmeticState>(
    a: &'a VecZnx<BE::BufRef<'b>, BE::ZnxWord, S>,
) -> VecZnx<&'a [u8], i64, S>
where
    BE: Backend<ZnxWord = i64> + 'b,
    for<'x> BE::BufRef<'x>: AsRef<[u8]>,
{
    poulpy_hal::oep::vec_znx_from_data_like(a, a.data.as_ref())
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
    a: &VecZnxBackendRef<'_, BE, impl ArithmeticState>,
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
    a: &VecZnxBackendRef<'_, BE, impl ArithmeticState>,
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

#[doc(hidden)]
pub trait FFT64VecZnxBigDefault<BE: Backend<ZnxWord = i64>>: Backend
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    fn vec_znx_big_from_small_default<R>(
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE, impl ArithmeticState>,
        a_col: usize,
    ) where
        BE: Backend<BigWord = i64, ZnxWord = i64>,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
    {
        let mut res = res.to_backend_mut();
        let a: VecZnx<&[u8], i64, _> = vec_znx_backend_ref_as_host_ref::<BE, _>(a);

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

    fn vec_znx_big_add_normal_default<R>(
        _module: &Module<BE>,
        res_base2k: usize,
        res: &mut R,
        res_col: usize,
        noise_infos: NoiseInfos,
        source: &mut Source,
    ) where
        BE: Backend<BigWord = i64, ZnxWord = i64>,
        R: VecZnxBigToBackendMut<BE>,
    {
        fft64_vec_znx_big_add_normal_ref::<_, BE>(res_base2k, res, res_col, noise_infos, source);
    }

    fn vec_znx_big_add_normal_seed_default<R>(
        module: &Module<BE>,
        res_base2k: usize,
        res: &mut R,
        res_col: usize,
        noise_infos: NoiseInfos,
        seed: [u8; 32],
    ) where
        BE: Backend<BigWord = i64, ZnxWord = i64>,
        R: VecZnxBigToBackendMut<BE>,
    {
        let mut source = Source::new(seed);
        Self::vec_znx_big_add_normal_default(module, res_base2k, res, res_col, noise_infos, &mut source);
    }

    fn vec_znx_big_add_into_default<R, A, C>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxAdd + ZnxCopy + ZnxZero,
        for<'x> BE::BufRef<'x>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
        C: VecZnxBigToBackendRef<BE>,
    {
        fft64_vec_znx_big_add_into::<_, _, _, BE>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_big_add_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxAddAssign,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        fft64_vec_znx_big_add_assign::<_, _, BE>(res, res_col, a, a_col);
    }

    fn vec_znx_big_add_small_into_default<R, A>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE, impl ArithmeticState>,
        b_col: usize,
    ) where
        BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxAdd + ZnxCopy + ZnxZero,
        for<'x> BE::BufRef<'x>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        fft64_vec_znx_big_add_small_into::<_, _, _, BE>(res, res_col, a, a_col, &b, b_col);
    }

    fn vec_znx_big_add_small_assign_default<R>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE, impl ArithmeticState>,
        a_col: usize,
    ) where
        BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxAddAssign,
        for<'x> BE::BufRef<'x>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
    {
        fft64_vec_znx_big_add_small_assign::<_, _, BE>(res, res_col, &a, a_col);
    }

    fn vec_znx_big_sub_default<R, A, C>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
        C: VecZnxBigToBackendRef<BE>,
    {
        fft64_vec_znx_big_sub::<_, _, _, BE>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_big_sub_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxSubAssign,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        fft64_vec_znx_big_sub_assign::<_, _, BE>(res, res_col, a, a_col);
    }

    fn vec_znx_big_sub_negate_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxSubNegateAssign + ZnxNegateAssign,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        fft64_vec_znx_big_sub_negate_assign::<_, _, BE>(res, res_col, a, a_col);
    }

    fn vec_znx_big_sub_small_a_default<R, C>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE, impl ArithmeticState>,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
        for<'x> BE::BufRef<'x>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
        C: VecZnxBigToBackendRef<BE>,
    {
        fft64_vec_znx_big_sub_small_a::<_, _, _, BE>(res, res_col, &a, a_col, b, b_col);
    }

    fn vec_znx_big_sub_small_assign_default<R>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE, impl ArithmeticState>,
        a_col: usize,
    ) where
        BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxSubAssign,
        for<'x> BE::BufRef<'x>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
    {
        fft64_vec_znx_big_sub_small_a_assign::<_, _, BE>(res, res_col, &a, a_col);
    }

    fn vec_znx_big_sub_small_b_default<R, A>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE, impl ArithmeticState>,
        b_col: usize,
    ) where
        BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
        for<'x> BE::BufRef<'x>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        fft64_vec_znx_big_sub_small_b::<_, _, _, BE>(res, res_col, a, a_col, &b, b_col);
    }

    fn vec_znx_big_sub_small_negate_assign_default<R>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE, impl ArithmeticState>,
        a_col: usize,
    ) where
        BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxSubNegateAssign + ZnxNegateAssign,
        for<'x> BE::BufRef<'x>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
    {
        fft64_vec_znx_big_sub_small_b_assign::<_, _, BE>(res, res_col, &a, a_col);
    }

    fn vec_znx_big_inner_sum_default<R, A>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        res_coeff: usize,
        a: &A,
        a_col: usize,
    ) where
        BE: Backend<BigWord = i64, ZnxWord = i64>,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        vec_znx_big_inner_sum_default_impl::<R, A, BE>(res, res_col, res_coeff, a, a_col);
    }

    fn vec_znx_scalar_product_default<R>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE, impl ArithmeticState>,
        a_col: usize,
        b: &ScalarZnxBackendRef<'_, BE>,
        b_col: usize,
    ) where
        BE: Backend<BigWord = i64, ZnxWord = i64>,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
        R: VecZnxBigToBackendMut<BE>,
    {
        vec_znx_scalar_product_default_impl::<R, BE>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_big_col_weighted_sum_default<R>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE, impl ArithmeticState>,
        weights: &ScalarZnxBackendRef<'_, BE>,
        weights_col: usize,
        cols: usize,
        coeffs: usize,
    ) where
        BE: Backend<BigWord = i64, ZnxWord = i64>,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
        R: VecZnxBigToBackendMut<BE>,
    {
        vec_znx_big_col_weighted_sum_default_impl::<R, BE>(res, res_col, a, weights, weights_col, cols, coeffs);
    }

    fn vec_znx_big_negate_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxNegate + ZnxZero,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        fft64_vec_znx_big_negate::<_, _, BE>(res, res_col, a, a_col);
    }

    fn vec_znx_big_negate_assign_default<R>(_module: &Module<BE>, res: &mut R, res_col: usize)
    where
        BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxNegateAssign,
        R: VecZnxBigToBackendMut<BE>,
    {
        fft64_vec_znx_big_negate_assign::<_, BE>(res, res_col);
    }

    fn vec_znx_big_normalize_tmp_bytes_default(module: &Module<BE>) -> usize
    where
        BE: Backend<BigWord = i64, ZnxWord = i64>,
    {
        fft64_vec_znx_big_normalize_tmp_bytes(module.n())
    }

    fn vec_znx_big_normalize_default<R, A>(
        module: &Module<BE>,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend<BigWord = i64, ZnxWord = i64>
            + ZnxZero
            + ZnxCopy
            + ZnxAddAssign
            + ZnxMulPowerOfTwoAssign
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly
            + ZnxNormalizeMiddleStep
            + ZnxNormalizeFinalStep
            + ZnxNormalizeFirstStep
            + ZnxExtractDigitAddMul
            + ZnxNormalizeDigit
            + ZnxNormalizeMiddleStepAssign
            + ZnxNormalizeFinalStepAssign,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
        R: VecZnxToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(
            scratch.borrow(),
            fft64_vec_znx_big_normalize_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        fft64_vec_znx_big_normalize::<_, _, BE>(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry);
    }

    fn vec_znx_big_automorphism_default<R, A>(_module: &Module<BE>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxAutomorphism + ZnxZero,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        fft64_vec_znx_big_automorphism::<_, _, BE>(k, res, res_col, a, a_col);
    }

    fn vec_znx_big_automorphism_assign_tmp_bytes_default(module: &Module<BE>) -> usize
    where
        BE: Backend<BigWord = i64, ZnxWord = i64>,
    {
        fft64_vec_znx_big_automorphism_assign_tmp_bytes(module.n())
    }

    fn vec_znx_big_automorphism_assign_default<R>(
        module: &Module<BE>,
        k: i64,
        res: &mut R,
        res_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxAutomorphism + ZnxCopy,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
        R: VecZnxBigToBackendMut<BE>,
    {
        let (tmp, _) = take_host_typed::<BE, i64>(
            scratch.borrow(),
            fft64_vec_znx_big_automorphism_assign_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        fft64_vec_znx_big_automorphism_assign::<_, BE>(k, res, res_col, tmp);
    }
}

impl<BE: Backend<ZnxWord = i64>> FFT64VecZnxBigDefault<BE> for BE
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
}

#[doc(hidden)]
pub trait NTT4x30VecZnxBigDefault<BE: Backend<ZnxWord = i64>>: Backend
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    fn vec_znx_big_from_small_default<R>(
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE, impl ArithmeticState>,
        a_col: usize,
    ) where
        BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        for<'x> BE::BufRef<'x>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
    {
        let a = vec_znx_backend_ref_as_host_ref::<BE, _>(a);
        ntt4x30_vec_znx_big_from_small::<_, _, BE>(res, res_col, &a, a_col);
    }

    fn vec_znx_big_add_normal_default<R>(
        _module: &Module<BE>,
        res_base2k: usize,
        res: &mut R,
        res_col: usize,
        noise_infos: NoiseInfos,
        source: &mut Source,
    ) where
        BE: Backend<BigWord = i128, ZnxWord = i64>,
        R: VecZnxBigToBackendMut<BE>,
    {
        ntt4x30_vec_znx_big_add_normal_ref::<_, BE>(res_base2k, res, res_col, noise_infos, source);
    }

    fn vec_znx_big_add_normal_seed_default<R>(
        module: &Module<BE>,
        res_base2k: usize,
        res: &mut R,
        res_col: usize,
        noise_infos: NoiseInfos,
        seed: [u8; 32],
    ) where
        BE: Backend<BigWord = i128, ZnxWord = i64>,
        R: VecZnxBigToBackendMut<BE>,
    {
        let mut source = Source::new(seed);
        Self::vec_znx_big_add_normal_default(module, res_base2k, res, res_col, noise_infos, &mut source);
    }

    fn vec_znx_big_add_into_default<R, A, C>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        for<'x> BE::BufRef<'x>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
        C: VecZnxBigToBackendRef<BE>,
    {
        ntt4x30_vec_znx_big_add_into::<_, _, _, BE>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_big_add_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        ntt4x30_vec_znx_big_add_assign::<_, _, BE>(res, res_col, a, a_col);
    }

    fn vec_znx_big_add_small_into_default<R, A>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE, impl ArithmeticState>,
        b_col: usize,
    ) where
        BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        for<'x> BE::BufRef<'x>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        let b = vec_znx_backend_ref_as_host_ref::<BE, _>(b);
        ntt4x30_vec_znx_big_add_small_into::<_, _, _, BE>(res, res_col, a, a_col, &b, b_col);
    }

    fn vec_znx_big_add_small_assign_default<R>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE, impl ArithmeticState>,
        a_col: usize,
    ) where
        BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        for<'x> BE::BufRef<'x>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
    {
        let a = vec_znx_backend_ref_as_host_ref::<BE, _>(a);
        ntt4x30_vec_znx_big_add_small_assign::<_, _, BE>(res, res_col, &a, a_col);
    }

    fn vec_znx_big_sub_default<R, A, C>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
        C: VecZnxBigToBackendRef<BE>,
    {
        ntt4x30_vec_znx_big_sub::<_, _, _, BE>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_big_sub_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        ntt4x30_vec_znx_big_sub_assign::<_, _, BE>(res, res_col, a, a_col);
    }

    fn vec_znx_big_sub_negate_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        ntt4x30_vec_znx_big_sub_negate_assign::<_, _, BE>(res, res_col, a, a_col);
    }

    fn vec_znx_big_sub_small_a_default<R, C>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE, impl ArithmeticState>,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        for<'x> BE::BufRef<'x>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
        C: VecZnxBigToBackendRef<BE>,
    {
        let a = vec_znx_backend_ref_as_host_ref::<BE, _>(a);
        ntt4x30_vec_znx_big_sub_small_a::<_, _, _, BE>(res, res_col, &a, a_col, b, b_col);
    }

    fn vec_znx_big_sub_small_assign_default<R>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE, impl ArithmeticState>,
        a_col: usize,
    ) where
        BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        for<'x> BE::BufRef<'x>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
    {
        let a = vec_znx_backend_ref_as_host_ref::<BE, _>(a);
        ntt4x30_vec_znx_big_sub_small_assign::<_, _, BE>(res, res_col, &a, a_col);
    }

    fn vec_znx_big_sub_small_b_default<R, A>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE, impl ArithmeticState>,
        b_col: usize,
    ) where
        BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        for<'x> BE::BufRef<'x>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        let b = vec_znx_backend_ref_as_host_ref::<BE, _>(b);
        ntt4x30_vec_znx_big_sub_small_b::<_, _, _, BE>(res, res_col, a, a_col, &b, b_col);
    }

    fn vec_znx_big_sub_small_negate_assign_default<R>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE, impl ArithmeticState>,
        a_col: usize,
    ) where
        BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        for<'x> BE::BufRef<'x>: AsRef<[u8]>,
        R: VecZnxBigToBackendMut<BE>,
    {
        let a = vec_znx_backend_ref_as_host_ref::<BE, _>(a);
        ntt4x30_vec_znx_big_sub_small_negate_assign::<_, _, BE>(res, res_col, &a, a_col);
    }

    fn vec_znx_big_inner_sum_default<R, A>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        res_coeff: usize,
        a: &A,
        a_col: usize,
    ) where
        BE: Backend<BigWord = i128, ZnxWord = i64>,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        vec_znx_big_inner_sum_default_impl::<R, A, BE>(res, res_col, res_coeff, a, a_col);
    }

    fn vec_znx_scalar_product_default<R>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE, impl ArithmeticState>,
        a_col: usize,
        b: &ScalarZnxBackendRef<'_, BE>,
        b_col: usize,
    ) where
        BE: Backend<BigWord = i128, ZnxWord = i64>,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
        R: VecZnxBigToBackendMut<BE>,
    {
        vec_znx_scalar_product_default_impl::<R, BE>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_big_col_weighted_sum_default<R>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE, impl ArithmeticState>,
        weights: &ScalarZnxBackendRef<'_, BE>,
        weights_col: usize,
        cols: usize,
        coeffs: usize,
    ) where
        BE: Backend<BigWord = i128, ZnxWord = i64>,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
        R: VecZnxBigToBackendMut<BE>,
    {
        vec_znx_big_col_weighted_sum_default_impl::<R, BE>(res, res_col, a, weights, weights_col, cols, coeffs);
    }

    fn vec_znx_big_negate_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        ntt4x30_vec_znx_big_negate::<_, _, BE>(res, res_col, a, a_col);
    }

    fn vec_znx_big_negate_assign_default<R>(_module: &Module<BE>, res: &mut R, res_col: usize)
    where
        BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        R: VecZnxBigToBackendMut<BE>,
    {
        ntt4x30_vec_znx_big_negate_assign::<_, BE>(res, res_col);
    }

    fn vec_znx_big_normalize_tmp_bytes_default(module: &Module<BE>) -> usize
    where
        BE: Backend<BigWord = i128, ZnxWord = i64> + I128NormalizeOps,
    {
        ntt4x30_vec_znx_big_normalize_tmp_bytes(module.n())
    }

    fn vec_znx_big_normalize_default<R, A>(
        module: &Module<BE>,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend<BigWord = i128, ZnxWord = i64> + I128NormalizeOps,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
        R: VecZnxToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        let (carry, _) = take_host_typed::<BE, i128>(
            scratch.borrow(),
            ntt4x30_vec_znx_big_normalize_tmp_bytes(module.n()) / size_of::<i128>(),
        );
        ntt4x30_vec_znx_big_normalize::<_, _, BE>(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry);
    }

    fn vec_znx_big_normalize_add_assign_default<R, A>(
        module: &Module<BE>,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend<BigWord = i128, ZnxWord = i64> + I128NormalizeOps,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
        R: VecZnxToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        let (carry, _) = take_host_typed::<BE, i128>(
            scratch.borrow(),
            ntt4x30_vec_znx_big_normalize_tmp_bytes(module.n()) / size_of::<i128>(),
        );
        ntt4x30_vec_znx_big_normalize_add_assign::<_, _, BE>(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry);
    }

    fn vec_znx_big_normalize_sub_assign_default<R, A>(
        module: &Module<BE>,
        res: &mut R,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &A,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend<BigWord = i128, ZnxWord = i64> + I128NormalizeOps,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
        R: VecZnxToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        let (carry, _) = take_host_typed::<BE, i128>(
            scratch.borrow(),
            ntt4x30_vec_znx_big_normalize_tmp_bytes(module.n()) / size_of::<i128>(),
        );
        ntt4x30_vec_znx_big_normalize_sub_assign::<_, _, BE>(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry);
    }

    fn vec_znx_big_automorphism_default<R, A>(_module: &Module<BE>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        R: VecZnxBigToBackendMut<BE>,
        A: VecZnxBigToBackendRef<BE>,
    {
        ntt4x30_vec_znx_big_automorphism::<_, _, BE>(k, res, res_col, a, a_col);
    }

    fn vec_znx_big_automorphism_assign_tmp_bytes_default(module: &Module<BE>) -> usize
    where
        BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
    {
        ntt4x30_vec_znx_big_automorphism_assign_tmp_bytes(module.n())
    }

    fn vec_znx_big_automorphism_assign_default<R>(
        module: &Module<BE>,
        k: i64,
        res: &mut R,
        res_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
        R: VecZnxBigToBackendMut<BE>,
    {
        let (tmp, _) = take_host_typed::<BE, i128>(
            scratch.borrow(),
            ntt4x30_vec_znx_big_automorphism_assign_tmp_bytes(module.n()) / size_of::<i128>(),
        );
        ntt4x30_vec_znx_big_automorphism_assign::<_, BE>(k, res, res_col, tmp);
    }
}

impl<BE: Backend<ZnxWord = i64>> NTT4x30VecZnxBigDefault<BE> for BE
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
}
