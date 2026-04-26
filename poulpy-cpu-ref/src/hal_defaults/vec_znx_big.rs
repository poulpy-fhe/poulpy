//! Backend extension points for extended-precision [`VecZnxBig`](poulpy_hal::layouts::VecZnxBig) operations.

#![allow(clippy::too_many_arguments)]

use std::mem::size_of;

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
    ntt120::vec_znx_big::{
        I128BigOps, I128NormalizeOps, ntt120_vec_znx_big_add_assign, ntt120_vec_znx_big_add_into,
        ntt120_vec_znx_big_add_normal_ref, ntt120_vec_znx_big_add_small_assign, ntt120_vec_znx_big_add_small_into,
        ntt120_vec_znx_big_automorphism, ntt120_vec_znx_big_automorphism_assign,
        ntt120_vec_znx_big_automorphism_assign_tmp_bytes, ntt120_vec_znx_big_from_small, ntt120_vec_znx_big_negate,
        ntt120_vec_znx_big_negate_assign, ntt120_vec_znx_big_normalize, ntt120_vec_znx_big_normalize_add_assign,
        ntt120_vec_znx_big_normalize_sub_assign, ntt120_vec_znx_big_normalize_tmp_bytes, ntt120_vec_znx_big_sub,
        ntt120_vec_znx_big_sub_assign, ntt120_vec_znx_big_sub_negate_assign, ntt120_vec_znx_big_sub_small_a,
        ntt120_vec_znx_big_sub_small_assign, ntt120_vec_znx_big_sub_small_b, ntt120_vec_znx_big_sub_small_negate_assign,
    },
    znx::{
        ZnxAdd, ZnxAddAssign, ZnxAutomorphism, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulPowerOfTwoAssign, ZnxNegate,
        ZnxNegateAssign, ZnxNormalizeDigit, ZnxNormalizeFinalStep, ZnxNormalizeFinalStepAssign, ZnxNormalizeFirstStep,
        ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepAssign, ZnxNormalizeMiddleStepCarryOnly,
        ZnxSub, ZnxSubAssign, ZnxSubNegateAssign, ZnxZero, znx_copy_ref, znx_zero_ref,
    },
};
use poulpy_hal::{
    api::TakeSlice,
    layouts::{
        Backend, Module, NoiseInfos, Scratch, VecZnx, VecZnxBig, VecZnxBigToMut, VecZnxBigToRef, VecZnxToMut, VecZnxToRef,
        ZnxInfos, ZnxView, ZnxViewMut,
    },
    source::Source,
};

#[doc(hidden)]
pub trait FFT64VecZnxBigDefaults<BE: Backend>: Backend {
    fn vec_znx_big_from_small_default<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i64>,
        R: VecZnxBigToMut<BE>,
        A: VecZnxToRef,
    {
        let mut res: VecZnxBig<&mut [u8], BE> = res.to_mut();
        let a: VecZnx<&[u8]> = a.to_ref();

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
        BE: Backend<ScalarBig = i64>,
        R: VecZnxBigToMut<BE>,
    {
        fft64_vec_znx_big_add_normal_ref(res_base2k, res, res_col, noise_infos, source);
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
        BE: Backend<ScalarBig = i64> + ZnxAdd + ZnxCopy + ZnxZero,
        R: VecZnxBigToMut<BE>,
        A: VecZnxBigToRef<BE>,
        C: VecZnxBigToRef<BE>,
    {
        fft64_vec_znx_big_add_into(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_big_add_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i64> + ZnxAddAssign,
        R: VecZnxBigToMut<BE>,
        A: VecZnxBigToRef<BE>,
    {
        fft64_vec_znx_big_add_assign(res, res_col, a, a_col);
    }

    fn vec_znx_big_add_small_into_default<R, A, C>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        BE: Backend<ScalarBig = i64> + ZnxAdd + ZnxCopy + ZnxZero,
        R: VecZnxBigToMut<BE>,
        A: VecZnxBigToRef<BE>,
        C: VecZnxToRef,
    {
        fft64_vec_znx_big_add_small_into(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_big_add_small_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i64> + ZnxAddAssign,
        R: VecZnxBigToMut<BE>,
        A: VecZnxToRef,
    {
        fft64_vec_znx_big_add_small_assign(res, res_col, a, a_col);
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
        BE: Backend<ScalarBig = i64> + ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
        R: VecZnxBigToMut<BE>,
        A: VecZnxBigToRef<BE>,
        C: VecZnxBigToRef<BE>,
    {
        fft64_vec_znx_big_sub(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_big_sub_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i64> + ZnxSubAssign,
        R: VecZnxBigToMut<BE>,
        A: VecZnxBigToRef<BE>,
    {
        fft64_vec_znx_big_sub_assign(res, res_col, a, a_col);
    }

    fn vec_znx_big_sub_negate_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i64> + ZnxSubNegateAssign + ZnxNegateAssign,
        R: VecZnxBigToMut<BE>,
        A: VecZnxBigToRef<BE>,
    {
        fft64_vec_znx_big_sub_negate_assign(res, res_col, a, a_col);
    }

    fn vec_znx_big_sub_small_a_default<R, A, C>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        BE: Backend<ScalarBig = i64> + ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
        R: VecZnxBigToMut<BE>,
        A: VecZnxToRef,
        C: VecZnxBigToRef<BE>,
    {
        fft64_vec_znx_big_sub_small_a(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_big_sub_small_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i64> + ZnxSubAssign,
        R: VecZnxBigToMut<BE>,
        A: VecZnxToRef,
    {
        fft64_vec_znx_big_sub_small_a_assign(res, res_col, a, a_col);
    }

    fn vec_znx_big_sub_small_b_default<R, A, C>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        BE: Backend<ScalarBig = i64> + ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
        R: VecZnxBigToMut<BE>,
        A: VecZnxBigToRef<BE>,
        C: VecZnxToRef,
    {
        fft64_vec_znx_big_sub_small_b(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_big_sub_small_negate_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i64> + ZnxSubNegateAssign + ZnxNegateAssign,
        R: VecZnxBigToMut<BE>,
        A: VecZnxToRef,
    {
        fft64_vec_znx_big_sub_small_b_assign(res, res_col, a, a_col);
    }

    fn vec_znx_big_negate_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i64> + ZnxNegate + ZnxZero,
        R: VecZnxBigToMut<BE>,
        A: VecZnxBigToRef<BE>,
    {
        fft64_vec_znx_big_negate(res, res_col, a, a_col);
    }

    fn vec_znx_big_negate_assign_default<R>(_module: &Module<BE>, res: &mut R, res_col: usize)
    where
        BE: Backend<ScalarBig = i64> + ZnxNegateAssign,
        R: VecZnxBigToMut<BE>,
    {
        fft64_vec_znx_big_negate_assign(res, res_col);
    }

    fn vec_znx_big_normalize_tmp_bytes_default(module: &Module<BE>) -> usize
    where
        BE: Backend<ScalarBig = i64>,
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
        scratch: &mut Scratch<BE>,
    ) where
        BE: Backend<ScalarBig = i64>
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
        Scratch<BE>: TakeSlice,
        R: VecZnxToMut,
        A: VecZnxBigToRef<BE>,
    {
        let (carry, _) = scratch.take_slice(fft64_vec_znx_big_normalize_tmp_bytes(module.n()) / size_of::<i64>());
        fft64_vec_znx_big_normalize(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry);
    }

    fn vec_znx_big_automorphism_default<R, A>(_module: &Module<BE>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i64> + ZnxAutomorphism + ZnxZero,
        R: VecZnxBigToMut<BE>,
        A: VecZnxBigToRef<BE>,
    {
        fft64_vec_znx_big_automorphism(k, res, res_col, a, a_col);
    }

    fn vec_znx_big_automorphism_assign_tmp_bytes_default(module: &Module<BE>) -> usize
    where
        BE: Backend<ScalarBig = i64>,
    {
        fft64_vec_znx_big_automorphism_assign_tmp_bytes(module.n())
    }

    fn vec_znx_big_automorphism_assign_default<R>(
        module: &Module<BE>,
        k: i64,
        res: &mut R,
        res_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        BE: Backend<ScalarBig = i64> + ZnxAutomorphism + ZnxCopy,
        Scratch<BE>: TakeSlice,
        R: VecZnxBigToMut<BE>,
    {
        let (tmp, _) = scratch.take_slice(fft64_vec_znx_big_automorphism_assign_tmp_bytes(module.n()) / size_of::<i64>());
        fft64_vec_znx_big_automorphism_assign(k, res, res_col, tmp);
    }
}

impl<BE: Backend> FFT64VecZnxBigDefaults<BE> for BE {}

#[doc(hidden)]
pub trait NTT120VecZnxBigDefaults<BE: Backend>: Backend {
    fn vec_znx_big_from_small_default<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        R: VecZnxBigToMut<BE>,
        A: VecZnxToRef,
    {
        ntt120_vec_znx_big_from_small(res, res_col, a, a_col);
    }

    fn vec_znx_big_add_normal_default<R>(
        _module: &Module<BE>,
        res_base2k: usize,
        res: &mut R,
        res_col: usize,
        noise_infos: NoiseInfos,
        source: &mut Source,
    ) where
        BE: Backend<ScalarBig = i128>,
        R: VecZnxBigToMut<BE>,
    {
        ntt120_vec_znx_big_add_normal_ref(res_base2k, res, res_col, noise_infos, source);
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
        BE: Backend<ScalarBig = i128> + I128BigOps,
        R: VecZnxBigToMut<BE>,
        A: VecZnxBigToRef<BE>,
        C: VecZnxBigToRef<BE>,
    {
        ntt120_vec_znx_big_add_into(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_big_add_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        R: VecZnxBigToMut<BE>,
        A: VecZnxBigToRef<BE>,
    {
        ntt120_vec_znx_big_add_assign(res, res_col, a, a_col);
    }

    fn vec_znx_big_add_small_into_default<R, A, C>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        R: VecZnxBigToMut<BE>,
        A: VecZnxBigToRef<BE>,
        C: VecZnxToRef,
    {
        ntt120_vec_znx_big_add_small_into(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_big_add_small_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        R: VecZnxBigToMut<BE>,
        A: VecZnxToRef,
    {
        ntt120_vec_znx_big_add_small_assign(res, res_col, a, a_col);
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
        BE: Backend<ScalarBig = i128> + I128BigOps,
        R: VecZnxBigToMut<BE>,
        A: VecZnxBigToRef<BE>,
        C: VecZnxBigToRef<BE>,
    {
        ntt120_vec_znx_big_sub(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_big_sub_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        R: VecZnxBigToMut<BE>,
        A: VecZnxBigToRef<BE>,
    {
        ntt120_vec_znx_big_sub_assign(res, res_col, a, a_col);
    }

    fn vec_znx_big_sub_negate_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        R: VecZnxBigToMut<BE>,
        A: VecZnxBigToRef<BE>,
    {
        ntt120_vec_znx_big_sub_negate_assign(res, res_col, a, a_col);
    }

    fn vec_znx_big_sub_small_a_default<R, A, C>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        R: VecZnxBigToMut<BE>,
        A: VecZnxToRef,
        C: VecZnxBigToRef<BE>,
    {
        ntt120_vec_znx_big_sub_small_a(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_big_sub_small_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        R: VecZnxBigToMut<BE>,
        A: VecZnxToRef,
    {
        ntt120_vec_znx_big_sub_small_assign(res, res_col, a, a_col);
    }

    fn vec_znx_big_sub_small_b_default<R, A, C>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &C,
        b_col: usize,
    ) where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        R: VecZnxBigToMut<BE>,
        A: VecZnxBigToRef<BE>,
        C: VecZnxToRef,
    {
        ntt120_vec_znx_big_sub_small_b(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_big_sub_small_negate_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        R: VecZnxBigToMut<BE>,
        A: VecZnxToRef,
    {
        ntt120_vec_znx_big_sub_small_negate_assign(res, res_col, a, a_col);
    }

    fn vec_znx_big_negate_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        R: VecZnxBigToMut<BE>,
        A: VecZnxBigToRef<BE>,
    {
        ntt120_vec_znx_big_negate(res, res_col, a, a_col);
    }

    fn vec_znx_big_negate_assign_default<R>(_module: &Module<BE>, res: &mut R, res_col: usize)
    where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        R: VecZnxBigToMut<BE>,
    {
        ntt120_vec_znx_big_negate_assign(res, res_col);
    }

    fn vec_znx_big_normalize_tmp_bytes_default(module: &Module<BE>) -> usize
    where
        BE: Backend<ScalarBig = i128> + I128NormalizeOps,
    {
        ntt120_vec_znx_big_normalize_tmp_bytes(module.n())
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
        scratch: &mut Scratch<BE>,
    ) where
        BE: Backend<ScalarBig = i128> + I128NormalizeOps,
        Scratch<BE>: TakeSlice,
        R: VecZnxToMut,
        A: VecZnxBigToRef<BE>,
    {
        let (carry, _) = scratch.take_slice(ntt120_vec_znx_big_normalize_tmp_bytes(module.n()) / size_of::<i128>());
        ntt120_vec_znx_big_normalize(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry);
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
        scratch: &mut Scratch<BE>,
    ) where
        BE: Backend<ScalarBig = i128> + I128NormalizeOps,
        Scratch<BE>: TakeSlice,
        R: VecZnxToMut,
        A: VecZnxBigToRef<BE>,
    {
        let (carry, _) = scratch.take_slice(ntt120_vec_znx_big_normalize_tmp_bytes(module.n()) / size_of::<i128>());
        ntt120_vec_znx_big_normalize_add_assign(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry);
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
        scratch: &mut Scratch<BE>,
    ) where
        BE: Backend<ScalarBig = i128> + I128NormalizeOps,
        Scratch<BE>: TakeSlice,
        R: VecZnxToMut,
        A: VecZnxBigToRef<BE>,
    {
        let (carry, _) = scratch.take_slice(ntt120_vec_znx_big_normalize_tmp_bytes(module.n()) / size_of::<i128>());
        ntt120_vec_znx_big_normalize_sub_assign(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry);
    }

    fn vec_znx_big_automorphism_default<R, A>(_module: &Module<BE>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        R: VecZnxBigToMut<BE>,
        A: VecZnxBigToRef<BE>,
    {
        ntt120_vec_znx_big_automorphism(k, res, res_col, a, a_col);
    }

    fn vec_znx_big_automorphism_assign_tmp_bytes_default(module: &Module<BE>) -> usize
    where
        BE: Backend<ScalarBig = i128> + I128BigOps,
    {
        ntt120_vec_znx_big_automorphism_assign_tmp_bytes(module.n())
    }

    fn vec_znx_big_automorphism_assign_default<R>(
        module: &Module<BE>,
        k: i64,
        res: &mut R,
        res_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        BE: Backend<ScalarBig = i128> + I128BigOps,
        Scratch<BE>: TakeSlice,
        R: VecZnxBigToMut<BE>,
    {
        let (tmp, _) = scratch.take_slice(ntt120_vec_znx_big_automorphism_assign_tmp_bytes(module.n()) / size_of::<i128>());
        ntt120_vec_znx_big_automorphism_assign(k, res, res_col, tmp);
    }
}

impl<BE: Backend> NTT120VecZnxBigDefaults<BE> for BE {}
