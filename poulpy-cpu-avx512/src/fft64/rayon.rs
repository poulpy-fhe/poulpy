//! Rayon-scheduled wrapper for the AVX-512 FFT64 backend.

use rayon::prelude::*;

use poulpy_cpu_ref::{
    hal_defaults::{BigWordHadamardProduct, FFT64ConvolutionDefault, FFT64ModuleDefault, FFT64SvpDefault, HalVecZnxDefault},
    reference::{
        fft64::{
            convolution::I64Ops,
            module::FFTModuleHandle,
            reim::{ReimArith, ReimFFTExecute, ReimFFTTable, ReimIFFTTable},
            reim4::{Reim4BlkMatVec, Reim4Convolution},
        },
        znx::{
            ZnxAdd, ZnxAddAssign, ZnxAutomorphism, ZnxAutomorphismRotate, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulAddPowerOfTwo,
            ZnxMulPowerOfTwo, ZnxMulPowerOfTwoAssign, ZnxNegate, ZnxNegateAssign, ZnxNormalizeDigit, ZnxNormalizeFinalStep,
            ZnxNormalizeFinalStepAssign, ZnxNormalizeFinalStepSub, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepAssign,
            ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepAssign,
            ZnxNormalizeMiddleStepCarryOnly, ZnxNormalizeMiddleStepSub, ZnxRotate, ZnxSub, ZnxSubAssign, ZnxSubNegateAssign,
            ZnxSwitchRing, ZnxZero,
        },
    },
};
use poulpy_hal::{
    api::{ScratchArenaTakeBasic, VecZnxDftApply, VecZnxDftZero, VmpApplyDftToDft},
    layouts::{
        DataView, DataViewMut, MatZnxBackendRef, Module, NoiseInfos, ScalarZnxBackendRef, ScratchArena, VecZnxBackendMut,
        VecZnxBackendRef, VecZnxBig, VecZnxBigBackendMut, VecZnxBigBackendRef, VecZnxDft, VecZnxDftBackendMut,
        VecZnxDftBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef, VmpPMat, VmpPMatBackendMut, VmpPMatBackendRef,
        ZnxView, ZnxViewMut,
    },
    oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
};

use super::FFT64Avx512Rayon;
use crate::{FFT64Avx512, execution::RayonTaskExecutor};

poulpy_hal::impl_backend_from!(FFT64Avx512Rayon, FFT64Avx512, RayonTaskExecutor);

fn base_module(module: &Module<FFT64Avx512Rayon>) -> &Module<FFT64Avx512> {
    module.reinterpret()
}

fn base_dft_ref<'a>(a: &'a VecZnxDftBackendRef<'_, FFT64Avx512Rayon>) -> VecZnxDftBackendRef<'a, FFT64Avx512> {
    VecZnxDft::from_data(&**a.data(), a.n(), a.cols(), a.size())
}

fn base_dft_mut<'a>(a: &'a mut VecZnxDftBackendMut<'_, FFT64Avx512Rayon>) -> VecZnxDftBackendMut<'a, FFT64Avx512> {
    let (n, cols, size) = (a.n(), a.cols(), a.size());
    VecZnxDft::from_data(&mut **a.data_mut(), n, cols, size)
}

fn base_big_mut<'a>(a: &'a mut VecZnxBigBackendMut<'_, FFT64Avx512Rayon>) -> VecZnxBigBackendMut<'a, FFT64Avx512> {
    let (n, cols, size) = (a.n(), a.cols(), a.size());
    VecZnxBig::from_data(&mut **a.data_mut(), n, cols, size)
}

fn base_big_ref<'a>(a: &'a VecZnxBigBackendRef<'_, FFT64Avx512Rayon>) -> VecZnxBigBackendRef<'a, FFT64Avx512> {
    VecZnxBig::from_data(&**a.data(), a.n(), a.cols(), a.size())
}

fn base_vmp_ref<'a>(a: &'a VmpPMatBackendRef<'_, FFT64Avx512Rayon>) -> VmpPMatBackendRef<'a, FFT64Avx512> {
    VmpPMat::from_data(&**a.data(), a.n(), a.rows(), a.cols_in(), a.cols_out(), a.size())
}

fn base_vmp_mut<'a>(a: &'a mut VmpPMatBackendMut<'_, FFT64Avx512Rayon>) -> VmpPMatBackendMut<'a, FFT64Avx512> {
    let (n, rows, cols_in, cols_out, size) = (a.n(), a.rows(), a.cols_in(), a.cols_out(), a.size());
    VmpPMat::from_data(&mut **a.data_mut(), n, rows, cols_in, cols_out, size)
}

fn parallel_chunk_len(len: usize) -> Option<usize> {
    if len < 1 << 15 || rayon::current_num_threads() < 2 {
        None
    } else {
        Some(len.div_ceil(rayon::current_num_threads()).next_multiple_of(64))
    }
}

#[inline]
fn parallel_limb_tasks(count: usize) -> bool {
    count > 1 && rayon::current_num_threads() > 1
}

macro_rules! parallel_binary {
    ($trait:ident, $method:ident) => {
        impl $trait for FFT64Avx512Rayon {
            #[inline(always)]
            fn $method(res: &mut [i64], a: &[i64], b: &[i64]) {
                let Some(chunk) = parallel_chunk_len(res.len()) else {
                    return <FFT64Avx512 as $trait>::$method(res, a, b);
                };
                res.par_chunks_mut(chunk)
                    .zip(a.par_chunks(chunk))
                    .zip(b.par_chunks(chunk))
                    .for_each(|((res, a), b)| <FFT64Avx512 as $trait>::$method(res, a, b));
            }
        }
    };
}

macro_rules! parallel_assign {
    ($trait:ident, $method:ident) => {
        impl $trait for FFT64Avx512Rayon {
            #[inline(always)]
            fn $method(res: &mut [i64], a: &[i64]) {
                let Some(chunk) = parallel_chunk_len(res.len()) else {
                    return <FFT64Avx512 as $trait>::$method(res, a);
                };
                res.par_chunks_mut(chunk)
                    .zip(a.par_chunks(chunk))
                    .for_each(|(res, a)| <FFT64Avx512 as $trait>::$method(res, a));
            }
        }
    };
}

macro_rules! parallel_unary {
    ($trait:ident, $method:ident) => {
        impl $trait for FFT64Avx512Rayon {
            #[inline(always)]
            fn $method(res: &mut [i64]) {
                let Some(chunk) = parallel_chunk_len(res.len()) else {
                    return <FFT64Avx512 as $trait>::$method(res);
                };
                res.par_chunks_mut(chunk)
                    .for_each(|res| <FFT64Avx512 as $trait>::$method(res));
            }
        }
    };
}

macro_rules! parallel_shift {
    ($trait:ident, $method:ident) => {
        impl $trait for FFT64Avx512Rayon {
            #[inline(always)]
            fn $method(k: i64, res: &mut [i64], a: &[i64]) {
                let Some(chunk) = parallel_chunk_len(res.len()) else {
                    return <FFT64Avx512 as $trait>::$method(k, res, a);
                };
                res.par_chunks_mut(chunk)
                    .zip(a.par_chunks(chunk))
                    .for_each(|(res, a)| <FFT64Avx512 as $trait>::$method(k, res, a));
            }
        }
    };
}

macro_rules! forward_znx {
    ($trait:ident, $method:ident($($arg:ident: $ty:ty),* $(,)?)) => {
        impl $trait for FFT64Avx512Rayon {
            #[inline(always)]
            fn $method($($arg: $ty),*) {
                <FFT64Avx512 as $trait>::$method($($arg),*)
            }
        }
    };
}

macro_rules! forward_znx_const {
    ($trait:ident, $method:ident($($arg:ident: $ty:ty),* $(,)?)) => {
        impl $trait for FFT64Avx512Rayon {
            #[inline(always)]
            fn $method<const OVERWRITE: bool>($($arg: $ty),*) {
                <FFT64Avx512 as $trait>::$method::<OVERWRITE>($($arg),*)
            }
        }
    };
}

parallel_binary!(ZnxAdd, znx_add);
parallel_assign!(ZnxAddAssign, znx_add_assign);
parallel_binary!(ZnxSub, znx_sub);
parallel_assign!(ZnxSubAssign, znx_sub_assign);
parallel_assign!(ZnxSubNegateAssign, znx_sub_negate_assign);
parallel_shift!(ZnxMulAddPowerOfTwo, znx_muladd_power_of_two);
parallel_shift!(ZnxMulPowerOfTwo, znx_mul_power_of_two);
impl ZnxMulPowerOfTwoAssign for FFT64Avx512Rayon {
    #[inline(always)]
    fn znx_mul_power_of_two_assign(k: i64, res: &mut [i64]) {
        let Some(chunk) = parallel_chunk_len(res.len()) else {
            return <FFT64Avx512 as ZnxMulPowerOfTwoAssign>::znx_mul_power_of_two_assign(k, res);
        };
        res.par_chunks_mut(chunk)
            .for_each(|res| <FFT64Avx512 as ZnxMulPowerOfTwoAssign>::znx_mul_power_of_two_assign(k, res));
    }
}
forward_znx!(ZnxAutomorphism, znx_automorphism(p: i64, res: &mut [i64], a: &[i64]));
forward_znx!(ZnxAutomorphismRotate, znx_automorphism_rotate(p: i64, k: i64, res: &mut [i64], a: &[i64]));
parallel_assign!(ZnxCopy, znx_copy);
parallel_assign!(ZnxNegate, znx_negate);
parallel_unary!(ZnxNegateAssign, znx_negate_assign);
forward_znx!(ZnxRotate, znx_rotate(p: i64, res: &mut [i64], src: &[i64]));
parallel_unary!(ZnxZero, znx_zero);
forward_znx!(ZnxSwitchRing, znx_switch_ring(res: &mut [i64], a: &[i64]));
forward_znx_const!(ZnxNormalizeFirstStep, znx_normalize_first_step(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]));
forward_znx_const!(ZnxNormalizeMiddleStep, znx_normalize_middle_step(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]));
forward_znx_const!(ZnxNormalizeFinalStep, znx_normalize_final_step(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]));
forward_znx!(ZnxNormalizeFirstStepCarryOnly, znx_normalize_first_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]));
forward_znx!(ZnxNormalizeFirstStepAssign, znx_normalize_first_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]));
forward_znx!(ZnxNormalizeMiddleStepCarryOnly, znx_normalize_middle_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]));
forward_znx!(ZnxNormalizeMiddleStepAssign, znx_normalize_middle_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]));
forward_znx!(ZnxNormalizeMiddleStepSub, znx_normalize_middle_step_sub(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]));
forward_znx!(ZnxNormalizeFinalStepSub, znx_normalize_final_step_sub(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]));
forward_znx!(ZnxNormalizeFinalStepAssign, znx_normalize_final_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]));
forward_znx!(ZnxExtractDigitAddMul, znx_extract_digit_addmul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]));
forward_znx!(ZnxNormalizeDigit, znx_normalize_digit(base2k: usize, res: &mut [i64], src: &mut [i64]));

impl ReimFFTExecute<ReimFFTTable<f64>, f64> for FFT64Avx512Rayon {
    #[inline(always)]
    fn reim_dft_execute(table: &ReimFFTTable<f64>, data: &mut [f64]) {
        <FFT64Avx512 as ReimFFTExecute<ReimFFTTable<f64>, f64>>::reim_dft_execute(table, data)
    }
}

impl ReimFFTExecute<ReimIFFTTable<f64>, f64> for FFT64Avx512Rayon {
    #[inline(always)]
    fn reim_dft_execute(table: &ReimIFFTTable<f64>, data: &mut [f64]) {
        <FFT64Avx512 as ReimFFTExecute<ReimIFFTTable<f64>, f64>>::reim_dft_execute(table, data)
    }
}

impl ReimArith for FFT64Avx512Rayon {
    #[inline(always)]
    fn reim_from_znx(res: &mut [f64], a: &[i64]) {
        <FFT64Avx512 as ReimArith>::reim_from_znx(res, a)
    }
    #[inline(always)]
    fn reim_from_znx_masked(res: &mut [f64], a: &[i64], mask: i64) {
        <FFT64Avx512 as ReimArith>::reim_from_znx_masked(res, a, mask)
    }
    #[inline(always)]
    fn reim_to_znx(res: &mut [i64], divisor: f64, a: &[f64]) {
        <FFT64Avx512 as ReimArith>::reim_to_znx(res, divisor, a)
    }
    #[inline(always)]
    fn reim_to_znx_assign(res: &mut [f64], divisor: f64) {
        <FFT64Avx512 as ReimArith>::reim_to_znx_assign(res, divisor)
    }
    #[inline(always)]
    fn reim_add(res: &mut [f64], a: &[f64], b: &[f64]) {
        <FFT64Avx512 as ReimArith>::reim_add(res, a, b)
    }
    #[inline(always)]
    fn reim_add_assign(res: &mut [f64], a: &[f64]) {
        <FFT64Avx512 as ReimArith>::reim_add_assign(res, a)
    }
    #[inline(always)]
    fn reim_sub(res: &mut [f64], a: &[f64], b: &[f64]) {
        <FFT64Avx512 as ReimArith>::reim_sub(res, a, b)
    }
    #[inline(always)]
    fn reim_sub_assign(res: &mut [f64], a: &[f64]) {
        <FFT64Avx512 as ReimArith>::reim_sub_assign(res, a)
    }
    #[inline(always)]
    fn reim_sub_negate_assign(res: &mut [f64], a: &[f64]) {
        <FFT64Avx512 as ReimArith>::reim_sub_negate_assign(res, a)
    }
    #[inline(always)]
    fn reim_negate(res: &mut [f64], a: &[f64]) {
        <FFT64Avx512 as ReimArith>::reim_negate(res, a)
    }
    #[inline(always)]
    fn reim_negate_assign(res: &mut [f64]) {
        <FFT64Avx512 as ReimArith>::reim_negate_assign(res)
    }
    #[inline(always)]
    fn reim_mul(res: &mut [f64], a: &[f64], b: &[f64]) {
        <FFT64Avx512 as ReimArith>::reim_mul(res, a, b)
    }
    #[inline(always)]
    fn reim_mul_assign(res: &mut [f64], a: &[f64]) {
        <FFT64Avx512 as ReimArith>::reim_mul_assign(res, a)
    }
    #[inline(always)]
    fn reim_addmul(res: &mut [f64], a: &[f64], b: &[f64]) {
        <FFT64Avx512 as ReimArith>::reim_addmul(res, a, b)
    }
    #[inline(always)]
    fn reim_copy(res: &mut [f64], a: &[f64]) {
        <FFT64Avx512 as ReimArith>::reim_copy(res, a)
    }
    #[inline(always)]
    fn reim_zero(res: &mut [f64]) {
        <FFT64Avx512 as ReimArith>::reim_zero(res)
    }
}

impl Reim4BlkMatVec for FFT64Avx512Rayon {
    #[inline(always)]
    fn reim4_extract_1blk_contiguous(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        <FFT64Avx512 as Reim4BlkMatVec>::reim4_extract_1blk_contiguous(m, rows, blk, dst, src)
    }
    #[inline(always)]
    fn reim4_save_1blk_contiguous(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        <FFT64Avx512 as Reim4BlkMatVec>::reim4_save_1blk_contiguous(m, rows, blk, dst, src)
    }
    #[inline(always)]
    fn reim4_save_1blk<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        <FFT64Avx512 as Reim4BlkMatVec>::reim4_save_1blk::<OVERWRITE>(m, blk, dst, src)
    }
    #[inline(always)]
    fn reim4_save_2blks<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        <FFT64Avx512 as Reim4BlkMatVec>::reim4_save_2blks::<OVERWRITE>(m, blk, dst, src)
    }
    #[inline(always)]
    fn reim4_mat1col_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        <FFT64Avx512 as Reim4BlkMatVec>::reim4_mat1col_prod(nrows, dst, u, v)
    }
    #[inline(always)]
    fn reim4_mat2cols_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        <FFT64Avx512 as Reim4BlkMatVec>::reim4_mat2cols_prod(nrows, dst, u, v)
    }
    #[inline(always)]
    fn reim4_mat2cols_2ndcol_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        <FFT64Avx512 as Reim4BlkMatVec>::reim4_mat2cols_2ndcol_prod(nrows, dst, u, v)
    }
}

impl Reim4Convolution for FFT64Avx512Rayon {
    #[inline(always)]
    fn reim4_convolution_1coeff(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
        <FFT64Avx512 as Reim4Convolution>::reim4_convolution_1coeff(k, dst, a, a_size, b, b_size)
    }
    #[inline(always)]
    fn reim4_convolution_2coeffs(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
        <FFT64Avx512 as Reim4Convolution>::reim4_convolution_2coeffs(k, dst, a, a_size, b, b_size)
    }
    #[inline(always)]
    fn reim4_convolution(dst: &mut [f64], dst_size: usize, offset: usize, a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
        <FFT64Avx512 as Reim4Convolution>::reim4_convolution(dst, dst_size, offset, a, a_size, b, b_size)
    }
    #[inline(always)]
    fn reim4_convolution_apply(
        m: usize,
        min_size: usize,
        offset: usize,
        dst: &mut [f64],
        dst_stride: usize,
        a: &[f64],
        a_size: usize,
        b: &[f64],
        b_size: usize,
        tmp: &mut [f64],
    ) {
        <FFT64Avx512 as Reim4Convolution>::reim4_convolution_apply(
            m, min_size, offset, dst, dst_stride, a, a_size, b, b_size, tmp,
        )
    }
    #[inline(always)]
    fn reim4_convolution_apply_accumulate(
        m: usize,
        min_size: usize,
        offset: usize,
        dst: &mut [f64],
        dst_stride: usize,
        a: &[f64],
        a_size: usize,
        b: &[f64],
        b_size: usize,
        tmp: &mut [f64],
    ) {
        <FFT64Avx512 as Reim4Convolution>::reim4_convolution_apply_accumulate(
            m, min_size, offset, dst, dst_stride, a, a_size, b, b_size, tmp,
        )
    }
    #[inline(always)]
    fn reim4_convolution_pairwise_apply(
        m: usize,
        min_size: usize,
        offset: usize,
        dst: &mut [f64],
        dst_stride: usize,
        a0: &[f64],
        a1: &[f64],
        a_size: usize,
        b0: &[f64],
        b1: &[f64],
        b_size: usize,
        tmp: &mut [f64],
    ) {
        <FFT64Avx512 as Reim4Convolution>::reim4_convolution_pairwise_apply(
            m, min_size, offset, dst, dst_stride, a0, a1, a_size, b0, b1, b_size, tmp,
        )
    }
    #[inline(always)]
    fn reim4_convolution_by_real_const_1coeff(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64]) {
        <FFT64Avx512 as Reim4Convolution>::reim4_convolution_by_real_const_1coeff(k, dst, a, a_size, b)
    }
    #[inline(always)]
    fn reim4_convolution_by_real_const_2coeffs(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64]) {
        <FFT64Avx512 as Reim4Convolution>::reim4_convolution_by_real_const_2coeffs(k, dst, a, a_size, b)
    }
}

impl I64Ops for FFT64Avx512Rayon {
    #[inline(always)]
    fn i64_hadamard_product(res: &mut [i64], a: &[i64], b: &[i64]) {
        <FFT64Avx512 as I64Ops>::i64_hadamard_product(res, a, b)
    }
    #[inline(always)]
    fn i64_extract_1blk_contiguous(n: usize, offset: usize, rows: usize, blk: usize, dst: &mut [i64], src: &[i64]) {
        <FFT64Avx512 as I64Ops>::i64_extract_1blk_contiguous(n, offset, rows, blk, dst, src)
    }
    #[inline(always)]
    fn i64_save_1blk_contiguous(n: usize, offset: usize, rows: usize, blk: usize, dst: &mut [i64], src: &[i64]) {
        <FFT64Avx512 as I64Ops>::i64_save_1blk_contiguous(n, offset, rows, blk, dst, src)
    }
    #[inline(always)]
    fn i64_convolution_by_const_1coeff(k: usize, dst: &mut [i64; 8], a: &[i64], a_size: usize, b: &[i64]) {
        <FFT64Avx512 as I64Ops>::i64_convolution_by_const_1coeff(k, dst, a, a_size, b)
    }
    #[inline(always)]
    fn i64_convolution_by_const_2coeffs(k: usize, dst: &mut [i64; 16], a: &[i64], a_size: usize, b: &[i64]) {
        <FFT64Avx512 as I64Ops>::i64_convolution_by_const_2coeffs(k, dst, a, a_size, b)
    }
}

impl BigWordHadamardProduct for FFT64Avx512Rayon {
    #[inline(always)]
    fn big_word_hadamard_product(res: &mut [i64], a: &[i64], b: &[i64]) {
        <FFT64Avx512 as BigWordHadamardProduct>::big_word_hadamard_product(res, a, b)
    }
}

unsafe impl HalVecZnxImpl<FFT64Avx512Rayon> for FFT64Avx512Rayon {
    poulpy_cpu_ref::hal_impl_vec_znx!();
    fn vec_znx_transpose_backend(module: &Module<Self>, res: &mut VecZnxBackendMut<'_, Self>, a: &VecZnxBackendRef<'_, Self>) {
        <Self as HalVecZnxDefault<Self>>::vec_znx_transpose_backend_default(module, res, a)
    }
}
unsafe impl HalModuleImpl<FFT64Avx512Rayon> for FFT64Avx512Rayon {
    poulpy_cpu_ref::hal_impl_module!(FFT64ModuleDefault);
}
unsafe impl HalVmpImpl<FFT64Avx512Rayon> for FFT64Avx512Rayon {
    #[inline(always)]
    fn vmp_prepare_tmp_bytes(module: &Module<Self>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        <FFT64Avx512 as HalVmpImpl<FFT64Avx512>>::vmp_prepare_tmp_bytes(base_module(module), rows, cols_in, cols_out, size)
    }

    #[inline(always)]
    fn vmp_prepare(
        module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &MatZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow().into_backend::<FFT64Avx512>();
        <FFT64Avx512 as HalVmpImpl<FFT64Avx512>>::vmp_prepare(base_module(module), &mut base_vmp_mut(res), a, &mut scratch)
    }

    #[inline(always)]
    fn vmp_apply_dft_tmp_bytes(
        module: &Module<Self>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        <FFT64Avx512 as HalVmpImpl<FFT64Avx512>>::vmp_apply_dft_tmp_bytes(
            base_module(module),
            res_size,
            a_size,
            b_rows,
            b_cols_in,
            b_cols_out,
            b_size,
        )
    }

    #[inline(always)]
    fn vmp_apply_dft<R>(
        module: &Module<Self>,
        res: &mut R,
        a: &VecZnxBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: VecZnxDftToBackendMut<Self>,
    {
        let cols_to_copy = a.cols().min(b.cols_in());
        let a_start_col = a.cols() - cols_to_copy;
        let a_dft_size = a.size().min(b.rows());
        let offset = b.cols_in() - cols_to_copy;
        scratch.consume(|scratch| {
            let (mut a_dft, mut scratch) = scratch.take_vec_znx_dft_scratch(module, b.cols_in(), a_dft_size);
            for col in 0..offset {
                module.vec_znx_dft_zero(&mut a_dft, col);
            }
            for col in 0..cols_to_copy {
                module.vec_znx_dft_apply(1, 0, &mut a_dft, offset + col, a, a_start_col + col);
            }
            let mut res = res.to_backend_mut();
            module.vmp_apply_dft_to_dft(&mut res, &a_dft.to_backend_ref(), b, 0, &mut scratch);
            ((), scratch)
        })
    }

    #[inline(always)]
    fn vmp_apply_dft_to_dft_tmp_bytes(
        module: &Module<Self>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        <FFT64Avx512 as HalVmpImpl<FFT64Avx512>>::vmp_apply_dft_to_dft_tmp_bytes(
            base_module(module),
            res_size,
            a_size,
            b_rows,
            b_cols_in,
            b_cols_out,
            b_size,
        )
    }

    #[inline(always)]
    fn vmp_apply_dft_to_dft(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow().into_backend::<FFT64Avx512>();
        <FFT64Avx512 as HalVmpImpl<FFT64Avx512>>::vmp_apply_dft_to_dft(
            base_module(module),
            &mut base_dft_mut(res),
            &base_dft_ref(a),
            &base_vmp_ref(b),
            limb_offset,
            &mut scratch,
        )
    }

    #[inline(always)]
    fn vmp_apply_dft_to_dft_accumulate_tmp_bytes(
        module: &Module<Self>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        <FFT64Avx512 as HalVmpImpl<FFT64Avx512>>::vmp_apply_dft_to_dft_accumulate_tmp_bytes(
            base_module(module),
            res_size,
            a_size,
            b_rows,
            b_cols_in,
            b_cols_out,
            b_size,
        )
    }

    #[inline(always)]
    fn vmp_apply_dft_to_dft_accumulate(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow().into_backend::<FFT64Avx512>();
        <FFT64Avx512 as HalVmpImpl<FFT64Avx512>>::vmp_apply_dft_to_dft_accumulate(
            base_module(module),
            &mut base_dft_mut(res),
            &base_dft_ref(a),
            &base_vmp_ref(b),
            limb_offset,
            &mut scratch,
        )
    }

    #[inline(always)]
    fn vmp_zero(module: &Module<Self>, res: &mut VmpPMatBackendMut<'_, Self>) {
        <FFT64Avx512 as HalVmpImpl<FFT64Avx512>>::vmp_zero(base_module(module), &mut base_vmp_mut(res))
    }
}
unsafe impl HalConvolutionImpl<FFT64Avx512Rayon> for FFT64Avx512Rayon {
    poulpy_cpu_ref::hal_impl_convolution!(FFT64ConvolutionDefault);
}
unsafe impl HalVecZnxBigImpl<FFT64Avx512Rayon> for FFT64Avx512Rayon {
    #[inline(always)]
    fn vec_znx_big_from_small_backend(
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <FFT64Avx512 as HalVecZnxBigImpl<FFT64Avx512>>::vec_znx_big_from_small_backend(&mut base_big_mut(res), res_col, a, a_col)
    }

    #[inline(always)]
    fn vec_znx_big_add_normal_backend(
        module: &Module<Self>,
        res_base2k: usize,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        noise_infos: NoiseInfos,
        seed: [u8; 32],
    ) {
        <FFT64Avx512 as HalVecZnxBigImpl<FFT64Avx512>>::vec_znx_big_add_normal_backend(
            base_module(module),
            res_base2k,
            &mut base_big_mut(res),
            res_col,
            noise_infos,
            seed,
        )
    }

    #[inline(always)]
    fn vec_znx_big_add_into(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBigBackendRef<'_, Self>,
        b_col: usize,
    ) {
        <FFT64Avx512 as HalVecZnxBigImpl<FFT64Avx512>>::vec_znx_big_add_into(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            &base_big_ref(a),
            a_col,
            &base_big_ref(b),
            b_col,
        )
    }

    #[inline(always)]
    fn vec_znx_big_add_assign(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <FFT64Avx512 as HalVecZnxBigImpl<FFT64Avx512>>::vec_znx_big_add_assign(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            &base_big_ref(a),
            a_col,
        )
    }

    #[inline(always)]
    fn vec_znx_big_add_small_into_backend(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
    ) {
        <FFT64Avx512 as HalVecZnxBigImpl<FFT64Avx512>>::vec_znx_big_add_small_into_backend(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            &base_big_ref(a),
            a_col,
            b,
            b_col,
        )
    }

    #[inline(always)]
    fn vec_znx_big_add_small_assign(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <FFT64Avx512 as HalVecZnxBigImpl<FFT64Avx512>>::vec_znx_big_add_small_assign(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            a,
            a_col,
        )
    }

    #[inline(always)]
    fn vec_znx_big_sub(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBigBackendRef<'_, Self>,
        b_col: usize,
    ) {
        <FFT64Avx512 as HalVecZnxBigImpl<FFT64Avx512>>::vec_znx_big_sub(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            &base_big_ref(a),
            a_col,
            &base_big_ref(b),
            b_col,
        )
    }

    #[inline(always)]
    fn vec_znx_big_sub_assign(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <FFT64Avx512 as HalVecZnxBigImpl<FFT64Avx512>>::vec_znx_big_sub_assign(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            &base_big_ref(a),
            a_col,
        )
    }

    #[inline(always)]
    fn vec_znx_big_sub_negate_assign(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <FFT64Avx512 as HalVecZnxBigImpl<FFT64Avx512>>::vec_znx_big_sub_negate_assign(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            &base_big_ref(a),
            a_col,
        )
    }

    #[inline(always)]
    fn vec_znx_big_sub_small_a_backend(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBigBackendRef<'_, Self>,
        b_col: usize,
    ) {
        <FFT64Avx512 as HalVecZnxBigImpl<FFT64Avx512>>::vec_znx_big_sub_small_a_backend(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            a,
            a_col,
            &base_big_ref(b),
            b_col,
        )
    }

    #[inline(always)]
    fn vec_znx_big_sub_small_assign(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <FFT64Avx512 as HalVecZnxBigImpl<FFT64Avx512>>::vec_znx_big_sub_small_assign(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            a,
            a_col,
        )
    }

    #[inline(always)]
    fn vec_znx_big_sub_small_b_backend(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
    ) {
        <FFT64Avx512 as HalVecZnxBigImpl<FFT64Avx512>>::vec_znx_big_sub_small_b_backend(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            &base_big_ref(a),
            a_col,
            b,
            b_col,
        )
    }

    #[inline(always)]
    fn vec_znx_big_sub_small_negate_assign(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <FFT64Avx512 as HalVecZnxBigImpl<FFT64Avx512>>::vec_znx_big_sub_small_negate_assign(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            a,
            a_col,
        )
    }

    #[inline(always)]
    fn vec_znx_big_inner_sum_backend(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        res_coeff: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <FFT64Avx512 as HalVecZnxBigImpl<FFT64Avx512>>::vec_znx_big_inner_sum_backend(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            res_coeff,
            &base_big_ref(a),
            a_col,
        )
    }

    #[inline(always)]
    fn vec_znx_big_col_weighted_sum(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        weights: &ScalarZnxBackendRef<'_, Self>,
        weights_col: usize,
        cols: usize,
        coeffs: usize,
    ) {
        <FFT64Avx512 as HalVecZnxBigImpl<FFT64Avx512>>::vec_znx_big_col_weighted_sum(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            a,
            weights,
            weights_col,
            cols,
            coeffs,
        )
    }

    #[inline(always)]
    fn vec_znx_scalar_product(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &ScalarZnxBackendRef<'_, Self>,
        b_col: usize,
    ) {
        <FFT64Avx512 as HalVecZnxBigImpl<FFT64Avx512>>::vec_znx_scalar_product(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            a,
            a_col,
            b,
            b_col,
        )
    }

    #[inline(always)]
    fn vec_znx_big_negate(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <FFT64Avx512 as HalVecZnxBigImpl<FFT64Avx512>>::vec_znx_big_negate(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            &base_big_ref(a),
            a_col,
        )
    }

    #[inline(always)]
    fn vec_znx_big_negate_assign(module: &Module<Self>, res: &mut VecZnxBigBackendMut<'_, Self>, res_col: usize) {
        <FFT64Avx512 as HalVecZnxBigImpl<FFT64Avx512>>::vec_znx_big_negate_assign(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
        )
    }

    #[inline(always)]
    fn vec_znx_big_normalize_tmp_bytes(module: &Module<Self>) -> usize {
        <FFT64Avx512 as HalVecZnxBigImpl<FFT64Avx512>>::vec_znx_big_normalize_tmp_bytes(base_module(module))
    }

    #[inline(always)]
    fn vec_znx_big_normalize(
        module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow().into_backend::<FFT64Avx512>();
        <FFT64Avx512 as HalVecZnxBigImpl<FFT64Avx512>>::vec_znx_big_normalize(
            base_module(module),
            res,
            res_base2k,
            res_offset,
            res_col,
            &base_big_ref(a),
            a_base2k,
            a_col,
            &mut scratch,
        )
    }

    #[inline(always)]
    fn vec_znx_big_automorphism(
        module: &Module<Self>,
        k: i64,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <FFT64Avx512 as HalVecZnxBigImpl<FFT64Avx512>>::vec_znx_big_automorphism(
            base_module(module),
            k,
            &mut base_big_mut(res),
            res_col,
            &base_big_ref(a),
            a_col,
        )
    }

    #[inline(always)]
    fn vec_znx_big_automorphism_assign_tmp_bytes(module: &Module<Self>) -> usize {
        <FFT64Avx512 as HalVecZnxBigImpl<FFT64Avx512>>::vec_znx_big_automorphism_assign_tmp_bytes(base_module(module))
    }

    #[inline(always)]
    fn vec_znx_big_automorphism_assign(
        module: &Module<Self>,
        k: i64,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow().into_backend::<FFT64Avx512>();
        <FFT64Avx512 as HalVecZnxBigImpl<FFT64Avx512>>::vec_znx_big_automorphism_assign(
            base_module(module),
            k,
            &mut base_big_mut(res),
            res_col,
            &mut scratch,
        )
    }
}
unsafe impl HalSvpImpl<FFT64Avx512Rayon> for FFT64Avx512Rayon {
    poulpy_cpu_ref::hal_impl_svp!(FFT64SvpDefault);
}
unsafe impl HalVecZnxDftImpl<FFT64Avx512Rayon> for FFT64Avx512Rayon {
    #[inline(always)]
    fn vec_znx_dft_apply(
        module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        if !parallel_limb_tasks(res.size()) {
            return <FFT64Avx512 as HalVecZnxDftImpl<FFT64Avx512>>::vec_znx_dft_apply(
                base_module(module),
                step,
                offset,
                &mut base_dft_mut(res),
                res_col,
                a,
                a_col,
            );
        }

        let n = res.n();
        let cols = res.cols();
        let a_size = a.size();
        let table = module.get_fft_table();
        res.raw_mut().par_chunks_mut(n * cols).enumerate().for_each(|(j, group)| {
            let dst = &mut group[n * res_col..][..n];
            let limb = offset + j * step;
            if limb < a_size {
                <FFT64Avx512 as ReimArith>::reim_from_znx(dst, a.at(a_col, limb));
                <FFT64Avx512 as ReimFFTExecute<ReimFFTTable<f64>, f64>>::reim_dft_execute(table, dst);
            } else {
                <FFT64Avx512 as ReimArith>::reim_zero(dst);
            }
        });
    }

    #[inline(always)]
    fn vec_znx_idft_apply_tmp_bytes(module: &Module<Self>) -> usize {
        <FFT64Avx512 as HalVecZnxDftImpl<FFT64Avx512>>::vec_znx_idft_apply_tmp_bytes(base_module(module))
    }

    #[inline(always)]
    fn vec_znx_idft_apply(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        if !parallel_limb_tasks(res.size()) {
            let mut scratch = scratch.borrow().into_backend::<FFT64Avx512>();
            return <FFT64Avx512 as HalVecZnxDftImpl<FFT64Avx512>>::vec_znx_idft_apply(
                base_module(module),
                &mut base_big_mut(res),
                res_col,
                &base_dft_ref(a),
                a_col,
                &mut scratch,
            );
        }

        let n = res.n();
        let res_cols = res.cols();
        let a_cols = a.cols();
        let min_size = res.size().min(a.size());
        let a_raw = a.raw();
        let table = module.get_ifft_table();
        let divisor = table.m() as f64;
        res.raw_mut().par_chunks_mut(n * res_cols).enumerate().for_each(|(j, group)| {
            let dst = &mut group[n * res_col..][..n];
            if j < min_size {
                let dst_f64 = bytemuck::cast_slice_mut(dst);
                let src = &a_raw[n * (j * a_cols + a_col)..][..n];
                <FFT64Avx512 as ReimArith>::reim_copy(dst_f64, src);
                <FFT64Avx512 as ReimFFTExecute<ReimIFFTTable<f64>, f64>>::reim_dft_execute(table, dst_f64);
                <FFT64Avx512 as ReimArith>::reim_to_znx_assign(dst_f64, divisor);
            } else {
                <FFT64Avx512 as ZnxZero>::znx_zero(dst);
            }
        });
    }

    #[inline(always)]
    fn vec_znx_idft_apply_tmpa(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, Self>,
        a_col: usize,
    ) {
        if !parallel_limb_tasks(res.size()) {
            return <FFT64Avx512 as HalVecZnxDftImpl<FFT64Avx512>>::vec_znx_idft_apply_tmpa(
                base_module(module),
                &mut base_big_mut(res),
                res_col,
                &mut base_dft_mut(a),
                a_col,
            );
        }

        let n = res.n();
        let res_cols = res.cols();
        let a_cols = a.cols();
        let min_size = res.size().min(a.size());
        let active_words = min_size * n * res_cols;
        let (res_active, res_zero) = res.raw_mut().split_at_mut(active_words);
        let table = module.get_ifft_table();
        let divisor = table.m() as f64;

        res_active
            .par_chunks_mut(n * res_cols)
            .zip(a.raw_mut().par_chunks_mut(n * a_cols))
            .for_each(|(res_group, a_group)| {
                let dst = &mut res_group[n * res_col..][..n];
                let src = &mut a_group[n * a_col..][..n];
                <FFT64Avx512 as ReimFFTExecute<ReimIFFTTable<f64>, f64>>::reim_dft_execute(table, src);
                <FFT64Avx512 as ReimArith>::reim_to_znx(dst, divisor, src);
            });
        res_zero
            .par_chunks_mut(n * res_cols)
            .for_each(|group| <FFT64Avx512 as ZnxZero>::znx_zero(&mut group[n * res_col..][..n]));
    }

    #[inline(always)]
    fn vec_znx_dft_add_into(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) {
        <FFT64Avx512 as HalVecZnxDftImpl<FFT64Avx512>>::vec_znx_dft_add_into(
            base_module(module),
            &mut base_dft_mut(res),
            res_col,
            &base_dft_ref(a),
            a_col,
            &base_dft_ref(b),
            b_col,
        )
    }

    #[inline(always)]
    fn vec_znx_dft_add_scaled_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        a_scale: i64,
    ) {
        <FFT64Avx512 as HalVecZnxDftImpl<FFT64Avx512>>::vec_znx_dft_add_scaled_assign(
            base_module(module),
            &mut base_dft_mut(res),
            res_col,
            &base_dft_ref(a),
            a_col,
            a_scale,
        )
    }

    #[inline(always)]
    fn vec_znx_dft_add_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <FFT64Avx512 as HalVecZnxDftImpl<FFT64Avx512>>::vec_znx_dft_add_assign(
            base_module(module),
            &mut base_dft_mut(res),
            res_col,
            &base_dft_ref(a),
            a_col,
        )
    }

    #[inline(always)]
    fn vec_znx_dft_sub(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) {
        <FFT64Avx512 as HalVecZnxDftImpl<FFT64Avx512>>::vec_znx_dft_sub(
            base_module(module),
            &mut base_dft_mut(res),
            res_col,
            &base_dft_ref(a),
            a_col,
            &base_dft_ref(b),
            b_col,
        )
    }

    #[inline(always)]
    fn vec_znx_dft_sub_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <FFT64Avx512 as HalVecZnxDftImpl<FFT64Avx512>>::vec_znx_dft_sub_assign(
            base_module(module),
            &mut base_dft_mut(res),
            res_col,
            &base_dft_ref(a),
            a_col,
        )
    }

    #[inline(always)]
    fn vec_znx_dft_sub_negate_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <FFT64Avx512 as HalVecZnxDftImpl<FFT64Avx512>>::vec_znx_dft_sub_negate_assign(
            base_module(module),
            &mut base_dft_mut(res),
            res_col,
            &base_dft_ref(a),
            a_col,
        )
    }

    #[inline(always)]
    fn vec_znx_dft_copy(
        module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <FFT64Avx512 as HalVecZnxDftImpl<FFT64Avx512>>::vec_znx_dft_copy(
            base_module(module),
            step,
            offset,
            &mut base_dft_mut(res),
            res_col,
            &base_dft_ref(a),
            a_col,
        )
    }

    #[inline(always)]
    fn vec_znx_dft_zero(module: &Module<Self>, res: &mut VecZnxDftBackendMut<'_, Self>, res_col: usize) {
        <FFT64Avx512 as HalVecZnxDftImpl<FFT64Avx512>>::vec_znx_dft_zero(base_module(module), &mut base_dft_mut(res), res_col)
    }

    type AutomorphismPlan = <FFT64Avx512 as HalVecZnxDftImpl<FFT64Avx512>>::AutomorphismPlan;

    #[inline(always)]
    fn vec_znx_dft_automorphism_plan(module: &Module<Self>, p: i64) -> Self::AutomorphismPlan {
        <FFT64Avx512 as HalVecZnxDftImpl<FFT64Avx512>>::vec_znx_dft_automorphism_plan(base_module(module), p)
    }

    #[inline(always)]
    fn vec_znx_dft_automorphism_with_plan(
        _module: &Module<Self>,
        plan: &Self::AutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        super::fft64_vec_znx_dft_automorphism_avx512::<Self>(plan, res, res_col, a, a_col);
    }
}

#[cfg(test)]
mod tests {
    use poulpy_cpu_ref::reference::znx::ZnxAdd;

    use super::FFT64Avx512Rayon;

    #[test]
    fn coefficient_add_matches_wrapping_arithmetic() {
        let a = vec![i64::MAX; 1 << 16];
        let b = vec![1; 1 << 16];
        let mut actual = vec![0; 1 << 16];
        <FFT64Avx512Rayon as ZnxAdd>::znx_add(&mut actual, &a, &b);
        assert!(actual.iter().all(|&x| x == i64::MIN));
    }
}
