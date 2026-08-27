//! Shared Rayon-scheduled wrapper for the FFT64 CPU backends.

#[macro_export]
macro_rules! rayon_parallel_binary {
    ($rayon:ty, $base:ty, $trait:ident, $method:ident) => {
        impl $trait for $rayon {
            #[inline(always)]
            fn $method(res: &mut [i64], a: &[i64], b: &[i64]) {
                let Some(chunk) = $crate::parallel_chunk_len::<$rayon>(res.len()) else {
                    return <$base as $trait>::$method(res, a, b);
                };
                res.par_chunks_mut(chunk)
                    .zip(a.par_chunks(chunk))
                    .zip(b.par_chunks(chunk))
                    .for_each(|((res, a), b)| <$base as $trait>::$method(res, a, b));
            }
        }
    };
}

#[macro_export]
macro_rules! rayon_parallel_assign {
    ($rayon:ty, $base:ty, $trait:ident, $method:ident) => {
        impl $trait for $rayon {
            #[inline(always)]
            fn $method(res: &mut [i64], a: &[i64]) {
                let Some(chunk) = $crate::parallel_chunk_len::<$rayon>(res.len()) else {
                    return <$base as $trait>::$method(res, a);
                };
                res.par_chunks_mut(chunk)
                    .zip(a.par_chunks(chunk))
                    .for_each(|(res, a)| <$base as $trait>::$method(res, a));
            }
        }
    };
}

#[macro_export]
macro_rules! rayon_parallel_unary {
    ($rayon:ty, $base:ty, $trait:ident, $method:ident) => {
        impl $trait for $rayon {
            #[inline(always)]
            fn $method(res: &mut [i64]) {
                let Some(chunk) = $crate::parallel_chunk_len::<$rayon>(res.len()) else {
                    return <$base as $trait>::$method(res);
                };
                res.par_chunks_mut(chunk)
                    .for_each(|res| <$base as $trait>::$method(res));
            }
        }
    };
}

#[macro_export]
macro_rules! rayon_parallel_shift {
    ($rayon:ty, $base:ty, $trait:ident, $method:ident) => {
        impl $trait for $rayon {
            #[inline(always)]
            fn $method(k: i64, res: &mut [i64], a: &[i64]) {
                let Some(chunk) = $crate::parallel_chunk_len::<$rayon>(res.len()) else {
                    return <$base as $trait>::$method(k, res, a);
                };
                res.par_chunks_mut(chunk)
                    .zip(a.par_chunks(chunk))
                    .for_each(|(res, a)| <$base as $trait>::$method(k, res, a));
            }
        }
    };
}

#[macro_export]
macro_rules! rayon_forward_znx {
    ($rayon:ty, $base:ty, $trait:ident, $method:ident($($arg:ident: $atype:ty),* $(,)?)) => {
        impl $trait for $rayon {
            #[inline(always)]
            fn $method($($arg: $atype),*) {
                <$base as $trait>::$method($($arg),*)
            }
        }
    };
}

#[macro_export]
macro_rules! rayon_forward_znx_const {
    ($rayon:ty, $base:ty, $trait:ident, $method:ident($($arg:ident: $atype:ty),* $(,)?)) => {
        impl $trait for $rayon {
            #[inline(always)]
            fn $method<const OVERWRITE: bool>($($arg: $atype),*) {
                <$base as $trait>::$method::<OVERWRITE>($($arg),*)
            }
        }
    };
}

/// Implements the Rayon-scheduled FFT64 backend `$rayon` on top of the serial
/// backend `$base`.
#[macro_export]
macro_rules! impl_fft64_rayon_backend {
    ($rayon:ty, $base:ty, $dft_automorphism:path) => {
        mod fft64_rayon_backend {
            #[allow(unused_imports)]
            use super::*;

use $crate::__private::rayon::prelude::*;

use $crate::__private::poulpy_cpu_ref::{
    hal_defaults::{
        BigWordHadamardProduct, FFT64ConvolutionDefault, FFT64ModuleDefault, FFT64SvpDefault, FFT64VmpDefault, HalVecZnxDefault,
    },
    reference::{
        fft64::{
            convolution::I64Ops,
            module::FFTModuleHandle,
            reim::{ReimArith, ReimFFTExecute, ReimFFTTable, ReimIFFTTable},
            reim4::{Reim4BlkMatVec, Reim4Convolution},
            vmp::{
                vmp_apply_dft_to_dft_tmp_bytes as fft64_vmp_apply_dft_to_dft_tmp_bytes, vmp_prepare as fft64_vmp_prepare,
                vmp_prepare_tmp_bytes as fft64_vmp_prepare_tmp_bytes,
            },
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
use $crate::__private::poulpy_hal::execution::{SerialTaskExecutor, TaskExecutor};
use $crate::__private::poulpy_hal::{
    api::{ScratchArenaTakeBasic, VecZnxDftApply, VecZnxDftZero, VmpApplyDftToDft},
    layouts::{
        DataView, DataViewMut, MatZnxBackendRef, Module, NoiseInfos, ScalarZnxBackendRef, ScratchArena, VecZnx, VecZnxBackendMut,
        VecZnxBackendRef, VecZnxBig, VecZnxBigBackendMut, VecZnxBigBackendRef, VecZnxDft, VecZnxDftBackendMut,
        VecZnxDftBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef, VmpPMatBackendMut, VmpPMatBackendRef, ZnxView,
        ZnxViewMut,
    },
    oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
};

use $crate::{RayonTaskExecutor, SendPtr};


$crate::__private::poulpy_hal::impl_backend_from!($rayon, $base, $crate::RayonTaskExecutor);

fn base_module(module: &Module<$rayon>) -> &Module<$base> {
    module.reinterpret()
}

fn base_dft_ref<'a>(a: &'a VecZnxDftBackendRef<'_, $rayon>) -> VecZnxDftBackendRef<'a, $base> {
    VecZnxDft::from_data(&**a.data(), a.n(), a.cols(), a.size())
}

fn base_dft_mut<'a>(a: &'a mut VecZnxDftBackendMut<'_, $rayon>) -> VecZnxDftBackendMut<'a, $base> {
    let (n, cols, size) = (a.n(), a.cols(), a.size());
    VecZnxDft::from_data(&mut **a.data_mut(), n, cols, size)
}

fn base_big_mut<'a>(a: &'a mut VecZnxBigBackendMut<'_, $rayon>) -> VecZnxBigBackendMut<'a, $base> {
    let (n, cols, size) = (a.n(), a.cols(), a.size());
    VecZnxBig::from_data(&mut **a.data_mut(), n, cols, size)
}

fn base_big_ref<'a>(a: &'a VecZnxBigBackendRef<'_, $rayon>) -> VecZnxBigBackendRef<'a, $base> {
    VecZnxBig::from_data(&**a.data(), a.n(), a.cols(), a.size())
}



$crate::rayon_parallel_binary!($rayon, $base, ZnxAdd, znx_add);
$crate::rayon_parallel_assign!($rayon, $base, ZnxAddAssign, znx_add_assign);
$crate::rayon_parallel_binary!($rayon, $base, ZnxSub, znx_sub);
$crate::rayon_parallel_assign!($rayon, $base, ZnxSubAssign, znx_sub_assign);
$crate::rayon_parallel_assign!($rayon, $base, ZnxSubNegateAssign, znx_sub_negate_assign);
$crate::rayon_parallel_shift!($rayon, $base, ZnxMulAddPowerOfTwo, znx_muladd_power_of_two);
$crate::rayon_parallel_shift!($rayon, $base, ZnxMulPowerOfTwo, znx_mul_power_of_two);
impl ZnxMulPowerOfTwoAssign for $rayon {
    #[inline(always)]
    fn znx_mul_power_of_two_assign(k: i64, res: &mut [i64]) {
        let Some(chunk) = $crate::parallel_chunk_len::<$rayon>(res.len()) else {
            return <$base as ZnxMulPowerOfTwoAssign>::znx_mul_power_of_two_assign(k, res);
        };
        res.par_chunks_mut(chunk)
            .for_each(|res| <$base as ZnxMulPowerOfTwoAssign>::znx_mul_power_of_two_assign(k, res));
    }
}
$crate::rayon_forward_znx!($rayon, $base, ZnxAutomorphism, znx_automorphism(p: i64, res: &mut [i64], a: &[i64]));
$crate::rayon_forward_znx!($rayon, $base, ZnxAutomorphismRotate, znx_automorphism_rotate(p: i64, k: i64, res: &mut [i64], a: &[i64]));
$crate::rayon_parallel_assign!($rayon, $base, ZnxCopy, znx_copy);
$crate::rayon_parallel_assign!($rayon, $base, ZnxNegate, znx_negate);
$crate::rayon_parallel_unary!($rayon, $base, ZnxNegateAssign, znx_negate_assign);
$crate::rayon_forward_znx!($rayon, $base, ZnxRotate, znx_rotate(p: i64, res: &mut [i64], src: &[i64]));
$crate::rayon_parallel_unary!($rayon, $base, ZnxZero, znx_zero);
$crate::rayon_forward_znx!($rayon, $base, ZnxSwitchRing, znx_switch_ring(res: &mut [i64], a: &[i64]));
$crate::rayon_forward_znx_const!($rayon, $base, ZnxNormalizeFirstStep, znx_normalize_first_step(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]));
$crate::rayon_forward_znx_const!($rayon, $base, ZnxNormalizeMiddleStep, znx_normalize_middle_step(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]));
$crate::rayon_forward_znx_const!($rayon, $base, ZnxNormalizeFinalStep, znx_normalize_final_step(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]));
$crate::rayon_forward_znx!($rayon, $base, ZnxNormalizeFirstStepCarryOnly, znx_normalize_first_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]));
$crate::rayon_forward_znx!($rayon, $base, ZnxNormalizeFirstStepAssign, znx_normalize_first_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]));
$crate::rayon_forward_znx!($rayon, $base, ZnxNormalizeMiddleStepCarryOnly, znx_normalize_middle_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]));
$crate::rayon_forward_znx!($rayon, $base, ZnxNormalizeMiddleStepAssign, znx_normalize_middle_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]));
$crate::rayon_forward_znx!($rayon, $base, ZnxNormalizeMiddleStepSub, znx_normalize_middle_step_sub(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]));
$crate::rayon_forward_znx!($rayon, $base, ZnxNormalizeFinalStepSub, znx_normalize_final_step_sub(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64]));
$crate::rayon_forward_znx!($rayon, $base, ZnxNormalizeFinalStepAssign, znx_normalize_final_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]));
$crate::rayon_forward_znx!($rayon, $base, ZnxExtractDigitAddMul, znx_extract_digit_addmul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]));
$crate::rayon_forward_znx!($rayon, $base, ZnxNormalizeDigit, znx_normalize_digit(base2k: usize, res: &mut [i64], src: &mut [i64]));

impl ReimFFTExecute<ReimFFTTable<f64>, f64> for $rayon {
    #[inline(always)]
    fn reim_dft_execute(table: &ReimFFTTable<f64>, data: &mut [f64]) {
        <$base as ReimFFTExecute<ReimFFTTable<f64>, f64>>::reim_dft_execute(table, data)
    }
}

impl ReimFFTExecute<ReimIFFTTable<f64>, f64> for $rayon {
    #[inline(always)]
    fn reim_dft_execute(table: &ReimIFFTTable<f64>, data: &mut [f64]) {
        <$base as ReimFFTExecute<ReimIFFTTable<f64>, f64>>::reim_dft_execute(table, data)
    }
}

impl ReimArith for $rayon {
    #[inline(always)]
    fn reim_from_znx(res: &mut [f64], a: &[i64]) {
        <$base as ReimArith>::reim_from_znx(res, a)
    }
    #[inline(always)]
    fn reim_from_znx_masked(res: &mut [f64], a: &[i64], mask: i64) {
        <$base as ReimArith>::reim_from_znx_masked(res, a, mask)
    }
    #[inline(always)]
    fn reim_to_znx(res: &mut [i64], divisor: f64, a: &[f64]) {
        <$base as ReimArith>::reim_to_znx(res, divisor, a)
    }
    #[inline(always)]
    fn reim_to_znx_assign(res: &mut [f64], divisor: f64) {
        <$base as ReimArith>::reim_to_znx_assign(res, divisor)
    }
    #[inline(always)]
    fn reim_add(res: &mut [f64], a: &[f64], b: &[f64]) {
        <$base as ReimArith>::reim_add(res, a, b)
    }
    #[inline(always)]
    fn reim_add_assign(res: &mut [f64], a: &[f64]) {
        <$base as ReimArith>::reim_add_assign(res, a)
    }
    #[inline(always)]
    fn reim_sub(res: &mut [f64], a: &[f64], b: &[f64]) {
        <$base as ReimArith>::reim_sub(res, a, b)
    }
    #[inline(always)]
    fn reim_sub_assign(res: &mut [f64], a: &[f64]) {
        <$base as ReimArith>::reim_sub_assign(res, a)
    }
    #[inline(always)]
    fn reim_sub_negate_assign(res: &mut [f64], a: &[f64]) {
        <$base as ReimArith>::reim_sub_negate_assign(res, a)
    }
    #[inline(always)]
    fn reim_negate(res: &mut [f64], a: &[f64]) {
        <$base as ReimArith>::reim_negate(res, a)
    }
    #[inline(always)]
    fn reim_negate_assign(res: &mut [f64]) {
        <$base as ReimArith>::reim_negate_assign(res)
    }
    #[inline(always)]
    fn reim_mul(res: &mut [f64], a: &[f64], b: &[f64]) {
        <$base as ReimArith>::reim_mul(res, a, b)
    }
    #[inline(always)]
    fn reim_mul_assign(res: &mut [f64], a: &[f64]) {
        <$base as ReimArith>::reim_mul_assign(res, a)
    }
    #[inline(always)]
    fn reim_addmul(res: &mut [f64], a: &[f64], b: &[f64]) {
        <$base as ReimArith>::reim_addmul(res, a, b)
    }
    #[inline(always)]
    fn reim_copy(res: &mut [f64], a: &[f64]) {
        <$base as ReimArith>::reim_copy(res, a)
    }
    #[inline(always)]
    fn reim_zero(res: &mut [f64]) {
        <$base as ReimArith>::reim_zero(res)
    }
}

impl Reim4BlkMatVec for $rayon {
    #[inline(always)]
    fn reim4_extract_1blk_contiguous(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        <$base as Reim4BlkMatVec>::reim4_extract_1blk_contiguous(m, rows, blk, dst, src)
    }
    #[inline(always)]
    fn reim4_save_1blk_contiguous(m: usize, rows: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        <$base as Reim4BlkMatVec>::reim4_save_1blk_contiguous(m, rows, blk, dst, src)
    }
    #[inline(always)]
    fn reim4_save_1blk<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        <$base as Reim4BlkMatVec>::reim4_save_1blk::<OVERWRITE>(m, blk, dst, src)
    }
    #[inline(always)]
    fn reim4_save_2blks<const OVERWRITE: bool>(m: usize, blk: usize, dst: &mut [f64], src: &[f64]) {
        <$base as Reim4BlkMatVec>::reim4_save_2blks::<OVERWRITE>(m, blk, dst, src)
    }
    #[inline(always)]
    fn reim4_mat1col_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        <$base as Reim4BlkMatVec>::reim4_mat1col_prod(nrows, dst, u, v)
    }
    #[inline(always)]
    fn reim4_mat2cols_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        <$base as Reim4BlkMatVec>::reim4_mat2cols_prod(nrows, dst, u, v)
    }
    #[inline(always)]
    fn reim4_mat2cols_2ndcol_prod(nrows: usize, dst: &mut [f64], u: &[f64], v: &[f64]) {
        <$base as Reim4BlkMatVec>::reim4_mat2cols_2ndcol_prod(nrows, dst, u, v)
    }
}

#[allow(clippy::too_many_arguments)]
fn parallel_reim4_convolution_apply<const PAIRWISE: bool, const ACC: bool>(
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
    let block_count = m / 4;
    let task_tmp_len = 8 * (min_size + (a_size + b_size) * PAIRWISE as usize);
    let dst_ptr = SendPtr::new(dst.as_mut_ptr());
    RayonTaskExecutor::for_each_chunked(block_count, tmp, task_tmp_len, |tmp, block| {
            let (a, b, out) = if PAIRWISE {
                let (a, rest) = tmp.split_at_mut(8 * a_size);
                let (b, out) = rest.split_at_mut(8 * b_size);
                <$base as ReimArith>::reim_add(
                    a,
                    &a0[block * 8 * a_size..][..8 * a_size],
                    &a1[block * 8 * a_size..][..8 * a_size],
                );
                <$base as ReimArith>::reim_add(
                    b,
                    &b0[block * 8 * b_size..][..8 * b_size],
                    &b1[block * 8 * b_size..][..8 * b_size],
                );
                (&*a, &*b, out)
            } else {
                (&a0[block * 8 * a_size..], &b0[block * 8 * b_size..], &mut tmp[..8 * min_size])
            };
            <$base as Reim4Convolution>::reim4_convolution(out, min_size, offset, a, a_size, b, b_size);
            unsafe {
                let dst = dst_ptr.get();
                for k in 0..min_size {
                    let base = dst.add(dst_stride * k + 4 * block);
                    if ACC {
                        for i in 0..4 {
                            *base.add(i) += out[8 * k + i];
                            *base.add(m + i) += out[8 * k + 4 + i];
                        }
                    } else {
                        std::ptr::copy_nonoverlapping(out.as_ptr().add(8 * k), base, 4);
                        std::ptr::copy_nonoverlapping(out.as_ptr().add(8 * k + 4), base.add(m), 4);
                    }
                }
            }
    });
}

impl Reim4Convolution for $rayon {
    #[inline(always)]
    fn reim4_convolution_1coeff(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
        <$base as Reim4Convolution>::reim4_convolution_1coeff(k, dst, a, a_size, b, b_size)
    }
    #[inline(always)]
    fn reim4_convolution_2coeffs(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
        <$base as Reim4Convolution>::reim4_convolution_2coeffs(k, dst, a, a_size, b, b_size)
    }
    #[inline(always)]
    fn reim4_convolution(dst: &mut [f64], dst_size: usize, offset: usize, a: &[f64], a_size: usize, b: &[f64], b_size: usize) {
        <$base as Reim4Convolution>::reim4_convolution(dst, dst_size, offset, a, a_size, b, b_size)
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
        if !RayonTaskExecutor::is_parallel() {
            return <$base as Reim4Convolution>::reim4_convolution_apply(
                m, min_size, offset, dst, dst_stride, a, a_size, b, b_size, tmp,
            );
        }
        parallel_reim4_convolution_apply::<false, false>(m, min_size, offset, dst, dst_stride, a, a, a_size, b, b, b_size, tmp);
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
        if !RayonTaskExecutor::is_parallel() {
            return <$base as Reim4Convolution>::reim4_convolution_apply_accumulate(
                m, min_size, offset, dst, dst_stride, a, a_size, b, b_size, tmp,
            );
        }
        parallel_reim4_convolution_apply::<false, true>(m, min_size, offset, dst, dst_stride, a, a, a_size, b, b, b_size, tmp);
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
        if !RayonTaskExecutor::is_parallel() {
            return <$base as Reim4Convolution>::reim4_convolution_pairwise_apply(
                m, min_size, offset, dst, dst_stride, a0, a1, a_size, b0, b1, b_size, tmp,
            );
        }
        parallel_reim4_convolution_apply::<true, false>(m, min_size, offset, dst, dst_stride, a0, a1, a_size, b0, b1, b_size, tmp);
    }
    #[inline(always)]
    fn reim4_convolution_by_real_const_1coeff(k: usize, dst: &mut [f64; 8], a: &[f64], a_size: usize, b: &[f64]) {
        <$base as Reim4Convolution>::reim4_convolution_by_real_const_1coeff(k, dst, a, a_size, b)
    }
    #[inline(always)]
    fn reim4_convolution_by_real_const_2coeffs(k: usize, dst: &mut [f64; 16], a: &[f64], a_size: usize, b: &[f64]) {
        <$base as Reim4Convolution>::reim4_convolution_by_real_const_2coeffs(k, dst, a, a_size, b)
    }
}

impl I64Ops for $rayon {
    #[inline(always)]
    fn i64_hadamard_product(res: &mut [i64], a: &[i64], b: &[i64]) {
        <$base as I64Ops>::i64_hadamard_product(res, a, b)
    }
    #[inline(always)]
    fn i64_extract_1blk_contiguous(n: usize, offset: usize, rows: usize, blk: usize, dst: &mut [i64], src: &[i64]) {
        <$base as I64Ops>::i64_extract_1blk_contiguous(n, offset, rows, blk, dst, src)
    }
    #[inline(always)]
    fn i64_save_1blk_contiguous(n: usize, offset: usize, rows: usize, blk: usize, dst: &mut [i64], src: &[i64]) {
        <$base as I64Ops>::i64_save_1blk_contiguous(n, offset, rows, blk, dst, src)
    }
    #[inline(always)]
    fn i64_convolution_by_const_1coeff(k: usize, dst: &mut [i64; 8], a: &[i64], a_size: usize, b: &[i64]) {
        <$base as I64Ops>::i64_convolution_by_const_1coeff(k, dst, a, a_size, b)
    }
    #[inline(always)]
    fn i64_convolution_by_const_2coeffs(k: usize, dst: &mut [i64; 16], a: &[i64], a_size: usize, b: &[i64]) {
        <$base as I64Ops>::i64_convolution_by_const_2coeffs(k, dst, a, a_size, b)
    }
}

impl BigWordHadamardProduct for $rayon {
    #[inline(always)]
    fn big_word_hadamard_product(res: &mut [i64], a: &[i64], b: &[i64]) {
        <$base as BigWordHadamardProduct>::big_word_hadamard_product(res, a, b)
    }
}

unsafe impl HalVecZnxImpl<$rayon> for $rayon {
    poulpy_cpu_ref::hal_impl_vec_znx_without_normalize!();
    fn vec_znx_transpose_backend(module: &Module<Self>, res: &mut VecZnxBackendMut<'_, Self>, a: &VecZnxBackendRef<'_, Self>) {
        <Self as HalVecZnxDefault<Self>>::vec_znx_transpose_backend_default(module, res, a)
    }

    fn vec_znx_normalize_backend(
        module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let (carry, _) = $crate::take_scratch::<Self, i64>(scratch.borrow(), 3 * module.n());
        $crate::normalize::vec_znx_normalize_par::<$base, $rayon>(res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry);
    }

    fn vec_znx_normalize_assign_backend(
        module: &Module<Self>,
        base2k: usize,
        a: &mut VecZnxBackendMut<'_, Self>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let (carry, _) = $crate::take_scratch::<Self, i64>(scratch.borrow(), 3 * module.n());
        $crate::normalize::vec_znx_normalize_assign_par::<$base, $rayon>(base2k, a, a_col, carry);
    }
}
unsafe impl HalModuleImpl<$rayon> for $rayon {
    poulpy_cpu_ref::hal_impl_module!(FFT64ModuleDefault);
}
unsafe impl HalVmpImpl<$rayon> for $rayon {
    fn vmp_prepare_tmp_bytes(module: &Module<Self>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        <$rayon as $crate::__private::poulpy_hal::execution::ScratchWorkers>::PREPARE
            * <Self as FFT64VmpDefault<Self>>::vmp_prepare_tmp_bytes_default(module, rows, cols_in, cols_out, size)
    }

    fn vmp_prepare(
        module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &MatZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = fft64_vmp_prepare_tmp_bytes(module.n());
        let rows = a.cols_in() * a.rows();
        let workers = $crate::workers_within(
            rows.min(<$rayon as $crate::__private::poulpy_hal::execution::ScratchWorkers>::PREPARE),
            per_worker,
            scratch.available(),
        );
        let (tmp, _) = $crate::take_scratch::<Self, f64>(scratch.borrow(), workers * per_worker / core::mem::size_of::<f64>());
        fft64_vmp_prepare::<Self>(module.get_fft_table(), res, a, tmp);
    }

    fn vmp_apply_dft_tmp_bytes(
        module: &Module<Self>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        <$base as HalVmpImpl<$base>>::vmp_apply_dft_tmp_bytes(
            base_module(module),
            res_size,
            a_size,
            b_rows,
            b_cols_in,
            b_cols_out,
            b_size,
        )
    }

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

    fn vmp_apply_dft_to_dft_tmp_bytes(
        module: &Module<Self>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        <$rayon as $crate::__private::poulpy_hal::execution::ScratchWorkers>::VMP
            * <Self as FFT64VmpDefault<Self>>::vmp_apply_dft_to_dft_tmp_bytes_default(
                module, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size,
            )
    }

    fn vmp_apply_dft_to_dft(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        if RayonTaskExecutor::should_serialize_inner() {
            return <Self as FFT64VmpDefault<Self>>::vmp_apply_dft_to_dft_with_kernel_default::<$base, SerialTaskExecutor>(
                module,
                res,
                a,
                b,
                limb_offset,
                1,
                scratch,
            );
        }
        let mut scratch = scratch.borrow();
        <Self as FFT64VmpDefault<Self>>::vmp_apply_dft_to_dft_with_kernel_default::<$base, RayonTaskExecutor>(
            module,
            res,
            a,
            b,
            limb_offset,
            <$rayon as $crate::__private::poulpy_hal::execution::ScratchWorkers>::VMP,
            &mut scratch,
        )
    }

    fn vmp_apply_dft_to_dft_accumulate_tmp_bytes(
        module: &Module<Self>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        <Self as FFT64VmpDefault<Self>>::vmp_apply_dft_to_dft_accumulate_tmp_bytes_default(
            module, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size,
        ) + (<$rayon as $crate::__private::poulpy_hal::execution::ScratchWorkers>::VMP - 1) * fft64_vmp_apply_dft_to_dft_tmp_bytes(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft_accumulate(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        if RayonTaskExecutor::should_serialize_inner() {
            return <Self as FFT64VmpDefault<Self>>::vmp_apply_dft_to_dft_accumulate_with_kernel_default::<$base, SerialTaskExecutor>(
                module,
                res,
                a,
                b,
                limb_offset,
                1,
                scratch,
            );
        }
        let mut scratch = scratch.borrow();
        <Self as FFT64VmpDefault<Self>>::vmp_apply_dft_to_dft_accumulate_with_kernel_default::<$base, RayonTaskExecutor>(
            module,
            res,
            a,
            b,
            limb_offset,
            <$rayon as $crate::__private::poulpy_hal::execution::ScratchWorkers>::VMP,
            &mut scratch,
        )
    }

    fn vmp_zero(module: &Module<Self>, res: &mut VmpPMatBackendMut<'_, Self>) {
        <Self as FFT64VmpDefault<Self>>::vmp_zero_default(module, res)
    }
}
unsafe impl HalConvolutionImpl<$rayon> for $rayon {
    poulpy_cpu_ref::hal_impl_convolution!(FFT64ConvolutionDefault);
}
unsafe impl HalVecZnxBigImpl<$rayon> for $rayon {
    fn vec_znx_big_from_small_backend(
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <$base as HalVecZnxBigImpl<$base>>::vec_znx_big_from_small_backend(&mut base_big_mut(res), res_col, a, a_col)
    }

    fn vec_znx_big_add_normal_backend(
        module: &Module<Self>,
        res_base2k: usize,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        noise_infos: NoiseInfos,
        seed: [u8; 32],
    ) {
        <$base as HalVecZnxBigImpl<$base>>::vec_znx_big_add_normal_backend(
            base_module(module),
            res_base2k,
            &mut base_big_mut(res),
            res_col,
            noise_infos,
            seed,
        )
    }

    fn vec_znx_big_add_into(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBigBackendRef<'_, Self>,
        b_col: usize,
    ) {
        <$base as HalVecZnxBigImpl<$base>>::vec_znx_big_add_into(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            &base_big_ref(a),
            a_col,
            &base_big_ref(b),
            b_col,
        )
    }

    fn vec_znx_big_add_assign(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <$base as HalVecZnxBigImpl<$base>>::vec_znx_big_add_assign(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            &base_big_ref(a),
            a_col,
        )
    }

    fn vec_znx_big_add_small_into_backend(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
    ) {
        <$base as HalVecZnxBigImpl<$base>>::vec_znx_big_add_small_into_backend(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            &base_big_ref(a),
            a_col,
            b,
            b_col,
        )
    }

    fn vec_znx_big_add_small_assign(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <$base as HalVecZnxBigImpl<$base>>::vec_znx_big_add_small_assign(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            a,
            a_col,
        )
    }

    fn vec_znx_big_sub(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBigBackendRef<'_, Self>,
        b_col: usize,
    ) {
        <$base as HalVecZnxBigImpl<$base>>::vec_znx_big_sub(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            &base_big_ref(a),
            a_col,
            &base_big_ref(b),
            b_col,
        )
    }

    fn vec_znx_big_sub_assign(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <$base as HalVecZnxBigImpl<$base>>::vec_znx_big_sub_assign(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            &base_big_ref(a),
            a_col,
        )
    }

    fn vec_znx_big_sub_negate_assign(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <$base as HalVecZnxBigImpl<$base>>::vec_znx_big_sub_negate_assign(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            &base_big_ref(a),
            a_col,
        )
    }

    fn vec_znx_big_sub_small_a_backend(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBigBackendRef<'_, Self>,
        b_col: usize,
    ) {
        <$base as HalVecZnxBigImpl<$base>>::vec_znx_big_sub_small_a_backend(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            a,
            a_col,
            &base_big_ref(b),
            b_col,
        )
    }

    fn vec_znx_big_sub_small_assign(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <$base as HalVecZnxBigImpl<$base>>::vec_znx_big_sub_small_assign(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            a,
            a_col,
        )
    }

    fn vec_znx_big_sub_small_b_backend(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
    ) {
        <$base as HalVecZnxBigImpl<$base>>::vec_znx_big_sub_small_b_backend(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            &base_big_ref(a),
            a_col,
            b,
            b_col,
        )
    }

    fn vec_znx_big_sub_small_negate_assign(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <$base as HalVecZnxBigImpl<$base>>::vec_znx_big_sub_small_negate_assign(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            a,
            a_col,
        )
    }

    fn vec_znx_big_inner_sum_backend(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        res_coeff: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <$base as HalVecZnxBigImpl<$base>>::vec_znx_big_inner_sum_backend(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            res_coeff,
            &base_big_ref(a),
            a_col,
        )
    }

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
        <$base as HalVecZnxBigImpl<$base>>::vec_znx_big_col_weighted_sum(
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

    fn vec_znx_scalar_product(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &ScalarZnxBackendRef<'_, Self>,
        b_col: usize,
    ) {
        <$base as HalVecZnxBigImpl<$base>>::vec_znx_scalar_product(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            a,
            a_col,
            b,
            b_col,
        )
    }

    fn vec_znx_big_negate(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <$base as HalVecZnxBigImpl<$base>>::vec_znx_big_negate(
            base_module(module),
            &mut base_big_mut(res),
            res_col,
            &base_big_ref(a),
            a_col,
        )
    }

    fn vec_znx_big_negate_assign(module: &Module<Self>, res: &mut VecZnxBigBackendMut<'_, Self>, res_col: usize) {
        <$base as HalVecZnxBigImpl<$base>>::vec_znx_big_negate_assign(base_module(module), &mut base_big_mut(res), res_col)
    }

    fn vec_znx_big_normalize_tmp_bytes(module: &Module<Self>) -> usize {
        <$base as HalVecZnxBigImpl<$base>>::vec_znx_big_normalize_tmp_bytes(base_module(module))
    }

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
        let (carry, _) = $crate::take_scratch::<Self, i64>(scratch.borrow(), 3 * module.n());
        let a_vec: VecZnxBackendRef<'_, $base> = VecZnx::from_data(&**a.data(), a.n(), a.cols(), a.size());
        $crate::normalize::vec_znx_normalize_par::<$base, $rayon>(res, res_base2k, res_offset, res_col, &a_vec, a_base2k, a_col, carry);
    }

    fn vec_znx_big_automorphism(
        module: &Module<Self>,
        k: i64,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <$base as HalVecZnxBigImpl<$base>>::vec_znx_big_automorphism(
            base_module(module),
            k,
            &mut base_big_mut(res),
            res_col,
            &base_big_ref(a),
            a_col,
        )
    }

    fn vec_znx_big_automorphism_assign_tmp_bytes(module: &Module<Self>) -> usize {
        <$base as HalVecZnxBigImpl<$base>>::vec_znx_big_automorphism_assign_tmp_bytes(base_module(module))
    }

    fn vec_znx_big_automorphism_assign(
        module: &Module<Self>,
        k: i64,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow().into_backend::<$base>();
        <$base as HalVecZnxBigImpl<$base>>::vec_znx_big_automorphism_assign(
            base_module(module),
            k,
            &mut base_big_mut(res),
            res_col,
            &mut scratch,
        )
    }
}
unsafe impl HalSvpImpl<$rayon> for $rayon {
    poulpy_cpu_ref::hal_impl_svp!(FFT64SvpDefault);
}
unsafe impl HalVecZnxDftImpl<$rayon> for $rayon {

    fn vec_znx_idft_normalize_consume_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        3 * module.n() * core::mem::size_of::<i64>()
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_idft_normalize_consume(
        module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_base2k: usize,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, Self>,
        a_col: usize,
        a_base2k: usize,
        addend: Option<(&VecZnxBackendRef<'_, Self>, usize)>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let n = a.n();
        let a_cols = a.cols();
        let a_size = a.size();
        let table = module.get_ifft_table();
        let divisor = table.m() as f64;
        // In-place inverse FFT per limb; the buffer becomes `VecZnx` layout.
        if $crate::parallel_limb_tasks(a_size) {
            a.raw_mut().par_chunks_mut(n * a_cols).for_each(|group| {
                let slot = &mut group[n * a_col..][..n];
                <$base as ReimFFTExecute<ReimIFFTTable<f64>, f64>>::reim_dft_execute(table, slot);
                <$base as ReimArith>::reim_to_znx_assign(slot, divisor);
            });
        } else {
            for group in a.raw_mut().chunks_mut(n * a_cols) {
                let slot = &mut group[n * a_col..][..n];
                <$base as ReimFFTExecute<ReimIFFTTable<f64>, f64>>::reim_dft_execute(table, slot);
                <$base as ReimArith>::reim_to_znx_assign(slot, divisor);
            }
        }
        let (carry, _) = $crate::take_scratch::<Self, i64>(scratch.borrow(), 3 * n);
        if let Some((add, add_col)) = addend {
            let mut big: VecZnxBigBackendMut<'_, $base> = VecZnxBig::from_data(&mut **a.data_mut(), n, a_cols, a_size);
            let mut big_ref = &mut big;
            $crate::__private::poulpy_cpu_ref::reference::fft64::vec_znx_big::vec_znx_big_add_small_assign::<_, _, $base>(
                &mut big_ref,
                a_col,
                &add,
                add_col,
            );
        }
        let a_vec: VecZnxBackendRef<'_, $base> = VecZnx::from_data(&**a.data(), n, a_cols, a_size);
        $crate::normalize::vec_znx_normalize_par::<$base, $rayon>(res, res_base2k, 0, res_col, &a_vec, a_base2k, a_col, carry);
    }
    fn vec_znx_dft_apply(
        module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        if !$crate::parallel_limb_tasks(res.size()) {
            return <$base as HalVecZnxDftImpl<$base>>::vec_znx_dft_apply(
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
                <$base as ReimArith>::reim_from_znx(dst, a.at(a_col, limb));
                <$base as ReimFFTExecute<ReimFFTTable<f64>, f64>>::reim_dft_execute(table, dst);
            } else {
                <$base as ReimArith>::reim_zero(dst);
            }
        });
    }

    fn vec_znx_idft_apply_tmp_bytes(module: &Module<Self>) -> usize {
        <$base as HalVecZnxDftImpl<$base>>::vec_znx_idft_apply_tmp_bytes(base_module(module))
    }

    fn vec_znx_idft_apply(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        if !$crate::parallel_limb_tasks(res.size()) {
            let mut scratch = scratch.borrow().into_backend::<$base>();
            return <$base as HalVecZnxDftImpl<$base>>::vec_znx_idft_apply(
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
                let dst_f64 = $crate::__private::bytemuck::cast_slice_mut(dst);
                let src = &a_raw[n * (j * a_cols + a_col)..][..n];
                <$base as ReimArith>::reim_copy(dst_f64, src);
                <$base as ReimFFTExecute<ReimIFFTTable<f64>, f64>>::reim_dft_execute(table, dst_f64);
                <$base as ReimArith>::reim_to_znx_assign(dst_f64, divisor);
            } else {
                <$base as ZnxZero>::znx_zero(dst);
            }
        });
    }

    fn vec_znx_idft_apply_tmpa(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, Self>,
        a_col: usize,
    ) {
        if !$crate::parallel_limb_tasks(res.size()) {
            return <$base as HalVecZnxDftImpl<$base>>::vec_znx_idft_apply_tmpa(
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
                <$base as ReimFFTExecute<ReimIFFTTable<f64>, f64>>::reim_dft_execute(table, src);
                <$base as ReimArith>::reim_to_znx(dst, divisor, src);
            });
        res_zero
            .par_chunks_mut(n * res_cols)
            .for_each(|group| <$base as ZnxZero>::znx_zero(&mut group[n * res_col..][..n]));
    }

    fn vec_znx_dft_add_into(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) {
        <$base as HalVecZnxDftImpl<$base>>::vec_znx_dft_add_into(
            base_module(module),
            &mut base_dft_mut(res),
            res_col,
            &base_dft_ref(a),
            a_col,
            &base_dft_ref(b),
            b_col,
        )
    }

    fn vec_znx_dft_add_scaled_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        a_scale: i64,
    ) {
        <$base as HalVecZnxDftImpl<$base>>::vec_znx_dft_add_scaled_assign(
            base_module(module),
            &mut base_dft_mut(res),
            res_col,
            &base_dft_ref(a),
            a_col,
            a_scale,
        )
    }

    fn vec_znx_dft_add_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <$base as HalVecZnxDftImpl<$base>>::vec_znx_dft_add_assign(
            base_module(module),
            &mut base_dft_mut(res),
            res_col,
            &base_dft_ref(a),
            a_col,
        )
    }

    fn vec_znx_dft_sub(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) {
        <$base as HalVecZnxDftImpl<$base>>::vec_znx_dft_sub(
            base_module(module),
            &mut base_dft_mut(res),
            res_col,
            &base_dft_ref(a),
            a_col,
            &base_dft_ref(b),
            b_col,
        )
    }

    fn vec_znx_dft_sub_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <$base as HalVecZnxDftImpl<$base>>::vec_znx_dft_sub_assign(
            base_module(module),
            &mut base_dft_mut(res),
            res_col,
            &base_dft_ref(a),
            a_col,
        )
    }

    fn vec_znx_dft_sub_negate_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <$base as HalVecZnxDftImpl<$base>>::vec_znx_dft_sub_negate_assign(
            base_module(module),
            &mut base_dft_mut(res),
            res_col,
            &base_dft_ref(a),
            a_col,
        )
    }

    fn vec_znx_dft_copy(
        module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <$base as HalVecZnxDftImpl<$base>>::vec_znx_dft_copy(
            base_module(module),
            step,
            offset,
            &mut base_dft_mut(res),
            res_col,
            &base_dft_ref(a),
            a_col,
        )
    }

    fn vec_znx_dft_zero(module: &Module<Self>, res: &mut VecZnxDftBackendMut<'_, Self>, res_col: usize) {
        <$base as HalVecZnxDftImpl<$base>>::vec_znx_dft_zero(base_module(module), &mut base_dft_mut(res), res_col)
    }

    type AutomorphismPlan = <$base as HalVecZnxDftImpl<$base>>::AutomorphismPlan;

    fn vec_znx_dft_automorphism_plan(module: &Module<Self>, p: i64) -> Self::AutomorphismPlan {
        <$base as HalVecZnxDftImpl<$base>>::vec_znx_dft_automorphism_plan(base_module(module), p)
    }

    fn vec_znx_dft_automorphism_with_plan(
        module: &Module<Self>,
        plan: &Self::AutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        $dft_automorphism(module, plan, res, res_col, a, a_col);
    }

    fn vec_znx_dft_automorphism_add_with_plan(
        _module: &Module<Self>,
        plan: &Self::AutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        if $crate::RayonTaskExecutor::should_serialize_inner() {
            $crate::__private::poulpy_cpu_ref::reference::fft64::vec_znx_dft::vec_znx_dft_automorphism_add::<
                $base,
                poulpy_hal::execution::SerialTaskExecutor,
            >(plan, &mut base_dft_mut(res), res_col, &base_dft_ref(a), a_col);
        } else {
            $crate::__private::poulpy_cpu_ref::reference::fft64::vec_znx_dft::vec_znx_dft_automorphism_add::<
                $base,
                $crate::RayonTaskExecutor,
            >(plan, &mut base_dft_mut(res), res_col, &base_dft_ref(a), a_col);
        }
    }
}

#[cfg(test)]
mod fft64_rayon_tests {
    #[allow(unused_imports)]
    use super::*;
    use $crate::__private::poulpy_cpu_ref::reference::znx::ZnxAdd;

    #[test]
    fn coefficient_add_matches_wrapping_arithmetic() {
        let a = vec![i64::MAX; 1 << 16];
        let b = vec![1; 1 << 16];
        let mut actual = vec![0; 1 << 16];
        <$rayon as ZnxAdd>::znx_add(&mut actual, &a, &b);
        assert!(actual.iter().all(|&x| x == i64::MIN));
    }
}

        }
    };
}
