use poulpy_hal::{
    api::{VecZnxDftApply, VecZnxDftZero, VmpApplyDftToDft},
    layouts::{
        Backend, Module, NoiseInfos, VecZnxBackendMut, VecZnxBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef, ZnxInfos,
    },
    oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
};

use crate::{
    FFT64Ref,
    hal_defaults::{
        FFT64ConvolutionDefault, FFT64ModuleDefault, FFT64SvpDefault, FFT64VecZnxBigDefault, FFT64VecZnxDftDefault,
        FFT64VmpDefault, HalVecZnxDefault,
    },
    reference::{
        fft64::{
            convolution::I64Ops,
            reim::{ReimArith, ReimFFTExecute, ReimFFTTable, ReimIFFTTable},
            reim4::{Reim4BlkMatVec, Reim4Convolution},
        },
        znx::{
            ZnxAdd, ZnxAddAssign, ZnxAutomorphism, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulAddPowerOfTwo, ZnxMulPowerOfTwo,
            ZnxMulPowerOfTwoAssign, ZnxNegate, ZnxNegateAssign, ZnxNormalizeDigit, ZnxNormalizeFinalStep,
            ZnxNormalizeFinalStepAssign, ZnxNormalizeFinalStepSub, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepAssign,
            ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepAssign,
            ZnxNormalizeMiddleStepCarryOnly, ZnxNormalizeMiddleStepSub, ZnxRotate, ZnxSub, ZnxSubAssign, ZnxSubNegateAssign,
            ZnxSwitchRing, ZnxZero,
        },
    },
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct DelegatingFFT64Ref;

poulpy_hal::impl_backend_from!(DelegatingFFT64Ref, FFT64Ref);

macro_rules! impl_forward_znx_trait {
    ($trait_name:ident, $method:ident($($arg:ident : $ty:ty),*)) => {
        impl $trait_name for DelegatingFFT64Ref {
            #[inline(always)]
            fn $method($($arg : $ty),*) {
                <FFT64Ref as $trait_name>::$method($($arg),*)
            }
        }
    };
}

macro_rules! impl_forward_znx_const_trait {
    ($trait_name:ident, $method:ident($($arg:ident : $ty:ty),*)) => {
        impl $trait_name for DelegatingFFT64Ref {
            #[inline(always)]
            fn $method<const OVERWRITE: bool>($($arg : $ty),*) {
                <FFT64Ref as $trait_name>::$method::<OVERWRITE>($($arg),*)
            }
        }
    };
}

impl_forward_znx_trait!(ZnxAdd, znx_add(res: &mut [i64], a: &[i64], b: &[i64]));
impl_forward_znx_trait!(ZnxAddAssign, znx_add_assign(res: &mut [i64], a: &[i64]));
impl_forward_znx_trait!(ZnxSub, znx_sub(res: &mut [i64], a: &[i64], b: &[i64]));
impl_forward_znx_trait!(ZnxSubAssign, znx_sub_assign(res: &mut [i64], a: &[i64]));
impl_forward_znx_trait!(ZnxSubNegateAssign, znx_sub_negate_assign(res: &mut [i64], a: &[i64]));
impl_forward_znx_trait!(ZnxMulAddPowerOfTwo, znx_muladd_power_of_two(k: i64, res: &mut [i64], a: &[i64]));
impl_forward_znx_trait!(ZnxMulPowerOfTwo, znx_mul_power_of_two(k: i64, res: &mut [i64], a: &[i64]));
impl_forward_znx_trait!(ZnxMulPowerOfTwoAssign, znx_mul_power_of_two_assign(k: i64, res: &mut [i64]));
impl_forward_znx_trait!(ZnxAutomorphism, znx_automorphism(p: i64, res: &mut [i64], a: &[i64]));
impl_forward_znx_trait!(ZnxCopy, znx_copy(res: &mut [i64], a: &[i64]));
impl_forward_znx_trait!(ZnxNegate, znx_negate(res: &mut [i64], src: &[i64]));
impl_forward_znx_trait!(ZnxNegateAssign, znx_negate_assign(res: &mut [i64]));
impl_forward_znx_trait!(ZnxRotate, znx_rotate(p: i64, res: &mut [i64], src: &[i64]));
impl_forward_znx_trait!(ZnxZero, znx_zero(res: &mut [i64]));
impl_forward_znx_trait!(ZnxSwitchRing, znx_switch_ring(res: &mut [i64], a: &[i64]));
impl_forward_znx_const_trait!(
    ZnxNormalizeFirstStep,
    znx_normalize_first_step(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64])
);
impl_forward_znx_const_trait!(
    ZnxNormalizeMiddleStep,
    znx_normalize_middle_step(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64])
);
impl_forward_znx_const_trait!(
    ZnxNormalizeFinalStep,
    znx_normalize_final_step(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64])
);
impl_forward_znx_trait!(
    ZnxNormalizeFirstStepCarryOnly,
    znx_normalize_first_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64])
);
impl_forward_znx_trait!(
    ZnxNormalizeFirstStepAssign,
    znx_normalize_first_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64])
);
impl_forward_znx_trait!(
    ZnxNormalizeMiddleStepCarryOnly,
    znx_normalize_middle_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64])
);
impl_forward_znx_trait!(
    ZnxNormalizeMiddleStepAssign,
    znx_normalize_middle_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64])
);
impl_forward_znx_trait!(
    ZnxNormalizeMiddleStepSub,
    znx_normalize_middle_step_sub(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64])
);
impl_forward_znx_trait!(
    ZnxNormalizeFinalStepSub,
    znx_normalize_final_step_sub(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64])
);
impl_forward_znx_trait!(
    ZnxNormalizeFinalStepAssign,
    znx_normalize_final_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64])
);
impl_forward_znx_trait!(
    ZnxExtractDigitAddMul,
    znx_extract_digit_addmul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64])
);
impl_forward_znx_trait!(ZnxNormalizeDigit, znx_normalize_digit(base2k: usize, res: &mut [i64], src: &mut [i64]));

impl ReimFFTExecute<ReimFFTTable<f64>, f64> for DelegatingFFT64Ref {
    #[inline(always)]
    fn reim_dft_execute(table: &ReimFFTTable<f64>, data: &mut [f64]) {
        <FFT64Ref as ReimFFTExecute<ReimFFTTable<f64>, f64>>::reim_dft_execute(table, data)
    }
}

impl ReimFFTExecute<ReimIFFTTable<f64>, f64> for DelegatingFFT64Ref {
    #[inline(always)]
    fn reim_dft_execute(table: &ReimIFFTTable<f64>, data: &mut [f64]) {
        <FFT64Ref as ReimFFTExecute<ReimIFFTTable<f64>, f64>>::reim_dft_execute(table, data)
    }
}

impl ReimArith for DelegatingFFT64Ref {}
impl Reim4BlkMatVec for DelegatingFFT64Ref {}
impl Reim4Convolution for DelegatingFFT64Ref {}
impl I64Ops for DelegatingFFT64Ref {}

unsafe impl HalVecZnxImpl<DelegatingFFT64Ref> for DelegatingFFT64Ref {
    crate::hal_impl_vec_znx!();
}

unsafe impl HalModuleImpl<DelegatingFFT64Ref> for DelegatingFFT64Ref {
    crate::hal_impl_module!(FFT64ModuleDefault);
}

unsafe impl HalVmpImpl<DelegatingFFT64Ref> for DelegatingFFT64Ref {
    crate::hal_impl_vmp!(FFT64VmpDefault);
}

unsafe impl HalConvolutionImpl<DelegatingFFT64Ref> for DelegatingFFT64Ref {
    crate::hal_impl_convolution!(FFT64ConvolutionDefault);
}

unsafe impl HalVecZnxBigImpl<DelegatingFFT64Ref> for DelegatingFFT64Ref {
    crate::hal_impl_vec_znx_big!(FFT64VecZnxBigDefault);
}

unsafe impl HalSvpImpl<DelegatingFFT64Ref> for DelegatingFFT64Ref {
    crate::hal_impl_svp!(FFT64SvpDefault);
}

unsafe impl HalVecZnxDftImpl<DelegatingFFT64Ref> for DelegatingFFT64Ref {
    crate::hal_impl_vec_znx_dft!(FFT64VecZnxDftDefault);
}
