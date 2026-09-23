use poulpy_hal::{
    layouts::{Module, VecZnxBackendMut, VecZnxBackendRef},
    oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
};

use crate::{
    FFT64Ref,
    hal_defaults::{
        BigWordHadamardProduct, FFT64ConvolutionDefault, FFT64ModuleDefault, FFT64SvpDefault, FFT64VecZnxBigDefault,
        FFT64VecZnxDftDefault, FFT64VmpDefault, HalVecZnxDefault,
    },
    reference::{
        fft64::{
            convolution::I64Ops,
            reim::{ReimArith, ReimFFTExecute, ReimFFTTable, ReimIFFTTable},
            reim4::{Reim4BlkMatVec, Reim4Convolution},
        },
        znx::{
            ZnxAdd, ZnxAddAssign, ZnxAutomorphism, ZnxAutomorphismRotate, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulAddPowerOfTwo,
            ZnxMulPowerOfTwo, ZnxMulPowerOfTwoAssign, ZnxNegate, ZnxNegateAssign, ZnxNormalizeDigit, ZnxNormalizeFinalStep,
            ZnxNormalizeFinalStepAssign, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepAssign, ZnxNormalizeFirstStepCarryOnly,
            ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepAssign, ZnxNormalizeMiddleStepCarryOnly, ZnxRotate, ZnxSub,
            ZnxSubAssign, ZnxSubNegateAssign, ZnxSwitchRing, ZnxZero,
        },
    },
};

// The two fixtures reuse the same independently implemented HAL surface. Core
// tests specialize the first marker; randomized parity injects samples into the
// second without modifying a production backend or sharing prepared storage.
macro_rules! impl_fft64_delegating_backend {
    ($be:ident) => {
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
        pub struct $be;

        impl poulpy_hal::execution::ScratchWorkers for $be {}

        poulpy_hal::impl_backend_from!($be, FFT64Ref);

        impl poulpy_hal::layouts::MaxBase2k for $be {
            fn max_base2k(n: usize, products: usize, failure_bits: usize) -> Option<usize> {
                <FFT64Ref as poulpy_hal::layouts::MaxBase2k>::max_base2k(n, products, failure_bits)
            }
        }

        impl ZnxAdd for $be {
            #[inline(always)]
            fn znx_add(res: &mut [i64], a: &[i64], b: &[i64]) {
                <FFT64Ref as ZnxAdd>::znx_add(res, a, b)
            }
        }
        impl ZnxAddAssign for $be {
            #[inline(always)]
            fn znx_add_assign(res: &mut [i64], a: &[i64]) {
                <FFT64Ref as ZnxAddAssign>::znx_add_assign(res, a)
            }
        }
        impl ZnxSub for $be {
            #[inline(always)]
            fn znx_sub(res: &mut [i64], a: &[i64], b: &[i64]) {
                <FFT64Ref as ZnxSub>::znx_sub(res, a, b)
            }
        }
        impl ZnxSubAssign for $be {
            #[inline(always)]
            fn znx_sub_assign(res: &mut [i64], a: &[i64]) {
                <FFT64Ref as ZnxSubAssign>::znx_sub_assign(res, a)
            }
        }
        impl ZnxSubNegateAssign for $be {
            #[inline(always)]
            fn znx_sub_negate_assign(res: &mut [i64], a: &[i64]) {
                <FFT64Ref as ZnxSubNegateAssign>::znx_sub_negate_assign(res, a)
            }
        }
        impl ZnxMulAddPowerOfTwo for $be {
            #[inline(always)]
            fn znx_muladd_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
                <FFT64Ref as ZnxMulAddPowerOfTwo>::znx_muladd_power_of_two(k, res, a)
            }
        }
        impl ZnxMulPowerOfTwo for $be {
            #[inline(always)]
            fn znx_mul_power_of_two(k: i64, res: &mut [i64], a: &[i64]) {
                <FFT64Ref as ZnxMulPowerOfTwo>::znx_mul_power_of_two(k, res, a)
            }
        }
        impl ZnxMulPowerOfTwoAssign for $be {
            #[inline(always)]
            fn znx_mul_power_of_two_assign(k: i64, res: &mut [i64]) {
                <FFT64Ref as ZnxMulPowerOfTwoAssign>::znx_mul_power_of_two_assign(k, res)
            }
        }
        impl ZnxAutomorphism for $be {
            #[inline(always)]
            fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
                <FFT64Ref as ZnxAutomorphism>::znx_automorphism(p, res, a)
            }
        }
        impl ZnxAutomorphismRotate for $be {
            #[inline(always)]
            fn znx_automorphism_rotate(p: i64, k: i64, res: &mut [i64], a: &[i64]) {
                <FFT64Ref as ZnxAutomorphismRotate>::znx_automorphism_rotate(p, k, res, a)
            }
        }
        impl ZnxCopy for $be {
            #[inline(always)]
            fn znx_copy(res: &mut [i64], a: &[i64]) {
                <FFT64Ref as ZnxCopy>::znx_copy(res, a)
            }
        }
        impl ZnxNegate for $be {
            #[inline(always)]
            fn znx_negate(res: &mut [i64], src: &[i64]) {
                <FFT64Ref as ZnxNegate>::znx_negate(res, src)
            }
        }
        impl ZnxNegateAssign for $be {
            #[inline(always)]
            fn znx_negate_assign(res: &mut [i64]) {
                <FFT64Ref as ZnxNegateAssign>::znx_negate_assign(res)
            }
        }
        impl ZnxRotate for $be {
            #[inline(always)]
            fn znx_rotate(p: i64, res: &mut [i64], src: &[i64]) {
                <FFT64Ref as ZnxRotate>::znx_rotate(p, res, src)
            }
        }
        impl ZnxZero for $be {
            #[inline(always)]
            fn znx_zero(res: &mut [i64]) {
                <FFT64Ref as ZnxZero>::znx_zero(res)
            }
        }
        impl ZnxSwitchRing for $be {
            #[inline(always)]
            fn znx_switch_ring(res: &mut [i64], a: &[i64]) {
                <FFT64Ref as ZnxSwitchRing>::znx_switch_ring(res, a)
            }
        }
        impl ZnxNormalizeFirstStep for $be {
            #[inline(always)]
            fn znx_normalize_first_step<const OVERWRITE: bool>(
                base2k: usize,
                lsh: usize,
                x: &mut [i64],
                a: &[i64],
                carry: &mut [i64],
            ) {
                <FFT64Ref as ZnxNormalizeFirstStep>::znx_normalize_first_step::<OVERWRITE>(base2k, lsh, x, a, carry)
            }
        }
        impl ZnxNormalizeMiddleStep for $be {
            #[inline(always)]
            fn znx_normalize_middle_step<const OVERWRITE: bool>(
                base2k: usize,
                lsh: usize,
                x: &mut [i64],
                a: &[i64],
                carry: &mut [i64],
            ) {
                <FFT64Ref as ZnxNormalizeMiddleStep>::znx_normalize_middle_step::<OVERWRITE>(base2k, lsh, x, a, carry)
            }
        }
        impl ZnxNormalizeFinalStep for $be {
            #[inline(always)]
            fn znx_normalize_final_step<const OVERWRITE: bool>(
                base2k: usize,
                lsh: usize,
                x: &mut [i64],
                a: &[i64],
                carry: &mut [i64],
            ) {
                <FFT64Ref as ZnxNormalizeFinalStep>::znx_normalize_final_step::<OVERWRITE>(base2k, lsh, x, a, carry)
            }
        }
        impl ZnxNormalizeFirstStepCarryOnly for $be {
            #[inline(always)]
            fn znx_normalize_first_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
                <FFT64Ref as ZnxNormalizeFirstStepCarryOnly>::znx_normalize_first_step_carry_only(base2k, lsh, x, carry)
            }
        }
        impl ZnxNormalizeFirstStepAssign for $be {
            #[inline(always)]
            fn znx_normalize_first_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
                <FFT64Ref as ZnxNormalizeFirstStepAssign>::znx_normalize_first_step_assign(base2k, lsh, x, carry)
            }
        }
        impl ZnxNormalizeMiddleStepCarryOnly for $be {
            #[inline(always)]
            fn znx_normalize_middle_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64]) {
                <FFT64Ref as ZnxNormalizeMiddleStepCarryOnly>::znx_normalize_middle_step_carry_only(base2k, lsh, x, carry)
            }
        }
        impl ZnxNormalizeMiddleStepAssign for $be {
            #[inline(always)]
            fn znx_normalize_middle_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
                <FFT64Ref as ZnxNormalizeMiddleStepAssign>::znx_normalize_middle_step_assign(base2k, lsh, x, carry)
            }
        }
        impl ZnxNormalizeFinalStepAssign for $be {
            #[inline(always)]
            fn znx_normalize_final_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
                <FFT64Ref as ZnxNormalizeFinalStepAssign>::znx_normalize_final_step_assign(base2k, lsh, x, carry)
            }
        }
        impl ZnxExtractDigitAddMul for $be {
            #[inline(always)]
            fn znx_extract_digit_addmul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
                <FFT64Ref as ZnxExtractDigitAddMul>::znx_extract_digit_addmul(base2k, lsh, res, src);
            }
        }

        impl crate::reference::normalization::I64NormalizeOps for $be {}
        impl ZnxNormalizeDigit for $be {
            #[inline(always)]
            fn znx_normalize_digit(base2k: usize, res: &mut [i64], src: &mut [i64]) {
                <FFT64Ref as ZnxNormalizeDigit>::znx_normalize_digit(base2k, res, src)
            }
        }

        impl ReimFFTExecute<ReimFFTTable<f64>, f64> for $be {
            #[inline(always)]
            fn reim_dft_execute(table: &ReimFFTTable<f64>, data: &mut [f64]) {
                <FFT64Ref as ReimFFTExecute<ReimFFTTable<f64>, f64>>::reim_dft_execute(table, data)
            }
        }

        impl ReimFFTExecute<ReimIFFTTable<f64>, f64> for $be {
            #[inline(always)]
            fn reim_dft_execute(table: &ReimIFFTTable<f64>, data: &mut [f64]) {
                <FFT64Ref as ReimFFTExecute<ReimIFFTTable<f64>, f64>>::reim_dft_execute(table, data)
            }
        }

        impl ReimArith for $be {}
        impl Reim4BlkMatVec for $be {}
        impl Reim4Convolution for $be {}
        impl I64Ops for $be {}
        impl BigWordHadamardProduct for $be {
            #[inline(always)]
            fn big_word_hadamard_product(res: &mut [i64], a: &[i64], b: &[i64]) {
                Self::i64_hadamard_product(res, a, b)
            }
        }

        unsafe impl HalVecZnxImpl for $be {
            crate::hal_impl_vec_znx!();
        }

        unsafe impl HalModuleImpl for $be {
            crate::hal_impl_module!(FFT64ModuleDefault);
        }

        unsafe impl HalVmpImpl for $be {
            crate::hal_impl_vmp!(FFT64VmpDefault);
        }

        unsafe impl HalConvolutionImpl for $be {
            crate::hal_impl_convolution!(FFT64ConvolutionDefault);
        }

        unsafe impl HalVecZnxBigImpl for $be {
            crate::hal_impl_vec_znx_big!(FFT64VecZnxBigDefault);
        }

        unsafe impl HalSvpImpl for $be {
            crate::hal_impl_svp!(FFT64SvpDefault);
        }

        unsafe impl HalVecZnxDftImpl for $be {
            crate::hal_impl_vec_znx_dft!(FFT64VecZnxDftDefault);
        }
    };
}

#[cfg(test)]
impl_fft64_delegating_backend!(DelegatingFFT64Ref);
impl_fft64_delegating_backend!(ControlledSamplingFFT64Ref);

#[test]
fn test_normalization_kernels_delegating_fft64_ref() {
    crate::test_suite::normalization::test_normalization_kernels::<DelegatingFFT64Ref>();
}

poulpy_core::impl_core_reference_full!(ControlledSamplingFFT64Ref);

#[cfg(test)]
impl_fft64_delegating_backend!(DifferentSamplingFFT64Ref);
#[cfg(test)]
poulpy_core::impl_core_reference_full!(DifferentSamplingFFT64Ref);
