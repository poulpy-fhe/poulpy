use std::mem::{ManuallyDrop, size_of};

use ::rayon::prelude::*;
use bytemuck::{cast_slice, cast_slice_mut};

use poulpy_cpu_ref::{
    hal_defaults::{BigWordHadamardProduct, HalVecZnxDefault, NTT4x30VecZnxBigDefault},
    reference::{
        ntt4x30::{I128BigOps, I128NormalizeOps, vec_znx_big::AssignOp},
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
    execution::{SerialTaskExecutor, TaskExecutor},
    layouts::{
        CnvDftAccTerm, CnvPVecL, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecR, CnvPVecRBackendMut, CnvPVecRBackendRef,
        DataView, DataViewMut, MatZnxBackendRef, Module, NoiseInfos, ScalarZnxBackendRef, ScratchArena, SvpPPol,
        SvpPPolBackendMut, SvpPPolBackendRef, VecZnxBackendMut, VecZnxBackendRef, VecZnxBig, VecZnxBigBackendMut, VecZnxDft,
        VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef, VmpPMat, VmpPMatBackendMut,
        VmpPMatBackendRef, ZnxView, ZnxViewMut,
    },
    oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
};

use super::{NTT3x42Ifma, NTT3x42IfmaRayon, NTT3x42IfmaRayonExecutor};
use poulpy_cpu_rayon::{RayonTaskExecutor, SendPtr};

fn base_module(module: &Module<NTT3x42IfmaRayon>) -> &Module<NTT3x42Ifma> {
    module.reinterpret()
}

fn base_dft_ref<'a>(a: &'a VecZnxDftBackendRef<'_, NTT3x42IfmaRayon>) -> VecZnxDftBackendRef<'a, NTT3x42Ifma> {
    VecZnxDft::from_data(&**a.data(), a.n(), a.cols(), a.size())
}

pub(crate) fn base_dft_mut<'a>(a: &'a mut VecZnxDftBackendMut<'_, NTT3x42IfmaRayon>) -> VecZnxDftBackendMut<'a, NTT3x42Ifma> {
    let (n, cols, size) = (a.n(), a.cols(), a.size());
    VecZnxDft::from_data(&mut **a.data_mut(), n, cols, size)
}

fn base_big_mut<'a>(a: &'a mut VecZnxBigBackendMut<'_, NTT3x42IfmaRayon>) -> VecZnxBigBackendMut<'a, NTT3x42Ifma> {
    let (n, cols, size) = (a.n(), a.cols(), a.size());
    VecZnxBig::from_data(&mut **a.data_mut(), n, cols, size)
}

fn base_big_ref<'a>(
    a: &'a poulpy_hal::layouts::VecZnxBigBackendRef<'_, NTT3x42IfmaRayon>,
) -> poulpy_hal::layouts::VecZnxBigBackendRef<'a, NTT3x42Ifma> {
    VecZnxBig::from_data(&**a.data(), a.n(), a.cols(), a.size())
}

fn base_svp_ref<'a>(a: &'a SvpPPolBackendRef<'_, NTT3x42IfmaRayon>) -> SvpPPolBackendRef<'a, NTT3x42Ifma> {
    SvpPPol::from_data(&**a.data(), a.n(), a.cols())
}

fn base_svp_mut<'a>(a: &'a mut SvpPPolBackendMut<'_, NTT3x42IfmaRayon>) -> SvpPPolBackendMut<'a, NTT3x42Ifma> {
    let (n, cols) = (a.n(), a.cols());
    SvpPPol::from_data(&mut **a.data_mut(), n, cols)
}

fn base_vmp_ref<'a>(a: &'a VmpPMatBackendRef<'_, NTT3x42IfmaRayon>) -> VmpPMatBackendRef<'a, NTT3x42Ifma> {
    VmpPMat::from_data(&**a.data(), a.n(), a.rows(), a.cols_in(), a.cols_out(), a.size())
}

fn base_vmp_mut<'a>(a: &'a mut VmpPMatBackendMut<'_, NTT3x42IfmaRayon>) -> VmpPMatBackendMut<'a, NTT3x42Ifma> {
    let (n, rows, cols_in, cols_out, size) = (a.n(), a.rows(), a.cols_in(), a.cols_out(), a.size());
    VmpPMat::from_data(&mut **a.data_mut(), n, rows, cols_in, cols_out, size)
}

pub(crate) fn base_cnv_l_ref<'a>(a: &'a CnvPVecLBackendRef<'_, NTT3x42IfmaRayon>) -> CnvPVecLBackendRef<'a, NTT3x42Ifma> {
    CnvPVecL::from_data(&**a.data(), a.n(), a.cols(), a.size())
}

fn base_cnv_l_mut<'a>(a: &'a mut CnvPVecLBackendMut<'_, NTT3x42IfmaRayon>) -> CnvPVecLBackendMut<'a, NTT3x42Ifma> {
    let (n, cols, size) = (a.n(), a.cols(), a.size());
    CnvPVecL::from_data(&mut **a.data_mut(), n, cols, size)
}

pub(crate) fn base_cnv_r_ref<'a>(a: &'a CnvPVecRBackendRef<'_, NTT3x42IfmaRayon>) -> CnvPVecRBackendRef<'a, NTT3x42Ifma> {
    CnvPVecR::from_data(&**a.data(), a.n(), a.cols(), a.size())
}

fn base_cnv_r_mut<'a>(a: &'a mut CnvPVecRBackendMut<'_, NTT3x42IfmaRayon>) -> CnvPVecRBackendMut<'a, NTT3x42Ifma> {
    let (n, cols, size) = (a.n(), a.cols(), a.size());
    CnvPVecR::from_data(&mut **a.data_mut(), n, cols, size)
}

macro_rules! forward_znx {
    ($trait:ident, $method:ident($($arg:ident: $ty:ty),* $(,)?)) => {
        impl $trait for NTT3x42IfmaRayon {
            #[inline(always)]
            fn $method($($arg: $ty),*) {
                <NTT3x42Ifma as $trait>::$method($($arg),*)
            }
        }
    };
}

macro_rules! forward_znx_const {
    ($trait:ident, $method:ident($($arg:ident: $ty:ty),* $(,)?)) => {
        impl $trait for NTT3x42IfmaRayon {
            #[inline(always)]
            fn $method<const OVERWRITE: bool>($($arg: $ty),*) {
                <NTT3x42Ifma as $trait>::$method::<OVERWRITE>($($arg),*)
            }
        }
    };
}

use poulpy_cpu_rayon::{parallel_chunk_len, parallel_limb_tasks};

macro_rules! parallel_binary {
    ($trait:ident, $method:ident) => {
        impl $trait for NTT3x42IfmaRayon {
            fn $method(res: &mut [i64], a: &[i64], b: &[i64]) {
                let Some(chunk) = parallel_chunk_len::<Self>(res.len()) else {
                    return <NTT3x42Ifma as $trait>::$method(res, a, b);
                };
                res.par_chunks_mut(chunk)
                    .zip(a.par_chunks(chunk))
                    .zip(b.par_chunks(chunk))
                    .for_each(|((res, a), b)| <NTT3x42Ifma as $trait>::$method(res, a, b));
            }
        }
    };
}

macro_rules! parallel_assign {
    ($trait:ident, $method:ident) => {
        impl $trait for NTT3x42IfmaRayon {
            fn $method(res: &mut [i64], a: &[i64]) {
                let Some(chunk) = parallel_chunk_len::<Self>(res.len()) else {
                    return <NTT3x42Ifma as $trait>::$method(res, a);
                };
                res.par_chunks_mut(chunk)
                    .zip(a.par_chunks(chunk))
                    .for_each(|(res, a)| <NTT3x42Ifma as $trait>::$method(res, a));
            }
        }
    };
}

macro_rules! parallel_unary {
    ($trait:ident, $method:ident) => {
        impl $trait for NTT3x42IfmaRayon {
            fn $method(res: &mut [i64]) {
                let Some(chunk) = parallel_chunk_len::<Self>(res.len()) else {
                    return <NTT3x42Ifma as $trait>::$method(res);
                };
                res.par_chunks_mut(chunk)
                    .for_each(|res| <NTT3x42Ifma as $trait>::$method(res));
            }
        }
    };
}

macro_rules! parallel_shift {
    ($trait:ident, $method:ident) => {
        impl $trait for NTT3x42IfmaRayon {
            fn $method(k: i64, res: &mut [i64], a: &[i64]) {
                let Some(chunk) = parallel_chunk_len::<Self>(res.len()) else {
                    return <NTT3x42Ifma as $trait>::$method(k, res, a);
                };
                res.par_chunks_mut(chunk)
                    .zip(a.par_chunks(chunk))
                    .for_each(|(res, a)| <NTT3x42Ifma as $trait>::$method(k, res, a));
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

impl ZnxMulPowerOfTwoAssign for NTT3x42IfmaRayon {
    fn znx_mul_power_of_two_assign(k: i64, res: &mut [i64]) {
        let Some(chunk) = parallel_chunk_len::<Self>(res.len()) else {
            return <NTT3x42Ifma as ZnxMulPowerOfTwoAssign>::znx_mul_power_of_two_assign(k, res);
        };
        res.par_chunks_mut(chunk)
            .for_each(|res| <NTT3x42Ifma as ZnxMulPowerOfTwoAssign>::znx_mul_power_of_two_assign(k, res));
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
forward_znx_const!(
    ZnxNormalizeFirstStep,
    znx_normalize_first_step(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64])
);
forward_znx_const!(
    ZnxNormalizeMiddleStep,
    znx_normalize_middle_step(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64])
);
forward_znx_const!(
    ZnxNormalizeFinalStep,
    znx_normalize_final_step(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64])
);
forward_znx!(
    ZnxNormalizeFirstStepCarryOnly,
    znx_normalize_first_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64])
);
forward_znx!(
    ZnxNormalizeFirstStepAssign,
    znx_normalize_first_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64])
);
forward_znx!(
    ZnxNormalizeMiddleStepCarryOnly,
    znx_normalize_middle_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64])
);
forward_znx!(
    ZnxNormalizeMiddleStepAssign,
    znx_normalize_middle_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64])
);
forward_znx!(
    ZnxNormalizeMiddleStepSub,
    znx_normalize_middle_step_sub(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64])
);
forward_znx!(
    ZnxNormalizeFinalStepSub,
    znx_normalize_final_step_sub(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64])
);
forward_znx!(
    ZnxNormalizeFinalStepAssign,
    znx_normalize_final_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64])
);
forward_znx!(
    ZnxExtractDigitAddMul,
    znx_extract_digit_addmul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64])
);
forward_znx!(ZnxNormalizeDigit, znx_normalize_digit(base2k: usize, res: &mut [i64], src: &mut [i64]));

unsafe impl HalModuleImpl<NTT3x42IfmaRayon> for NTT3x42IfmaRayon {
    fn new(n: u64) -> Module<NTT3x42IfmaRayon> {
        let module = ManuallyDrop::new(super::module::module_new(n));
        unsafe { Module::from_raw_parts(module.as_mut_ptr(), n) }
    }
}

unsafe impl HalVecZnxImpl<NTT3x42IfmaRayon> for NTT3x42IfmaRayon {
    poulpy_cpu_ref::hal_impl_vec_znx_without_normalize!();

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
        let (carry, _) = poulpy_cpu_rayon::take_scratch::<Self, i64>(scratch.borrow(), 3 * module.n());
        poulpy_cpu_rayon::normalize::vec_znx_normalize_par::<NTT3x42Ifma, Self>(
            res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry,
        );
    }

    fn vec_znx_normalize_assign_backend(
        module: &Module<Self>,
        base2k: usize,
        a: &mut VecZnxBackendMut<'_, Self>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let (carry, _) = poulpy_cpu_rayon::take_scratch::<Self, i64>(scratch.borrow(), 3 * module.n());
        poulpy_cpu_rayon::normalize::vec_znx_normalize_assign_par::<NTT3x42Ifma, Self>(base2k, a, a_col, carry);
    }

    fn vec_znx_transpose_backend(module: &Module<Self>, res: &mut VecZnxBackendMut<'_, Self>, a: &VecZnxBackendRef<'_, Self>) {
        <Self as HalVecZnxDefault<Self>>::vec_znx_transpose_backend_default(module, res, a)
    }
}

macro_rules! forward_i128_big {
    ($method:ident($($arg:ident: $ty:ty),* $(,)?)) => {
        #[inline(always)]
        fn $method($($arg: $ty),*) {
            <NTT3x42Ifma as I128BigOps>::$method($($arg),*)
        }
    };
}

impl I128BigOps for NTT3x42IfmaRayon {
    forward_i128_big!(i128_hadamard_product_i64(res: &mut [i128], a: &[i64], b: &[i64]));
    forward_i128_big!(i128_add(res: &mut [i128], a: &[i128], b: &[i128]));
    forward_i128_big!(i128_add_assign(res: &mut [i128], a: &[i128]));
    forward_i128_big!(i128_add_small(res: &mut [i128], a: &[i128], b: &[i64]));
    forward_i128_big!(i128_add_small_assign(res: &mut [i128], a: &[i64]));
    forward_i128_big!(i128_sub(res: &mut [i128], a: &[i128], b: &[i128]));
    forward_i128_big!(i128_sub_assign(res: &mut [i128], a: &[i128]));
    forward_i128_big!(i128_sub_negate_assign(res: &mut [i128], a: &[i128]));
    forward_i128_big!(i128_sub_small_a(res: &mut [i128], a: &[i64], b: &[i128]));
    forward_i128_big!(i128_sub_small_b(res: &mut [i128], a: &[i128], b: &[i64]));
    forward_i128_big!(i128_sub_small_assign(res: &mut [i128], a: &[i64]));
    forward_i128_big!(i128_sub_small_negate_assign(res: &mut [i128], a: &[i64]));
    forward_i128_big!(i128_negate(res: &mut [i128], a: &[i128]));
    forward_i128_big!(i128_negate_assign(res: &mut [i128]));
    forward_i128_big!(i128_neg_from_small(res: &mut [i128], a: &[i64]));
    forward_i128_big!(i128_from_small(res: &mut [i128], a: &[i64]));
}

impl I128NormalizeOps for NTT3x42IfmaRayon {
    #[inline(always)]
    fn nfc_middle_step(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
        <NTT3x42Ifma as I128NormalizeOps>::nfc_middle_step(base2k, lsh, res, a, carry)
    }

    #[inline(always)]
    fn nfc_middle_step_into<O: AssignOp>(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
        <NTT3x42Ifma as I128NormalizeOps>::nfc_middle_step_into::<O>(base2k, lsh, res, a, carry)
    }

    #[inline(always)]
    fn nfc_middle_step_assign(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        <NTT3x42Ifma as I128NormalizeOps>::nfc_middle_step_assign(base2k, lsh, res, carry)
    }

    #[inline(always)]
    fn nfc_final_step_assign(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        <NTT3x42Ifma as I128NormalizeOps>::nfc_final_step_assign(base2k, lsh, res, carry)
    }

    #[inline(always)]
    fn nfc_final_step_into<O: AssignOp>(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        <NTT3x42Ifma as I128NormalizeOps>::nfc_final_step_into::<O>(base2k, lsh, res, carry)
    }
}

impl BigWordHadamardProduct for NTT3x42IfmaRayon {
    fn big_word_hadamard_product(res: &mut [i128], a: &[i64], b: &[i64]) {
        <Self as I128BigOps>::i128_hadamard_product_i64(res, a, b)
    }
}

unsafe impl HalVecZnxBigImpl<NTT3x42IfmaRayon> for NTT3x42IfmaRayon {
    poulpy_cpu_ref::hal_impl_vec_znx_big_without_normalize!(NTT4x30VecZnxBigDefault);

    fn vec_znx_big_normalize(
        module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, Self>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let (carry, _) = poulpy_cpu_rayon::take_scratch::<Self, i128>(scratch.borrow(), 3 * module.n());
        poulpy_cpu_rayon::normalize::ntt4x30_vec_znx_big_normalize_par::<NTT3x42Ifma, Self>(
            res,
            res_base2k,
            res_offset,
            res_col,
            &base_big_ref(a),
            a_base2k,
            a_col,
            carry,
        );
    }
}

unsafe impl HalVecZnxDftImpl<NTT3x42IfmaRayon> for NTT3x42IfmaRayon {
    fn vec_znx_idft_normalize_consume_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::IDFT) * 3 * module.n() * size_of::<u64>()
            + 3 * module.n() * size_of::<i128>()
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
        let n = module.n();
        let workers = poulpy_cpu_rayon::workers_within(
            a.size().min(<Self as poulpy_hal::execution::ScratchWorkers>::IDFT),
            3 * n * size_of::<u64>(),
            scratch.available().saturating_sub(3 * n * size_of::<i128>()),
        );
        let arena = scratch.borrow();
        let (tmp, arena) = crate::hal_impl::take_host_typed::<Self, u64>(arena, workers * 3 * n);
        let (carry, _) = crate::hal_impl::take_host_typed::<Self, i128>(arena, 3 * n);
        let mut a_base = base_dft_mut(a);
        super::vec_znx_dft::idft_compact_in_place_ifma::<NTT3x42IfmaRayonExecutor>(base_module(module), &mut a_base, a_col, tmp);
        let (a_cols, a_size) = (a.cols(), a.size());
        if let Some((add, add_col)) = addend {
            let mut big: poulpy_hal::layouts::VecZnxBigBackendMut<'_, NTT3x42Ifma> =
                VecZnxBig::from_data(&mut **a.data_mut(), n, a_cols, a_size);
            let mut big_ref = &mut big;
            poulpy_cpu_ref::reference::ntt4x30::vec_znx_big::ntt4x30_vec_znx_big_add_small_assign::<_, _, NTT3x42Ifma>(
                &mut big_ref,
                a_col,
                &add,
                add_col,
            );
        }
        let big_ref: poulpy_hal::layouts::VecZnxBigBackendRef<'_, NTT3x42Ifma> =
            VecZnxBig::from_data(&**a.data(), n, a_cols, a_size);
        poulpy_cpu_rayon::normalize::ntt4x30_vec_znx_big_normalize_par::<NTT3x42Ifma, Self>(
            res, res_base2k, 0, res_col, &big_ref, a_base2k, a_col, carry,
        );
    }
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
        if parallel_limb_tasks(res.size()) {
            let n = res.n();
            let cols = res.cols();
            let res_size = res.size();
            let a_size = a.size();
            let min_steps = res_size.min(a_size.div_ceil(step));
            let limb_group_words = 2 * n * cols;
            let res = cast_slice_mut::<_, u64>(res.data_mut());
            res.par_chunks_mut(limb_group_words).enumerate().for_each_init(
                || vec![0u64; 3 * n],
                |scratch, (j, group)| {
                    let dst = &mut group[2 * n * res_col..][..2 * n];
                    let limb = offset + j * step;
                    let src = (j < min_steps && limb < a_size).then(|| a.at(a_col, limb));
                    super::vec_znx_dft::vec_znx_dft_apply_limb(base_module(module), dst, src, scratch);
                },
            );
            return;
        }
        let mut res = base_dft_mut(res);
        <NTT3x42Ifma as HalVecZnxDftImpl<NTT3x42Ifma>>::vec_znx_dft_apply(
            base_module(module),
            step,
            offset,
            &mut res,
            res_col,
            a,
            a_col,
        )
    }

    fn vec_znx_idft_apply_tmp_bytes(module: &Module<Self>) -> usize {
        <NTT3x42Ifma as HalVecZnxDftImpl<NTT3x42Ifma>>::vec_znx_idft_apply_tmp_bytes(base_module(module)).max(
            poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::IDFT) * 3 * module.n() * size_of::<u64>(),
        )
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
        if parallel_limb_tasks(res.size()) {
            let n = res.n();
            let res_cols = res.cols();
            let min_size = res.size().min(a.size());
            let a_cols = a.cols();
            let src = cast_slice::<_, u64>(a.data());
            let size = res.size();
            let per_worker = 3 * n;
            let workers = poulpy_cpu_rayon::workers_within(
                size.min(<Self as poulpy_hal::execution::ScratchWorkers>::IDFT),
                per_worker * size_of::<u64>(),
                scratch.available(),
            );
            let (worker_tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), workers * per_worker);
            let res_ptr = SendPtr::new(res.raw_mut().as_mut_ptr());
            RayonTaskExecutor::for_each_chunked(size, worker_tmp, per_worker, |scratch, j| {
                let dst = unsafe { std::slice::from_raw_parts_mut(res_ptr.get().add(n * (j * res_cols + res_col)), n) };
                let src = (j < min_size).then(|| &src[2 * n * (j * a_cols + a_col)..][..2 * n]);
                super::vec_znx_dft::vec_znx_idft_apply_limb(base_module(module), dst, src, scratch);
            });
            return;
        }
        let mut res = base_big_mut(res);
        let a = base_dft_ref(a);
        let mut scratch = scratch.borrow().into_backend::<NTT3x42Ifma>();
        <NTT3x42Ifma as HalVecZnxDftImpl<NTT3x42Ifma>>::vec_znx_idft_apply(
            base_module(module),
            &mut res,
            res_col,
            &a,
            a_col,
            &mut scratch,
        )
    }

    #[inline(always)]
    fn vec_znx_idft_apply_tmpa(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, Self>,
        a_col: usize,
    ) {
        if parallel_limb_tasks(res.size()) {
            let n = res.n();
            let res_cols = res.cols();
            let min_size = res.size().min(a.size());
            let a_cols = a.cols();
            let src = cast_slice::<_, u64>(a.data_mut());
            res.raw_mut().par_chunks_mut(n * res_cols).enumerate().for_each_init(
                || vec![0u64; 3 * n],
                |scratch, (j, group)| {
                    let dst = &mut group[n * res_col..][..n];
                    let src = (j < min_size).then(|| &src[2 * n * (j * a_cols + a_col)..][..2 * n]);
                    super::vec_znx_dft::vec_znx_idft_apply_tmpa_limb_ifma(base_module(module), dst, src, scratch);
                },
            );
            return;
        }
        let mut res = base_big_mut(res);
        let mut a = base_dft_mut(a);
        <NTT3x42Ifma as HalVecZnxDftImpl<NTT3x42Ifma>>::vec_znx_idft_apply_tmpa(
            base_module(module),
            &mut res,
            res_col,
            &mut a,
            a_col,
        )
    }

    fn vec_znx_dft_add_into(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) {
        let mut res = base_dft_mut(res);
        super::vec_znx_dft::vec_znx_dft_add_into::<NTT3x42IfmaRayonExecutor>(
            &mut res,
            res_col,
            &base_dft_ref(a),
            a_col,
            &base_dft_ref(b),
            b_col,
        )
    }

    fn vec_znx_dft_add_scaled_assign(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        a_scale: i64,
    ) {
        let mut res = base_dft_mut(res);
        super::vec_znx_dft::vec_znx_dft_add_scaled_assign::<NTT3x42IfmaRayonExecutor>(
            &mut res,
            res_col,
            &base_dft_ref(a),
            a_col,
            a_scale,
        )
    }

    fn vec_znx_dft_add_assign(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let mut res = base_dft_mut(res);
        super::vec_znx_dft::vec_znx_dft_add_assign::<NTT3x42IfmaRayonExecutor>(&mut res, res_col, &base_dft_ref(a), a_col)
    }

    fn vec_znx_dft_sub(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) {
        let mut res = base_dft_mut(res);
        super::vec_znx_dft::vec_znx_dft_sub::<NTT3x42IfmaRayonExecutor>(
            &mut res,
            res_col,
            &base_dft_ref(a),
            a_col,
            &base_dft_ref(b),
            b_col,
        )
    }

    fn vec_znx_dft_sub_assign(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let mut res = base_dft_mut(res);
        super::vec_znx_dft::vec_znx_dft_sub_assign::<NTT3x42IfmaRayonExecutor>(&mut res, res_col, &base_dft_ref(a), a_col)
    }

    fn vec_znx_dft_sub_negate_assign(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let mut res = base_dft_mut(res);
        super::vec_znx_dft::vec_znx_dft_sub_negate_assign::<NTT3x42IfmaRayonExecutor>(&mut res, res_col, &base_dft_ref(a), a_col)
    }

    fn vec_znx_dft_copy(
        _module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let mut res = base_dft_mut(res);
        super::vec_znx_dft::vec_znx_dft_copy::<NTT3x42IfmaRayonExecutor>(step, offset, &mut res, res_col, &base_dft_ref(a), a_col)
    }

    fn vec_znx_dft_zero(_module: &Module<Self>, res: &mut VecZnxDftBackendMut<'_, Self>, res_col: usize) {
        let mut res = base_dft_mut(res);
        super::vec_znx_dft::vec_znx_dft_zero::<NTT3x42IfmaRayonExecutor>(&mut res, res_col)
    }

    type AutomorphismPlan = <NTT3x42Ifma as HalVecZnxDftImpl<NTT3x42Ifma>>::AutomorphismPlan;

    fn vec_znx_dft_automorphism_plan(module: &Module<Self>, p: i64) -> Self::AutomorphismPlan {
        <NTT3x42Ifma as HalVecZnxDftImpl<NTT3x42Ifma>>::vec_znx_dft_automorphism_plan(base_module(module), p)
    }

    fn vec_znx_dft_automorphism_with_plan(
        _module: &Module<Self>,
        plan: &Self::AutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let mut res = base_dft_mut(res);
        super::vec_znx_dft::vec_znx_dft_automorphism::<NTT3x42IfmaRayonExecutor>(plan, &mut res, res_col, &base_dft_ref(a), a_col)
    }

    fn vec_znx_dft_automorphism_add_with_plan(
        _module: &Module<Self>,
        plan: &Self::AutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let mut res = base_dft_mut(res);
        super::vec_znx_dft::vec_znx_dft_automorphism_add::<NTT3x42IfmaRayonExecutor>(
            plan,
            &mut res,
            res_col,
            &base_dft_ref(a),
            a_col,
        );
    }
}

unsafe impl HalSvpImpl<NTT3x42IfmaRayon> for NTT3x42IfmaRayon {
    fn svp_prepare(
        module: &Module<Self>,
        res: &mut SvpPPolBackendMut<'_, Self>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <NTT3x42Ifma as HalSvpImpl<NTT3x42Ifma>>::svp_prepare(base_module(module), &mut base_svp_mut(res), res_col, a, a_col)
    }

    fn svp_ppol_copy_backend(
        module: &Module<Self>,
        res: &mut SvpPPolBackendMut<'_, Self>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <NTT3x42Ifma as HalSvpImpl<NTT3x42Ifma>>::svp_ppol_copy_backend(
            base_module(module),
            &mut base_svp_mut(res),
            res_col,
            &base_svp_ref(a),
            a_col,
        )
    }

    fn svp_apply_dft(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
    ) {
        super::svp::svp_apply_dft::<NTT3x42IfmaRayonExecutor>(
            base_module(module),
            &mut base_dft_mut(res),
            res_col,
            &base_svp_ref(a),
            a_col,
            b,
            b_col,
        )
    }

    fn svp_apply_dft_to_dft(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) {
        super::svp::svp_apply_dft_to_dft::<NTT3x42IfmaRayonExecutor>(
            base_module(module),
            &mut base_dft_mut(res),
            res_col,
            &base_svp_ref(a),
            a_col,
            &base_dft_ref(b),
            b_col,
        )
    }

    fn svp_apply_dft_to_dft_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self>,
        a_col: usize,
    ) {
        super::svp::svp_apply_dft_to_dft_assign::<NTT3x42IfmaRayonExecutor>(
            base_module(module),
            &mut base_dft_mut(res),
            res_col,
            &base_svp_ref(a),
            a_col,
        )
    }
}

unsafe impl HalVmpImpl<NTT3x42IfmaRayon> for NTT3x42IfmaRayon {
    fn vmp_prepare_tmp_bytes(module: &Module<Self>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::PREPARE)
            * <NTT3x42Ifma as HalVmpImpl<NTT3x42Ifma>>::vmp_prepare_tmp_bytes(base_module(module), rows, cols_in, cols_out, size)
    }

    fn vmp_prepare(
        module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &MatZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = super::vmp::vmp_prepare_tmp_bytes_ifma(module.n());
        let rows = a.cols_in() * a.rows();
        let workers = poulpy_cpu_rayon::workers_within(
            rows.min(<Self as poulpy_hal::execution::ScratchWorkers>::PREPARE),
            per_worker,
            scratch.available(),
        );
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), workers * per_worker / size_of::<u64>());
        super::vmp::vmp_prepare_ifma::<NTT3x42IfmaRayonExecutor>(base_module(module), &mut base_vmp_mut(res), a, tmp);
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
        <NTT3x42Ifma as HalVmpImpl<NTT3x42Ifma>>::vmp_apply_dft_tmp_bytes(
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
        poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::VMP)
            * <NTT3x42Ifma as HalVmpImpl<NTT3x42Ifma>>::vmp_apply_dft_to_dft_tmp_bytes(
                base_module(module),
                res_size,
                a_size,
                b_rows,
                b_cols_in,
                b_cols_out,
                b_size,
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
        let per_worker = super::vmp::vmp_apply_tmp_bytes_ifma(a.size(), b.rows(), b.cols_in());
        let bytes = poulpy_cpu_rayon::workers_within(
            <Self as poulpy_hal::execution::ScratchWorkers>::VMP,
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        if NTT3x42IfmaRayonExecutor::should_serialize_inner() {
            super::vmp::vmp_apply_dft_to_dft_ifma::<SerialTaskExecutor>(
                base_module(module),
                &mut base_dft_mut(res),
                &base_dft_ref(a),
                &base_vmp_ref(b),
                limb_offset,
                tmp,
            )
        } else {
            super::vmp::vmp_apply_dft_to_dft_ifma::<NTT3x42IfmaRayonExecutor>(
                base_module(module),
                &mut base_dft_mut(res),
                &base_dft_ref(a),
                &base_vmp_ref(b),
                limb_offset,
                tmp,
            )
        }
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
        poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::VMP)
            * <NTT3x42Ifma as HalVmpImpl<NTT3x42Ifma>>::vmp_apply_dft_to_dft_accumulate_tmp_bytes(
                base_module(module),
                res_size,
                a_size,
                b_rows,
                b_cols_in,
                b_cols_out,
                b_size,
            )
    }

    fn vmp_apply_dft_to_dft_accumulate(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = super::vmp::vmp_apply_tmp_bytes_ifma(a.size(), b.rows(), b.cols_in());
        let bytes = poulpy_cpu_rayon::workers_within(
            <Self as poulpy_hal::execution::ScratchWorkers>::VMP,
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        if NTT3x42IfmaRayonExecutor::should_serialize_inner() {
            super::vmp::vmp_apply_dft_to_dft_accumulate_ifma::<SerialTaskExecutor>(
                base_module(module),
                &mut base_dft_mut(res),
                &base_dft_ref(a),
                &base_vmp_ref(b),
                limb_offset,
                tmp,
            )
        } else {
            super::vmp::vmp_apply_dft_to_dft_accumulate_ifma::<NTT3x42IfmaRayonExecutor>(
                base_module(module),
                &mut base_dft_mut(res),
                &base_dft_ref(a),
                &base_vmp_ref(b),
                limb_offset,
                tmp,
            )
        }
    }

    fn vmp_zero(module: &Module<Self>, res: &mut VmpPMatBackendMut<'_, Self>) {
        <NTT3x42Ifma as HalVmpImpl<NTT3x42Ifma>>::vmp_zero(base_module(module), &mut base_vmp_mut(res))
    }
}

unsafe impl poulpy_core::oep::GGLWEProductDigitsStridedImpl<NTT3x42IfmaRayon> for NTT3x42IfmaRayon {
    fn gglwe_product_digits_strided_tmp_bytes(
        _module: &Module<Self>,
        _res_size: usize,
        a_cols: usize,
        a_size: usize,
        dsize: usize,
        pmat_rows: usize,
        pmat_cols_in: usize,
        _pmat_cols_out: usize,
        _pmat_size: usize,
    ) -> usize {
        poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::VMP)
            * super::vmp::vmp_apply_digits_strided_tmp_bytes_ifma(a_cols, a_size, dsize, pmat_rows, pmat_cols_in)
    }

    fn gglwe_product_digits_strided(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        dsize: usize,
        product_limbs: usize,
        pmat: &VmpPMatBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = Self::gglwe_product_digits_strided_tmp_bytes(
            module,
            res.size(),
            a.cols(),
            a.size(),
            dsize,
            pmat.rows(),
            pmat.cols_in(),
            pmat.cols_out(),
            pmat.size(),
        );
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        super::vmp::vmp_apply_dft_to_dft_digits_strided_ifma::<NTT3x42IfmaRayonExecutor>(
            base_module(module),
            &mut base_dft_mut(res),
            &base_dft_ref(a),
            dsize,
            product_limbs,
            &base_vmp_ref(pmat),
            tmp,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn vmp_apply_digits_strided_known_zero_prefix(
    module: &Module<NTT3x42IfmaRayon>,
    res: &mut VecZnxDftBackendMut<'_, NTT3x42IfmaRayon>,
    a: &VecZnxDftBackendRef<'_, NTT3x42IfmaRayon>,
    dsize: usize,
    zero_prefix: usize,
    product_limbs: usize,
    pmat: &VmpPMatBackendRef<'_, NTT3x42IfmaRayon>,
    scratch: &mut ScratchArena<'_, NTT3x42IfmaRayon>,
) {
    let bytes = <NTT3x42IfmaRayon as poulpy_core::oep::GGLWEProductDigitsStridedImpl<NTT3x42IfmaRayon>>::gglwe_product_digits_strided_tmp_bytes(
        module,
        res.size(),
        a.cols(),
        a.size(),
        dsize,
        pmat.rows(),
        pmat.cols_in(),
        pmat.cols_out(),
        pmat.size(),
    );
    let (tmp, _) = crate::hal_impl::take_host_typed::<NTT3x42IfmaRayon, u64>(scratch.borrow(), bytes / size_of::<u64>());
    super::vmp::vmp_apply_dft_to_dft_digits_strided_ifma_known_zero_prefix::<NTT3x42IfmaRayonExecutor>(
        &mut base_dft_mut(res),
        &base_dft_ref(a),
        dsize,
        product_limbs,
        &base_vmp_ref(pmat),
        zero_prefix,
        tmp,
    );
}

unsafe impl HalConvolutionImpl<NTT3x42IfmaRayon> for NTT3x42IfmaRayon {
    fn cnv_prepare_left_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
        poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::PREPARE)
            * <NTT3x42Ifma as HalConvolutionImpl<NTT3x42Ifma>>::cnv_prepare_left_tmp_bytes(base_module(module), res_size, a_size)
    }

    fn cnv_prepare_left(
        module: &Module<Self>,
        res: &mut CnvPVecLBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        mask: i64,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = super::convolution::cnv_prepare_left_tmp_bytes(module.n());
        let bytes = poulpy_cpu_rayon::workers_within(
            res.size().min(<Self as poulpy_hal::execution::ScratchWorkers>::PREPARE),
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        super::convolution::cnv_prepare_left::<NTT3x42IfmaRayonExecutor>(
            base_module(module),
            &mut base_cnv_l_mut(res),
            a,
            mask,
            tmp,
        )
    }

    fn cnv_prepare_right_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
        poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::PREPARE)
            * <NTT3x42Ifma as HalConvolutionImpl<NTT3x42Ifma>>::cnv_prepare_right_tmp_bytes(base_module(module), res_size, a_size)
    }

    fn cnv_prepare_right(
        module: &Module<Self>,
        res: &mut CnvPVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        mask: i64,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = super::convolution::cnv_prepare_right_tmp_bytes(module.n());
        let bytes = poulpy_cpu_rayon::workers_within(
            res.size().min(<Self as poulpy_hal::execution::ScratchWorkers>::PREPARE),
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        super::convolution::cnv_prepare_right::<NTT3x42IfmaRayonExecutor>(
            base_module(module),
            &mut base_cnv_r_mut(res),
            a,
            mask,
            tmp,
        )
    }

    fn cnv_apply_dft_tmp_bytes(module: &Module<Self>, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize {
        poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::APPLY)
            * <NTT3x42Ifma as HalConvolutionImpl<NTT3x42Ifma>>::cnv_apply_dft_tmp_bytes(
                base_module(module),
                cnv_offset,
                res_size,
                a_size,
                b_size,
            )
    }

    fn cnv_by_const_apply_tmp_bytes(
        module: &Module<Self>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        <NTT3x42Ifma as HalConvolutionImpl<NTT3x42Ifma>>::cnv_by_const_apply_tmp_bytes(
            base_module(module),
            cnv_offset,
            res_size,
            a_size,
            b_size,
        )
    }

    fn cnv_by_const_apply(
        _module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
        b_coeff: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = super::convolution::cnv_by_const_apply_tmp_bytes(res.size(), a.size(), b.size());
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        if NTT3x42IfmaRayonExecutor::should_serialize_inner() {
            super::convolution::cnv_by_const_apply::<SerialTaskExecutor>(
                cnv_offset,
                &mut base_big_mut(res),
                res_col,
                a,
                a_col,
                b,
                b_col,
                b_coeff,
                tmp,
            )
        } else {
            super::convolution::cnv_by_const_apply::<NTT3x42IfmaRayonExecutor>(
                cnv_offset,
                &mut base_big_mut(res),
                res_col,
                a,
                a_col,
                b,
                b_col,
                b_coeff,
                tmp,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_by_const_apply_add(
        _module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
        b_coeff: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = super::convolution::cnv_by_const_apply_tmp_bytes(res.size(), a.size(), b.size());
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        if NTT3x42IfmaRayonExecutor::should_serialize_inner() {
            super::convolution::cnv_by_const_apply_add::<SerialTaskExecutor>(
                cnv_offset,
                &mut base_big_mut(res),
                res_col,
                a,
                a_col,
                b,
                b_col,
                b_coeff,
                tmp,
            )
        } else {
            super::convolution::cnv_by_const_apply_add::<NTT3x42IfmaRayonExecutor>(
                cnv_offset,
                &mut base_big_mut(res),
                res_col,
                a,
                a_col,
                b,
                b_col,
                b_coeff,
                tmp,
            )
        }
    }

    fn cnv_apply_dft(
        _module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, Self>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, Self>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = super::convolution::cnv_apply_dft_ifma_tmp_bytes(res.size(), a.size(), b.size());
        let bytes = poulpy_cpu_rayon::workers_within(
            <Self as poulpy_hal::execution::ScratchWorkers>::APPLY,
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        unsafe {
            super::convolution::cnv_apply_dft_ifma::<NTT3x42IfmaRayonExecutor>(
                &mut base_dft_mut(res),
                cnv_offset,
                res_col,
                &base_cnv_l_ref(a),
                a_col,
                &base_cnv_r_ref(b),
                b_col,
                tmp,
            );
        }
    }

    fn cnv_apply_dft_accumulate(
        _module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, Self>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, Self>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = super::convolution::cnv_apply_dft_ifma_tmp_bytes(res.size(), a.size(), b.size());
        let bytes = poulpy_cpu_rayon::workers_within(
            <Self as poulpy_hal::execution::ScratchWorkers>::APPLY,
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        unsafe {
            super::convolution::cnv_apply_dft_accumulate_ifma::<NTT3x42IfmaRayonExecutor>(
                &mut base_dft_mut(res),
                cnv_offset,
                res_col,
                &base_cnv_l_ref(a),
                a_col,
                &base_cnv_r_ref(b),
                b_col,
                tmp,
            );
        }
    }

    fn cnv_accumulate_dft_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::APPLY)
            * super::convolution::cnv_accumulate_dft_ifma_tmp_bytes(res_size, a_size, b_size)
    }

    fn cnv_accumulate_dft<'a>(
        _module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        terms: &[CnvDftAccTerm<'a, Self>],
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Self: HalVecZnxDftImpl<Self> + 'a,
    {
        let base_terms: Vec<_> = terms
            .iter()
            .map(|term| CnvDftAccTerm {
                a: base_cnv_l_ref(&term.a),
                a_col: term.a_col,
                b: base_cnv_r_ref(&term.b),
                b_col: term.b_col,
            })
            .collect();
        let a_size = base_terms.iter().map(|term| term.a.size()).max().unwrap_or(0);
        let b_size = base_terms.iter().map(|term| term.b.size()).max().unwrap_or(0);
        let per_worker = super::convolution::cnv_accumulate_dft_ifma_tmp_bytes(res.size(), a_size, b_size);
        let bytes = poulpy_cpu_rayon::workers_within(
            <Self as poulpy_hal::execution::ScratchWorkers>::APPLY,
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        unsafe {
            super::convolution::cnv_accumulate_dft_ifma::<NTT3x42IfmaRayonExecutor>(
                &mut base_dft_mut(res),
                cnv_offset,
                res_col,
                &base_terms,
                tmp,
            );
        }
    }

    fn cnv_pairwise_apply_dft_tmp_bytes(
        module: &Module<Self>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::APPLY)
            * <NTT3x42Ifma as HalConvolutionImpl<NTT3x42Ifma>>::cnv_pairwise_apply_dft_tmp_bytes(
                base_module(module),
                cnv_offset,
                res_size,
                a_size,
                b_size,
            )
    }

    fn cnv_pairwise_apply_dft(
        _module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, Self>,
        b: &CnvPVecRBackendRef<'_, Self>,
        i: usize,
        j: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = super::convolution::cnv_pairwise_apply_dft_ifma_tmp_bytes(res.size(), a.size(), b.size());
        let bytes = poulpy_cpu_rayon::workers_within(
            <Self as poulpy_hal::execution::ScratchWorkers>::APPLY,
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        unsafe {
            super::convolution::cnv_pairwise_apply_dft_ifma::<NTT3x42IfmaRayonExecutor>(
                &mut base_dft_mut(res),
                cnv_offset,
                res_col,
                &base_cnv_l_ref(a),
                &base_cnv_r_ref(b),
                i,
                j,
                tmp,
            );
        }
    }

    fn cnv_prepare_self_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
        poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::PREPARE)
            * <NTT3x42Ifma as HalConvolutionImpl<NTT3x42Ifma>>::cnv_prepare_self_tmp_bytes(base_module(module), res_size, a_size)
    }

    fn cnv_prepare_self(
        module: &Module<Self>,
        left: &mut CnvPVecLBackendMut<'_, Self>,
        right: &mut CnvPVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        mask: i64,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = super::convolution::cnv_prepare_self_tmp_bytes(module.n());
        let bytes = poulpy_cpu_rayon::workers_within(
            left.size().min(<Self as poulpy_hal::execution::ScratchWorkers>::PREPARE),
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        super::convolution::cnv_prepare_self::<NTT3x42IfmaRayonExecutor>(
            base_module(module),
            &mut base_cnv_l_mut(left),
            &mut base_cnv_r_mut(right),
            a,
            mask,
            tmp,
        )
    }
}

impl poulpy_hal::execution::ScratchWorkers for NTT3x42IfmaRayon {
    const PREPARE: usize = 4;
    const APPLY: usize = 8;
    const VMP: usize = 8;
    const IDFT: usize = 8;
}

impl poulpy_cpu_rayon::RayonTuning for NTT3x42IfmaRayon {
    const COEFF_MIN_LEN: usize = 1 << 15;
    const COEFF_MIN_TASK: usize = 1 << 13;
    const NORMALIZE_MIN_TASK: usize = 1 << 12;
}

#[cfg(test)]
mod tests {
    use poulpy_cpu_ref::reference::znx::{ZnxAdd, ZnxMulPowerOfTwo};

    use super::NTT3x42IfmaRayon;

    #[test]
    fn parallel_coefficient_ops_match_wrapping_arithmetic() {
        let len = (1 << 15) + 17;
        let a: Vec<i64> = (0..len).map(|i| i as i64 * 17 - 31).collect();
        let b: Vec<i64> = (0..len).map(|i| i as i64 * -7 + 19).collect();
        let mut actual = vec![0; len];
        <NTT3x42IfmaRayon as ZnxAdd>::znx_add(&mut actual, &a, &b);
        let expected: Vec<_> = a.iter().zip(&b).map(|(&a, &b)| a.wrapping_add(b)).collect();
        assert_eq!(actual, expected);

        <NTT3x42IfmaRayon as ZnxMulPowerOfTwo>::znx_mul_power_of_two(11, &mut actual, &a);
        let expected: Vec<_> = a.iter().map(|&a| a.wrapping_mul(1 << 11)).collect();
        assert_eq!(actual, expected);
    }
}
