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
    layouts::{
        CnvPVecL, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecR, CnvPVecRBackendMut, CnvPVecRBackendRef, DataView,
        DataViewMut, MatZnxBackendRef, Module, NoiseInfos, ScalarZnxBackendRef, ScratchArena, SvpPPol, SvpPPolBackendMut,
        SvpPPolBackendRef, VecZnxBackendMut, VecZnxBackendRef, VecZnxBig, VecZnxBigBackendMut, VecZnxDft, VecZnxDftBackendMut,
        VecZnxDftBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef, VmpPMat, VmpPMatBackendMut, VmpPMatBackendRef,
        ZnxView, ZnxViewMut,
    },
    oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
};

use super::{NTT3x42Ifma, NTT3x42IfmaRayon, NTT3x42IfmaRayonExecutor};

fn base_module(module: &Module<NTT3x42IfmaRayon>) -> &Module<NTT3x42Ifma> {
    module.reinterpret()
}

fn base_dft_ref<'a>(a: &'a VecZnxDftBackendRef<'_, NTT3x42IfmaRayon>) -> VecZnxDftBackendRef<'a, NTT3x42Ifma> {
    VecZnxDft::from_data(&**a.data(), a.n(), a.cols(), a.size())
}

fn base_dft_mut<'a>(a: &'a mut VecZnxDftBackendMut<'_, NTT3x42IfmaRayon>) -> VecZnxDftBackendMut<'a, NTT3x42Ifma> {
    let (n, cols, size) = (a.n(), a.cols(), a.size());
    VecZnxDft::from_data(&mut **a.data_mut(), n, cols, size)
}

fn base_big_mut<'a>(a: &'a mut VecZnxBigBackendMut<'_, NTT3x42IfmaRayon>) -> VecZnxBigBackendMut<'a, NTT3x42Ifma> {
    let (n, cols, size) = (a.n(), a.cols(), a.size());
    VecZnxBig::from_data(&mut **a.data_mut(), n, cols, size)
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

fn base_cnv_l_ref<'a>(a: &'a CnvPVecLBackendRef<'_, NTT3x42IfmaRayon>) -> CnvPVecLBackendRef<'a, NTT3x42Ifma> {
    CnvPVecL::from_data(&**a.data(), a.n(), a.cols(), a.size())
}

fn base_cnv_l_mut<'a>(a: &'a mut CnvPVecLBackendMut<'_, NTT3x42IfmaRayon>) -> CnvPVecLBackendMut<'a, NTT3x42Ifma> {
    let (n, cols, size) = (a.n(), a.cols(), a.size());
    CnvPVecL::from_data(&mut **a.data_mut(), n, cols, size)
}

fn base_cnv_r_ref<'a>(a: &'a CnvPVecRBackendRef<'_, NTT3x42IfmaRayon>) -> CnvPVecRBackendRef<'a, NTT3x42Ifma> {
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

fn parallel_chunk_len(len: usize) -> Option<usize> {
    // TODO: Replace this provisional cutoff with a benchmark-derived work model.
    if len < 1 << 15 || ::rayon::current_num_threads() < 2 {
        None
    } else {
        Some(len.div_ceil(::rayon::current_num_threads()).next_multiple_of(64))
    }
}

fn parallel_limb_tasks(count: usize) -> bool {
    count > 1 && ::rayon::current_num_threads() > 1
}

macro_rules! parallel_binary {
    ($trait:ident, $method:ident) => {
        impl $trait for NTT3x42IfmaRayon {
            fn $method(res: &mut [i64], a: &[i64], b: &[i64]) {
                let Some(chunk) = parallel_chunk_len(res.len()) else {
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
                let Some(chunk) = parallel_chunk_len(res.len()) else {
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
                let Some(chunk) = parallel_chunk_len(res.len()) else {
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
                let Some(chunk) = parallel_chunk_len(res.len()) else {
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
        let Some(chunk) = parallel_chunk_len(res.len()) else {
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
    poulpy_cpu_ref::hal_impl_vec_znx!();

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
    poulpy_cpu_ref::hal_impl_vec_znx_big!(NTT4x30VecZnxBigDefault);
}

unsafe impl HalVecZnxDftImpl<NTT3x42IfmaRayon> for NTT3x42IfmaRayon {
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
        <NTT3x42Ifma as HalVecZnxDftImpl<NTT3x42Ifma>>::vec_znx_idft_apply_tmp_bytes(base_module(module))
    }

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
            res.raw_mut().par_chunks_mut(n * res_cols).enumerate().for_each_init(
                || vec![0u64; 3 * n],
                |scratch, (j, group)| {
                    let dst = &mut group[n * res_col..][..n];
                    let src = (j < min_size).then(|| &src[2 * n * (j * a_cols + a_col)..][..2 * n]);
                    super::vec_znx_dft::vec_znx_idft_apply_limb(base_module(module), dst, src, scratch);
                },
            );
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
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) {
        let mut res = base_dft_mut(res);
        <NTT3x42Ifma as HalVecZnxDftImpl<NTT3x42Ifma>>::vec_znx_dft_add_into(
            base_module(module),
            &mut res,
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
        let mut res = base_dft_mut(res);
        <NTT3x42Ifma as HalVecZnxDftImpl<NTT3x42Ifma>>::vec_znx_dft_add_scaled_assign(
            base_module(module),
            &mut res,
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
        let mut res = base_dft_mut(res);
        <NTT3x42Ifma as HalVecZnxDftImpl<NTT3x42Ifma>>::vec_znx_dft_add_assign(
            base_module(module),
            &mut res,
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
        let mut res = base_dft_mut(res);
        <NTT3x42Ifma as HalVecZnxDftImpl<NTT3x42Ifma>>::vec_znx_dft_sub(
            base_module(module),
            &mut res,
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
        let mut res = base_dft_mut(res);
        <NTT3x42Ifma as HalVecZnxDftImpl<NTT3x42Ifma>>::vec_znx_dft_sub_assign(
            base_module(module),
            &mut res,
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
        let mut res = base_dft_mut(res);
        <NTT3x42Ifma as HalVecZnxDftImpl<NTT3x42Ifma>>::vec_znx_dft_sub_negate_assign(
            base_module(module),
            &mut res,
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
        let mut res = base_dft_mut(res);
        <NTT3x42Ifma as HalVecZnxDftImpl<NTT3x42Ifma>>::vec_znx_dft_copy(
            base_module(module),
            step,
            offset,
            &mut res,
            res_col,
            &base_dft_ref(a),
            a_col,
        )
    }

    fn vec_znx_dft_zero(module: &Module<Self>, res: &mut VecZnxDftBackendMut<'_, Self>, res_col: usize) {
        let mut res = base_dft_mut(res);
        <NTT3x42Ifma as HalVecZnxDftImpl<NTT3x42Ifma>>::vec_znx_dft_zero(base_module(module), &mut res, res_col)
    }

    type AutomorphismPlan = <NTT3x42Ifma as HalVecZnxDftImpl<NTT3x42Ifma>>::AutomorphismPlan;

    fn vec_znx_dft_automorphism_plan(module: &Module<Self>, p: i64) -> Self::AutomorphismPlan {
        <NTT3x42Ifma as HalVecZnxDftImpl<NTT3x42Ifma>>::vec_znx_dft_automorphism_plan(base_module(module), p)
    }

    fn vec_znx_dft_automorphism_with_plan(
        module: &Module<Self>,
        plan: &Self::AutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let mut res = base_dft_mut(res);
        <NTT3x42Ifma as HalVecZnxDftImpl<NTT3x42Ifma>>::vec_znx_dft_automorphism_with_plan(
            base_module(module),
            plan,
            &mut res,
            res_col,
            &base_dft_ref(a),
            a_col,
        )
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
        <NTT3x42Ifma as HalSvpImpl<NTT3x42Ifma>>::svp_apply_dft(
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
        <NTT3x42Ifma as HalSvpImpl<NTT3x42Ifma>>::svp_apply_dft_to_dft(
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
        <NTT3x42Ifma as HalSvpImpl<NTT3x42Ifma>>::svp_apply_dft_to_dft_assign(
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
        <NTT3x42Ifma as HalVmpImpl<NTT3x42Ifma>>::vmp_prepare_tmp_bytes(base_module(module), rows, cols_in, cols_out, size)
    }

    fn vmp_prepare(
        module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &MatZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow().into_backend::<NTT3x42Ifma>();
        <NTT3x42Ifma as HalVmpImpl<NTT3x42Ifma>>::vmp_prepare(base_module(module), &mut base_vmp_mut(res), a, &mut scratch)
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
        <NTT3x42Ifma as HalVmpImpl<NTT3x42Ifma>>::vmp_apply_dft_to_dft_tmp_bytes(
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
        let bytes = super::vmp::vmp_apply_tmp_bytes_ifma(a.size(), b.rows(), b.cols_in());
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        super::vmp::vmp_apply_dft_to_dft_ifma::<NTT3x42IfmaRayonExecutor>(
            base_module(module),
            &mut base_dft_mut(res),
            &base_dft_ref(a),
            &base_vmp_ref(b),
            limb_offset,
            tmp,
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
        <NTT3x42Ifma as HalVmpImpl<NTT3x42Ifma>>::vmp_apply_dft_to_dft_accumulate_tmp_bytes(
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
        let bytes = super::vmp::vmp_apply_tmp_bytes_ifma(a.size(), b.rows(), b.cols_in());
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        super::vmp::vmp_apply_dft_to_dft_accumulate_ifma::<NTT3x42IfmaRayonExecutor>(
            base_module(module),
            &mut base_dft_mut(res),
            &base_dft_ref(a),
            &base_vmp_ref(b),
            limb_offset,
            tmp,
        )
    }

    fn vmp_apply_dft_to_dft_digits_strided(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        dsize: usize,
        pmat: &VmpPMatBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: poulpy_hal::api::VecZnxDftCopy<Self>,
    {
        let bytes = super::vmp::vmp_apply_digits_strided_tmp_bytes_ifma(a.cols(), a.size(), dsize, pmat.rows(), pmat.cols_in());
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        super::vmp::vmp_apply_dft_to_dft_digits_strided_ifma::<NTT3x42IfmaRayonExecutor>(
            base_module(module),
            &mut base_dft_mut(res),
            &base_dft_ref(a),
            dsize,
            &base_vmp_ref(pmat),
            tmp,
        )
    }

    fn vmp_zero(module: &Module<Self>, res: &mut VmpPMatBackendMut<'_, Self>) {
        <NTT3x42Ifma as HalVmpImpl<NTT3x42Ifma>>::vmp_zero(base_module(module), &mut base_vmp_mut(res))
    }
}

unsafe impl HalConvolutionImpl<NTT3x42IfmaRayon> for NTT3x42IfmaRayon {
    fn cnv_prepare_left_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
        <NTT3x42Ifma as HalConvolutionImpl<NTT3x42Ifma>>::cnv_prepare_left_tmp_bytes(base_module(module), res_size, a_size)
    }

    fn cnv_prepare_left(
        module: &Module<Self>,
        res: &mut CnvPVecLBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        mask: i64,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow().into_backend::<NTT3x42Ifma>();
        <NTT3x42Ifma as HalConvolutionImpl<NTT3x42Ifma>>::cnv_prepare_left(
            base_module(module),
            &mut base_cnv_l_mut(res),
            a,
            mask,
            &mut scratch,
        )
    }

    fn cnv_prepare_right_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
        <NTT3x42Ifma as HalConvolutionImpl<NTT3x42Ifma>>::cnv_prepare_right_tmp_bytes(base_module(module), res_size, a_size)
    }

    fn cnv_prepare_right(
        module: &Module<Self>,
        res: &mut CnvPVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        mask: i64,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow().into_backend::<NTT3x42Ifma>();
        <NTT3x42Ifma as HalConvolutionImpl<NTT3x42Ifma>>::cnv_prepare_right(
            base_module(module),
            &mut base_cnv_r_mut(res),
            a,
            mask,
            &mut scratch,
        )
    }

    fn cnv_apply_dft_tmp_bytes(module: &Module<Self>, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize {
        <NTT3x42Ifma as HalConvolutionImpl<NTT3x42Ifma>>::cnv_apply_dft_tmp_bytes(
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
        module: &Module<Self>,
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
        let mut scratch = scratch.borrow().into_backend::<NTT3x42Ifma>();
        <NTT3x42Ifma as HalConvolutionImpl<NTT3x42Ifma>>::cnv_by_const_apply(
            base_module(module),
            cnv_offset,
            &mut base_big_mut(res),
            res_col,
            a,
            a_col,
            b,
            b_col,
            b_coeff,
            &mut scratch,
        )
    }

    fn cnv_apply_dft(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, Self>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, Self>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow().into_backend::<NTT3x42Ifma>();
        <NTT3x42Ifma as HalConvolutionImpl<NTT3x42Ifma>>::cnv_apply_dft(
            base_module(module),
            cnv_offset,
            &mut base_dft_mut(res),
            res_col,
            &base_cnv_l_ref(a),
            a_col,
            &base_cnv_r_ref(b),
            b_col,
            &mut scratch,
        )
    }

    fn cnv_apply_dft_accumulate(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, Self>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, Self>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow().into_backend::<NTT3x42Ifma>();
        <NTT3x42Ifma as HalConvolutionImpl<NTT3x42Ifma>>::cnv_apply_dft_accumulate(
            base_module(module),
            cnv_offset,
            &mut base_dft_mut(res),
            res_col,
            &base_cnv_l_ref(a),
            a_col,
            &base_cnv_r_ref(b),
            b_col,
            &mut scratch,
        )
    }

    fn cnv_pairwise_apply_dft_tmp_bytes(
        module: &Module<Self>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        <NTT3x42Ifma as HalConvolutionImpl<NTT3x42Ifma>>::cnv_pairwise_apply_dft_tmp_bytes(
            base_module(module),
            cnv_offset,
            res_size,
            a_size,
            b_size,
        )
    }

    fn cnv_pairwise_apply_dft(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, Self>,
        b: &CnvPVecRBackendRef<'_, Self>,
        i: usize,
        j: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow().into_backend::<NTT3x42Ifma>();
        <NTT3x42Ifma as HalConvolutionImpl<NTT3x42Ifma>>::cnv_pairwise_apply_dft(
            base_module(module),
            cnv_offset,
            &mut base_dft_mut(res),
            res_col,
            &base_cnv_l_ref(a),
            &base_cnv_r_ref(b),
            i,
            j,
            &mut scratch,
        )
    }

    fn cnv_tensor_rank1_dft_tmp_bytes(
        module: &Module<Self>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        <NTT3x42Ifma as HalConvolutionImpl<NTT3x42Ifma>>::cnv_tensor_rank1_dft_tmp_bytes(
            base_module(module),
            cnv_offset,
            res_size,
            a_size,
            b_size,
        )
    }

    fn cnv_tensor_rank1_dft_is_fused(module: &Module<Self>) -> bool {
        <NTT3x42Ifma as HalConvolutionImpl<NTT3x42Ifma>>::cnv_tensor_rank1_dft_is_fused(base_module(module))
    }

    fn cnv_tensor_rank1_dft(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &CnvPVecLBackendRef<'_, Self>,
        b: &CnvPVecRBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow().into_backend::<NTT3x42Ifma>();
        <NTT3x42Ifma as HalConvolutionImpl<NTT3x42Ifma>>::cnv_tensor_rank1_dft(
            base_module(module),
            cnv_offset,
            &mut base_dft_mut(res),
            &base_cnv_l_ref(a),
            &base_cnv_r_ref(b),
            &mut scratch,
        )
    }

    fn cnv_prepare_self_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
        <NTT3x42Ifma as HalConvolutionImpl<NTT3x42Ifma>>::cnv_prepare_self_tmp_bytes(base_module(module), res_size, a_size)
    }

    fn cnv_prepare_self(
        module: &Module<Self>,
        left: &mut CnvPVecLBackendMut<'_, Self>,
        right: &mut CnvPVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        mask: i64,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow().into_backend::<NTT3x42Ifma>();
        <NTT3x42Ifma as HalConvolutionImpl<NTT3x42Ifma>>::cnv_prepare_self(
            base_module(module),
            &mut base_cnv_l_mut(left),
            &mut base_cnv_r_mut(right),
            a,
            mask,
            &mut scratch,
        )
    }
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
