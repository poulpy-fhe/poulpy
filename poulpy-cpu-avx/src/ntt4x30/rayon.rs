//! Rayon-scheduled wrapper for the AVX2 NTT4x30 backend.

use std::mem::size_of;

use bytemuck::{cast_slice, cast_slice_mut};
use rayon::prelude::*;

use poulpy_cpu_ref::{
    hal_defaults::{BigWordHadamardProduct, HalVecZnxDefault, NTT4x30ModuleDefault, NTT4x30VecZnxBigDefault, NTT4x30VmpDefault},
    reference::{
        ntt4x30::{
            I128BigOps, I128NormalizeOps, NttAdd, NttAddAssign, NttCFromB, NttCopy, NttDFTExecute, NttExtract1BlkContiguous,
            NttFromZnx64, NttMulBbb, NttMulBbc, NttMulBbc1ColX2, NttMulBbc2ColsX2, NttNegate, NttNegateAssign, NttPackLeft1BlkX2,
            NttPackRight1BlkX2, NttPairwisePackLeft1BlkX2, NttPairwisePackRight1BlkX2, NttSub, NttSubAssign, NttSubNegateAssign,
            NttToZnx128, NttZero,
            mat_vec::{BbbMeta, BbcMeta},
            ntt::{NttTable, NttTableInv},
            primes::Primes30,
            vec_znx_big::AssignOp,
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
    execution::{SerialTaskExecutor, TaskExecutor},
    layouts::{
        Backend, DataView, DataViewMut, FitsIn, MatZnxBackendRef, Module, NoiseInfos, NormalizationState, Normalized, ScalarZnx,
        ScalarZnxBackendRef, ScratchArena, SvpPPol, SvpPPolBackendMut, SvpPPolBackendRef, VecZnxBackendMut, VecZnxBackendRef,
        VecZnxBig, VecZnxBigBackendMut, VecZnxDft, VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftToBackendMut,
        VecZnxDftToBackendRef, VmpPMat, VmpPMatBackendMut, VmpPMatBackendRef, ZnxView, ZnxViewMut,
    },
    oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
};

use super::{NTT4x30Avx, NTT4x30AvxRayon};
use poulpy_cpu_rayon::{RayonTaskExecutor, SendPtr, parallel_limb_tasks};

poulpy_hal::impl_backend_from!(NTT4x30AvxRayon, NTT4x30Avx, RayonTaskExecutor);

fn base_module(module: &Module<NTT4x30AvxRayon>) -> &Module<NTT4x30Avx> {
    module.reinterpret()
}

fn base_dft_ref<'a>(a: &'a VecZnxDftBackendRef<'_, NTT4x30AvxRayon>) -> VecZnxDftBackendRef<'a, NTT4x30Avx> {
    VecZnxDft::from_data(&**a.data(), a.n(), a.cols(), a.size())
}

fn base_dft_mut<'a>(a: &'a mut VecZnxDftBackendMut<'_, NTT4x30AvxRayon>) -> VecZnxDftBackendMut<'a, NTT4x30Avx> {
    let (n, cols, size) = (a.n(), a.cols(), a.size());
    VecZnxDft::from_data(&mut **a.data_mut(), n, cols, size)
}

fn base_znx_ref<'a, S: NormalizationState>(
    a: &'a VecZnxBackendRef<'_, NTT4x30AvxRayon, S>,
) -> VecZnxBackendRef<'a, NTT4x30Avx, S> {
    poulpy_hal::oep::vec_znx_from_data_like(a, &**a.data())
}

fn base_znx_mut<'a, S: NormalizationState>(
    a: &'a mut VecZnxBackendMut<'_, NTT4x30AvxRayon, S>,
) -> VecZnxBackendMut<'a, NTT4x30Avx, S> {
    poulpy_hal::oep::vec_znx_map_data_mut(a, |d| &mut **d)
}

fn base_scalar_ref<'a>(a: &'a ScalarZnxBackendRef<'_, NTT4x30AvxRayon>) -> ScalarZnxBackendRef<'a, NTT4x30Avx> {
    ScalarZnx::from_data(&**a.data(), a.n(), a.cols())
}

fn base_svp_ref<'a>(a: &'a SvpPPolBackendRef<'_, NTT4x30AvxRayon>) -> SvpPPolBackendRef<'a, NTT4x30Avx> {
    SvpPPol::from_data(&**a.data(), a.n(), a.cols())
}

fn base_svp_mut<'a>(a: &'a mut SvpPPolBackendMut<'_, NTT4x30AvxRayon>) -> SvpPPolBackendMut<'a, NTT4x30Avx> {
    let (n, cols) = (a.n(), a.cols());
    SvpPPol::from_data(&mut **a.data_mut(), n, cols)
}

fn base_big_mut<'a>(a: &'a mut VecZnxBigBackendMut<'_, NTT4x30AvxRayon>) -> VecZnxBigBackendMut<'a, NTT4x30Avx> {
    let (n, cols, size) = (a.n(), a.cols(), a.size());
    VecZnxBig::from_data(&mut **a.data_mut(), n, cols, size)
}

fn base_big_ref<'a>(
    a: &'a poulpy_hal::layouts::VecZnxBigBackendRef<'_, NTT4x30AvxRayon>,
) -> poulpy_hal::layouts::VecZnxBigBackendRef<'a, NTT4x30Avx> {
    VecZnxBig::from_data(&**a.data(), a.n(), a.cols(), a.size())
}

fn base_vmp_ref<'a>(a: &'a VmpPMatBackendRef<'_, NTT4x30AvxRayon>) -> VmpPMatBackendRef<'a, NTT4x30Avx> {
    VmpPMat::from_data(&**a.data(), a.n(), a.rows(), a.cols_in(), a.cols_out(), a.size())
}

fn base_vmp_mut<'a>(a: &'a mut VmpPMatBackendMut<'_, NTT4x30AvxRayon>) -> VmpPMatBackendMut<'a, NTT4x30Avx> {
    let (n, rows, cols_in, cols_out, size) = (a.n(), a.rows(), a.cols_in(), a.cols_out(), a.size());
    VmpPMat::from_data(&mut **a.data_mut(), n, rows, cols_in, cols_out, size)
}

use poulpy_cpu_rayon::parallel_chunk_len;

macro_rules! parallel_binary {
    ($trait:ident, $method:ident) => {
        impl $trait for NTT4x30AvxRayon {
            fn $method(res: &mut [i64], a: &[i64], b: &[i64]) {
                let Some(chunk) = parallel_chunk_len::<Self>(res.len()) else {
                    return <NTT4x30Avx as $trait>::$method(res, a, b);
                };
                res.par_chunks_mut(chunk)
                    .zip(a.par_chunks(chunk))
                    .zip(b.par_chunks(chunk))
                    .for_each(|((res, a), b)| <NTT4x30Avx as $trait>::$method(res, a, b));
            }
        }
    };
}

macro_rules! parallel_assign {
    ($trait:ident, $method:ident) => {
        impl $trait for NTT4x30AvxRayon {
            fn $method(res: &mut [i64], a: &[i64]) {
                let Some(chunk) = parallel_chunk_len::<Self>(res.len()) else {
                    return <NTT4x30Avx as $trait>::$method(res, a);
                };
                res.par_chunks_mut(chunk)
                    .zip(a.par_chunks(chunk))
                    .for_each(|(res, a)| <NTT4x30Avx as $trait>::$method(res, a));
            }
        }
    };
}

macro_rules! parallel_unary {
    ($trait:ident, $method:ident) => {
        impl $trait for NTT4x30AvxRayon {
            fn $method(res: &mut [i64]) {
                let Some(chunk) = parallel_chunk_len::<Self>(res.len()) else {
                    return <NTT4x30Avx as $trait>::$method(res);
                };
                res.par_chunks_mut(chunk)
                    .for_each(|res| <NTT4x30Avx as $trait>::$method(res));
            }
        }
    };
}

macro_rules! parallel_shift {
    ($trait:ident, $method:ident) => {
        impl $trait for NTT4x30AvxRayon {
            fn $method(k: i64, res: &mut [i64], a: &[i64]) {
                let Some(chunk) = parallel_chunk_len::<Self>(res.len()) else {
                    return <NTT4x30Avx as $trait>::$method(k, res, a);
                };
                res.par_chunks_mut(chunk)
                    .zip(a.par_chunks(chunk))
                    .for_each(|(res, a)| <NTT4x30Avx as $trait>::$method(k, res, a));
            }
        }
    };
}

macro_rules! forward_znx {
    ($trait:ident, $method:ident($($arg:ident: $ty:ty),* $(,)?)) => {
        impl $trait for NTT4x30AvxRayon {
            #[inline(always)]
            fn $method($($arg: $ty),*) {
                <NTT4x30Avx as $trait>::$method($($arg),*)
            }
        }
    };
}

macro_rules! forward_znx_const {
    ($trait:ident, $method:ident($($arg:ident: $ty:ty),* $(,)?)) => {
        impl $trait for NTT4x30AvxRayon {
            #[inline(always)]
            fn $method<const OVERWRITE: bool>($($arg: $ty),*) {
                <NTT4x30Avx as $trait>::$method::<OVERWRITE>($($arg),*)
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

impl ZnxMulPowerOfTwoAssign for NTT4x30AvxRayon {
    fn znx_mul_power_of_two_assign(k: i64, res: &mut [i64]) {
        let Some(chunk) = parallel_chunk_len::<Self>(res.len()) else {
            return <NTT4x30Avx as ZnxMulPowerOfTwoAssign>::znx_mul_power_of_two_assign(k, res);
        };
        res.par_chunks_mut(chunk)
            .for_each(|res| <NTT4x30Avx as ZnxMulPowerOfTwoAssign>::znx_mul_power_of_two_assign(k, res));
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

impl NttDFTExecute<NttTable<Primes30>> for NTT4x30AvxRayon {
    fn ntt_dft_execute(table: &NttTable<Primes30>, data: &mut [u64]) {
        <NTT4x30Avx as NttDFTExecute<NttTable<Primes30>>>::ntt_dft_execute(table, data)
    }
}

impl NttDFTExecute<NttTableInv<Primes30>> for NTT4x30AvxRayon {
    fn ntt_dft_execute(table: &NttTableInv<Primes30>, data: &mut [u64]) {
        <NTT4x30Avx as NttDFTExecute<NttTableInv<Primes30>>>::ntt_dft_execute(table, data)
    }
}

impl NttFromZnx64 for NTT4x30AvxRayon {
    fn ntt_from_znx64(res: &mut [u64], a: &[i64]) {
        <NTT4x30Avx as NttFromZnx64>::ntt_from_znx64(res, a)
    }
    fn ntt_from_znx64_masked(res: &mut [u64], a: &[i64], mask: i64) {
        <NTT4x30Avx as NttFromZnx64>::ntt_from_znx64_masked(res, a, mask)
    }
}

impl NttToZnx128 for NTT4x30AvxRayon {
    fn ntt_to_znx128(res: &mut [i128], divisor_is_n: usize, a: &[u64]) {
        <NTT4x30Avx as NttToZnx128>::ntt_to_znx128(res, divisor_is_n, a)
    }
}

macro_rules! forward_ntt_binary {
    ($trait:ident, $method:ident) => {
        impl $trait for NTT4x30AvxRayon {
            fn $method(res: &mut [u64], a: &[u64], b: &[u64]) {
                <NTT4x30Avx as $trait>::$method(res, a, b)
            }
        }
    };
}
macro_rules! forward_ntt_assign {
    ($trait:ident, $method:ident) => {
        impl $trait for NTT4x30AvxRayon {
            fn $method(res: &mut [u64], a: &[u64]) {
                <NTT4x30Avx as $trait>::$method(res, a)
            }
        }
    };
}

forward_ntt_binary!(NttAdd, ntt_add);
forward_ntt_assign!(NttAddAssign, ntt_add_assign);
forward_ntt_binary!(NttSub, ntt_sub);
forward_ntt_assign!(NttSubAssign, ntt_sub_assign);
forward_ntt_assign!(NttSubNegateAssign, ntt_sub_negate_assign);
forward_ntt_assign!(NttNegate, ntt_negate);
impl NttNegateAssign for NTT4x30AvxRayon {
    fn ntt_negate_assign(res: &mut [u64]) {
        <NTT4x30Avx as NttNegateAssign>::ntt_negate_assign(res)
    }
}
impl NttZero for NTT4x30AvxRayon {
    fn ntt_zero(res: &mut [u64]) {
        <NTT4x30Avx as NttZero>::ntt_zero(res)
    }
}
forward_ntt_assign!(NttCopy, ntt_copy);

impl NttMulBbb for NTT4x30AvxRayon {
    fn ntt_mul_bbb(meta: &BbbMeta<Primes30>, ell: usize, res: &mut [u64], a: &[u64], b: &[u64]) {
        <NTT4x30Avx as NttMulBbb>::ntt_mul_bbb(meta, ell, res, a, b)
    }
}
impl NttMulBbc for NTT4x30AvxRayon {
    fn ntt_mul_bbc(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], a: &[u32], b: &[u32]) {
        <NTT4x30Avx as NttMulBbc>::ntt_mul_bbc(meta, ell, res, a, b)
    }
}
impl NttCFromB for NTT4x30AvxRayon {
    fn ntt_c_from_b(n: usize, res: &mut [u32], a: &[u64]) {
        <NTT4x30Avx as NttCFromB>::ntt_c_from_b(n, res, a)
    }
}
impl NttMulBbc1ColX2 for NTT4x30AvxRayon {
    fn ntt_mul_bbc_1col_x2(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], a: &[u32], b: &[u32]) {
        <NTT4x30Avx as NttMulBbc1ColX2>::ntt_mul_bbc_1col_x2(meta, ell, res, a, b)
    }
    fn ntt_mul_bbc_tile4_x2(meta: &BbcMeta<Primes30>, len: usize, res: &mut [u64], a: &[u32], b: &[u32]) {
        <NTT4x30Avx as NttMulBbc1ColX2>::ntt_mul_bbc_tile4_x2(meta, len, res, a, b)
    }
}
impl NttMulBbc2ColsX2 for NTT4x30AvxRayon {
    fn ntt_mul_bbc_2cols_x2(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], a: &[u32], b: &[u32]) {
        <NTT4x30Avx as NttMulBbc2ColsX2>::ntt_mul_bbc_2cols_x2(meta, ell, res, a, b)
    }
}
impl NttExtract1BlkContiguous for NTT4x30AvxRayon {
    fn ntt_extract_1blk_contiguous(n: usize, rows: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
        <NTT4x30Avx as NttExtract1BlkContiguous>::ntt_extract_1blk_contiguous(n, rows, blk, dst, src)
    }
}
impl NttPackLeft1BlkX2 for NTT4x30AvxRayon {
    fn ntt_pack_left_1blk_x2(dst: &mut [u32], a: &[u64], rows: usize, stride: usize, blk: usize) {
        <NTT4x30Avx as NttPackLeft1BlkX2>::ntt_pack_left_1blk_x2(dst, a, rows, stride, blk)
    }
}
impl NttPackRight1BlkX2 for NTT4x30AvxRayon {
    fn ntt_pack_right_1blk_x2(dst: &mut [u32], a: &[u32], rows: usize, stride: usize, blk: usize) {
        <NTT4x30Avx as NttPackRight1BlkX2>::ntt_pack_right_1blk_x2(dst, a, rows, stride, blk)
    }
}
impl NttPairwisePackLeft1BlkX2 for NTT4x30AvxRayon {
    fn ntt_pairwise_pack_left_1blk_x2(dst: &mut [u32], a: &[u64], b: &[u64], rows: usize, stride: usize, blk: usize) {
        <NTT4x30Avx as NttPairwisePackLeft1BlkX2>::ntt_pairwise_pack_left_1blk_x2(dst, a, b, rows, stride, blk)
    }
}
impl NttPairwisePackRight1BlkX2 for NTT4x30AvxRayon {
    fn ntt_pairwise_pack_right_1blk_x2(dst: &mut [u32], a: &[u32], b: &[u32], rows: usize, stride: usize, blk: usize) {
        <NTT4x30Avx as NttPairwisePackRight1BlkX2>::ntt_pairwise_pack_right_1blk_x2(dst, a, b, rows, stride, blk)
    }
}

macro_rules! forward_i128_big {
    ($method:ident($($arg:ident: $ty:ty),* $(,)?)) => {
        fn $method($($arg: $ty),*) { <NTT4x30Avx as I128BigOps>::$method($($arg),*) }
    };
}

impl I128BigOps for NTT4x30AvxRayon {
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

impl I128NormalizeOps for NTT4x30AvxRayon {
    fn nfc_middle_step(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
        <NTT4x30Avx as I128NormalizeOps>::nfc_middle_step(base2k, lsh, res, a, carry)
    }
    fn nfc_middle_step_into<O: AssignOp>(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
        <NTT4x30Avx as I128NormalizeOps>::nfc_middle_step_into::<O>(base2k, lsh, res, a, carry)
    }
    fn nfc_middle_step_assign(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        <NTT4x30Avx as I128NormalizeOps>::nfc_middle_step_assign(base2k, lsh, res, carry)
    }
    fn nfc_final_step_assign(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        <NTT4x30Avx as I128NormalizeOps>::nfc_final_step_assign(base2k, lsh, res, carry)
    }
    fn nfc_final_step_into<O: AssignOp>(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        <NTT4x30Avx as I128NormalizeOps>::nfc_final_step_into::<O>(base2k, lsh, res, carry)
    }
}

impl BigWordHadamardProduct for NTT4x30AvxRayon {
    fn big_word_hadamard_product(res: &mut [i128], a: &[i64], b: &[i64]) {
        <Self as I128BigOps>::i128_hadamard_product_i64(res, a, b)
    }
}

unsafe impl HalVecZnxImpl<NTT4x30AvxRayon> for NTT4x30AvxRayon {
    poulpy_cpu_ref::hal_impl_vec_znx_without_normalize!();

    fn vec_znx_normalize_backend(
        module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self, impl NormalizationState>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self, impl NormalizationState>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let (carry, _) = poulpy_cpu_rayon::take_scratch::<Self, i64>(scratch.borrow(), 3 * module.n());
        poulpy_cpu_rayon::normalize::vec_znx_normalize_par::<NTT4x30Avx, Self>(
            res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry,
        );
    }

    fn vec_znx_normalize_assign_backend(
        module: &Module<Self>,
        base2k: usize,
        a: &mut VecZnxBackendMut<'_, Self, impl NormalizationState>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let (carry, _) = poulpy_cpu_rayon::take_scratch::<Self, i64>(scratch.borrow(), 3 * module.n());
        poulpy_cpu_rayon::normalize::vec_znx_normalize_assign_par::<NTT4x30Avx, Self>(base2k, a, a_col, carry);
    }
    fn vec_znx_transpose_backend<S: NormalizationState>(
        module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self, S>,
        a: &VecZnxBackendRef<'_, Self, impl FitsIn<S>>,
    ) {
        <Self as HalVecZnxDefault<Self>>::vec_znx_transpose_backend_default(module, res, a)
    }
}
unsafe impl HalModuleImpl<NTT4x30AvxRayon> for NTT4x30AvxRayon {
    poulpy_cpu_ref::hal_impl_module!(NTT4x30ModuleDefault);
}
unsafe impl HalVmpImpl<NTT4x30AvxRayon> for NTT4x30AvxRayon {
    fn vmp_apply_dft_tmp_bytes(
        module: &Module<Self>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        let a_dft_size = a_size.min(b_rows);
        <Self as Backend>::bytes_of_vec_znx_dft(module.n(), b_cols_in, a_dft_size)
            + Self::vmp_apply_dft_to_dft_tmp_bytes(module, res_size, a_dft_size, b_rows, b_cols_in, b_cols_out, b_size)
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

    fn vmp_prepare_tmp_bytes(module: &Module<Self>, _rows: usize, _cols_in: usize, _cols_out: usize, _size: usize) -> usize {
        super::vmp::vmp_prepare_tmp_bytes_avx(module.n())
    }

    fn vmp_prepare(
        module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &MatZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = super::vmp::vmp_prepare_tmp_bytes_avx(module.n());
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        super::vmp::vmp_prepare_avx_pm(base_module(module), &mut base_vmp_mut(res), a, tmp);
    }

    fn vmp_apply_dft_to_dft_tmp_bytes(
        _module: &Module<Self>,
        _res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize {
        poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::VMP)
            * super::vmp::vmp_apply_tmp_bytes_avx(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = super::vmp::vmp_apply_tmp_bytes_avx(a.size(), b.rows(), b.cols_in());
        let bytes = poulpy_cpu_rayon::workers_within(
            <Self as poulpy_hal::execution::ScratchWorkers>::VMP,
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        if RayonTaskExecutor::should_serialize_inner() {
            super::vmp::vmp_apply_dft_to_dft_avx::<SerialTaskExecutor>(
                base_module(module),
                &mut base_dft_mut(res),
                &base_dft_ref(a),
                &base_vmp_ref(b),
                limb_offset,
                tmp,
            );
        } else {
            super::vmp::vmp_apply_dft_to_dft_avx::<RayonTaskExecutor>(
                base_module(module),
                &mut base_dft_mut(res),
                &base_dft_ref(a),
                &base_vmp_ref(b),
                limb_offset,
                tmp,
            );
        }
    }

    fn vmp_apply_dft_to_dft_accumulate_tmp_bytes(
        _module: &Module<Self>,
        _res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize {
        poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::VMP)
            * super::vmp::vmp_apply_tmp_bytes_avx(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft_accumulate(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = super::vmp::vmp_apply_tmp_bytes_avx(a.size(), b.rows(), b.cols_in());
        let bytes = poulpy_cpu_rayon::workers_within(
            <Self as poulpy_hal::execution::ScratchWorkers>::VMP,
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        if RayonTaskExecutor::should_serialize_inner() {
            super::vmp::vmp_apply_dft_to_dft_accumulate_avx::<SerialTaskExecutor>(
                base_module(module),
                &mut base_dft_mut(res),
                &base_dft_ref(a),
                &base_vmp_ref(b),
                limb_offset,
                tmp,
            );
        } else {
            super::vmp::vmp_apply_dft_to_dft_accumulate_avx::<RayonTaskExecutor>(
                base_module(module),
                &mut base_dft_mut(res),
                &base_dft_ref(a),
                &base_vmp_ref(b),
                limb_offset,
                tmp,
            );
        }
    }

    fn vmp_extract_selected_rows(
        module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &VmpPMatBackendRef<'_, Self>,
        first_row: usize,
        row_step: usize,
    ) {
        <NTT4x30Avx as HalVmpImpl<NTT4x30Avx>>::vmp_extract_selected_rows(
            base_module(module),
            &mut base_vmp_mut(res),
            &base_vmp_ref(a),
            first_row,
            row_step,
        )
    }

    fn vmp_zero(module: &Module<Self>, res: &mut VmpPMatBackendMut<'_, Self>) {
        <Self as NTT4x30VmpDefault<Self>>::vmp_zero_default(module, res)
    }
}

unsafe impl poulpy_core::oep::GGLWEProductDigitsStridedImpl<NTT4x30AvxRayon> for NTT4x30AvxRayon {
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
        super::vmp::vmp_apply_digits_strided_tmp_bytes_avx(
            a_cols,
            a_size,
            dsize,
            pmat_rows,
            pmat_cols_in,
            poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::VMP),
        )
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
        let metadata_bytes = 4 * dsize * size_of::<u64>();
        let per_worker =
            super::vmp::vmp_apply_digits_strided_tmp_bytes_avx(a.cols(), a.size(), dsize, pmat.rows(), pmat.cols_in(), 1)
                - metadata_bytes;
        let workers = poulpy_cpu_rayon::workers_within(
            <Self as poulpy_hal::execution::ScratchWorkers>::VMP,
            per_worker,
            scratch.available().saturating_sub(metadata_bytes),
        );
        let bytes = metadata_bytes + workers * per_worker;
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        super::vmp::vmp_apply_dft_to_dft_digits_strided_avx::<RayonTaskExecutor>(
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
    module: &Module<NTT4x30AvxRayon>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30AvxRayon>,
    a: &VecZnxDftBackendRef<'_, NTT4x30AvxRayon>,
    dsize: usize,
    zero_prefix: usize,
    product_limbs: usize,
    pmat: &VmpPMatBackendRef<'_, NTT4x30AvxRayon>,
    scratch: &mut ScratchArena<'_, NTT4x30AvxRayon>,
) {
    let bytes = <NTT4x30AvxRayon as poulpy_core::oep::GGLWEProductDigitsStridedImpl<NTT4x30AvxRayon>>::gglwe_product_digits_strided_tmp_bytes(
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
    let (tmp, _) = crate::hal_impl::take_host_typed::<NTT4x30AvxRayon, u64>(scratch.borrow(), bytes / size_of::<u64>());
    super::vmp::vmp_apply_dft_to_dft_digits_strided_avx_known_zero_prefix::<RayonTaskExecutor>(
        base_module(module),
        &mut base_dft_mut(res),
        &base_dft_ref(a),
        dsize,
        product_limbs,
        &base_vmp_ref(pmat),
        zero_prefix,
        tmp,
    );
}

unsafe impl HalConvolutionImpl<NTT4x30AvxRayon> for NTT4x30AvxRayon {
    fn cnv_prepare_left_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::PREPARE)
            * super::convolution::cnv_prepare_tmp_bytes(module.n())
    }

    fn cnv_prepare_left(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::CnvPVecLBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        mask: i64,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = super::convolution::cnv_prepare_tmp_bytes(module.n());
        let bytes = poulpy_cpu_rayon::workers_within(
            res.size().min(<Self as poulpy_hal::execution::ScratchWorkers>::PREPARE),
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        super::convolution::cnv_prepare_left::<_, RayonTaskExecutor>(module, res, a, mask, tmp);
    }

    fn cnv_prepare_right_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::PREPARE)
            * super::convolution::cnv_prepare_tmp_bytes(module.n())
    }

    fn cnv_prepare_right(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::CnvPVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        mask: i64,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = super::convolution::cnv_prepare_tmp_bytes(module.n());
        let bytes = poulpy_cpu_rayon::workers_within(
            res.size().min(<Self as poulpy_hal::execution::ScratchWorkers>::PREPARE),
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        super::convolution::cnv_prepare_right::<_, RayonTaskExecutor>(module, res, a, mask, tmp);
    }

    fn cnv_apply_dft_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        super::convolution::cnv_apply_dft_tmp_bytes(res_size, a_size, b_size)
    }

    fn cnv_by_const_apply_tmp_bytes(
        module: &Module<Self>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        let _ = (module, cnv_offset);
        super::convolution::cnv_by_const_apply_tmp_bytes(res_size, a_size, b_size)
    }

    #[allow(clippy::too_many_arguments)]
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
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        if RayonTaskExecutor::should_serialize_inner() {
            super::convolution::cnv_by_const_apply::<_, SerialTaskExecutor>(
                cnv_offset, res, res_col, a, a_col, b, b_col, b_coeff,
            );
        } else {
            super::convolution::cnv_by_const_apply::<_, RayonTaskExecutor>(cnv_offset, res, res_col, a, a_col, b, b_col, b_coeff);
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
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        if RayonTaskExecutor::should_serialize_inner() {
            super::convolution::cnv_by_const_apply_add::<_, SerialTaskExecutor>(
                cnv_offset, res, res_col, a, a_col, b, b_col, b_coeff,
            );
        } else {
            super::convolution::cnv_by_const_apply_add::<_, RayonTaskExecutor>(
                cnv_offset, res, res_col, a, a_col, b, b_col, b_coeff,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
        a_col: usize,
        b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
        b_col: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        unsafe {
            super::convolution::cnv_apply_dft::<_, RayonTaskExecutor>(module, cnv_offset, res, res_col, a, a_col, b, b_col)
        };
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft_accumulate(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
        a_col: usize,
        b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
        b_col: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        unsafe {
            super::convolution::cnv_apply_dft_accumulate::<_, RayonTaskExecutor>(
                module, cnv_offset, res, res_col, a, a_col, b, b_col,
            )
        };
    }

    fn cnv_accumulate_dft_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        _a_size: usize,
        _b_size: usize,
    ) -> usize {
        poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::APPLY)
            * super::convolution::cnv_accumulate_dft_avx_tmp_bytes(res_size)
    }

    fn cnv_accumulate_dft<'a>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        terms: &[poulpy_hal::layouts::CnvDftAccTerm<'a, Self>],
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Self: HalVecZnxDftImpl<Self> + 'a,
    {
        let per_worker = super::convolution::cnv_accumulate_dft_avx_tmp_bytes(res.size());
        let bytes = poulpy_cpu_rayon::workers_within(
            <Self as poulpy_hal::execution::ScratchWorkers>::APPLY,
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        unsafe {
            super::convolution::cnv_accumulate_dft_avx::<_, RayonTaskExecutor>(module, cnv_offset, res, res_col, terms, tmp)
        };
    }

    fn cnv_pairwise_apply_dft_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        super::convolution::cnv_apply_dft_tmp_bytes(res_size, a_size, b_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_pairwise_apply_dft(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
        b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
        i: usize,
        j: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        unsafe {
            super::convolution::cnv_pairwise_apply_dft::<_, RayonTaskExecutor>(module, cnv_offset, res, res_col, a, b, i, j)
        };
    }

    fn cnv_prepare_self_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::PREPARE)
            * super::convolution::cnv_prepare_tmp_bytes(module.n())
    }

    fn cnv_prepare_self(
        module: &Module<Self>,
        left: &mut poulpy_hal::layouts::CnvPVecLBackendMut<'_, Self>,
        right: &mut poulpy_hal::layouts::CnvPVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        mask: i64,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = super::convolution::cnv_prepare_tmp_bytes(module.n());
        let bytes = poulpy_cpu_rayon::workers_within(
            left.size().min(<Self as poulpy_hal::execution::ScratchWorkers>::PREPARE),
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        super::convolution::cnv_prepare_self::<_, RayonTaskExecutor>(module, left, right, a, mask, tmp);
    }
}
unsafe impl HalVecZnxBigImpl<NTT4x30AvxRayon> for NTT4x30AvxRayon {
    poulpy_cpu_ref::hal_impl_vec_znx_big_without_normalize!(NTT4x30VecZnxBigDefault);

    fn vec_znx_big_normalize(
        module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self, impl NormalizationState>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, Self>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let (carry, _) = poulpy_cpu_rayon::take_scratch::<Self, i128>(scratch.borrow(), 3 * module.n());
        poulpy_cpu_rayon::normalize::ntt4x30_vec_znx_big_normalize_par::<NTT4x30Avx, Self>(
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
unsafe impl HalSvpImpl<NTT4x30AvxRayon> for NTT4x30AvxRayon {
    fn svp_prepare(
        module: &Module<Self>,
        res: &mut SvpPPolBackendMut<'_, Self>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <NTT4x30Avx as HalSvpImpl<NTT4x30Avx>>::svp_prepare(
            base_module(module),
            &mut base_svp_mut(res),
            res_col,
            &base_scalar_ref(a),
            a_col,
        );
    }

    fn svp_ppol_copy_backend(
        module: &Module<Self>,
        res: &mut SvpPPolBackendMut<'_, Self>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <NTT4x30Avx as HalSvpImpl<NTT4x30Avx>>::svp_ppol_copy_backend(
            base_module(module),
            &mut base_svp_mut(res),
            res_col,
            &base_svp_ref(a),
            a_col,
        );
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
        <NTT4x30Avx as HalSvpImpl<NTT4x30Avx>>::svp_apply_dft(
            base_module(module),
            &mut base_dft_mut(res),
            res_col,
            &base_svp_ref(a),
            a_col,
            &base_znx_ref(b),
            b_col,
        );
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
        <NTT4x30Avx as HalSvpImpl<NTT4x30Avx>>::svp_apply_dft_to_dft(
            base_module(module),
            &mut base_dft_mut(res),
            res_col,
            &base_svp_ref(a),
            a_col,
            &base_dft_ref(b),
            b_col,
        );
    }

    fn svp_apply_dft_to_dft_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <NTT4x30Avx as HalSvpImpl<NTT4x30Avx>>::svp_apply_dft_to_dft_assign(
            base_module(module),
            &mut base_dft_mut(res),
            res_col,
            &base_svp_ref(a),
            a_col,
        );
    }
}
unsafe impl HalVecZnxDftImpl<NTT4x30AvxRayon> for NTT4x30AvxRayon {
    fn vec_znx_idft_normalize_consume_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
        <NTT4x30Avx as HalVecZnxDftImpl<NTT4x30Avx>>::vec_znx_idft_normalize_consume_tmp_bytes(
            base_module(module),
            res_size,
            a_size,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_idft_normalize_consume(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self, impl poulpy_hal::layouts::NormalizationState>,
        res_base2k: usize,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, Self>,
        a_col: usize,
        a_base2k: usize,
        addend: Option<(&VecZnxBackendRef<'_, Self, impl NormalizationState>, usize)>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut base_res = base_znx_mut(res);
        let mut base_a = base_dft_mut(a);
        let mut base_scratch = scratch.borrow().into_backend::<NTT4x30Avx>();
        if let Some((add, add_col)) = addend {
            let base_add = base_znx_ref(add);
            <NTT4x30Avx as HalVecZnxDftImpl<NTT4x30Avx>>::vec_znx_idft_normalize_consume(
                base_module(module),
                &mut base_res,
                res_base2k,
                res_col,
                &mut base_a,
                a_col,
                a_base2k,
                Some((&base_add, add_col)),
                &mut base_scratch,
            );
        } else {
            <NTT4x30Avx as HalVecZnxDftImpl<NTT4x30Avx>>::vec_znx_idft_normalize_consume(
                base_module(module),
                &mut base_res,
                res_base2k,
                res_col,
                &mut base_a,
                a_col,
                a_base2k,
                None::<(&VecZnxBackendRef<'_, NTT4x30Avx, Normalized>, usize)>,
                &mut base_scratch,
            );
        }
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
        if !parallel_limb_tasks(res.size()) {
            return <NTT4x30Avx as HalVecZnxDftImpl<NTT4x30Avx>>::vec_znx_dft_apply(
                base_module(module),
                step,
                offset,
                &mut base_dft_mut(res),
                res_col,
                &base_znx_ref(a),
                a_col,
            );
        }

        let n = res.n();
        let cols = res.cols();
        let a_size = a.size();
        let module = base_module(module);
        let data: &mut [u32] = cast_slice_mut(res.raw_mut());
        data.par_chunks_mut(4 * n * cols).enumerate().for_each_init(
            || vec![0u64; 4 * n],
            |tmp, (limb, group)| {
                let src_limb = offset + limb * step;
                super::vec_znx_dft::dft_limb(
                    module,
                    &mut group[4 * n * res_col..][..4 * n],
                    (src_limb < a_size).then(|| a.at(a_col, src_limb)),
                    tmp,
                );
            },
        );
    }

    fn vec_znx_idft_apply_tmp_bytes(module: &Module<Self>) -> usize {
        <NTT4x30Avx as HalVecZnxDftImpl<NTT4x30Avx>>::vec_znx_idft_apply_tmp_bytes(base_module(module)).max(
            poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::IDFT) * 4 * module.n() * size_of::<u64>(),
        )
    }

    fn vec_znx_idft_apply(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        if !parallel_limb_tasks(res.size()) {
            let mut base_scratch = scratch.borrow().into_backend::<NTT4x30Avx>();
            return <NTT4x30Avx as HalVecZnxDftImpl<NTT4x30Avx>>::vec_znx_idft_apply(
                base_module(module),
                &mut base_big_mut(res),
                res_col,
                &base_dft_ref(a),
                a_col,
                &mut base_scratch,
            );
        }

        let n = res.n();
        let res_cols = res.cols();
        let a_cols = a.cols();
        let size = res.size();
        let min_size = size.min(a.size());
        let a_data: &[u32] = cast_slice(a.raw());
        let per_worker = 4 * n;
        let workers = poulpy_cpu_rayon::workers_within(
            size.min(<Self as poulpy_hal::execution::ScratchWorkers>::IDFT),
            per_worker * size_of::<u64>(),
            scratch.available(),
        );
        let (worker_tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), workers * per_worker);
        let res_ptr = SendPtr::new(res.raw_mut().as_mut_ptr());
        let module = base_module(module);
        RayonTaskExecutor::for_each_chunked(size, worker_tmp, per_worker, |tmp, limb| {
            let dst = unsafe { std::slice::from_raw_parts_mut(res_ptr.get().add(n * (limb * res_cols + res_col)), n) };
            if limb < min_size {
                super::vec_znx_dft::idft_limb(
                    module,
                    dst,
                    super::vec_znx_dft::packed_limb(a_data, n, a_cols, a_col, limb),
                    tmp,
                );
            } else {
                dst.fill(0);
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
        if !parallel_limb_tasks(res.size()) {
            return <NTT4x30Avx as HalVecZnxDftImpl<NTT4x30Avx>>::vec_znx_idft_apply_tmpa(
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
        let a_data: &[u32] = cast_slice(a.raw());
        let module = base_module(module);
        res.raw_mut().par_chunks_mut(n * res_cols).enumerate().for_each_init(
            || vec![0u64; 4 * n],
            |tmp, (limb, group)| {
                let dst = &mut group[n * res_col..][..n];
                if limb < min_size {
                    super::vec_znx_dft::idft_limb(
                        module,
                        dst,
                        super::vec_znx_dft::packed_limb(a_data, n, a_cols, a_col, limb),
                        tmp,
                    );
                } else {
                    dst.fill(0);
                }
            },
        );
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
        <NTT4x30Avx as HalVecZnxDftImpl<NTT4x30Avx>>::vec_znx_dft_add_into(
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
        <NTT4x30Avx as HalVecZnxDftImpl<NTT4x30Avx>>::vec_znx_dft_add_scaled_assign(
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
        <NTT4x30Avx as HalVecZnxDftImpl<NTT4x30Avx>>::vec_znx_dft_add_assign(
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
        <NTT4x30Avx as HalVecZnxDftImpl<NTT4x30Avx>>::vec_znx_dft_sub(
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
        <NTT4x30Avx as HalVecZnxDftImpl<NTT4x30Avx>>::vec_znx_dft_sub_assign(
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
        <NTT4x30Avx as HalVecZnxDftImpl<NTT4x30Avx>>::vec_znx_dft_sub_negate_assign(
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
        <NTT4x30Avx as HalVecZnxDftImpl<NTT4x30Avx>>::vec_znx_dft_copy(
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
        <NTT4x30Avx as HalVecZnxDftImpl<NTT4x30Avx>>::vec_znx_dft_zero(base_module(module), &mut base_dft_mut(res), res_col)
    }

    type AutomorphismPlan = <NTT4x30Avx as HalVecZnxDftImpl<NTT4x30Avx>>::AutomorphismPlan;

    fn vec_znx_dft_automorphism_plan(module: &Module<Self>, p: i64) -> Self::AutomorphismPlan {
        <NTT4x30Avx as HalVecZnxDftImpl<NTT4x30Avx>>::vec_znx_dft_automorphism_plan(base_module(module), p)
    }

    fn vec_znx_dft_automorphism_with_plan(
        _module: &Module<Self>,
        plan: &Self::AutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        super::vec_znx_dft::vec_znx_dft_automorphism(plan, &mut base_dft_mut(res), res_col, &base_dft_ref(a), a_col);
    }

    fn vec_znx_dft_automorphism_add_with_plan(
        _module: &Module<Self>,
        plan: &Self::AutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        if RayonTaskExecutor::should_serialize_inner() {
            super::vec_znx_dft::vec_znx_dft_automorphism_add::<SerialTaskExecutor>(
                plan,
                &mut base_dft_mut(res),
                res_col,
                &base_dft_ref(a),
                a_col,
            );
        } else {
            super::vec_znx_dft::vec_znx_dft_automorphism_add::<RayonTaskExecutor>(
                plan,
                &mut base_dft_mut(res),
                res_col,
                &base_dft_ref(a),
                a_col,
            );
        }
    }
}

impl poulpy_hal::execution::ScratchWorkers for NTT4x30AvxRayon {
    const PREPARE: usize = 4;
    const APPLY: usize = 8;
    const VMP: usize = 8;
    const IDFT: usize = 8;
}

impl poulpy_cpu_rayon::RayonTuning for NTT4x30AvxRayon {
    const COEFF_MIN_LEN: usize = 1 << 15;
    const COEFF_MIN_TASK: usize = 1 << 13;
    const NORMALIZE_MIN_TASK: usize = 1 << 12;
}

#[cfg(test)]
mod tests {
    use poulpy_cpu_ref::reference::znx::ZnxAdd;
    use poulpy_hal::{layouts::Module, test_suite::convolution::test_convolution_by_const};

    use super::NTT4x30AvxRayon;

    #[test]
    fn coefficient_add_matches_wrapping_arithmetic() {
        let a = vec![i64::MAX; 1 << 16];
        let b = vec![1; 1 << 16];
        let mut actual = vec![0; 1 << 16];
        <NTT4x30AvxRayon as ZnxAdd>::znx_add(&mut actual, &a, &b);
        assert!(actual.iter().all(|&x| x == i64::MIN));
    }

    #[test]
    fn convolution_by_const() {
        test_convolution_by_const(&Module::<NTT4x30AvxRayon>::new(1 << 8), 50);
    }
}
