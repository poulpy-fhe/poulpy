//! Rayon-scheduled wrapper for the NEON NTT4x30 backend.

use std::mem::size_of;

use bytemuck::{cast_slice, cast_slice_mut};
use rayon::prelude::*;

use poulpy_cpu_portable::{
    hal_defaults::{BigWordHadamardProduct, HalVecZnxDefault, NTT4x30ModuleDefault, NTT4x30VecZnxBigDefault},
    kernels::{
        ntt4x30::{
            I128BigOps, I128NormalizeOps, NttDFTExecute,
            ntt::{NttTable, NttTableInv},
            primes::Primes30,
            vec_znx_big::AssignOp,
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
use poulpy_hal::{
    execution::{SerialTaskExecutor, TaskExecutor},
    layouts::{
        DataView, DataViewMut, MatZnxBackendRef, Module, ScalarZnx, ScalarZnxBackendRef, ScratchArena, SvpPPol,
        SvpPPolBackendMut, SvpPPolBackendRef, VecZnx, VecZnxBackendMut, VecZnxBackendRef, VecZnxBig, VecZnxBigBackendMut,
        VecZnxDft, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMat, VmpPMatBackendMut, VmpPMatBackendRef, ZnxView, ZnxViewMut,
    },
    oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
};

use super::{NTT4x30Neon, NTT4x30NeonRayon};
use crate::neon::ntt4x30_ntt32::{intt32, intt32_crt, intt32_plane, ntt32_plane};
use poulpy_cpu_portable::kernels::ntt4x30::vec_znx_dft::{NttPlan, NttPlanNew};
use poulpy_cpu_rayon::{RayonTaskExecutor, SendPtr};
use poulpy_hal::layouts::Ring;

poulpy_hal::impl_backend_from!(NTT4x30NeonRayon<R>, NTT4x30Neon<R>, RayonTaskExecutor; generic R: Ring);

impl<R: Ring> poulpy_hal::layouts::MaxBase2k for NTT4x30NeonRayon<R>
where
    NTT4x30Neon<R>: poulpy_hal::layouts::MaxBase2k,
{
    fn max_base2k(n: usize, products: usize, failure_bits: usize, squaring: bool) -> Option<usize> {
        <NTT4x30Neon<R> as poulpy_hal::layouts::MaxBase2k>::max_base2k(n, products, failure_bits, squaring)
    }
}

fn base_module<R: Ring>(module: &Module<NTT4x30NeonRayon<R>>) -> &Module<NTT4x30Neon<R>> {
    module.reinterpret()
}

fn base_dft_ref<'a, R: Ring>(a: &'a VecZnxDftBackendRef<'_, NTT4x30NeonRayon<R>>) -> VecZnxDftBackendRef<'a, NTT4x30Neon<R>> {
    VecZnxDft::from_shape(&**a.data(), a.shape())
}

fn base_dft_mut<'a, R: Ring>(a: &'a mut VecZnxDftBackendMut<'_, NTT4x30NeonRayon<R>>) -> VecZnxDftBackendMut<'a, NTT4x30Neon<R>> {
    let shape = a.shape();
    VecZnxDft::from_shape(&mut **a.data_mut(), shape)
}

fn base_znx_ref<'a, R: Ring>(a: &'a VecZnxBackendRef<'_, NTT4x30NeonRayon<R>>) -> VecZnxBackendRef<'a, NTT4x30Neon<R>> {
    VecZnx::from_shape(&**a.data(), a.shape())
}

fn base_scalar_ref<'a, R: Ring>(a: &'a ScalarZnxBackendRef<'_, NTT4x30NeonRayon<R>>) -> ScalarZnxBackendRef<'a, NTT4x30Neon<R>> {
    ScalarZnx::from_data(&**a.data(), a.n(), a.cols())
}

fn base_svp_ref<'a, R: Ring>(a: &'a SvpPPolBackendRef<'_, NTT4x30NeonRayon<R>>) -> SvpPPolBackendRef<'a, NTT4x30Neon<R>> {
    SvpPPol::from_data(&**a.data(), a.n(), a.cols(), a.hint())
}

fn base_svp_mut<'a, R: Ring>(a: &'a mut SvpPPolBackendMut<'_, NTT4x30NeonRayon<R>>) -> SvpPPolBackendMut<'a, NTT4x30Neon<R>> {
    let (n, cols, hint) = (a.n(), a.cols(), a.hint());
    SvpPPol::from_data(&mut **a.data_mut(), n, cols, hint)
}

fn base_big_mut<'a, R: Ring>(a: &'a mut VecZnxBigBackendMut<'_, NTT4x30NeonRayon<R>>) -> VecZnxBigBackendMut<'a, NTT4x30Neon<R>> {
    let shape = a.shape();
    VecZnxBig::from_shape(&mut **a.data_mut(), shape)
}

fn base_big_ref<'a, R: Ring>(
    a: &'a poulpy_hal::layouts::VecZnxBigBackendRef<'_, NTT4x30NeonRayon<R>>,
) -> poulpy_hal::layouts::VecZnxBigBackendRef<'a, NTT4x30Neon<R>> {
    VecZnxBig::from_shape(&**a.data(), a.shape())
}

fn base_vmp_ref<'a, R: Ring>(a: &'a VmpPMatBackendRef<'_, NTT4x30NeonRayon<R>>) -> VmpPMatBackendRef<'a, NTT4x30Neon<R>> {
    VmpPMat::from_data(&**a.data(), a.n(), a.rows(), a.cols_in(), a.cols_out(), a.size(), a.hint())
}

fn base_vmp_mut<'a, R: Ring>(a: &'a mut VmpPMatBackendMut<'_, NTT4x30NeonRayon<R>>) -> VmpPMatBackendMut<'a, NTT4x30Neon<R>> {
    let (n, rows, cols_in, cols_out, size, hint) = (a.n(), a.rows(), a.cols_in(), a.cols_out(), a.size(), a.hint());
    VmpPMat::from_data(&mut **a.data_mut(), n, rows, cols_in, cols_out, size, hint)
}

use poulpy_cpu_rayon::{parallel_chunk_len, parallel_limb_tasks};

impl<R: Ring> super::vec_znx_dft::PackedDft for Module<NTT4x30NeonRayon<R>>
where
    NTT4x30Neon<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
{
    #[inline(always)]
    fn packed_dft_limb(&self, n: usize, dst: &mut [u32], src: &[i64], prepared: bool) {
        if !plane_tasks(n) {
            return base_module(self).packed_dft_limb(n, dst, src, prepared);
        }
        assert!(dst.len() >= 4 * n);
        let table = super::vec_znx_dft::packed_table(base_module(self), n);
        let dst = SendPtr::new(dst.as_mut_ptr());
        join4(|p| {
            let plane = unsafe { std::slice::from_raw_parts_mut(dst.get().add(p * n), n) };
            ntt32_plane(table, p, plane, src, prepared)
        });
    }
}

/// Ring degree from which the four planes of a limb are transformed as separate tasks.
///
/// A ciphertext has few limbs at a large `base2k`, fewer than the pool has threads.
const PLANE_TASKS_MIN_N: usize = 1 << 13;

#[inline]
fn plane_tasks(n: usize) -> bool {
    n >= PLANE_TASKS_MIN_N && rayon::current_num_threads() > 1
}

/// Whether the limbs of a transform-domain vector are worth one task each.
#[inline]
fn limb_tasks(n: usize, size: usize) -> bool {
    n >= PLANE_TASKS_MIN_N && parallel_limb_tasks(size)
}

/// Runs `task` on `0..4` as four tasks.
#[inline]
fn join4(task: impl Fn(usize) + Sync) {
    rayon::join(|| rayon::join(|| task(0), || task(1)), || rayon::join(|| task(2), || task(3)));
}

/// Inverse transform of one packed limb, its planes and its reconstruction split into tasks at large degrees.
///
/// # Safety
/// Same contract as [`intt32`].
unsafe fn idft_limb_planes<R: Ring>(module: &Module<NTT4x30Neon<R>>, n: usize, dst: *mut i128, src: *const u32, work: *mut u32)
where
    NTT4x30Neon<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
{
    let table = super::vec_znx_dft::packed_table(module, n);
    if !plane_tasks(n) {
        return unsafe { intt32(table, dst, src, work) };
    }
    let (dst, src, work) = (SendPtr::new(dst), SendPtr::new(src as *mut u32), SendPtr::new(work));
    join4(|p| unsafe { intt32_plane(table, p, src.get(), work.get()) });
    let quarter = n / 4;
    join4(|k| unsafe { intt32_crt(table, dst.get(), work.get(), k * quarter, (k + 1) * quarter) });
}

macro_rules! parallel_binary {
    ($trait:ident, $method:ident) => {
        impl<R: Ring> $trait for NTT4x30NeonRayon<R> {
            fn $method(res: &mut [i64], a: &[i64], b: &[i64]) {
                let Some(chunk) = parallel_chunk_len::<Self>(res.len()) else {
                    return <NTT4x30Neon<R> as $trait>::$method(res, a, b);
                };
                res.par_chunks_mut(chunk)
                    .zip(a.par_chunks(chunk))
                    .zip(b.par_chunks(chunk))
                    .for_each(|((res, a), b)| <NTT4x30Neon<R> as $trait>::$method(res, a, b));
            }
        }
    };
}

macro_rules! parallel_assign {
    ($trait:ident, $method:ident) => {
        impl<R: Ring> $trait for NTT4x30NeonRayon<R> {
            fn $method(res: &mut [i64], a: &[i64]) {
                let Some(chunk) = parallel_chunk_len::<Self>(res.len()) else {
                    return <NTT4x30Neon<R> as $trait>::$method(res, a);
                };
                res.par_chunks_mut(chunk)
                    .zip(a.par_chunks(chunk))
                    .for_each(|(res, a)| <NTT4x30Neon<R> as $trait>::$method(res, a));
            }
        }
    };
}

macro_rules! parallel_unary {
    ($trait:ident, $method:ident) => {
        impl<R: Ring> $trait for NTT4x30NeonRayon<R> {
            fn $method(res: &mut [i64]) {
                let Some(chunk) = parallel_chunk_len::<Self>(res.len()) else {
                    return <NTT4x30Neon<R> as $trait>::$method(res);
                };
                res.par_chunks_mut(chunk)
                    .for_each(|res| <NTT4x30Neon<R> as $trait>::$method(res));
            }
        }
    };
}

macro_rules! parallel_shift {
    ($trait:ident, $method:ident) => {
        impl<R: Ring> $trait for NTT4x30NeonRayon<R> {
            fn $method(k: i64, res: &mut [i64], a: &[i64]) {
                let Some(chunk) = parallel_chunk_len::<Self>(res.len()) else {
                    return <NTT4x30Neon<R> as $trait>::$method(k, res, a);
                };
                res.par_chunks_mut(chunk)
                    .zip(a.par_chunks(chunk))
                    .for_each(|(res, a)| <NTT4x30Neon<R> as $trait>::$method(k, res, a));
            }
        }
    };
}

macro_rules! forward_znx {
    ($trait:ident, $method:ident($($arg:ident: $ty:ty),* $(,)?)) => {
        impl<R: Ring> $trait for NTT4x30NeonRayon<R> {
            #[inline(always)]
            fn $method($($arg: $ty),*) {
                <NTT4x30Neon<R> as $trait>::$method($($arg),*)
            }
        }
    };
}

macro_rules! forward_znx_const {
    ($trait:ident, $method:ident($($arg:ident: $ty:ty),* $(,)?)) => {
        impl<R: Ring> $trait for NTT4x30NeonRayon<R> {
            #[inline(always)]
            fn $method<const OVERWRITE: bool>($($arg: $ty),*) {
                <NTT4x30Neon<R> as $trait>::$method::<OVERWRITE>($($arg),*)
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

impl<R: Ring> ZnxMulPowerOfTwoAssign for NTT4x30NeonRayon<R> {
    fn znx_mul_power_of_two_assign(k: i64, res: &mut [i64]) {
        let Some(chunk) = parallel_chunk_len::<Self>(res.len()) else {
            return <NTT4x30Neon<R> as ZnxMulPowerOfTwoAssign>::znx_mul_power_of_two_assign(k, res);
        };
        res.par_chunks_mut(chunk)
            .for_each(|res| <NTT4x30Neon<R> as ZnxMulPowerOfTwoAssign>::znx_mul_power_of_two_assign(k, res));
    }
}

impl<R: Ring> ZnxAutomorphism for NTT4x30NeonRayon<R>
where
    NTT4x30Neon<R>: ZnxAutomorphism,
{
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        <NTT4x30Neon<R> as ZnxAutomorphism>::znx_automorphism(p, res, a)
    }

    #[inline(always)]
    fn znx_automorphism_i128(p: i64, res: &mut [i128], a: &[i128]) {
        <NTT4x30Neon<R> as ZnxAutomorphism>::znx_automorphism_i128(p, res, a)
    }
}

impl ZnxAutomorphismRotate for NTT4x30NeonRayon {
    #[inline(always)]
    fn znx_automorphism_rotate(p: i64, k: i64, res: &mut [i64], a: &[i64]) {
        <NTT4x30Neon as ZnxAutomorphismRotate>::znx_automorphism_rotate(p, k, res, a)
    }
}
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
forward_znx!(ZnxNormalizeFinalStepAssign, znx_normalize_final_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]));
impl<R: Ring> ZnxExtractDigitAddMul for NTT4x30NeonRayon<R> {
    #[inline(always)]
    fn znx_extract_digit_addmul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
        <NTT4x30Neon<R> as ZnxExtractDigitAddMul>::znx_extract_digit_addmul(base2k, lsh, res, src);
    }
}

impl<R: Ring> poulpy_cpu_portable::kernels::normalization::I64NormalizeOps for NTT4x30NeonRayon<R> {
    #[inline(always)]
    fn znx_normalize_floor<const CARRY_IN: bool, const ROUND: bool>(base2k: usize, lsh: usize, a: &[i64], carry: &mut [i64]) {
        <NTT4x30Neon<R> as poulpy_cpu_portable::kernels::normalization::I64NormalizeOps>::znx_normalize_floor::<CARRY_IN, ROUND>(
            base2k, lsh, a, carry,
        );
    }

    #[inline(always)]
    fn znx_normalize_round<const CARRY_IN: bool, const PAD: bool>(
        base2k: usize,
        lsh: usize,
        padding: usize,
        res: &mut [i64],
        a: &[i64],
        carry: &mut [i64],
    ) {
        <NTT4x30Neon<R> as poulpy_cpu_portable::kernels::normalization::I64NormalizeOps>::znx_normalize_round::<CARRY_IN, PAD>(
            base2k, lsh, padding, res, a, carry,
        );
    }

    #[inline(always)]
    fn znx_normalize_round_assign<const CARRY_IN: bool>(
        base2k: usize,
        lsh: usize,
        padding: usize,
        res: &mut [i64],
        carry: &mut [i64],
    ) {
        <NTT4x30Neon<R> as poulpy_cpu_portable::kernels::normalization::I64NormalizeOps>::znx_normalize_round_assign::<CARRY_IN>(
            base2k, lsh, padding, res, carry,
        );
    }

    #[inline(always)]
    fn znx_extract_digit_mul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64]) {
        <NTT4x30Neon<R> as poulpy_cpu_portable::kernels::normalization::I64NormalizeOps>::znx_extract_digit_mul(
            base2k, lsh, res, src,
        );
    }

    #[inline(always)]
    fn znx_extract_digit_addmul_normalize<const OVERWRITE: bool>(
        base2k: usize,
        lsh: usize,
        res_base2k: usize,
        res: &mut [i64],
        src: &mut [i64],
        carry: &mut [i64],
    ) {
        <NTT4x30Neon<R> as poulpy_cpu_portable::kernels::normalization::I64NormalizeOps>::znx_extract_digit_addmul_normalize::<
            OVERWRITE,
        >(base2k, lsh, res_base2k, res, src, carry);
    }
}
forward_znx!(ZnxNormalizeDigit, znx_normalize_digit(base2k: usize, res: &mut [i64], src: &mut [i64]));

impl<R: Ring> NttDFTExecute<NttTable<Primes30, R>> for NTT4x30NeonRayon<R>
where
    NTT4x30Neon<R>: NttDFTExecute<NttTable<Primes30, R>>,
{
    fn ntt_dft_execute(table: &NttTable<Primes30, R>, data: &mut [u64]) {
        <NTT4x30Neon<R> as NttDFTExecute<NttTable<Primes30, R>>>::ntt_dft_execute(table, data)
    }

    fn ntt_automorphism_plan(n: usize, p: i64) -> poulpy_cpu_portable::kernels::ntt4x30::vec_znx_dft::NttAutomorphismPlan {
        <NTT4x30Neon<R> as NttDFTExecute<NttTable<Primes30, R>>>::ntt_automorphism_plan(n, p)
    }
}

impl<R: Ring> NttDFTExecute<NttTableInv<Primes30, R>> for NTT4x30NeonRayon<R>
where
    NTT4x30Neon<R>: NttDFTExecute<NttTableInv<Primes30, R>>,
{
    fn ntt_dft_execute(table: &NttTableInv<Primes30, R>, data: &mut [u64]) {
        <NTT4x30Neon<R> as NttDFTExecute<NttTableInv<Primes30, R>>>::ntt_dft_execute(table, data)
    }

    fn ntt_automorphism_plan(n: usize, p: i64) -> poulpy_cpu_portable::kernels::ntt4x30::vec_znx_dft::NttAutomorphismPlan {
        <NTT4x30Neon<R> as NttDFTExecute<NttTableInv<Primes30, R>>>::ntt_automorphism_plan(n, p)
    }
}

macro_rules! forward_i128_big {
    ($method:ident($($arg:ident: $ty:ty),* $(,)?)) => {
        fn $method($($arg: $ty),*) { <NTT4x30Neon as I128BigOps>::$method($($arg),*) }
    };
}

impl<R: Ring> I128BigOps for NTT4x30NeonRayon<R> {
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

impl<R: Ring> I128NormalizeOps for NTT4x30NeonRayon<R> {
    #[inline(always)]
    fn nfc_normalize_floor<const CARRY_IN: bool, const ROUND: bool>(base2k: usize, lsh: usize, a: &[i128], carry: &mut [i128]) {
        <NTT4x30Neon<R> as I128NormalizeOps>::nfc_normalize_floor::<CARRY_IN, ROUND>(base2k, lsh, a, carry);
    }

    #[inline(always)]
    fn nfc_normalize_round<const CARRY_IN: bool, const PAD: bool>(
        base2k: usize,
        lsh: usize,
        padding: usize,
        res: &mut [i64],
        a: &[i128],
        carry: &mut [i128],
    ) {
        <NTT4x30Neon<R> as I128NormalizeOps>::nfc_normalize_round::<CARRY_IN, PAD>(base2k, lsh, padding, res, a, carry);
    }

    #[inline(always)]
    fn nfc_add_small_carry(carry: &mut [i128], a: &[i64]) {
        <NTT4x30Neon<R> as I128NormalizeOps>::nfc_add_small_carry(carry, a);
    }

    #[inline(always)]
    fn znx_extract_digit_addmul_i128(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i128]) {
        <NTT4x30Neon<R> as I128NormalizeOps>::znx_extract_digit_addmul_i128(base2k, lsh, res, src);
    }

    const FUSE_NORMALIZE: bool = <NTT4x30Neon<R> as poulpy_cpu_portable::kernels::ntt4x30::I128NormalizeOps>::FUSE_NORMALIZE;

    fn znx_extract_digit_mul_i128(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i128]) {
        <NTT4x30Neon<R> as poulpy_cpu_portable::kernels::ntt4x30::I128NormalizeOps>::znx_extract_digit_mul_i128(
            base2k, lsh, res, src,
        )
    }

    fn znx_extract_digit_addmul_normalize_i128<const OVERWRITE: bool>(
        base2k: usize,
        lsh: usize,
        res_base2k: usize,
        res: &mut [i64],
        src: &mut [i128],
        carry: &mut [i128],
    ) {
        <NTT4x30Neon<R> as poulpy_cpu_portable::kernels::ntt4x30::I128NormalizeOps>::znx_extract_digit_addmul_normalize_i128::<
            OVERWRITE,
        >(base2k, lsh, res_base2k, res, src, carry)
    }

    fn nfc_middle_step(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
        <NTT4x30Neon<R> as I128NormalizeOps>::nfc_middle_step(base2k, lsh, res, a, carry)
    }
    fn nfc_middle_step_into<O: AssignOp>(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
        <NTT4x30Neon<R> as I128NormalizeOps>::nfc_middle_step_into::<O>(base2k, lsh, res, a, carry)
    }
    fn nfc_middle_step_assign(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        <NTT4x30Neon<R> as I128NormalizeOps>::nfc_middle_step_assign(base2k, lsh, res, carry)
    }
    fn nfc_final_step_assign(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        <NTT4x30Neon<R> as I128NormalizeOps>::nfc_final_step_assign(base2k, lsh, res, carry)
    }
    fn nfc_final_step_into<O: AssignOp>(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        <NTT4x30Neon<R> as I128NormalizeOps>::nfc_final_step_into::<O>(base2k, lsh, res, carry)
    }
}

impl<R: Ring> BigWordHadamardProduct for NTT4x30NeonRayon<R> {
    fn big_word_hadamard_product(res: &mut [i128], a: &[i64], b: &[i64]) {
        <Self as I128BigOps>::i128_hadamard_product_i64(res, a, b)
    }
}

unsafe impl poulpy_hal::oep::HalVecZnxMonomialImpl for NTT4x30NeonRayon {
    poulpy_cpu_portable::hal_impl_vec_znx_monomial!();
}

unsafe impl poulpy_hal::oep::HalVecZnxCIImpl for NTT4x30NeonRayon {
    poulpy_cpu_portable::hal_impl_vec_znx_ci!();
}

unsafe impl<R: Ring> HalVecZnxImpl for NTT4x30NeonRayon<R>
where
    NTT4x30Neon<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>> + ZnxAutomorphism,
{
    poulpy_cpu_portable::hal_impl_vec_znx_without_normalize!();

    fn vec_znx_normalize(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let (carry, _) = poulpy_cpu_rayon::take_scratch::<Self, i64>(scratch.borrow(), 3 * res.n());
        poulpy_cpu_rayon::normalize::vec_znx_normalize_par::<NTT4x30Neon<R>, Self>(
            res, res_base2k, res_k, res_offset, res_col, a, a_base2k, a_col, carry,
        );
    }

    fn vec_znx_normalize_assign(
        _module: &Module<Self>,
        base2k: usize,
        k: usize,
        a_offset: i64,
        a: &mut VecZnxBackendMut<'_, Self>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let (carry, _) = poulpy_cpu_rayon::take_scratch::<Self, i64>(scratch.borrow(), 3 * a.n());
        poulpy_cpu_rayon::normalize::vec_znx_normalize_assign_par::<NTT4x30Neon<R>, Self>(base2k, k, a_offset, a, a_col, carry);
    }
}
unsafe impl<R: Ring> HalModuleImpl for NTT4x30NeonRayon<R>
where
    NttPlan<Primes30, R>: NttPlanNew,
{
    poulpy_cpu_portable::hal_impl_module!(NTT4x30ModuleDefault);
}
unsafe impl<R: Ring> HalVmpImpl for NTT4x30NeonRayon<R>
where
    NTT4x30Neon<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>> + ZnxAutomorphism,
{
    fn vmp_prepare_tmp_bytes(module: &Module<Self>, _rows: usize, _cols_in: usize, _cols_out: usize, _size: usize) -> usize {
        super::vmp::vmp_prepare_tmp_bytes_neon(module.n())
    }

    fn vmp_prepare(
        module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &MatZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = super::vmp::vmp_prepare_tmp_bytes_neon(res.n());
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        super::vmp::vmp_prepare_neon_pm(base_module(module), &mut base_vmp_mut::<R>(res), a, tmp);
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
            * super::vmp::vmp_apply_tmp_bytes_neon(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = super::vmp::vmp_apply_tmp_bytes_neon(a.size(), b.rows(), b.cols_in());
        let bytes = poulpy_cpu_rayon::workers_within(
            <Self as poulpy_hal::execution::ScratchWorkers>::VMP,
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        if RayonTaskExecutor::should_serialize_inner() {
            super::vmp::vmp_apply_dft_to_dft_neon::<_, SerialTaskExecutor>(
                base_module(module),
                &mut base_dft_mut::<R>(res),
                &base_dft_ref::<R>(a),
                &base_vmp_ref::<R>(b),
                limb_offset,
                tmp,
            );
        } else {
            super::vmp::vmp_apply_dft_to_dft_neon::<_, RayonTaskExecutor>(
                base_module(module),
                &mut base_dft_mut::<R>(res),
                &base_dft_ref::<R>(a),
                &base_vmp_ref::<R>(b),
                limb_offset,
                tmp,
            );
        }
    }

    fn vmp_apply_dft_to_dft_add_tmp_bytes(
        _module: &Module<Self>,
        _res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize {
        poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::VMP)
            * super::vmp::vmp_apply_tmp_bytes_neon(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft_add(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = super::vmp::vmp_apply_tmp_bytes_neon(a.size(), b.rows(), b.cols_in());
        let bytes = poulpy_cpu_rayon::workers_within(
            <Self as poulpy_hal::execution::ScratchWorkers>::VMP,
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        if RayonTaskExecutor::should_serialize_inner() {
            super::vmp::vmp_apply_dft_to_dft_add_neon::<_, SerialTaskExecutor>(
                base_module(module),
                &mut base_dft_mut::<R>(res),
                &base_dft_ref::<R>(a),
                &base_vmp_ref::<R>(b),
                limb_offset,
                tmp,
            );
        } else {
            super::vmp::vmp_apply_dft_to_dft_add_neon::<_, RayonTaskExecutor>(
                base_module(module),
                &mut base_dft_mut::<R>(res),
                &base_dft_ref::<R>(a),
                &base_vmp_ref::<R>(b),
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
        NTT4x30Neon::<R>::vmp_extract_selected_rows(
            base_module(module),
            &mut base_vmp_mut::<R>(res),
            &base_vmp_ref::<R>(a),
            first_row,
            row_step,
        )
    }

    fn vmp_zero(_module: &Module<Self>, res: &mut VmpPMatBackendMut<'_, Self>) {
        res.data_mut().fill(Default::default());
    }
}

unsafe impl<R: Ring> HalConvolutionImpl for NTT4x30NeonRayon<R>
where
    NTT4x30Neon<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>> + ZnxAutomorphism,
{
    fn cnv_prepare_left_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::PREPARE)
            * super::convolution::cnv_prepare_tmp_bytes(module.n())
    }

    fn cnv_prepare_left(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::CnvPVecLBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = super::convolution::cnv_prepare_tmp_bytes(res.n());
        let bytes = poulpy_cpu_rayon::workers_within(
            res.size().min(<Self as poulpy_hal::execution::ScratchWorkers>::PREPARE),
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        super::convolution::cnv_prepare_left::<_, RayonTaskExecutor>(module, res, a, tmp);
    }

    fn cnv_prepare_right_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::PREPARE)
            * super::convolution::cnv_prepare_tmp_bytes(module.n())
    }

    fn cnv_prepare_right(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::CnvPVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = super::convolution::cnv_prepare_tmp_bytes(res.n());
        let bytes = poulpy_cpu_rayon::workers_within(
            res.size().min(<Self as poulpy_hal::execution::ScratchWorkers>::PREPARE),
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        super::convolution::cnv_prepare_right::<_, RayonTaskExecutor>(module, res, a, tmp);
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
            poulpy_cpu_portable::kernels::ntt4x30::convolution::ntt4x30_cnv_by_const_apply_portable::<Self, SerialTaskExecutor>(
                cnv_offset,
                res,
                res_col,
                a,
                a_col,
                b,
                b_col,
                b_coeff,
                &mut [],
            );
        } else {
            poulpy_cpu_portable::kernels::ntt4x30::convolution::ntt4x30_cnv_by_const_apply_portable::<Self, RayonTaskExecutor>(
                cnv_offset,
                res,
                res_col,
                a,
                a_col,
                b,
                b_col,
                b_coeff,
                &mut [],
            );
        }
    }

    fn cnv_by_const_apply_add_tmp_bytes(
        module: &Module<Self>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        Self::cnv_by_const_apply_tmp_bytes(module, cnv_offset, res_size, a_size, b_size)
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
            poulpy_cpu_portable::kernels::ntt4x30::convolution::ntt4x30_cnv_by_const_apply_add_portable::<Self, SerialTaskExecutor>(
                cnv_offset,
                res,
                res_col,
                a,
                a_col,
                b,
                b_col,
                b_coeff,
                &mut [],
            );
        } else {
            poulpy_cpu_portable::kernels::ntt4x30::convolution::ntt4x30_cnv_by_const_apply_add_portable::<Self, RayonTaskExecutor>(
                cnv_offset,
                res,
                res_col,
                a,
                a_col,
                b,
                b_col,
                b_coeff,
                &mut [],
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

    fn cnv_apply_dft_add_tmp_bytes(
        module: &Module<Self>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        Self::cnv_apply_dft_tmp_bytes(module, cnv_offset, res_size, a_size, b_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft_add(
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
            super::convolution::cnv_apply_dft_add::<_, RayonTaskExecutor>(module, cnv_offset, res, res_col, a, a_col, b, b_col)
        };
    }

    fn cnv_apply_dft_sum_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        _a_size: usize,
        _b_size: usize,
    ) -> usize {
        poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::APPLY)
            * super::convolution::cnv_apply_dft_sum_neon_tmp_bytes(res_size)
    }

    fn cnv_apply_dft_sum(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        terms: &[poulpy_hal::layouts::CnvDftAccTerm<'_, Self>],
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = super::convolution::cnv_apply_dft_sum_neon_tmp_bytes(res.size());
        let bytes = poulpy_cpu_rayon::workers_within(
            <Self as poulpy_hal::execution::ScratchWorkers>::APPLY,
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        unsafe {
            super::convolution::cnv_apply_dft_sum_neon::<_, RayonTaskExecutor>(module, cnv_offset, res, res_col, terms, tmp)
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
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = super::convolution::cnv_prepare_tmp_bytes(left.n());
        let bytes = poulpy_cpu_rayon::workers_within(
            left.size().min(<Self as poulpy_hal::execution::ScratchWorkers>::PREPARE),
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        super::convolution::cnv_prepare_self::<_, RayonTaskExecutor>(module, left, right, a, tmp);
    }
}
unsafe impl<R: Ring> HalVecZnxBigImpl for NTT4x30NeonRayon<R>
where
    NTT4x30Neon<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>> + ZnxAutomorphism,
{
    poulpy_cpu_portable::hal_impl_vec_znx_big_without_normalize!(NTT4x30VecZnxBigDefault);

    fn vec_znx_big_normalize(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_col: usize,
        a: &poulpy_hal::layouts::VecZnxBigBackendRef<'_, Self>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let (carry, _) = poulpy_cpu_rayon::take_scratch::<Self, i128>(scratch.borrow(), 3 * res.n());
        poulpy_cpu_rayon::normalize::ntt4x30_vec_znx_big_normalize_par::<NTT4x30Neon<R>, Self>(
            res,
            res_base2k,
            res_k,
            res_offset,
            res_col,
            &base_big_ref::<R>(a),
            a_base2k,
            a_col,
            carry,
        );
    }
}
unsafe impl<R: Ring> HalSvpImpl for NTT4x30NeonRayon<R>
where
    NTT4x30Neon<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>> + ZnxAutomorphism,
{
    fn svp_prepare(
        module: &Module<Self>,
        res: &mut SvpPPolBackendMut<'_, Self>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        NTT4x30Neon::<R>::svp_prepare(
            base_module(module),
            &mut base_svp_mut::<R>(res),
            res_col,
            &base_scalar_ref::<R>(a),
            a_col,
        );
    }

    fn svp_ppol_copy(
        module: &Module<Self>,
        res: &mut SvpPPolBackendMut<'_, Self>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self>,
        a_col: usize,
    ) {
        NTT4x30Neon::<R>::svp_ppol_copy(
            base_module(module),
            &mut base_svp_mut::<R>(res),
            res_col,
            &base_svp_ref::<R>(a),
            a_col,
        );
    }

    fn svp_apply_dft_tmp_bytes(_module: &Module<Self>, _b_size: usize) -> usize {
        0
    }

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_dft(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut base_scratch = scratch.borrow().into_backend::<NTT4x30Neon<R>>();
        NTT4x30Neon::<R>::svp_apply_dft(
            base_module(module),
            &mut base_dft_mut::<R>(res),
            res_col,
            &base_svp_ref::<R>(a),
            a_col,
            &base_znx_ref::<R>(b),
            b_col,
            &mut base_scratch,
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
        NTT4x30Neon::<R>::svp_apply_dft_to_dft(
            base_module(module),
            &mut base_dft_mut::<R>(res),
            res_col,
            &base_svp_ref::<R>(a),
            a_col,
            &base_dft_ref::<R>(b),
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
        NTT4x30Neon::<R>::svp_apply_dft_to_dft_assign(
            base_module(module),
            &mut base_dft_mut::<R>(res),
            res_col,
            &base_svp_ref::<R>(a),
            a_col,
        );
    }
}
unsafe impl<R: Ring> HalVecZnxDftImpl for NTT4x30NeonRayon<R>
where
    NTT4x30Neon<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>> + ZnxAutomorphism,
{
    fn vec_znx_idft_normalize_consume_tmp_bytes(module: &Module<Self>, _res_size: usize, a_size: usize) -> usize {
        let workers = poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::IDFT).min(a_size.max(1));
        workers * super::vec_znx_dft::idft_tmp_words(module.n()) * size_of::<u64>() + 3 * module.n() * size_of::<i128>()
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_idft_normalize_consume(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, Self>,
        a_col: usize,
        a_base2k: usize,
        addend: Option<(&VecZnxBackendRef<'_, Self>, usize)>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let n = a.n();
        poulpy_hal::layouts::check_degree::<NTT4x30Neon<R>>(module.n(), n);
        assert_eq!(res.n(), n, "vec_znx_idft_normalize_consume: res.n():{} != a.n():{n}", res.n());
        let cols = a.cols();
        let size = a.size();
        let per_worker = super::vec_znx_dft::idft_tmp_words(n);
        let (carry, arena) = crate::hal_impl::take_host_typed::<Self, i128>(scratch.borrow(), 3 * n);
        let workers = poulpy_cpu_rayon::workers_within(
            size.clamp(1, <Self as poulpy_hal::execution::ScratchWorkers>::IDFT),
            per_worker * size_of::<u64>(),
            arena.available(),
        );
        let (worker_tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(arena, workers * per_worker);

        // Each limb is replaced by its coefficients: the planes move to the worker's buffer during the transform.
        let base = base_module(module);
        let data = SendPtr::new(cast_slice_mut::<_, u32>(a.raw_mut()).as_mut_ptr());
        RayonTaskExecutor::for_each_chunked(size, worker_tmp, per_worker, |tmp, limb| {
            let work: &mut [u32] = cast_slice_mut(tmp);
            assert!(work.len() >= 4 * n);
            let slot = unsafe { data.get().add(4 * n * (limb * cols + a_col)) };
            unsafe { idft_limb_planes(base, n, slot as *mut i128, slot, work.as_mut_ptr()) };
        });

        if let Some((add, add_col)) = addend {
            if add.n() == n {
                let big: &mut [i128] = cast_slice_mut(a.raw_mut());
                big.par_chunks_mut(n * cols)
                    .take(size.min(add.size()))
                    .enumerate()
                    .for_each(|(limb, group)| {
                        <NTT4x30Neon<R> as I128BigOps>::i128_add_small_assign(
                            &mut group[n * a_col..][..n],
                            add.at(add_col, limb),
                        );
                    });
            } else {
                let a_shape = a.shape();
                let mut big: VecZnxBigBackendMut<'_, Self> = VecZnxBig::from_shape(&mut **a.data_mut(), a_shape);
                let mut big_ref = &mut big;
                poulpy_cpu_portable::kernels::ntt4x30::vec_znx_big::ntt4x30_vec_znx_big_add_small_assign_portable::<_, _, Self>(
                    &mut big_ref,
                    a_col,
                    &add,
                    add_col,
                );
            }
        }
        let a_shape = a.shape();
        let big_ref: poulpy_hal::layouts::VecZnxBigBackendRef<'_, NTT4x30Neon<R>> = VecZnxBig::from_shape(&**a.data(), a_shape);
        poulpy_cpu_rayon::normalize::ntt4x30_vec_znx_big_normalize_par::<NTT4x30Neon<R>, Self>(
            res, res_base2k, res_k, res_offset, res_col, &big_ref, a_base2k, a_col, carry,
        );
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
        poulpy_hal::layouts::assert_dense(a, "vec_znx_dft_apply");
        assert!(step >= 1, "vec_znx_dft_apply: step must be >= 1");
        if !parallel_limb_tasks(res.size()) {
            return NTT4x30Neon::<R>::vec_znx_dft_apply(
                base_module(module),
                step,
                offset,
                &mut base_dft_mut::<R>(res),
                res_col,
                &base_znx_ref::<R>(a),
                a_col,
            );
        }

        poulpy_hal::layouts::check_degree::<NTT4x30Neon<R>>(module.n(), res.n());
        assert!(a.n() == res.n(), "vec_znx_dft_apply: a.n() != res.n()");
        let n = res.n();
        let cols = res.cols();
        let a_size = a.size();
        let data: &mut [u32] = cast_slice_mut(res.raw_mut());
        data.par_chunks_mut(4 * n * cols).enumerate().for_each(|(limb, group)| {
            let src_limb = offset + limb * step;
            super::vec_znx_dft::dft_limb(
                module,
                n,
                &mut group[4 * n * res_col..][..4 * n],
                (src_limb < a_size).then(|| a.at(a_col, src_limb)),
            );
        });
    }

    fn vec_znx_idft_apply_tmp_bytes(module: &Module<Self>) -> usize {
        NTT4x30Neon::<R>::vec_znx_idft_apply_tmp_bytes(base_module(module)).max(
            poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::IDFT)
                * super::vec_znx_dft::idft_tmp_words(module.n())
                * size_of::<u64>(),
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
            let mut base_scratch = scratch.borrow().into_backend::<NTT4x30Neon<R>>();
            return NTT4x30Neon::<R>::vec_znx_idft_apply(
                base_module(module),
                &mut base_big_mut::<R>(res),
                res_col,
                &base_dft_ref::<R>(a),
                a_col,
                &mut base_scratch,
            );
        }

        poulpy_hal::layouts::check_degree::<NTT4x30Neon<R>>(module.n(), res.n());
        assert_eq!(a.n(), res.n(), "vec_znx_idft_apply: a.n():{} != res.n():{}", a.n(), res.n());
        let n = res.n();
        let res_cols = res.cols();
        let a_cols = a.cols();
        let size = res.size();
        let min_size = size.min(a.size());
        let a_data: &[u32] = cast_slice(a.raw());
        let per_worker = super::vec_znx_dft::idft_tmp_words(n);
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
                let src = super::vec_znx_dft::packed_limb(a_data, n, a_cols, a_col, limb);
                let work: &mut [u32] = cast_slice_mut(tmp);
                assert!(src.len() >= 4 * n && work.len() >= 4 * n);
                unsafe { idft_limb_planes(module, n, dst.as_mut_ptr(), src.as_ptr(), work.as_mut_ptr()) };
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
            return NTT4x30Neon::<R>::vec_znx_idft_apply_tmpa(
                base_module(module),
                &mut base_big_mut::<R>(res),
                res_col,
                &mut base_dft_mut::<R>(a),
                a_col,
            );
        }

        poulpy_hal::layouts::check_degree::<NTT4x30Neon<R>>(module.n(), res.n());
        assert_eq!(
            a.n(),
            res.n(),
            "vec_znx_idft_apply_tmpa: a.n():{} != res.n():{}",
            a.n(),
            res.n()
        );
        let n = res.n();
        let res_cols = res.cols();
        let a_cols = a.cols();
        let min_size = res.size().min(a.size());
        let module = base_module(module);
        let (res_active, res_zero) = res.raw_mut().split_at_mut(min_size * n * res_cols);
        let a_data: &mut [u32] = cast_slice_mut(a.raw_mut());
        res_active
            .par_chunks_mut(n * res_cols)
            .zip(a_data.par_chunks_mut(4 * n * a_cols))
            .for_each(|(res_group, a_group)| {
                let dst = &mut res_group[n * res_col..][..n];
                let src = a_group[4 * n * a_col..][..4 * n].as_mut_ptr();
                unsafe { idft_limb_planes(module, n, dst.as_mut_ptr(), src, src) };
            });
        res_zero
            .par_chunks_mut(n * res_cols)
            .for_each(|group| group[n * res_col..][..n].fill(0));
    }

    fn vec_znx_dft_add(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) {
        if limb_tasks(res.n(), res.size()) {
            super::vec_znx_dft::vec_znx_dft_add::<R, RayonTaskExecutor>(
                &mut base_dft_mut::<R>(res),
                res_col,
                &base_dft_ref::<R>(a),
                a_col,
                &base_dft_ref::<R>(b),
                b_col,
            )
        } else {
            super::vec_znx_dft::vec_znx_dft_add::<R, SerialTaskExecutor>(
                &mut base_dft_mut::<R>(res),
                res_col,
                &base_dft_ref::<R>(a),
                a_col,
                &base_dft_ref::<R>(b),
                b_col,
            )
        }
    }

    fn vec_znx_dft_add_assign(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        if limb_tasks(res.n(), res.size()) {
            super::vec_znx_dft::vec_znx_dft_add_assign::<R, RayonTaskExecutor>(
                &mut base_dft_mut::<R>(res),
                res_col,
                &base_dft_ref::<R>(a),
                a_col,
            )
        } else {
            super::vec_znx_dft::vec_znx_dft_add_assign::<R, SerialTaskExecutor>(
                &mut base_dft_mut::<R>(res),
                res_col,
                &base_dft_ref::<R>(a),
                a_col,
            )
        }
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
        if limb_tasks(res.n(), res.size()) {
            super::vec_znx_dft::vec_znx_dft_sub::<R, RayonTaskExecutor>(
                &mut base_dft_mut::<R>(res),
                res_col,
                &base_dft_ref::<R>(a),
                a_col,
                &base_dft_ref::<R>(b),
                b_col,
            )
        } else {
            super::vec_znx_dft::vec_znx_dft_sub::<R, SerialTaskExecutor>(
                &mut base_dft_mut::<R>(res),
                res_col,
                &base_dft_ref::<R>(a),
                a_col,
                &base_dft_ref::<R>(b),
                b_col,
            )
        }
    }

    fn vec_znx_dft_sub_assign(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        if limb_tasks(res.n(), res.size()) {
            super::vec_znx_dft::vec_znx_dft_sub_assign::<R, RayonTaskExecutor>(
                &mut base_dft_mut::<R>(res),
                res_col,
                &base_dft_ref::<R>(a),
                a_col,
            )
        } else {
            super::vec_znx_dft::vec_znx_dft_sub_assign::<R, SerialTaskExecutor>(
                &mut base_dft_mut::<R>(res),
                res_col,
                &base_dft_ref::<R>(a),
                a_col,
            )
        }
    }

    fn vec_znx_dft_sub_negate_assign(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        if limb_tasks(res.n(), res.size()) {
            super::vec_znx_dft::vec_znx_dft_sub_negate_assign::<R, RayonTaskExecutor>(
                &mut base_dft_mut::<R>(res),
                res_col,
                &base_dft_ref::<R>(a),
                a_col,
            )
        } else {
            super::vec_znx_dft::vec_znx_dft_sub_negate_assign::<R, SerialTaskExecutor>(
                &mut base_dft_mut::<R>(res),
                res_col,
                &base_dft_ref::<R>(a),
                a_col,
            )
        }
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
        if limb_tasks(res.n(), res.size()) {
            super::vec_znx_dft::vec_znx_dft_copy::<R, RayonTaskExecutor>(
                step,
                offset,
                &mut base_dft_mut::<R>(res),
                res_col,
                &base_dft_ref::<R>(a),
                a_col,
            )
        } else {
            super::vec_znx_dft::vec_znx_dft_copy::<R, SerialTaskExecutor>(
                step,
                offset,
                &mut base_dft_mut::<R>(res),
                res_col,
                &base_dft_ref::<R>(a),
                a_col,
            )
        }
    }

    fn vec_znx_dft_zero(module: &Module<Self>, res: &mut VecZnxDftBackendMut<'_, Self>, res_col: usize) {
        NTT4x30Neon::<R>::vec_znx_dft_zero(base_module(module), &mut base_dft_mut::<R>(res), res_col)
    }

    type AutomorphismPlan = <NTT4x30Neon<R> as HalVecZnxDftImpl>::AutomorphismPlan;

    fn vec_znx_dft_automorphism_plan(module: &Module<Self>, n: usize, p: i64) -> Self::AutomorphismPlan {
        NTT4x30Neon::<R>::vec_znx_dft_automorphism_plan(base_module(module), n, p)
    }

    fn vec_znx_dft_automorphism_with_plan(
        _module: &Module<Self>,
        plan: &Self::AutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        super::vec_znx_dft::vec_znx_dft_automorphism(plan, &mut base_dft_mut::<R>(res), res_col, &base_dft_ref::<R>(a), a_col);
    }

    fn vec_znx_dft_automorphism_add_with_plan_tmp_bytes(_module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        0
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_dft_automorphism_add_with_plan(
        _module: &Module<Self>,
        plan: &Self::AutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let _ = scratch;
        if RayonTaskExecutor::should_serialize_inner() {
            super::vec_znx_dft::vec_znx_dft_automorphism_add::<_, SerialTaskExecutor>(
                plan,
                &mut base_dft_mut::<R>(res),
                res_col,
                &base_dft_ref::<R>(a),
                a_col,
            );
        } else {
            super::vec_znx_dft::vec_znx_dft_automorphism_add::<_, RayonTaskExecutor>(
                plan,
                &mut base_dft_mut::<R>(res),
                res_col,
                &base_dft_ref::<R>(a),
                a_col,
            );
        }
    }
}

impl<R: Ring> poulpy_hal::execution::ScratchWorkers for NTT4x30NeonRayon<R> {
    const PREPARE: usize = 32;
    const APPLY: usize = 32;
    const VMP: usize = 32;
    const IDFT: usize = 32;
}

impl<R: Ring> poulpy_cpu_rayon::RayonTuning for NTT4x30NeonRayon<R> {
    const COEFF_MIN_LEN: usize = 1 << 17;
    const COEFF_MIN_TASK: usize = 1 << 13;
    const NORMALIZE_MIN_TASK: usize = 1 << 12;
}

#[cfg(test)]
mod tests {
    use poulpy_cpu_portable::kernels::znx::ZnxAdd;
    use poulpy_hal::{layouts::Module, test_suite::convolution::test_convolution_by_const};

    use super::NTT4x30NeonRayon;

    #[test]
    fn coefficient_add_matches_wrapping_arithmetic() {
        let a = vec![i64::MAX; 1 << 16];
        let b = vec![1; 1 << 16];
        let mut actual = vec![0; 1 << 16];
        <NTT4x30NeonRayon as ZnxAdd>::znx_add(&mut actual, &a, &b);
        assert!(actual.iter().all(|&x| x == i64::MIN));
    }

    #[test]
    fn convolution_by_const() {
        test_convolution_by_const(&Module::<NTT4x30NeonRayon>::new(1 << 8), 1 << 8, 50);
    }
}
