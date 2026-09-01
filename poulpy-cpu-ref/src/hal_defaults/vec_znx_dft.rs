//! Backend extension points for DFT-domain [`poulpy_hal::layouts::VecZnxDft`] operations.

use std::mem::size_of;

use crate::reference::{
    fft64::{
        module::FFTModuleHandle,
        reim::{ReimArith, ReimFFTExecute, ReimFFTTable, ReimIFFTTable},
        vec_znx_big::{
            vec_znx_big_add_small_assign as fft64_vec_znx_big_add_small_assign,
            vec_znx_big_normalize as fft64_default_vec_znx_big_normalize,
        },
        vec_znx_dft::{
            Fft64AutomorphismPlan, build_fft64_automorphism_plan, vec_znx_dft_add_assign as fft64_vec_znx_dft_add_assign,
            vec_znx_dft_add_into as fft64_vec_znx_dft_add_into,
            vec_znx_dft_add_scaled_assign as fft64_vec_znx_dft_add_scaled_assign, vec_znx_dft_apply as fft64_vec_znx_dft_apply,
            vec_znx_dft_automorphism as fft64_vec_znx_dft_automorphism,
            vec_znx_dft_automorphism_add as fft64_vec_znx_dft_automorphism_add, vec_znx_dft_copy as fft64_vec_znx_dft_copy,
            vec_znx_dft_sub as fft64_vec_znx_dft_sub, vec_znx_dft_sub_assign as fft64_vec_znx_dft_sub_assign,
            vec_znx_dft_sub_negate_assign as fft64_vec_znx_dft_sub_negate_assign, vec_znx_dft_zero as fft64_vec_znx_dft_zero,
            vec_znx_idft_apply as fft64_vec_znx_idft_apply, vec_znx_idft_apply_tmpa as fft64_vec_znx_idft_apply_tmpa,
        },
    },
    ntt4x30::{
        NttAdd, NttAddAssign, NttCopy, NttDFTExecute, NttFromZnx64, NttNegate, NttNegateAssign, NttSub, NttSubAssign,
        NttSubNegateAssign, NttToZnx128, NttZero,
        ntt::{NttTable, NttTableInv},
        primes::Primes30,
        types::Q120bScalar,
        vec_znx_big::{
            I128BigOps, I128NormalizeOps, ntt4x30_vec_znx_big_add_small_assign,
            ntt4x30_vec_znx_big_normalize as ntt4x30_default_vec_znx_big_normalize,
        },
        vec_znx_dft::{
            NttAutomorphismPlan, NttModuleHandle, build_ntt4x30_automorphism_plan,
            ntt4x30_vec_znx_dft_add_assign as ntt4x30_default_vec_znx_dft_add_assign,
            ntt4x30_vec_znx_dft_add_into as ntt4x30_default_vec_znx_dft_add_into,
            ntt4x30_vec_znx_dft_add_scaled_assign as ntt4x30_default_vec_znx_dft_add_scaled_assign,
            ntt4x30_vec_znx_dft_apply as ntt4x30_default_vec_znx_dft_apply,
            ntt4x30_vec_znx_dft_automorphism as ntt4x30_default_vec_znx_dft_automorphism,
            ntt4x30_vec_znx_dft_automorphism_add as ntt4x30_default_vec_znx_dft_automorphism_add,
            ntt4x30_vec_znx_dft_copy as ntt4x30_default_vec_znx_dft_copy,
            ntt4x30_vec_znx_dft_sub as ntt4x30_default_vec_znx_dft_sub,
            ntt4x30_vec_znx_dft_sub_assign as ntt4x30_default_vec_znx_dft_sub_assign,
            ntt4x30_vec_znx_dft_sub_negate_assign as ntt4x30_default_vec_znx_dft_sub_negate_assign,
            ntt4x30_vec_znx_dft_zero as ntt4x30_default_vec_znx_dft_zero,
            ntt4x30_vec_znx_idft_apply as ntt4x30_default_vec_znx_idft_apply,
            ntt4x30_vec_znx_idft_apply_tmp_bytes as ntt4x30_default_vec_znx_idft_apply_tmp_bytes,
            ntt4x30_vec_znx_idft_apply_tmpa as ntt4x30_default_vec_znx_idft_apply_tmpa,
        },
    },
    znx::{
        ZnxAddAssign, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulPowerOfTwoAssign, ZnxNormalizeDigit, ZnxNormalizeFinalStep,
        ZnxNormalizeFinalStepAssign, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeMiddleStep,
        ZnxNormalizeMiddleStepAssign, ZnxNormalizeMiddleStepCarryOnly, ZnxZero,
    },
};
use poulpy_hal::{
    api::HostBufMut,
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, NormalizationState, ScratchArena, VecZnxBackendMut, VecZnxBackendRef,
        VecZnxBig, VecZnxBigBackendMut, VecZnxDftBackendMut, VecZnxDftBackendRef,
    },
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

#[doc(hidden)]
pub trait FFT64VecZnxDftDefault<BE: Backend<ZnxWord = i64>>: Backend
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    type AutomorphismPlanDefault: Send + Sync;

    fn vec_znx_dft_apply_default(
        module: &Module<BE>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        Module<BE>: FFTModuleHandle<f64>,
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64> + 'static,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8], ZnxWord = i64>,
    {
        fft64_vec_znx_dft_apply::<BE>(module.get_fft_table(), step, offset, res, res_col, a, a_col);
    }

    fn vec_znx_idft_apply_tmp_bytes_default(_module: &Module<BE>) -> usize
    where
        BE: Backend<DftWord = f64, ZnxWord = i64>,
    {
        0
    }

    fn vec_znx_idft_apply_default(
        module: &Module<BE>,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: FFTModuleHandle<f64>,
        BE: Backend<DftWord = f64, BigWord = i64, ZnxWord = i64> + ReimArith + ReimFFTExecute<ReimIFFTTable<f64>, f64> + ZnxZero,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        let _ = scratch;
        fft64_vec_znx_idft_apply::<BE>(module.get_ifft_table(), res, res_col, a, a_col);
    }

    fn vec_znx_idft_apply_tmpa_default(
        module: &Module<BE>,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, BE>,
        a_col: usize,
    ) where
        Module<BE>: FFTModuleHandle<f64>,
        BE: Backend<DftWord = f64, BigWord = i64, ZnxWord = i64> + ReimArith + ReimFFTExecute<ReimIFFTTable<f64>, f64> + ZnxZero,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    {
        fft64_vec_znx_idft_apply_tmpa::<BE>(module.get_ifft_table(), res, res_col, a, a_col);
    }
    fn vec_znx_idft_normalize_consume_tmp_bytes_default(module: &Module<BE>, _res_size: usize, a_size: usize) -> usize
    where
        BE: Backend<DftWord = f64, BigWord = i64, ZnxWord = i64>,
    {
        BE::bytes_of_vec_znx_big(module.n(), 1, a_size) + 3 * module.n() * size_of::<i64>()
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_idft_normalize_consume_default(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE, impl NormalizationState>,
        res_base2k: usize,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, BE>,
        a_col: usize,
        a_base2k: usize,
        addend: Option<(&VecZnxBackendRef<'_, BE, impl NormalizationState>, usize)>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: FFTModuleHandle<f64>,
        BE: Backend<DftWord = f64, BigWord = i64, ZnxWord = i64>
            + ReimArith
            + ReimFFTExecute<ReimIFFTTable<f64>, f64>
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
            + ZnxNormalizeMiddleStepAssign
            + ZnxNormalizeFinalStepAssign
            + ZnxNormalizeDigit,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
        BE: 'static,
    {
        let n = module.n();
        let a_size = a.size();
        let arena = scratch.borrow();
        let (big_bytes, arena) = take_host_typed::<BE, u8>(arena, BE::bytes_of_vec_znx_big(n, 1, a_size));
        let (carry, _) = take_host_typed::<BE, i64>(arena, 3 * n);
        {
            let mut big: VecZnxBigBackendMut<'_, BE> = VecZnxBig::from_data(&mut *big_bytes, n, 1, a_size);
            fft64_vec_znx_idft_apply_tmpa::<BE>(module.get_ifft_table(), &mut big, 0, a, a_col);
            if let Some((add, add_col)) = addend {
                let mut big_ref: &mut VecZnxBigBackendMut<'_, BE> = &mut big;
                fft64_vec_znx_big_add_small_assign::<_, _, BE>(&mut big_ref, 0, &add, add_col);
            }
        }
        let big_ref: poulpy_hal::layouts::VecZnxBigBackendRef<'_, BE> = VecZnxBig::from_data(&*big_bytes, n, 1, a_size);
        let mut res_ref: &mut VecZnxBackendMut<'_, BE, _> = res;
        fft64_default_vec_znx_big_normalize::<_, _, BE>(&mut res_ref, res_base2k, 0, res_col, &&big_ref, a_base2k, 0, carry);
    }

    fn vec_znx_dft_add_into_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    ) where
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_add_into::<BE>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_dft_add_scaled_assign_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
        a_scale: i64,
    ) where
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_add_scaled_assign::<BE>(res, res_col, a, a_col, a_scale);
    }

    fn vec_znx_dft_add_assign_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_add_assign::<BE>(res, res_col, a, a_col);
    }

    fn vec_znx_dft_sub_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    ) where
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_sub::<BE>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_dft_sub_assign_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_sub_assign::<BE>(res, res_col, a, a_col);
    }

    fn vec_znx_dft_sub_negate_assign_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_sub_negate_assign::<BE>(res, res_col, a, a_col);
    }

    fn vec_znx_dft_copy_default(
        _module: &Module<BE>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_copy::<BE>(step, offset, res, res_col, a, a_col);
    }

    fn vec_znx_dft_zero_default(_module: &Module<BE>, res: &mut VecZnxDftBackendMut<'_, BE>, res_col: usize)
    where
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    {
        fft64_vec_znx_dft_zero::<BE>(res, res_col);
    }

    fn vec_znx_dft_automorphism_plan_default(module: &Module<BE>, p: i64) -> Fft64AutomorphismPlan
    where
        BE: Backend<DftWord = f64, ZnxWord = i64>,
    {
        build_fft64_automorphism_plan(module.n(), p)
    }

    fn vec_znx_dft_automorphism_with_plan_default(
        _module: &Module<BE>,
        plan: &Fft64AutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_automorphism::<BE>(plan, res, res_col, a, a_col);
    }

    fn vec_znx_dft_automorphism_add_with_plan_default(
        _module: &Module<BE>,
        plan: &Fft64AutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<DftWord = f64, ZnxWord = i64>,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_automorphism_add::<BE, poulpy_hal::execution::SerialTaskExecutor>(plan, res, res_col, a, a_col);
    }
}

impl<BE: Backend<ZnxWord = i64>> FFT64VecZnxDftDefault<BE> for BE
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    type AutomorphismPlanDefault = Fft64AutomorphismPlan;
}

#[doc(hidden)]
pub trait NTT4x30VecZnxDftDefault<BE: Backend<ZnxWord = i64>>: Backend
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    /// NTT4x30 automorphism plan type. Implementation lands as a follow-up
    /// step; the placeholder unit type keeps the OEP wiring consistent
    /// across backends.
    type AutomorphismPlanDefault: Send + Sync;

    fn vec_znx_dft_apply_default(
        module: &Module<BE>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttDFTExecute<NttTable<Primes30>> + NttFromZnx64 + NttZero + 'static,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8], ZnxWord = i64>,
    {
        ntt4x30_default_vec_znx_dft_apply::<BE>(module, step, offset, res, res_col, a, a_col);
    }

    fn vec_znx_idft_apply_tmp_bytes_default(module: &Module<BE>) -> usize
    where
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>,
    {
        ntt4x30_default_vec_znx_idft_apply_tmp_bytes(module.n())
    }

    fn vec_znx_idft_apply_default(
        module: &Module<BE>,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<DftWord = Q120bScalar, BigWord = i128, ZnxWord = i64>
            + NttDFTExecute<NttTableInv<Primes30>>
            + NttToZnx128
            + NttCopy,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let (tmp, _) = take_host_typed::<BE, u64>(
            scratch.borrow(),
            ntt4x30_default_vec_znx_idft_apply_tmp_bytes(module.n()) / size_of::<u64>(),
        );
        ntt4x30_default_vec_znx_idft_apply::<BE>(module, res, res_col, a, a_col, tmp);
    }

    fn vec_znx_idft_apply_tmpa_default(
        module: &Module<BE>,
        res: &mut VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, BE>,
        a_col: usize,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<DftWord = Q120bScalar, BigWord = i128, ZnxWord = i64> + NttDFTExecute<NttTableInv<Primes30>> + NttToZnx128,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    {
        ntt4x30_default_vec_znx_idft_apply_tmpa::<BE>(module, res, res_col, a, a_col);
    }
    fn vec_znx_idft_normalize_consume_tmp_bytes_default(module: &Module<BE>, _res_size: usize, a_size: usize) -> usize
    where
        BE: Backend<DftWord = Q120bScalar, BigWord = i128, ZnxWord = i64>,
    {
        BE::bytes_of_vec_znx_big(module.n(), 1, a_size) + 3 * module.n() * size_of::<i128>()
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_idft_normalize_consume_default(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE, impl NormalizationState>,
        res_base2k: usize,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, BE>,
        a_col: usize,
        a_base2k: usize,
        addend: Option<(&VecZnxBackendRef<'_, BE, impl NormalizationState>, usize)>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<DftWord = Q120bScalar, BigWord = i128, ZnxWord = i64>
            + NttDFTExecute<NttTableInv<Primes30>>
            + NttToZnx128
            + I128BigOps
            + I128NormalizeOps,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
        BE: 'static,
    {
        let n = module.n();
        let a_size = a.size();
        let arena = scratch.borrow();
        let (big_bytes, arena) = take_host_typed::<BE, u8>(arena, BE::bytes_of_vec_znx_big(n, 1, a_size));
        let (carry, _) = take_host_typed::<BE, i128>(arena, 3 * n);
        {
            let mut big: VecZnxBigBackendMut<'_, BE> = VecZnxBig::from_data(&mut *big_bytes, n, 1, a_size);
            ntt4x30_default_vec_znx_idft_apply_tmpa::<BE>(module, &mut big, 0, a, a_col);
            if let Some((add, add_col)) = addend {
                let mut big_ref: &mut VecZnxBigBackendMut<'_, BE> = &mut big;
                ntt4x30_vec_znx_big_add_small_assign::<_, _, BE>(&mut big_ref, 0, &add, add_col);
            }
        }
        let big_ref: poulpy_hal::layouts::VecZnxBigBackendRef<'_, BE> = VecZnxBig::from_data(&*big_bytes, n, 1, a_size);
        let mut res_ref: &mut VecZnxBackendMut<'_, BE, _> = res;
        ntt4x30_default_vec_znx_big_normalize::<_, _, BE>(&mut res_ref, res_base2k, 0, res_col, &&big_ref, a_base2k, 0, carry);
    }

    fn vec_znx_dft_add_into_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    ) where
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttAdd + NttCopy + NttZero,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt4x30_default_vec_znx_dft_add_into::<BE>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_dft_add_scaled_assign_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
        a_scale: i64,
    ) where
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttAddAssign,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt4x30_default_vec_znx_dft_add_scaled_assign::<BE>(res, res_col, a, a_col, a_scale);
    }

    fn vec_znx_dft_add_assign_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttAddAssign,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt4x30_default_vec_znx_dft_add_assign::<BE>(res, res_col, a, a_col);
    }

    fn vec_znx_dft_sub_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    ) where
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttSub + NttNegate + NttCopy + NttZero,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt4x30_default_vec_znx_dft_sub::<BE>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_dft_sub_assign_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttSubAssign,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt4x30_default_vec_znx_dft_sub_assign::<BE>(res, res_col, a, a_col);
    }

    fn vec_znx_dft_sub_negate_assign_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttSubNegateAssign + NttNegateAssign,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt4x30_default_vec_znx_dft_sub_negate_assign::<BE>(res, res_col, a, a_col);
    }

    fn vec_znx_dft_copy_default(
        _module: &Module<BE>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttCopy + NttZero,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt4x30_default_vec_znx_dft_copy::<BE>(step, offset, res, res_col, a, a_col);
    }

    fn vec_znx_dft_zero_default(_module: &Module<BE>, res: &mut VecZnxDftBackendMut<'_, BE>, res_col: usize)
    where
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttZero,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    {
        ntt4x30_default_vec_znx_dft_zero::<BE>(res, res_col);
    }

    fn vec_znx_dft_automorphism_plan_default(module: &Module<BE>, p: i64) -> NttAutomorphismPlan
    where
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>,
    {
        build_ntt4x30_automorphism_plan(module.n(), p)
    }

    fn vec_znx_dft_automorphism_with_plan_default(
        _module: &Module<BE>,
        plan: &NttAutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttZero,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt4x30_default_vec_znx_dft_automorphism::<BE>(plan, res, res_col, a, a_col);
    }

    fn vec_znx_dft_automorphism_add_with_plan_default(
        _module: &Module<BE>,
        plan: &NttAutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttAddAssign,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt4x30_default_vec_znx_dft_automorphism_add::<BE, poulpy_hal::execution::SerialTaskExecutor>(
            plan, res, res_col, a, a_col,
        );
    }
}

impl<BE: Backend<ZnxWord = i64>> NTT4x30VecZnxDftDefault<BE> for BE
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    type AutomorphismPlanDefault = NttAutomorphismPlan;
}
