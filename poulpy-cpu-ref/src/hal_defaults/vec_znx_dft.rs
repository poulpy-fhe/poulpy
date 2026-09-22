//! Backend extension points for DFT-domain [`poulpy_hal::layouts::VecZnxDft`] operations.

use std::mem::size_of;

use crate::reference::{
    fft64::{
        module::FFTModuleHandle,
        reim::{ReimArith, ReimFFTExecute, ReimFFTTable, ReimIFFTTable},
        vec_znx_dft::{
            Fft64AutomorphismPlan, build_fft64_automorphism_plan, vec_znx_dft_add as fft64_vec_znx_dft_add,
            vec_znx_dft_add_assign as fft64_vec_znx_dft_add_assign, vec_znx_dft_apply as fft64_vec_znx_dft_apply,
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
        vec_znx_dft::{
            NttAutomorphismPlan, NttModuleHandle, build_ntt4x30_automorphism_plan,
            ntt4x30_vec_znx_dft_add as ntt4x30_default_vec_znx_dft_add,
            ntt4x30_vec_znx_dft_add_assign as ntt4x30_default_vec_znx_dft_add_assign,
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
    znx::ZnxZero,
};
use poulpy_hal::{
    api::HostBufMut,
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ScratchArena, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut,
        VecZnxDftBackendRef, check_degree,
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
pub trait FFT64VecZnxDftDefault: Backend<ZnxWord = i64>
where
    Self::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    type AutomorphismPlanDefault: Send + Sync;

    fn vec_znx_dft_apply_default(
        module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Module<Self>: FFTModuleHandle<f64>,
        Self: Backend<DftWord = f64, ZnxWord = i64>
            + ReimArith
            + ReimFFTExecute<ReimFFTTable<f64>, f64>
            + ReimFFTExecute<ReimIFFTTable<f64>, f64>
            + 'static,
        for<'x> Self: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8], ZnxWord = i64>,
    {
        let n: usize = res.n();
        check_degree::<Self>(module.n(), n);
        fft64_vec_znx_dft_apply::<Self>(module.get_fft_plan(n), step, offset, res, res_col, a, a_col);
    }

    fn vec_znx_idft_apply_tmp_bytes_default(_module: &Module<Self>) -> usize
    where
        Self: Backend<DftWord = f64, ZnxWord = i64>,
    {
        0
    }

    fn vec_znx_idft_apply_default(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: FFTModuleHandle<f64>,
        Self: Backend<DftWord = f64, BigWord = i64, ZnxWord = i64>
            + ReimArith
            + ReimFFTExecute<ReimFFTTable<f64>, f64>
            + ReimFFTExecute<ReimIFFTTable<f64>, f64>
            + ZnxZero,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        let _ = scratch;
        let n: usize = res.n();
        check_degree::<Self>(module.n(), n);
        fft64_vec_znx_idft_apply::<Self>(module.get_fft_plan(n), res, res_col, a, a_col);
    }

    fn vec_znx_idft_apply_tmpa_default(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, Self>,
        a_col: usize,
    ) where
        Module<Self>: FFTModuleHandle<f64>,
        Self: Backend<DftWord = f64, BigWord = i64, ZnxWord = i64>
            + ReimArith
            + ReimFFTExecute<ReimFFTTable<f64>, f64>
            + ReimFFTExecute<ReimIFFTTable<f64>, f64>
            + ZnxZero,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
    {
        let n: usize = res.n();
        check_degree::<Self>(module.n(), n);
        fft64_vec_znx_idft_apply_tmpa::<Self>(module.get_fft_plan(n), res, res_col, a, a_col);
    }
    fn vec_znx_dft_add_default(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) where
        Self: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_add::<Self>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_dft_add_assign_default(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_add_assign::<Self>(res, res_col, a, a_col);
    }

    fn vec_znx_dft_sub_default(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) where
        Self: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_sub::<Self>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_dft_sub_assign_default(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_sub_assign::<Self>(res, res_col, a, a_col);
    }

    fn vec_znx_dft_sub_negate_assign_default(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_sub_negate_assign::<Self>(res, res_col, a, a_col);
    }

    fn vec_znx_dft_copy_default(
        _module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_copy::<Self>(step, offset, res, res_col, a, a_col);
    }

    fn vec_znx_dft_zero_default(_module: &Module<Self>, res: &mut VecZnxDftBackendMut<'_, Self>, res_col: usize)
    where
        Self: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
    {
        fft64_vec_znx_dft_zero::<Self>(res, res_col);
    }

    fn vec_znx_dft_automorphism_plan_default(_module: &Module<Self>, n: usize, p: i64) -> Fft64AutomorphismPlan
    where
        Module<Self>: FFTModuleHandle<f64>,
        Self: Backend<DftWord = f64, ZnxWord = i64>,
    {
        build_fft64_automorphism_plan::<Self>(n, p)
    }

    fn vec_znx_dft_automorphism_with_plan_default(
        module: &Module<Self>,
        plan: &Fft64AutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        check_degree::<Self>(module.n(), res.n());
        fft64_vec_znx_dft_automorphism::<Self>(plan, res, res_col, a, a_col);
    }

    fn vec_znx_dft_automorphism_add_with_plan_default(
        module: &Module<Self>,
        plan: &Fft64AutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: Backend<DftWord = f64, ZnxWord = i64>,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        check_degree::<Self>(module.n(), res.n());
        fft64_vec_znx_dft_automorphism_add::<Self, poulpy_hal::execution::SerialTaskExecutor>(plan, res, res_col, a, a_col);
    }
}

impl<BE: Backend<ZnxWord = i64>> FFT64VecZnxDftDefault for BE
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    type AutomorphismPlanDefault = Fft64AutomorphismPlan;
}

#[doc(hidden)]
pub trait NTT4x30VecZnxDftDefault: Backend<ZnxWord = i64>
where
    Self::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    /// NTT4x30 automorphism plan type. Implementation lands as a follow-up
    /// step; the placeholder unit type keeps the OEP wiring consistent
    /// across backends.
    type AutomorphismPlanDefault: Send + Sync;

    fn vec_znx_dft_apply_default(
        module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Module<Self>: NttModuleHandle,
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64>
            + NttDFTExecute<NttTable<Primes30, <Module<Self> as crate::reference::ntt4x30::vec_znx_dft::NttModuleHandle>::Ring>>
            + NttFromZnx64
            + NttZero
            + 'static,
        for<'x> Self: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8], ZnxWord = i64>,
    {
        ntt4x30_default_vec_znx_dft_apply::<Self>(module, step, offset, res, res_col, a, a_col);
    }

    fn vec_znx_idft_apply_tmp_bytes_default(module: &Module<Self>) -> usize
    where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64>,
    {
        ntt4x30_default_vec_znx_idft_apply_tmp_bytes(module.n())
    }

    fn vec_znx_idft_apply_default(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: NttModuleHandle,
        Self: Backend<DftWord = Q120bScalar, BigWord = i128, ZnxWord = i64>
            + NttDFTExecute<NttTableInv<Primes30, <Module<Self> as crate::reference::ntt4x30::vec_znx_dft::NttModuleHandle>::Ring>>
            + NttToZnx128
            + NttCopy,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
    {
        let (tmp, _) = take_host_typed::<Self, u64>(
            scratch.borrow(),
            ntt4x30_default_vec_znx_idft_apply_tmp_bytes(res.n()) / size_of::<u64>(),
        );
        ntt4x30_default_vec_znx_idft_apply::<Self>(module, res, res_col, a, a_col, tmp);
    }

    fn vec_znx_idft_apply_tmpa_default(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, Self>,
        a_col: usize,
    ) where
        Module<Self>: NttModuleHandle,
        Self: Backend<DftWord = Q120bScalar, BigWord = i128, ZnxWord = i64>
            + NttDFTExecute<NttTableInv<Primes30, <Module<Self> as crate::reference::ntt4x30::vec_znx_dft::NttModuleHandle>::Ring>>
            + NttToZnx128,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
    {
        ntt4x30_default_vec_znx_idft_apply_tmpa::<Self>(module, res, res_col, a, a_col);
    }
    fn vec_znx_dft_add_default(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttAdd + NttCopy + NttZero,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt4x30_default_vec_znx_dft_add::<Self>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_dft_add_assign_default(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttAddAssign,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt4x30_default_vec_znx_dft_add_assign::<Self>(res, res_col, a, a_col);
    }

    fn vec_znx_dft_sub_default(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttSub + NttNegate + NttCopy + NttZero,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt4x30_default_vec_znx_dft_sub::<Self>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_dft_sub_assign_default(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttSubAssign,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt4x30_default_vec_znx_dft_sub_assign::<Self>(res, res_col, a, a_col);
    }

    fn vec_znx_dft_sub_negate_assign_default(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttSubNegateAssign + NttNegateAssign,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt4x30_default_vec_znx_dft_sub_negate_assign::<Self>(res, res_col, a, a_col);
    }

    fn vec_znx_dft_copy_default(
        _module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttCopy + NttZero,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt4x30_default_vec_znx_dft_copy::<Self>(step, offset, res, res_col, a, a_col);
    }

    fn vec_znx_dft_zero_default(_module: &Module<Self>, res: &mut VecZnxDftBackendMut<'_, Self>, res_col: usize)
    where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttZero,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
    {
        ntt4x30_default_vec_znx_dft_zero::<Self>(res, res_col);
    }

    fn vec_znx_dft_automorphism_plan_default(_module: &Module<Self>, n: usize, p: i64) -> NttAutomorphismPlan
    where
        Self: Backend<ZnxWord = i64>,
        Module<Self>: NttModuleHandle,
    {
        if Self::CYCLOTOMIC_ORDER_FACTOR == 4 {
            NttAutomorphismPlan {
                p,
                perm: crate::reference::conjugate_invariant::ntt_automorphism_permutation(n, p)
                    .into_iter()
                    .map(|x| x as u32)
                    .collect(),
            }
        } else {
            build_ntt4x30_automorphism_plan(n, p)
        }
    }

    fn vec_znx_dft_automorphism_with_plan_default(
        module: &Module<Self>,
        plan: &NttAutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttZero,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        check_degree::<Self>(module.n(), res.n());
        ntt4x30_default_vec_znx_dft_automorphism::<Self>(plan, res, res_col, a, a_col);
    }

    fn vec_znx_dft_automorphism_add_with_plan_default(
        module: &Module<Self>,
        plan: &NttAutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttAddAssign,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        check_degree::<Self>(module.n(), res.n());
        ntt4x30_default_vec_znx_dft_automorphism_add::<Self, poulpy_hal::execution::SerialTaskExecutor>(
            plan, res, res_col, a, a_col,
        );
    }
}

impl<BE: Backend<ZnxWord = i64>> NTT4x30VecZnxDftDefault for BE
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    type AutomorphismPlanDefault = NttAutomorphismPlan;
}
