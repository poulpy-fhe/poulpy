//! Backend extension points for DFT-domain [`poulpy_hal::layouts::VecZnxDft`] operations.

use crate::kernels::fft64::ring_arith::Fft64RingArith;
use std::mem::size_of;

use crate::kernels::{
    fft64::{
        module::FFTModuleHandle,
        reim::ReimArith,
        vec_znx_dft::{
            Fft64AutomorphismPlan, vec_znx_dft_add_assign_portable as fft64_vec_znx_dft_add_assign,
            vec_znx_dft_add_portable as fft64_vec_znx_dft_add, vec_znx_dft_apply_portable as fft64_vec_znx_dft_apply,
            vec_znx_dft_automorphism_add_portable as fft64_vec_znx_dft_automorphism_add,
            vec_znx_dft_automorphism_portable as fft64_vec_znx_dft_automorphism,
            vec_znx_dft_copy_portable as fft64_vec_znx_dft_copy, vec_znx_dft_sub_assign_portable as fft64_vec_znx_dft_sub_assign,
            vec_znx_dft_sub_negate_assign_portable as fft64_vec_znx_dft_sub_negate_assign,
            vec_znx_dft_sub_portable as fft64_vec_znx_dft_sub, vec_znx_dft_zero_portable as fft64_vec_znx_dft_zero,
            vec_znx_idft_apply_portable as fft64_vec_znx_idft_apply,
        },
    },
    ntt4x30::{
        NttAdd, NttAddAssign, NttCopy, NttDFTExecute, NttFromZnx64, NttNegate, NttNegateAssign, NttSub, NttSubAssign,
        NttSubNegateAssign, NttToZnx128, NttZero,
        ntt::{NttTable, NttTableInv},
        primes::Primes30,
        types::Q120bScalar,
        vec_znx_dft::{
            NttAutomorphismPlan, NttModuleHandle,
            ntt4x30_vec_znx_dft_add_assign_portable as ntt4x30_default_vec_znx_dft_add_assign,
            ntt4x30_vec_znx_dft_add_portable as ntt4x30_default_vec_znx_dft_add,
            ntt4x30_vec_znx_dft_apply_portable as ntt4x30_default_vec_znx_dft_apply,
            ntt4x30_vec_znx_dft_automorphism_add_portable as ntt4x30_default_vec_znx_dft_automorphism_add,
            ntt4x30_vec_znx_dft_automorphism_portable as ntt4x30_default_vec_znx_dft_automorphism,
            ntt4x30_vec_znx_dft_copy_portable as ntt4x30_default_vec_znx_dft_copy,
            ntt4x30_vec_znx_dft_sub_assign_portable as ntt4x30_default_vec_znx_dft_sub_assign,
            ntt4x30_vec_znx_dft_sub_negate_assign_portable as ntt4x30_default_vec_znx_dft_sub_negate_assign,
            ntt4x30_vec_znx_dft_sub_portable as ntt4x30_default_vec_znx_dft_sub,
            ntt4x30_vec_znx_dft_zero_portable as ntt4x30_default_vec_znx_dft_zero,
            ntt4x30_vec_znx_idft_apply_portable as ntt4x30_default_vec_znx_idft_apply,
            ntt4x30_vec_znx_idft_apply_tmp_bytes_portable as ntt4x30_default_vec_znx_idft_apply_tmp_bytes,
        },
    },
    znx::ZnxZero,
};
use poulpy_hal::{
    api::HostBufMut,
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, Ring, ScratchArena, Standard, VecZnxBackendRef, VecZnxBigBackendMut,
        VecZnxDftBackendMut, VecZnxDftBackendRef, check_degree,
    },
};

use super::take_host_typed;

#[doc(hidden)]
pub trait FFT64VecZnxDftDefault<Rg: Ring = Standard>: Backend<ZnxWord = i64>
where
    Self::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    type AutomorphismPlanDefault: Send + Sync;

    fn vec_znx_dft_apply_default(
        module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self, Rg>,
        a_col: usize,
    ) where
        Module<Self>: FFTModuleHandle<f64, Rg>,
        Self: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + Fft64RingArith<Rg> + 'static,
        for<'x> Self: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8], ZnxWord = i64>,
    {
        let n: usize = res.n();
        check_degree::<Self>(module.n(), n);
        fft64_vec_znx_dft_apply::<Self, Rg>(module.get_fft_plan(n), step, offset, res, res_col, a, a_col);
    }

    fn vec_znx_idft_apply_tmp_bytes_default(_module: &Module<Self>) -> usize
    where
        Self: Backend<DftWord = f64, ZnxWord = i64>,
    {
        0
    }

    fn vec_znx_idft_apply_default(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: FFTModuleHandle<f64, Rg>,
        Self: Backend<DftWord = f64, BigWord = i64, ZnxWord = i64> + ReimArith + Fft64RingArith<Rg> + ZnxZero,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        let _ = scratch;
        let n: usize = res.n();
        check_degree::<Self>(module.n(), n);
        fft64_vec_znx_idft_apply::<Self, Rg>(module.get_fft_plan(n), res, res_col, a, a_col);
    }

    fn vec_znx_dft_add_default(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self, Rg>,
        b_col: usize,
    ) where
        Self: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_add::<Self, Rg>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_dft_add_assign_default(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
    ) where
        Self: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_add_assign::<Self, Rg>(res, res_col, a, a_col);
    }

    fn vec_znx_dft_sub_default(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self, Rg>,
        b_col: usize,
    ) where
        Self: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_sub::<Self, Rg>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_dft_sub_assign_default(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
    ) where
        Self: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_sub_assign::<Self, Rg>(res, res_col, a, a_col);
    }

    fn vec_znx_dft_sub_negate_assign_default(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
    ) where
        Self: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_sub_negate_assign::<Self, Rg>(res, res_col, a, a_col);
    }

    fn vec_znx_dft_copy_default(
        _module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
    ) where
        Self: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vec_znx_dft_copy::<Self, Rg>(step, offset, res, res_col, a, a_col);
    }

    fn vec_znx_dft_zero_default(_module: &Module<Self>, res: &mut VecZnxDftBackendMut<'_, Self, Rg>, res_col: usize)
    where
        Self: Backend<DftWord = f64, ZnxWord = i64> + ReimArith,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
    {
        fft64_vec_znx_dft_zero::<Self, Rg>(res, res_col);
    }

    fn vec_znx_dft_automorphism_plan_default(_module: &Module<Self>, n: usize, p: i64) -> Fft64AutomorphismPlan<Rg>
    where
        Module<Self>: FFTModuleHandle<f64, Rg>,
        Self: Backend<DftWord = f64, ZnxWord = i64> + Fft64RingArith<Rg>,
    {
        <Self as Fft64RingArith<Rg>>::fft64_automorphism_plan(n, p)
    }

    fn vec_znx_dft_automorphism_with_plan_default(
        module: &Module<Self>,
        plan: &Fft64AutomorphismPlan<Rg>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
    ) where
        Self: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + Fft64RingArith<Rg>,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        check_degree::<Self>(module.n(), res.n());
        fft64_vec_znx_dft_automorphism::<Self, Rg>(plan, res, res_col, a, a_col);
    }

    fn vec_znx_dft_automorphism_add_with_plan_default(
        module: &Module<Self>,
        plan: &Fft64AutomorphismPlan<Rg>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
    ) where
        Self: Backend<DftWord = f64, ZnxWord = i64> + Fft64RingArith<Rg>,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        check_degree::<Self>(module.n(), res.n());
        fft64_vec_znx_dft_automorphism_add::<Self, poulpy_hal::execution::SerialTaskExecutor, Rg>(plan, res, res_col, a, a_col);
    }
}

impl<BE: Backend<ZnxWord = i64>, Rg: Ring> FFT64VecZnxDftDefault<Rg> for BE
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    type AutomorphismPlanDefault = Fft64AutomorphismPlan<Rg>;
}

#[doc(hidden)]
pub trait NTT4x30VecZnxDftDefault<Rg: Ring = Standard>: Backend<ZnxWord = i64>
where
    Self::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    type AutomorphismPlanDefault: Send + Sync;

    fn vec_znx_dft_apply_default(
        module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self, Rg>,
        a_col: usize,
    ) where
        Module<Self>: NttModuleHandle<Rg>,
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64>
            + NttDFTExecute<NttTable<Primes30, Rg>>
            + NttFromZnx64
            + NttZero
            + 'static,
        for<'x> Self: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8], ZnxWord = i64>,
    {
        ntt4x30_default_vec_znx_dft_apply::<Self, Rg>(module, step, offset, res, res_col, a, a_col);
    }

    fn vec_znx_idft_apply_tmp_bytes_default(module: &Module<Self>) -> usize
    where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64>,
    {
        ntt4x30_default_vec_znx_idft_apply_tmp_bytes(module.n())
    }

    fn vec_znx_idft_apply_default(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: NttModuleHandle<Rg>,
        Self: Backend<DftWord = Q120bScalar, BigWord = i128, ZnxWord = i64>
            + NttDFTExecute<NttTableInv<Primes30, Rg>>
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
        ntt4x30_default_vec_znx_idft_apply::<Self, Rg>(module, res, res_col, a, a_col, tmp);
    }

    fn vec_znx_dft_add_default(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self, Rg>,
        b_col: usize,
    ) where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttAdd + NttCopy + NttZero,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt4x30_default_vec_znx_dft_add::<Self, Rg>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_dft_add_assign_default(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
    ) where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttAddAssign,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt4x30_default_vec_znx_dft_add_assign::<Self, Rg>(res, res_col, a, a_col);
    }

    fn vec_znx_dft_sub_default(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self, Rg>,
        b_col: usize,
    ) where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttSub + NttNegate + NttCopy + NttZero,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt4x30_default_vec_znx_dft_sub::<Self, Rg>(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_dft_sub_assign_default(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
    ) where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttSubAssign,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt4x30_default_vec_znx_dft_sub_assign::<Self, Rg>(res, res_col, a, a_col);
    }

    fn vec_znx_dft_sub_negate_assign_default(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
    ) where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttSubNegateAssign + NttNegateAssign,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt4x30_default_vec_znx_dft_sub_negate_assign::<Self, Rg>(res, res_col, a, a_col);
    }

    fn vec_znx_dft_copy_default(
        _module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
    ) where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttCopy + NttZero,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt4x30_default_vec_znx_dft_copy::<Self, Rg>(step, offset, res, res_col, a, a_col);
    }

    fn vec_znx_dft_zero_default(_module: &Module<Self>, res: &mut VecZnxDftBackendMut<'_, Self, Rg>, res_col: usize)
    where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttZero,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
    {
        ntt4x30_default_vec_znx_dft_zero::<Self, Rg>(res, res_col);
    }

    fn vec_znx_dft_automorphism_plan_default(_module: &Module<Self>, n: usize, p: i64) -> NttAutomorphismPlan<Rg>
    where
        Self: Backend<ZnxWord = i64> + NttDFTExecute<NttTable<Primes30, Rg>>,
        Module<Self>: NttModuleHandle<Rg>,
    {
        <Self as NttDFTExecute<NttTable<Primes30, Rg>>>::ntt_automorphism_plan(n, p)
    }

    fn vec_znx_dft_automorphism_with_plan_default(
        module: &Module<Self>,
        plan: &NttAutomorphismPlan<Rg>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
    ) where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttZero,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        check_degree::<Self>(module.n(), res.n());
        ntt4x30_default_vec_znx_dft_automorphism::<Self, Rg>(plan, res, res_col, a, a_col);
    }

    fn vec_znx_dft_automorphism_add_with_plan_default(
        module: &Module<Self>,
        plan: &NttAutomorphismPlan<Rg>,
        res: &mut VecZnxDftBackendMut<'_, Self, Rg>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self, Rg>,
        a_col: usize,
    ) where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttAddAssign,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        check_degree::<Self>(module.n(), res.n());
        ntt4x30_default_vec_znx_dft_automorphism_add::<Self, poulpy_hal::execution::SerialTaskExecutor, Rg>(
            plan, res, res_col, a, a_col,
        );
    }
}

impl<BE: Backend<ZnxWord = i64>, Rg: Ring> NTT4x30VecZnxDftDefault<Rg> for BE
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    type AutomorphismPlanDefault = NttAutomorphismPlan<Rg>;
}
