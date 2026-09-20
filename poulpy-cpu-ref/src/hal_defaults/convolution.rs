//! Backend extension points for bivariate convolution operations.

use std::mem::size_of;

use crate::reference::{
    fft64::{
        convolution::{
            I64Ops, convolution_apply_dft, convolution_apply_dft_add, convolution_apply_dft_tmp_bytes,
            convolution_by_const_apply, convolution_by_const_apply_add, convolution_by_const_apply_tmp_bytes,
            convolution_pairwise_apply_dft, convolution_pairwise_apply_dft_tmp_bytes, convolution_prepare_left,
            convolution_prepare_right, convolution_prepare_self,
        },
        module::FFTModuleHandle,
        reim::{ReimArith, ReimFFTExecute, ReimFFTTable, ReimIFFTTable},
        reim4::{Reim4BlkMatVec, Reim4Convolution},
    },
    ntt4x30::{
        NttAddAssign, NttCFromB, NttDFTExecute, NttFromZnx64, NttMulBbc1ColX2, NttPackLeft1BlkX2,
        convolution::{
            CNV_ACC_GROUP, ntt4x30_cnv_apply_dft, ntt4x30_cnv_apply_dft_add, ntt4x30_cnv_apply_dft_sum,
            ntt4x30_cnv_apply_dft_sum_tmp_bytes, ntt4x30_cnv_apply_dft_tmp_bytes, ntt4x30_cnv_by_const_apply,
            ntt4x30_cnv_by_const_apply_add, ntt4x30_cnv_by_const_apply_tmp_bytes, ntt4x30_cnv_pairwise_apply_dft,
            ntt4x30_cnv_pairwise_apply_dft_tmp_bytes, ntt4x30_cnv_prepare_left, ntt4x30_cnv_prepare_left_tmp_bytes,
            ntt4x30_cnv_prepare_right, ntt4x30_cnv_prepare_right_tmp_bytes, ntt4x30_cnv_prepare_self,
            ntt4x30_cnv_prepare_self_tmp_bytes,
        },
        ntt::NttTable,
        primes::Primes30,
        types::Q120bScalar,
        vec_znx_dft::NttModuleHandle,
    },
};
use poulpy_hal::{
    api::{HostBufMut, ModuleN, VecZnxDftBytesOf},
    execution::{ScratchWorkers, SerialTaskExecutor, scratch_workers, scratch_workers_within},
    layouts::{
        Backend, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef, HostDataRef, Module,
        ScratchArena, VecZnxBackendRef, VecZnxBigToBackendMut, VecZnxDft, VecZnxDftToBackendMut, check_degree,
        vec_znx_dft_backend_mut_from_mut,
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
pub trait FFT64ConvolutionDefault: Backend<ZnxWord = i64> + ScratchWorkers
where
    Self::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    fn cnv_prepare_left_tmp_bytes_default(module: &Module<Self>, res_size: usize, a_size: usize) -> usize
    where
        Self: Backend<DftWord = f64, ZnxWord = i64>,
    {
        Self::bytes_of_vec_znx_dft(module.n(), 1, res_size.min(a_size))
    }

    fn cnv_prepare_left_default(
        module: &Module<Self>,
        res: &mut CnvPVecLBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: FFTModuleHandle<f64> + ModuleN + VecZnxDftBytesOf,
        Self: Backend<DftWord = f64, ZnxWord = i64>
            + ReimArith
            + Reim4BlkMatVec
            + ReimFFTExecute<ReimFFTTable<f64>, f64>
            + ReimFFTExecute<ReimIFFTTable<f64>, f64>
            + 'static,
        for<'x> Self: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8], ZnxWord = i64>,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
    {
        let n: usize = res.n();
        check_degree::<Self>(module.n(), n);
        assert_eq!(a.n(), n, "cnv_prepare_left: a.n():{} != res.n():{n}", a.n());
        let tmp_size = res.size().min(a.size());
        let (tmp_bytes, _) = take_host_typed::<Self, u8>(scratch.borrow(), Self::bytes_of_vec_znx_dft(n, 1, tmp_size));
        let mut tmp = VecZnxDft::from_data(tmp_bytes, n, 1, tmp_size);
        let mut tmp_ref = vec_znx_dft_backend_mut_from_mut::<Self>(&mut tmp);
        convolution_prepare_left::<Self>(module.get_fft_plan(n), res, a, &mut tmp_ref);
    }

    fn cnv_prepare_right_tmp_bytes_default(module: &Module<Self>, res_size: usize, a_size: usize) -> usize
    where
        Self: Backend<DftWord = f64, ZnxWord = i64>,
    {
        Self::bytes_of_vec_znx_dft(module.n(), 1, res_size.min(a_size))
    }

    fn cnv_prepare_right_default(
        module: &Module<Self>,
        res: &mut CnvPVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: FFTModuleHandle<f64> + ModuleN + VecZnxDftBytesOf,
        Self: Backend<DftWord = f64, ZnxWord = i64>
            + ReimArith
            + Reim4BlkMatVec
            + ReimFFTExecute<ReimFFTTable<f64>, f64>
            + ReimFFTExecute<ReimIFFTTable<f64>, f64>
            + 'static,
        for<'x> Self: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8], ZnxWord = i64>,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
    {
        let n: usize = res.n();
        check_degree::<Self>(module.n(), n);
        assert_eq!(a.n(), n, "cnv_prepare_right: a.n():{} != res.n():{n}", a.n());
        let tmp_size = res.size().min(a.size());
        let (tmp_bytes, _) = take_host_typed::<Self, u8>(scratch.borrow(), Self::bytes_of_vec_znx_dft(n, 1, tmp_size));
        let mut tmp = VecZnxDft::from_data(tmp_bytes, n, 1, tmp_size);
        let mut tmp_ref = vec_znx_dft_backend_mut_from_mut::<Self>(&mut tmp);
        convolution_prepare_right::<Self>(module.get_fft_plan(n), res, a, &mut tmp_ref);
    }

    fn cnv_apply_dft_tmp_bytes_default(
        module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize
    where
        Self: Backend<DftWord = f64, ZnxWord = i64>,
    {
        reim4_block_workers::<Self>(module.n()) * convolution_apply_dft_tmp_bytes(res_size, a_size, b_size)
    }

    fn cnv_by_const_apply_tmp_bytes_default(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize
    where
        Self: Backend<BigWord = i64, ZnxWord = i64>,
    {
        convolution_by_const_apply_tmp_bytes(res_size, a_size, b_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_by_const_apply_default<R>(
        _module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
        b_coeff: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Self: Backend<BigWord = i64, ZnxWord = i64> + I64Ops + 'static,
        for<'x> Self: Backend<BufRef<'x> = &'x [u8], ZnxWord = i64>,
        for<'x> <Self as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
        R: VecZnxBigToBackendMut<Self>,
    {
        let mut res_ref = res.to_backend_mut();
        let bytes = convolution_by_const_apply_tmp_bytes(res_ref.size(), a.size(), b.size());
        let (tmp, _) = take_host_typed::<Self, i64>(scratch.borrow(), bytes / size_of::<i64>());
        convolution_by_const_apply::<Self>(cnv_offset, &mut res_ref, res_col, a, a_col, b, b_col, b_coeff, tmp);
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_by_const_apply_add_default<R>(
        _module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
        b_coeff: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Self: Backend<BigWord = i64, ZnxWord = i64> + I64Ops + 'static,
        for<'x> Self: Backend<BufRef<'x> = &'x [u8], ZnxWord = i64>,
        for<'x> <Self as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
        R: VecZnxBigToBackendMut<Self>,
    {
        let mut res_ref = res.to_backend_mut();
        let bytes = convolution_by_const_apply_tmp_bytes(res_ref.size(), a.size(), b.size());
        let (tmp, _) = take_host_typed::<Self, i64>(scratch.borrow(), bytes / size_of::<i64>());
        convolution_by_const_apply_add::<Self>(cnv_offset, &mut res_ref, res_col, a, a_col, b, b_col, b_coeff, tmp);
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft_default<R>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, Self>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, Self>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: FFTModuleHandle<f64>,
        Self: Backend<DftWord = f64, ZnxWord = i64> + Reim4BlkMatVec + Reim4Convolution,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
        for<'x> <Self as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        R: VecZnxDftToBackendMut<Self>,
    {
        let mut res_ref = res.to_backend_mut();
        let per_worker = convolution_apply_dft_tmp_bytes(res_ref.size(), a.size(), b.size());
        let bytes = reim4_block_workers_within::<Self>(res_ref.n(), per_worker, scratch.available()) * per_worker;
        let (tmp, _) = take_host_typed::<Self, f64>(scratch.borrow(), bytes / size_of::<f64>());
        convolution_apply_dft::<Self>(
            cnv_offset,
            &mut res_ref,
            res_col,
            a,
            a_col,
            b,
            b_col,
            module.get_fft_plan(module.n()).is_conjugate_invariant(),
            tmp,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft_add_default<R>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, Self>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, Self>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: FFTModuleHandle<f64>,
        Self: Backend<DftWord = f64, ZnxWord = i64> + Reim4BlkMatVec + Reim4Convolution,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
        for<'x> <Self as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        R: VecZnxDftToBackendMut<Self>,
    {
        let mut res_ref = res.to_backend_mut();
        let per_worker = convolution_apply_dft_tmp_bytes(res_ref.size(), a.size(), b.size());
        let bytes = reim4_block_workers_within::<Self>(res_ref.n(), per_worker, scratch.available()) * per_worker;
        let (tmp, _) = take_host_typed::<Self, f64>(scratch.borrow(), bytes / size_of::<f64>());
        convolution_apply_dft_add::<Self>(
            cnv_offset,
            &mut res_ref,
            res_col,
            a,
            a_col,
            b,
            b_col,
            module.get_fft_plan(module.n()).is_conjugate_invariant(),
            tmp,
        );
    }

    fn cnv_pairwise_apply_dft_tmp_bytes_default(
        module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize
    where
        Self: Backend<DftWord = f64, ZnxWord = i64>,
    {
        reim4_block_workers::<Self>(module.n()) * convolution_pairwise_apply_dft_tmp_bytes(res_size, a_size, b_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_pairwise_apply_dft_default<R>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, Self>,
        b: &CnvPVecRBackendRef<'_, Self>,
        i: usize,
        j: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: FFTModuleHandle<f64>,
        Self: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + Reim4BlkMatVec + Reim4Convolution,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
        for<'x> <Self as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        R: VecZnxDftToBackendMut<Self>,
    {
        let mut res_ref = res.to_backend_mut();
        let per_worker = convolution_pairwise_apply_dft_tmp_bytes(res_ref.size(), a.size(), b.size());
        let bytes = reim4_block_workers_within::<Self>(res_ref.n(), per_worker, scratch.available()) * per_worker;
        let (tmp, _) = take_host_typed::<Self, f64>(scratch.borrow(), bytes / size_of::<f64>());
        convolution_pairwise_apply_dft::<Self>(
            cnv_offset,
            &mut res_ref,
            res_col,
            a,
            b,
            i,
            j,
            module.get_fft_plan(module.n()).is_conjugate_invariant(),
            tmp,
        );
    }

    fn cnv_prepare_self_tmp_bytes_default(module: &Module<Self>, res_size: usize, a_size: usize) -> usize
    where
        Self: Backend<DftWord = f64, ZnxWord = i64>,
    {
        Self::bytes_of_vec_znx_dft(module.n(), 1, res_size.min(a_size))
    }

    fn cnv_prepare_self_default(
        module: &Module<Self>,
        left: &mut CnvPVecLBackendMut<'_, Self>,
        right: &mut CnvPVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: FFTModuleHandle<f64> + ModuleN + VecZnxDftBytesOf,
        Self: Backend<DftWord = f64, ZnxWord = i64>
            + ReimArith
            + Reim4BlkMatVec
            + ReimFFTExecute<ReimFFTTable<f64>, f64>
            + ReimFFTExecute<ReimIFFTTable<f64>, f64>
            + 'static,
        for<'x> Self: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8], ZnxWord = i64>,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
    {
        let n: usize = left.n();
        check_degree::<Self>(module.n(), n);
        assert_eq!(right.n(), n, "cnv_prepare_self: right.n():{} != left.n():{n}", right.n());
        assert_eq!(a.n(), n, "cnv_prepare_self: a.n():{} != left.n():{n}", a.n());
        let tmp_size = left.size().min(a.size());
        let (tmp_bytes, _) = take_host_typed::<Self, u8>(scratch.borrow(), Self::bytes_of_vec_znx_dft(n, 1, tmp_size));
        let mut tmp = VecZnxDft::from_data(tmp_bytes, n, 1, tmp_size);
        let mut tmp_ref = vec_znx_dft_backend_mut_from_mut::<Self>(&mut tmp);
        convolution_prepare_self::<Self>(module.get_fft_plan(n), left, right, a, &mut tmp_ref);
    }
}

impl<BE: Backend<ZnxWord = i64> + ScratchWorkers> FFT64ConvolutionDefault for BE where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut
{
}

/// Worker slices for the block-parallel reim4 convolution kernels at degree `n`.
fn reim4_block_workers<BE: Backend + ScratchWorkers>(n: usize) -> usize {
    scratch_workers::<BE::TaskExecutor>((n / 8).min(BE::APPLY))
}

fn reim4_block_workers_within<BE: Backend + ScratchWorkers>(n: usize, per_worker: usize, available: usize) -> usize {
    scratch_workers_within::<BE::TaskExecutor>((n / 8).min(BE::APPLY), per_worker, available)
}

/// Worker slices for the block-group parallel convolution kernels at degree `n`.
fn cnv_group_workers<BE: Backend + ScratchWorkers>(n: usize) -> usize {
    scratch_workers::<BE::TaskExecutor>(cnv_groups(n).min(BE::APPLY))
}

fn cnv_group_workers_within<BE: Backend + ScratchWorkers>(n: usize, per_worker: usize, available: usize) -> usize {
    scratch_workers_within::<BE::TaskExecutor>(cnv_groups(n).min(BE::APPLY), per_worker, available)
}

fn cnv_groups(n: usize) -> usize {
    (n / 2).div_ceil(CNV_ACC_GROUP)
}

#[doc(hidden)]
pub trait NTT4x30ConvolutionDefault: Backend<ZnxWord = i64> + ScratchWorkers
where
    Self::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    fn cnv_prepare_left_tmp_bytes_default(module: &Module<Self>, res_size: usize, _a_size: usize) -> usize
    where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64>,
    {
        scratch_workers::<Self::TaskExecutor>(res_size.min(Self::PREPARE)) * ntt4x30_cnv_prepare_left_tmp_bytes(module.n())
    }

    fn cnv_prepare_left_default(
        module: &Module<Self>,
        res: &mut CnvPVecLBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: NttModuleHandle,
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64>
            + NttFromZnx64
            + NttDFTExecute<NttTable<Primes30>>
            + NttPackLeft1BlkX2
            + 'static,
        for<'x> Self: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8], ZnxWord = i64>,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
    {
        let per_worker = ntt4x30_cnv_prepare_left_tmp_bytes(res.n());
        let bytes = scratch_workers_within::<Self::TaskExecutor>(res.size().min(Self::PREPARE), per_worker, scratch.available())
            * per_worker;
        let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        ntt4x30_cnv_prepare_left::<Self>(module, res, a, tmp);
    }

    fn cnv_prepare_right_tmp_bytes_default(module: &Module<Self>, res_size: usize, _a_size: usize) -> usize
    where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64>,
    {
        scratch_workers::<Self::TaskExecutor>(res_size.min(Self::PREPARE)) * ntt4x30_cnv_prepare_right_tmp_bytes(module.n())
    }

    fn cnv_prepare_right_default(
        module: &Module<Self>,
        res: &mut CnvPVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: NttModuleHandle,
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64>
            + NttFromZnx64
            + NttDFTExecute<NttTable<Primes30>>
            + NttCFromB
            + 'static,
        for<'x> Self: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8], ZnxWord = i64>,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
    {
        let per_worker = ntt4x30_cnv_prepare_right_tmp_bytes(res.n());
        let bytes = scratch_workers_within::<Self::TaskExecutor>(res.size().min(Self::PREPARE), per_worker, scratch.available())
            * per_worker;
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        ntt4x30_cnv_prepare_right::<Self>(module, res, a, tmp);
    }

    fn cnv_apply_dft_tmp_bytes_default(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize
    where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64>,
    {
        cnv_group_workers::<Self>(_module.n()) * ntt4x30_cnv_apply_dft_tmp_bytes(res_size, a_size, b_size)
    }

    fn cnv_by_const_apply_tmp_bytes_default(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize
    where
        Self: Backend<BigWord = i128, DftWord = Q120bScalar, ZnxWord = i64>,
    {
        ntt4x30_cnv_by_const_apply_tmp_bytes(res_size, a_size, b_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_by_const_apply_default<R>(
        _module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
        b_coeff: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Self: Backend<BigWord = i128, DftWord = Q120bScalar, ZnxWord = i64> + 'static,
        for<'x> Self: Backend<BufRef<'x> = &'x [u8], ZnxWord = i64>,
        for<'x> <Self as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
        R: VecZnxBigToBackendMut<Self>,
    {
        let mut res_ref = res.to_backend_mut();
        let bytes = ntt4x30_cnv_by_const_apply_tmp_bytes(0, 0, 0);
        let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        ntt4x30_cnv_by_const_apply::<Self, SerialTaskExecutor>(
            cnv_offset,
            &mut res_ref,
            res_col,
            a,
            a_col,
            b,
            b_col,
            b_coeff,
            tmp,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_by_const_apply_add_default<R>(
        _module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
        b_coeff: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Self: Backend<BigWord = i128, DftWord = Q120bScalar, ZnxWord = i64> + 'static,
        for<'x> Self: Backend<BufRef<'x> = &'x [u8], ZnxWord = i64>,
        for<'x> <Self as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
        R: VecZnxBigToBackendMut<Self>,
    {
        let mut res_ref = res.to_backend_mut();
        let bytes = ntt4x30_cnv_by_const_apply_tmp_bytes(0, 0, 0);
        let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        ntt4x30_cnv_by_const_apply_add::<Self, SerialTaskExecutor>(
            cnv_offset,
            &mut res_ref,
            res_col,
            a,
            a_col,
            b,
            b_col,
            b_coeff,
            tmp,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft_default<R>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, Self>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, Self>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: NttModuleHandle,
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttAddAssign + NttMulBbc1ColX2,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
        for<'x> <Self as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        R: VecZnxDftToBackendMut<Self>,
    {
        let mut res_ref = res.to_backend_mut();
        let per_worker = ntt4x30_cnv_apply_dft_tmp_bytes(res_ref.size(), a.size(), b.size());
        let bytes = cnv_group_workers_within::<Self>(res_ref.n(), per_worker, scratch.available()) * per_worker;
        let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        ntt4x30_cnv_apply_dft::<Self>(module, cnv_offset, &mut res_ref, res_col, a, a_col, b, b_col, tmp);
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft_add_default<R>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, Self>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, Self>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: NttModuleHandle,
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttAddAssign + NttMulBbc1ColX2,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
        for<'x> <Self as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        R: VecZnxDftToBackendMut<Self>,
    {
        let mut res_ref = res.to_backend_mut();
        let per_worker = ntt4x30_cnv_apply_dft_tmp_bytes(res_ref.size(), a.size(), b.size());
        let bytes = cnv_group_workers_within::<Self>(res_ref.n(), per_worker, scratch.available()) * per_worker;
        let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        ntt4x30_cnv_apply_dft_add::<Self>(module, cnv_offset, &mut res_ref, res_col, a, a_col, b, b_col, tmp);
    }

    fn cnv_apply_dft_sum_tmp_bytes_default(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize
    where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64>,
    {
        ntt4x30_cnv_apply_dft_sum_tmp_bytes(res_size, a_size, b_size)
    }

    fn cnv_apply_dft_sum_default<'a, R>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        terms: &[poulpy_hal::layouts::CnvDftAccTerm<'a, Self>],
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: NttModuleHandle,
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + 'a,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
        for<'x> <Self as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        R: VecZnxDftToBackendMut<Self>,
    {
        let mut res_ref = res.to_backend_mut();
        let bytes = ntt4x30_cnv_apply_dft_sum_tmp_bytes(res_ref.size(), 0, 0);
        let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        ntt4x30_cnv_apply_dft_sum::<Self>(module, cnv_offset, &mut res_ref, res_col, terms, tmp);
    }

    fn cnv_pairwise_apply_dft_tmp_bytes_default(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize
    where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64>,
    {
        cnv_group_workers::<Self>(_module.n()) * ntt4x30_cnv_pairwise_apply_dft_tmp_bytes(res_size, a_size, b_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_pairwise_apply_dft_default<R>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, Self>,
        b: &CnvPVecRBackendRef<'_, Self>,
        i: usize,
        j: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: NttModuleHandle,
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttAddAssign + NttMulBbc1ColX2,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
        for<'x> <Self as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        R: VecZnxDftToBackendMut<Self>,
    {
        let mut res_ref = res.to_backend_mut();
        let per_worker = ntt4x30_cnv_pairwise_apply_dft_tmp_bytes(res_ref.size(), a.size(), b.size());
        let bytes = cnv_group_workers_within::<Self>(res_ref.n(), per_worker, scratch.available()) * per_worker;
        let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        ntt4x30_cnv_pairwise_apply_dft::<Self>(module, cnv_offset, &mut res_ref, res_col, a, b, i, j, tmp);
    }

    fn cnv_prepare_self_tmp_bytes_default(module: &Module<Self>, res_size: usize, _a_size: usize) -> usize
    where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64>,
    {
        scratch_workers::<Self::TaskExecutor>(res_size.min(Self::PREPARE)) * ntt4x30_cnv_prepare_self_tmp_bytes(module.n())
    }

    fn cnv_prepare_self_default(
        module: &Module<Self>,
        left: &mut CnvPVecLBackendMut<'_, Self>,
        right: &mut CnvPVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: NttModuleHandle,
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64>
            + NttFromZnx64
            + NttDFTExecute<NttTable<Primes30>>
            + NttCFromB
            + NttPackLeft1BlkX2
            + 'static,
        for<'x> Self: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8], ZnxWord = i64>,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
    {
        let per_worker = ntt4x30_cnv_prepare_self_tmp_bytes(left.n());
        let bytes = scratch_workers_within::<Self::TaskExecutor>(left.size().min(Self::PREPARE), per_worker, scratch.available())
            * per_worker;
        let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        ntt4x30_cnv_prepare_self::<Self>(module, left, right, a, tmp);
    }
}

impl<BE: Backend<ZnxWord = i64> + ScratchWorkers> NTT4x30ConvolutionDefault for BE where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut
{
}
