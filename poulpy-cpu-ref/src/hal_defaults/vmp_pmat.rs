//! Backend extension points for vector-matrix product (VMP) operations
//! on [`VmpPMat`](poulpy_hal::layouts::VmpPMat).

use std::mem::size_of;

use crate::reference::{
    fft64::{
        module::FFTModuleHandle,
        reim::{ReimArith, ReimFFTExecute, ReimFFTTable, ReimIFFTTable},
        reim4::Reim4BlkMatVec,
        vmp::{
            vmp_apply_dft_to_dft_tmp_bytes as fft64_vmp_apply_dft_to_dft_tmp_bytes,
            vmp_apply_dft_to_dft_with_kernel as fft64_vmp_apply_dft_to_dft_with_kernel,
            vmp_extract_selected_rows as fft64_vmp_extract_selected_rows, vmp_prepare as fft64_vmp_prepare,
            vmp_prepare_tmp_bytes as fft64_vmp_prepare_tmp_bytes, vmp_zero as fft64_vmp_zero,
        },
    },
    ntt4x30::{
        NttCFromB, NttDFTExecute, NttExtract1BlkContiguous, NttFromZnx64, NttMulBbc1ColX2, NttMulBbc2ColsX2,
        ntt::NttTable,
        primes::Primes30,
        types::Q120bScalar,
        vec_znx_dft::NttModuleHandle,
        vmp::{
            ntt4x30_vmp_apply_dft_to_dft, ntt4x30_vmp_apply_dft_to_dft_tmp_bytes, ntt4x30_vmp_extract_selected_rows,
            ntt4x30_vmp_prepare, ntt4x30_vmp_prepare_tmp_bytes, ntt4x30_vmp_zero,
        },
    },
};
use poulpy_hal::{
    api::{HostBufMut, ModuleN, ScratchArenaTakeBasic, VecZnxDftAddAssign, VecZnxDftBytesOf, VecZnxDftZero},
    execution::TaskExecutor,
    layouts::{
        Backend, HostDataMut, HostDataRef, MatZnxBackendRef, Module, ScratchArena, VecZnxDftBackendMut, VecZnxDftBackendRef,
        VecZnxDftToBackendRef, VmpPMatBackendMut, VmpPMatBackendRef, check_degree,
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
pub trait FFT64VmpDefault: Backend<ZnxWord = i64>
where
    Self::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'x> Self::BufMut<'x>: HostDataMut,
    for<'x> Self::BufRef<'x>: HostDataRef,
{
    fn vmp_prepare_tmp_bytes_default(
        module: &Module<Self>,
        _rows: usize,
        _cols_in: usize,
        _cols_out: usize,
        _size: usize,
    ) -> usize
    where
        Self: Backend<DftWord = f64, ZnxWord = i64>,
    {
        fft64_vmp_prepare_tmp_bytes(module.n())
    }

    fn vmp_prepare_default(
        module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &MatZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: FFTModuleHandle<f64>,
        Self: Backend<DftWord = f64, ZnxWord = i64>
            + ReimArith
            + Reim4BlkMatVec
            + ReimFFTExecute<ReimFFTTable<f64>, f64>
            + ReimFFTExecute<ReimIFFTTable<f64>, f64>
            + 'static,
        for<'x> Self: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8], ZnxWord = i64>,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
    {
        let n: usize = res.n();
        check_degree::<Self>(module.n(), n);
        assert_eq!(a.n(), n, "vmp_prepare: a.n():{} != res.n():{n}", a.n());
        let bytes = fft64_vmp_prepare_tmp_bytes(n);
        let (tmp, _) = take_host_typed::<Self, f64>(scratch.borrow(), bytes / size_of::<f64>());
        fft64_vmp_prepare::<Self>(module.get_fft_plan(n), res, a, tmp);
    }

    fn vmp_apply_dft_to_dft_tmp_bytes_default(
        _module: &Module<Self>,
        _res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize
    where
        Self: Backend<DftWord = f64, ZnxWord = i64>,
    {
        fft64_vmp_apply_dft_to_dft_tmp_bytes(a_size, b_rows, b_cols_in)
    }

    #[inline(always)]
    fn vmp_apply_dft_to_dft_default(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: FFTModuleHandle<f64>,
        Self: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + Reim4BlkMatVec,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
    {
        Self::vmp_apply_dft_to_dft_with_kernel_default::<Self, Self::TaskExecutor>(_module, res, a, b, limb_offset, 1, scratch);
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn vmp_apply_dft_to_dft_with_kernel_default<KERNEL, E>(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        workers: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: FFTModuleHandle<f64>,
        Self: Backend<DftWord = f64, ZnxWord = i64>,
        KERNEL: ReimArith + Reim4BlkMatVec,
        E: TaskExecutor,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
    {
        let per_worker = fft64_vmp_apply_dft_to_dft_tmp_bytes(a.size(), b.rows(), b.cols_in());
        let bytes = workers.max(1).min(scratch.available() / per_worker.max(1)).max(1) * per_worker;
        let (tmp, _) = take_host_typed::<Self, f64>(scratch.borrow(), bytes / size_of::<f64>());
        fft64_vmp_apply_dft_to_dft_with_kernel::<Self, KERNEL, E>(
            res,
            a,
            b,
            limb_offset,
            _module.get_fft_plan(_module.n()).is_conjugate_invariant(),
            tmp,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn vmp_apply_dft_to_dft_add_with_kernel_default<KERNEL, E>(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        workers: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: FFTModuleHandle<f64> + VecZnxDftBytesOf + ModuleN + VecZnxDftAddAssign<Self> + VecZnxDftZero<Self>,
        Self: Backend<DftWord = f64, ZnxWord = i64>,
        KERNEL: ReimArith + Reim4BlkMatVec,
        E: TaskExecutor,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
    {
        let cols_out = res.cols();
        let res_size = res.size();
        let (mut tmp, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(res.n(), cols_out, res_size);
        for col in 0..cols_out {
            module.vec_znx_dft_zero(&mut tmp, col);
        }
        let per_worker = fft64_vmp_apply_dft_to_dft_tmp_bytes(a.size(), b.rows(), b.cols_in());
        let bytes = workers.max(1).min(scratch_1.available() / per_worker.max(1)).max(1) * per_worker;
        let (kernel_tmp, _) = take_host_typed::<Self, f64>(scratch_1, bytes / size_of::<f64>());
        fft64_vmp_apply_dft_to_dft_with_kernel::<Self, KERNEL, E>(
            &mut tmp,
            a,
            b,
            limb_offset,
            module.get_fft_plan(module.n()).is_conjugate_invariant(),
            kernel_tmp,
        );
        let tmp_ref = tmp.to_backend_ref();
        for col in 0..cols_out {
            module.vec_znx_dft_add_assign(res, col, &tmp_ref, col);
        }
    }

    fn vmp_extract_selected_rows_default(
        _module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &VmpPMatBackendRef<'_, Self>,
        first_row: usize,
        row_step: usize,
    ) where
        Self: Backend<DftWord = f64, ZnxWord = i64>,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        fft64_vmp_extract_selected_rows::<Self>(res, a, first_row, row_step);
    }

    fn vmp_zero_default(_module: &Module<Self>, res: &mut VmpPMatBackendMut<'_, Self>)
    where
        Self: Backend<DftWord = f64, ZnxWord = i64>,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
    {
        fft64_vmp_zero::<Self>(res);
    }
}

impl<BE: Backend<ZnxWord = i64>> FFT64VmpDefault for BE
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
}

#[doc(hidden)]
pub trait NTT4x30VmpDefault: Backend<ZnxWord = i64>
where
    Self::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'x> Self::BufMut<'x>: HostDataMut,
    for<'x> Self::BufRef<'x>: HostDataRef,
{
    fn vmp_prepare_tmp_bytes_default(
        module: &Module<Self>,
        _rows: usize,
        _cols_in: usize,
        _cols_out: usize,
        _size: usize,
    ) -> usize
    where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64>,
    {
        ntt4x30_vmp_prepare_tmp_bytes(module.n())
    }

    fn vmp_prepare_default(
        module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &MatZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: NttModuleHandle,
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttDFTExecute<NttTable<Primes30>> + NttFromZnx64 + NttCFromB,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
    {
        let bytes = ntt4x30_vmp_prepare_tmp_bytes(res.n());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        ntt4x30_vmp_prepare::<Self>(module, res, a, tmp);
    }

    fn vmp_apply_dft_to_dft_tmp_bytes_default(
        _module: &Module<Self>,
        _res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize
    where
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64>,
    {
        ntt4x30_vmp_apply_dft_to_dft_tmp_bytes(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft_default(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Module<Self>: NttModuleHandle,
        Self: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttExtract1BlkContiguous + NttMulBbc1ColX2 + NttMulBbc2ColsX2,
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
        for<'x> Self::BufMut<'x>: HostBufMut<'x>,
    {
        let bytes = ntt4x30_vmp_apply_dft_to_dft_tmp_bytes(a.size(), b.rows(), b.cols_in());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        ntt4x30_vmp_apply_dft_to_dft::<Self>(module, res, a, b, limb_offset, tmp);
    }

    fn vmp_extract_selected_rows_default(
        _module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &VmpPMatBackendRef<'_, Self>,
        first_row: usize,
        row_step: usize,
    ) where
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <Self as Backend>::BufRef<'x>: HostDataRef,
    {
        ntt4x30_vmp_extract_selected_rows::<Self>(res, a, first_row, row_step);
    }

    fn vmp_zero_default(_module: &Module<Self>, res: &mut VmpPMatBackendMut<'_, Self>)
    where
        for<'x> <Self as Backend>::BufMut<'x>: HostDataMut,
    {
        ntt4x30_vmp_zero::<Self>(res);
    }
}

impl<BE: Backend<ZnxWord = i64>> NTT4x30VmpDefault for BE
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
}
