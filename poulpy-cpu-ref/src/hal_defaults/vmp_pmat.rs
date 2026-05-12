//! Backend extension points for vector-matrix product (VMP) operations
//! on [`VmpPMat`](poulpy_hal::layouts::VmpPMat).

use std::mem::size_of;

use crate::reference::{
    fft64::{
        module::FFTModuleHandle,
        reim::{ReimArith, ReimFFTExecute, ReimFFTTable},
        reim4::Reim4BlkMatVec,
        vmp::{
            vmp_apply_dft_to_dft as fft64_vmp_apply_dft_to_dft,
            vmp_apply_dft_to_dft_tmp_bytes as fft64_vmp_apply_dft_to_dft_tmp_bytes, vmp_prepare as fft64_vmp_prepare,
            vmp_prepare_tmp_bytes as fft64_vmp_prepare_tmp_bytes, vmp_zero as fft64_vmp_zero,
        },
    },
    ntt120::{
        NttCFromB, NttDFTExecute, NttExtract1BlkContiguous, NttFromZnx64, NttMulBbc1ColX2, NttMulBbc2ColsX2,
        ntt::NttTable,
        primes::Primes30,
        types::Q120bScalar,
        vec_znx_dft::NttModuleHandle,
        vmp::{
            ntt120_vmp_apply_dft_to_dft, ntt120_vmp_apply_dft_to_dft_tmp_bytes, ntt120_vmp_prepare, ntt120_vmp_prepare_tmp_bytes,
            ntt120_vmp_zero,
        },
    },
};
use poulpy_hal::{
    api::{HostBufMut, ModuleN, ScratchArenaTakeBasic, VecZnxDftAddAssign, VecZnxDftBytesOf, VecZnxDftZero},
    layouts::{
        Backend, HostDataMut, HostDataRef, MatZnxBackendRef, Module, ScratchArena, VecZnxDftBackendMut, VecZnxDftBackendRef,
        VecZnxDftToBackendRef, VmpPMatBackendMut, VmpPMatBackendRef,
    },
};

#[inline]
fn take_host_typed<'a, BE, T>(arena: ScratchArena<'a, BE>, len: usize) -> (&'a mut [T], ScratchArena<'a, BE>)
where
    BE: Backend + 'a,
    BE::BufMut<'a>: HostBufMut<'a>,
    T: Copy,
{
    debug_assert!(
        BE::SCRATCH_ALIGN.is_multiple_of(std::mem::align_of::<T>()),
        "B::SCRATCH_ALIGN ({}) must be a multiple of align_of::<T>() ({})",
        BE::SCRATCH_ALIGN,
        std::mem::align_of::<T>()
    );
    let (buf, arena) = arena.take_region(len * std::mem::size_of::<T>());
    let bytes: &'a mut [u8] = buf.into_bytes();
    let slice = unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, len) };
    (slice, arena)
}

#[doc(hidden)]
pub trait FFT64VmpDefault<BE: Backend>: Backend
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
{
    fn vmp_prepare_tmp_bytes_default(module: &Module<BE>, _rows: usize, _cols_in: usize, _cols_out: usize, _size: usize) -> usize
    where
        BE: Backend<ScalarPrep = f64>,
    {
        fft64_vmp_prepare_tmp_bytes(module.n())
    }

    fn vmp_prepare_default<'s>(
        module: &Module<BE>,
        res: &mut VmpPMatBackendMut<'_, BE>,
        a: &MatZnxBackendRef<'_, BE>,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        Module<BE>: FFTModuleHandle<f64>,
        BE: 's,
        BE: Backend<ScalarPrep = f64> + ReimArith + Reim4BlkMatVec + ReimFFTExecute<ReimFFTTable<f64>, f64> + 'static,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let bytes = fft64_vmp_prepare_tmp_bytes(module.n());
        let (tmp, _) = take_host_typed::<BE, f64>(scratch.borrow(), bytes / size_of::<f64>());
        fft64_vmp_prepare(module.get_fft_table(), res, a, tmp);
    }

    fn vmp_apply_dft_to_dft_tmp_bytes_default(
        _module: &Module<BE>,
        _res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize
    where
        BE: Backend<ScalarPrep = f64>,
    {
        fft64_vmp_apply_dft_to_dft_tmp_bytes(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft_default<'s, 'b>(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        a: &VecZnxDftBackendRef<'_, BE>,
        b: &VmpPMatBackendRef<'b, BE>,
        limb_offset: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: 's,
        BE: 'b,
        BE: Backend<ScalarPrep = f64> + ReimArith + Reim4BlkMatVec,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let bytes = fft64_vmp_apply_dft_to_dft_tmp_bytes(a.size(), b.rows(), b.cols_in());
        let (tmp, _) = take_host_typed::<BE, f64>(scratch.borrow(), bytes / size_of::<f64>());
        fft64_vmp_apply_dft_to_dft(res, a, b, limb_offset, tmp);
    }

    fn vmp_apply_dft_to_dft_accumulate_tmp_bytes_default(
        module: &Module<BE>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        _b_size: usize,
    ) -> usize
    where
        BE: Backend<ScalarPrep = f64>,
        Module<BE>: VecZnxDftBytesOf,
    {
        module.bytes_of_vec_znx_dft(b_cols_out, res_size) + fft64_vmp_apply_dft_to_dft_tmp_bytes(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft_accumulate_default<'s, 'b>(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        a: &VecZnxDftBackendRef<'_, BE>,
        b: &VmpPMatBackendRef<'b, BE>,
        limb_offset: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        Module<BE>: VecZnxDftBytesOf + ModuleN + VecZnxDftAddAssign<BE> + VecZnxDftZero<BE>,
        BE: 's,
        BE: 'b,
        BE: Backend<ScalarPrep = f64> + ReimArith + Reim4BlkMatVec,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let cols_out = res.cols();
        let res_size = res.size();
        let (mut tmp, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, cols_out, res_size);
        for col in 0..cols_out {
            module.vec_znx_dft_zero(&mut tmp, col);
        }
        let bytes = fft64_vmp_apply_dft_to_dft_tmp_bytes(a.size(), b.rows(), b.cols_in());
        let (kernel_tmp, _) = take_host_typed::<BE, f64>(scratch_1, bytes / size_of::<f64>());
        fft64_vmp_apply_dft_to_dft(&mut tmp, a, b, limb_offset, kernel_tmp);
        let tmp_ref = tmp.to_backend_ref();
        for col in 0..cols_out {
            module.vec_znx_dft_add_assign(res, col, &tmp_ref, col);
        }
    }

    fn vmp_zero_default(_module: &Module<BE>, res: &mut VmpPMatBackendMut<'_, BE>)
    where
        BE: Backend<ScalarPrep = f64>,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    {
        fft64_vmp_zero(res);
    }
}

impl<BE: Backend> FFT64VmpDefault<BE> for BE
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
{
}

#[doc(hidden)]
pub trait NTT120VmpDefault<BE: Backend>: Backend
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
{
    fn vmp_prepare_tmp_bytes_default(module: &Module<BE>, _rows: usize, _cols_in: usize, _cols_out: usize, _size: usize) -> usize
    where
        BE: Backend<ScalarPrep = Q120bScalar>,
    {
        ntt120_vmp_prepare_tmp_bytes(module.n())
    }

    fn vmp_prepare_default<'s>(
        module: &Module<BE>,
        res: &mut VmpPMatBackendMut<'_, BE>,
        a: &MatZnxBackendRef<'_, BE>,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        Module<BE>: NttModuleHandle,
        BE: 's,
        BE: Backend<ScalarPrep = Q120bScalar> + NttDFTExecute<NttTable<Primes30>> + NttFromZnx64 + NttCFromB,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let bytes = ntt120_vmp_prepare_tmp_bytes(module.n());
        let (tmp, _) = take_host_typed::<BE, u64>(scratch.borrow(), bytes / size_of::<u64>());
        ntt120_vmp_prepare::<BE>(module, res, a, tmp);
    }

    fn vmp_apply_dft_to_dft_tmp_bytes_default(
        _module: &Module<BE>,
        _res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize
    where
        BE: Backend<ScalarPrep = Q120bScalar>,
    {
        ntt120_vmp_apply_dft_to_dft_tmp_bytes(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft_default<'s, 'b>(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        a: &VecZnxDftBackendRef<'_, BE>,
        b: &VmpPMatBackendRef<'b, BE>,
        limb_offset: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        Module<BE>: NttModuleHandle,
        BE: 's,
        BE: 'b,
        BE: Backend<ScalarPrep = Q120bScalar> + NttExtract1BlkContiguous + NttMulBbc1ColX2 + NttMulBbc2ColsX2,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let bytes = ntt120_vmp_apply_dft_to_dft_tmp_bytes(a.size(), b.rows(), b.cols_in());
        let (tmp, _) = take_host_typed::<BE, u64>(scratch.borrow(), bytes / size_of::<u64>());
        ntt120_vmp_apply_dft_to_dft::<BE>(module, res, a, b, limb_offset, tmp);
    }

    fn vmp_apply_dft_to_dft_accumulate_tmp_bytes_default(
        module: &Module<BE>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize
    where
        BE: Backend<ScalarPrep = Q120bScalar>,
        Module<BE>: VecZnxDftBytesOf,
    {
        module.bytes_of_vec_znx_dft(b_cols_out, res_size.min(b_size))
            + ntt120_vmp_apply_dft_to_dft_tmp_bytes(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft_accumulate_default<'s, 'b>(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        a: &VecZnxDftBackendRef<'_, BE>,
        b: &VmpPMatBackendRef<'b, BE>,
        limb_offset: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        Module<BE>: NttModuleHandle + VecZnxDftBytesOf + ModuleN + VecZnxDftAddAssign<BE> + VecZnxDftZero<BE>,
        BE: 's,
        BE: 'b,
        BE: Backend<ScalarPrep = Q120bScalar> + NttExtract1BlkContiguous + NttMulBbc1ColX2 + NttMulBbc2ColsX2,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let cols_out = res.cols();
        let res_size = res.size();
        let (mut tmp, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, cols_out, res_size);
        for col in 0..cols_out {
            module.vec_znx_dft_zero(&mut tmp, col);
        }
        let bytes = ntt120_vmp_apply_dft_to_dft_tmp_bytes(a.size(), b.rows(), b.cols_in());
        let (kernel_tmp, _) = take_host_typed::<BE, u64>(scratch_1, bytes / size_of::<u64>());
        ntt120_vmp_apply_dft_to_dft::<BE>(module, &mut tmp, a, b, limb_offset, kernel_tmp);
        let tmp_ref = tmp.to_backend_ref();
        for col in 0..cols_out {
            module.vec_znx_dft_add_assign(res, col, &tmp_ref, col);
        }
    }

    fn vmp_zero_default(_module: &Module<BE>, res: &mut VmpPMatBackendMut<'_, BE>)
    where
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    {
        ntt120_vmp_zero(res);
    }
}

impl<BE: Backend> NTT120VmpDefault<BE> for BE
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
{
}
