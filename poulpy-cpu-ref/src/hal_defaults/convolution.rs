//! Backend extension points for bivariate convolution operations.

use std::mem::size_of;

use crate::reference::{
    fft64::{
        convolution::{
            I64Ops, convolution_apply_dft, convolution_apply_dft_accumulate, convolution_apply_dft_tmp_bytes,
            convolution_by_const_apply, convolution_by_const_apply_tmp_bytes, convolution_pairwise_apply_dft,
            convolution_pairwise_apply_dft_tmp_bytes, convolution_prepare_left, convolution_prepare_right,
            convolution_prepare_self,
        },
        module::FFTModuleHandle,
        reim::{ReimArith, ReimFFTExecute, ReimFFTTable},
        reim4::{Reim4BlkMatVec, Reim4Convolution},
    },
    ntt4x30::{
        NttAddAssign, NttCFromB, NttDFTExecute, NttFromZnx64, NttMulBbc1ColX2, NttPackLeft1BlkX2,
        convolution::{
            ntt4x30_cnv_accumulate_dft, ntt4x30_cnv_accumulate_dft_tmp_bytes, ntt4x30_cnv_apply_dft,
            ntt4x30_cnv_apply_dft_accumulate, ntt4x30_cnv_apply_dft_tmp_bytes, ntt4x30_cnv_by_const_apply,
            ntt4x30_cnv_by_const_apply_tmp_bytes, ntt4x30_cnv_pairwise_apply_dft, ntt4x30_cnv_pairwise_apply_dft_tmp_bytes,
            ntt4x30_cnv_prepare_left, ntt4x30_cnv_prepare_left_tmp_bytes, ntt4x30_cnv_prepare_right,
            ntt4x30_cnv_prepare_right_tmp_bytes, ntt4x30_cnv_prepare_self, ntt4x30_cnv_prepare_self_tmp_bytes,
        },
        ntt::NttTable,
        primes::Primes30,
        types::Q120bScalar,
        vec_znx_dft::NttModuleHandle,
    },
};
use poulpy_hal::{
    api::{HostBufMut, ModuleN, VecZnxDftBytesOf},
    layouts::{
        Backend, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef, HostDataRef, Module,
        ScratchArena, VecZnxBackendRef, VecZnxBigToBackendMut, VecZnxDft, VecZnxDftToBackendMut,
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
pub trait FFT64ConvolutionDefault<BE: Backend<ZnxWord = i64>>: Backend
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    fn cnv_prepare_left_tmp_bytes_default(module: &Module<BE>, res_size: usize, a_size: usize) -> usize
    where
        BE: Backend<DftWord = f64, ZnxWord = i64>,
    {
        BE::bytes_of_vec_znx_dft(module.n(), 1, res_size.min(a_size))
    }

    fn cnv_prepare_left_default(
        module: &Module<BE>,
        res: &mut CnvPVecLBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: FFTModuleHandle<f64> + ModuleN + VecZnxDftBytesOf,
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + Reim4BlkMatVec + ReimFFTExecute<ReimFFTTable<f64>, f64> + 'static,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8], ZnxWord = i64>,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let tmp_size = res.size().min(a.size());
        let (tmp_bytes, _) = take_host_typed::<BE, u8>(scratch.borrow(), BE::bytes_of_vec_znx_dft(module.n(), 1, tmp_size));
        let mut tmp = VecZnxDft::from_data(tmp_bytes, module.n(), 1, tmp_size);
        let mut tmp_ref = vec_znx_dft_backend_mut_from_mut::<BE>(&mut tmp);
        convolution_prepare_left::<BE>(module.get_fft_table(), res, a, mask, &mut tmp_ref);
    }

    fn cnv_prepare_right_tmp_bytes_default(module: &Module<BE>, res_size: usize, a_size: usize) -> usize
    where
        BE: Backend<DftWord = f64, ZnxWord = i64>,
    {
        BE::bytes_of_vec_znx_dft(module.n(), 1, res_size.min(a_size))
    }

    fn cnv_prepare_right_default(
        module: &Module<BE>,
        res: &mut CnvPVecRBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: FFTModuleHandle<f64> + ModuleN + VecZnxDftBytesOf,
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + Reim4BlkMatVec + ReimFFTExecute<ReimFFTTable<f64>, f64> + 'static,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8], ZnxWord = i64>,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let tmp_size = res.size().min(a.size());
        let (tmp_bytes, _) = take_host_typed::<BE, u8>(scratch.borrow(), BE::bytes_of_vec_znx_dft(module.n(), 1, tmp_size));
        let mut tmp = VecZnxDft::from_data(tmp_bytes, module.n(), 1, tmp_size);
        let mut tmp_ref = vec_znx_dft_backend_mut_from_mut::<BE>(&mut tmp);
        convolution_prepare_right::<BE>(module.get_fft_table(), res, a, mask, &mut tmp_ref);
    }

    fn cnv_apply_dft_tmp_bytes_default(
        _module: &Module<BE>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize
    where
        BE: Backend<DftWord = f64, ZnxWord = i64>,
    {
        convolution_apply_dft_tmp_bytes(res_size, a_size, b_size)
    }

    fn cnv_by_const_apply_tmp_bytes_default(
        _module: &Module<BE>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize
    where
        BE: Backend<BigWord = i64, ZnxWord = i64>,
    {
        convolution_by_const_apply_tmp_bytes(res_size, a_size, b_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_by_const_apply_default<R>(
        _module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
        b_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend<BigWord = i64, ZnxWord = i64> + I64Ops + 'static,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8], ZnxWord = i64>,
        for<'x> <BE as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
        R: VecZnxBigToBackendMut<BE>,
    {
        let mut res_ref = res.to_backend_mut();
        let bytes = convolution_by_const_apply_tmp_bytes(res_ref.size(), a.size(), b.size());
        let (tmp, _) = take_host_typed::<BE, i64>(scratch.borrow(), bytes / size_of::<i64>());
        convolution_by_const_apply::<BE>(cnv_offset, &mut res_ref, res_col, a, a_col, b, b_col, b_coeff, tmp);
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft_default<R>(
        _module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, BE>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend<DftWord = f64, ZnxWord = i64> + Reim4BlkMatVec + Reim4Convolution,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
        for<'x> <BE as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        R: VecZnxDftToBackendMut<BE>,
    {
        let mut res_ref = res.to_backend_mut();
        let bytes = convolution_apply_dft_tmp_bytes(res_ref.size(), a.size(), b.size());
        let (tmp, _) = take_host_typed::<BE, f64>(scratch.borrow(), bytes / size_of::<f64>());
        convolution_apply_dft::<BE>(cnv_offset, &mut res_ref, res_col, a, a_col, b, b_col, tmp);
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft_accumulate_default<R>(
        _module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, BE>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend<DftWord = f64, ZnxWord = i64> + Reim4BlkMatVec + Reim4Convolution,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
        for<'x> <BE as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        R: VecZnxDftToBackendMut<BE>,
    {
        let mut res_ref = res.to_backend_mut();
        let bytes = convolution_apply_dft_tmp_bytes(res_ref.size(), a.size(), b.size());
        let (tmp, _) = take_host_typed::<BE, f64>(scratch.borrow(), bytes / size_of::<f64>());
        convolution_apply_dft_accumulate::<BE>(cnv_offset, &mut res_ref, res_col, a, a_col, b, b_col, tmp);
    }

    fn cnv_pairwise_apply_dft_tmp_bytes_default(
        _module: &Module<BE>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize
    where
        BE: Backend<DftWord = f64, ZnxWord = i64>,
    {
        convolution_pairwise_apply_dft_tmp_bytes(res_size, a_size, b_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_pairwise_apply_dft_default<R>(
        _module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, BE>,
        b: &CnvPVecRBackendRef<'_, BE>,
        i: usize,
        j: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + Reim4BlkMatVec + Reim4Convolution,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
        for<'x> <BE as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        R: VecZnxDftToBackendMut<BE>,
    {
        let mut res_ref = res.to_backend_mut();
        let bytes = convolution_pairwise_apply_dft_tmp_bytes(res_ref.size(), a.size(), b.size());
        let (tmp, _) = take_host_typed::<BE, f64>(scratch.borrow(), bytes / size_of::<f64>());
        convolution_pairwise_apply_dft::<BE>(cnv_offset, &mut res_ref, res_col, a, b, i, j, tmp);
    }

    fn cnv_prepare_self_tmp_bytes_default(module: &Module<BE>, res_size: usize, a_size: usize) -> usize
    where
        BE: Backend<DftWord = f64, ZnxWord = i64>,
    {
        BE::bytes_of_vec_znx_dft(module.n(), 1, res_size.min(a_size))
    }

    fn cnv_prepare_self_default(
        module: &Module<BE>,
        left: &mut CnvPVecLBackendMut<'_, BE>,
        right: &mut CnvPVecRBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: FFTModuleHandle<f64> + ModuleN + VecZnxDftBytesOf,
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + Reim4BlkMatVec + ReimFFTExecute<ReimFFTTable<f64>, f64> + 'static,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8], ZnxWord = i64>,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let tmp_size = left.size().min(a.size());
        let (tmp_bytes, _) = take_host_typed::<BE, u8>(scratch.borrow(), BE::bytes_of_vec_znx_dft(module.n(), 1, tmp_size));
        let mut tmp = VecZnxDft::from_data(tmp_bytes, module.n(), 1, tmp_size);
        let mut tmp_ref = vec_znx_dft_backend_mut_from_mut::<BE>(&mut tmp);
        convolution_prepare_self::<BE>(module.get_fft_table(), left, right, a, mask, &mut tmp_ref);
    }
}

impl<BE: Backend<ZnxWord = i64>> FFT64ConvolutionDefault<BE> for BE where BE::OwnedBuf: poulpy_hal::layouts::HostDataMut {}

#[doc(hidden)]
pub trait NTT4x30ConvolutionDefault<BE: Backend<ZnxWord = i64>>: Backend
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    fn cnv_prepare_left_tmp_bytes_default(module: &Module<BE>, _res_size: usize, _a_size: usize) -> usize
    where
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>,
    {
        ntt4x30_cnv_prepare_left_tmp_bytes(module.n())
    }

    fn cnv_prepare_left_default(
        module: &Module<BE>,
        res: &mut CnvPVecLBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>
            + NttFromZnx64
            + NttDFTExecute<NttTable<Primes30>>
            + NttPackLeft1BlkX2
            + 'static,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8], ZnxWord = i64>,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let bytes = ntt4x30_cnv_prepare_left_tmp_bytes(module.n());
        let (tmp, _) = take_host_typed::<BE, u8>(scratch.borrow(), bytes);
        ntt4x30_cnv_prepare_left::<BE>(module, res, a, mask, tmp);
    }

    fn cnv_prepare_right_tmp_bytes_default(module: &Module<BE>, _res_size: usize, _a_size: usize) -> usize
    where
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>,
    {
        ntt4x30_cnv_prepare_right_tmp_bytes(module.n())
    }

    fn cnv_prepare_right_default(
        module: &Module<BE>,
        res: &mut CnvPVecRBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>
            + NttFromZnx64
            + NttDFTExecute<NttTable<Primes30>>
            + NttCFromB
            + 'static,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8], ZnxWord = i64>,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let bytes = ntt4x30_cnv_prepare_right_tmp_bytes(module.n());
        let (tmp, _) = take_host_typed::<BE, u64>(scratch.borrow(), bytes / size_of::<u64>());
        ntt4x30_cnv_prepare_right::<BE>(module, res, a, mask, tmp);
    }

    fn cnv_apply_dft_tmp_bytes_default(
        _module: &Module<BE>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize
    where
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>,
    {
        ntt4x30_cnv_apply_dft_tmp_bytes(res_size, a_size, b_size)
    }

    fn cnv_by_const_apply_tmp_bytes_default(
        _module: &Module<BE>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize
    where
        BE: Backend<BigWord = i128, DftWord = Q120bScalar, ZnxWord = i64>,
    {
        ntt4x30_cnv_by_const_apply_tmp_bytes(res_size, a_size, b_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_by_const_apply_default<R>(
        _module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
        b_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend<BigWord = i128, DftWord = Q120bScalar, ZnxWord = i64> + 'static,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8], ZnxWord = i64>,
        for<'x> <BE as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
        R: VecZnxBigToBackendMut<BE>,
    {
        let mut res_ref = res.to_backend_mut();
        let bytes = ntt4x30_cnv_by_const_apply_tmp_bytes(0, 0, 0);
        let (tmp, _) = take_host_typed::<BE, u8>(scratch.borrow(), bytes);
        ntt4x30_cnv_by_const_apply::<BE>(cnv_offset, &mut res_ref, res_col, a, a_col, b, b_col, b_coeff, tmp);
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft_default<R>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, BE>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttAddAssign + NttMulBbc1ColX2,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
        for<'x> <BE as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        R: VecZnxDftToBackendMut<BE>,
    {
        let mut res_ref = res.to_backend_mut();
        let bytes = ntt4x30_cnv_apply_dft_tmp_bytes(res_ref.size(), a.size(), b.size());
        let (tmp, _) = take_host_typed::<BE, u8>(scratch.borrow(), bytes);
        ntt4x30_cnv_apply_dft::<BE>(module, cnv_offset, &mut res_ref, res_col, a, a_col, b, b_col, tmp);
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft_accumulate_default<R>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, BE>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttAddAssign + NttMulBbc1ColX2,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
        for<'x> <BE as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        R: VecZnxDftToBackendMut<BE>,
    {
        let mut res_ref = res.to_backend_mut();
        let bytes = ntt4x30_cnv_apply_dft_tmp_bytes(res_ref.size(), a.size(), b.size());
        let (tmp, _) = take_host_typed::<BE, u8>(scratch.borrow(), bytes);
        ntt4x30_cnv_apply_dft_accumulate::<BE>(module, cnv_offset, &mut res_ref, res_col, a, a_col, b, b_col, tmp);
    }

    fn cnv_accumulate_dft_tmp_bytes_default(
        _module: &Module<BE>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize
    where
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>,
    {
        ntt4x30_cnv_accumulate_dft_tmp_bytes(res_size, a_size, b_size)
    }

    fn cnv_accumulate_dft_default<'a, R>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        terms: &[poulpy_hal::layouts::CnvDftAccTerm<'a, BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + 'a,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
        for<'x> <BE as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        R: VecZnxDftToBackendMut<BE>,
    {
        let mut res_ref = res.to_backend_mut();
        let bytes = ntt4x30_cnv_accumulate_dft_tmp_bytes(res_ref.size(), 0, 0);
        let (tmp, _) = take_host_typed::<BE, u8>(scratch.borrow(), bytes);
        ntt4x30_cnv_accumulate_dft::<BE>(module, cnv_offset, &mut res_ref, res_col, terms, tmp);
    }

    fn cnv_pairwise_apply_dft_tmp_bytes_default(
        _module: &Module<BE>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize
    where
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>,
    {
        ntt4x30_cnv_pairwise_apply_dft_tmp_bytes(res_size, a_size, b_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_pairwise_apply_dft_default<R>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, BE>,
        b: &CnvPVecRBackendRef<'_, BE>,
        i: usize,
        j: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttAddAssign + NttMulBbc1ColX2,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
        for<'x> <BE as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        R: VecZnxDftToBackendMut<BE>,
    {
        let mut res_ref = res.to_backend_mut();
        let bytes = ntt4x30_cnv_pairwise_apply_dft_tmp_bytes(res_ref.size(), a.size(), b.size());
        let (tmp, _) = take_host_typed::<BE, u8>(scratch.borrow(), bytes);
        ntt4x30_cnv_pairwise_apply_dft::<BE>(module, cnv_offset, &mut res_ref, res_col, a, b, i, j, tmp);
    }

    fn cnv_prepare_self_tmp_bytes_default(module: &Module<BE>, _res_size: usize, _a_size: usize) -> usize
    where
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>,
    {
        ntt4x30_cnv_prepare_self_tmp_bytes(module.n())
    }

    fn cnv_prepare_self_default(
        module: &Module<BE>,
        left: &mut CnvPVecLBackendMut<'_, BE>,
        right: &mut CnvPVecRBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>
            + NttFromZnx64
            + NttDFTExecute<NttTable<Primes30>>
            + NttCFromB
            + NttPackLeft1BlkX2
            + 'static,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8], ZnxWord = i64>,
        for<'x> BE::BufMut<'x>: HostBufMut<'x>,
    {
        let bytes = ntt4x30_cnv_prepare_self_tmp_bytes(module.n());
        let (tmp, _) = take_host_typed::<BE, u8>(scratch.borrow(), bytes);
        ntt4x30_cnv_prepare_self::<BE>(module, left, right, a, mask, tmp);
    }
}

impl<BE: Backend<ZnxWord = i64>> NTT4x30ConvolutionDefault<BE> for BE where BE::OwnedBuf: poulpy_hal::layouts::HostDataMut {}
