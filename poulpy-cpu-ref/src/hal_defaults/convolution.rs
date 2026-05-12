//! Backend extension points for bivariate convolution operations.

use std::mem::size_of;

use crate::reference::{
    fft64::{
        convolution::{
            I64Ops, convolution_apply_dft, convolution_apply_dft_tmp_bytes, convolution_by_const_apply,
            convolution_by_const_apply_tmp_bytes, convolution_pairwise_apply_dft, convolution_pairwise_apply_dft_tmp_bytes,
            convolution_prepare_left, convolution_prepare_right, convolution_prepare_self,
        },
        module::FFTModuleHandle,
        reim::{ReimArith, ReimFFTExecute, ReimFFTTable},
        reim4::{Reim4BlkMatVec, Reim4Convolution},
    },
    ntt120::{
        NttAddAssign, NttCFromB, NttDFTExecute, NttFromZnx64, NttMulBbc1ColX2, NttMulBbc2ColsX2, NttPackLeft1BlkX2,
        NttPackRight1BlkX2, NttPairwisePackLeft1BlkX2, NttPairwisePackRight1BlkX2,
        convolution::{
            ntt120_cnv_apply_dft, ntt120_cnv_apply_dft_tmp_bytes, ntt120_cnv_by_const_apply, ntt120_cnv_by_const_apply_tmp_bytes,
            ntt120_cnv_pairwise_apply_dft, ntt120_cnv_pairwise_apply_dft_tmp_bytes, ntt120_cnv_prepare_left,
            ntt120_cnv_prepare_left_tmp_bytes, ntt120_cnv_prepare_right, ntt120_cnv_prepare_right_tmp_bytes,
            ntt120_cnv_prepare_self, ntt120_cnv_prepare_self_tmp_bytes,
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
pub trait FFT64ConvolutionDefault<BE: Backend>: Backend
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    fn cnv_prepare_left_tmp_bytes_default(module: &Module<BE>, res_size: usize, a_size: usize) -> usize
    where
        BE: Backend<ScalarPrep = f64>,
    {
        BE::bytes_of_vec_znx_dft(module.n(), 1, res_size.min(a_size))
    }

    fn cnv_prepare_left_default<'s, 'r>(
        module: &Module<BE>,
        res: &mut CnvPVecLBackendMut<'r, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        Module<BE>: FFTModuleHandle<f64> + ModuleN + VecZnxDftBytesOf,
        BE: Backend<ScalarPrep = f64> + ReimArith + Reim4BlkMatVec + ReimFFTExecute<ReimFFTTable<f64>, f64> + 'static,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let tmp_size = res.size().min(a.size());
        let (tmp_bytes, _) = take_host_typed::<BE, u8>(scratch.borrow(), BE::bytes_of_vec_znx_dft(module.n(), 1, tmp_size));
        let mut tmp = VecZnxDft::from_data(tmp_bytes, module.n(), 1, tmp_size);
        let mut tmp_ref = vec_znx_dft_backend_mut_from_mut::<BE>(&mut tmp);
        convolution_prepare_left(module.get_fft_table(), res, a, mask, &mut tmp_ref);
    }

    fn cnv_prepare_right_tmp_bytes_default(module: &Module<BE>, res_size: usize, a_size: usize) -> usize
    where
        BE: Backend<ScalarPrep = f64>,
    {
        BE::bytes_of_vec_znx_dft(module.n(), 1, res_size.min(a_size))
    }

    fn cnv_prepare_right_default<'s, 'r>(
        module: &Module<BE>,
        res: &mut CnvPVecRBackendMut<'r, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        Module<BE>: FFTModuleHandle<f64> + ModuleN + VecZnxDftBytesOf,
        BE: Backend<ScalarPrep = f64> + ReimArith + Reim4BlkMatVec + ReimFFTExecute<ReimFFTTable<f64>, f64> + 'static,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let tmp_size = res.size().min(a.size());
        let (tmp_bytes, _) = take_host_typed::<BE, u8>(scratch.borrow(), BE::bytes_of_vec_znx_dft(module.n(), 1, tmp_size));
        let mut tmp = VecZnxDft::from_data(tmp_bytes, module.n(), 1, tmp_size);
        let mut tmp_ref = vec_znx_dft_backend_mut_from_mut::<BE>(&mut tmp);
        convolution_prepare_right(module.get_fft_table(), res, a, mask, &mut tmp_ref);
    }

    fn cnv_apply_dft_tmp_bytes_default(
        _module: &Module<BE>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize
    where
        BE: Backend<ScalarPrep = f64>,
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
        BE: Backend<ScalarBig = i64>,
    {
        convolution_by_const_apply_tmp_bytes(res_size, a_size, b_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_by_const_apply_default<'s, R>(
        _module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
        b_coeff: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: Backend<ScalarBig = i64> + I64Ops + 'static,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8]>,
        for<'x> <BE as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        BE::BufMut<'s>: HostBufMut<'s>,
        R: VecZnxBigToBackendMut<BE>,
    {
        let mut res_ref = res.to_backend_mut();
        let bytes = convolution_by_const_apply_tmp_bytes(res_ref.size(), a.size(), b.size());
        let (tmp, _) = take_host_typed::<BE, i64>(scratch.borrow(), bytes / size_of::<i64>());
        convolution_by_const_apply(cnv_offset, &mut res_ref, res_col, a, a_col, b, b_col, b_coeff, tmp);
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft_default<'s, R>(
        _module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, BE>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, BE>,
        b_col: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: Backend<ScalarPrep = f64> + Reim4BlkMatVec + Reim4Convolution,
        BE::BufMut<'s>: HostBufMut<'s>,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
        for<'x> <BE as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        R: VecZnxDftToBackendMut<BE>,
    {
        let mut res_ref = res.to_backend_mut();
        let bytes = convolution_apply_dft_tmp_bytes(res_ref.size(), a.size(), b.size());
        let (tmp, _) = take_host_typed::<BE, f64>(scratch.borrow(), bytes / size_of::<f64>());
        convolution_apply_dft(cnv_offset, &mut res_ref, res_col, a, a_col, b, b_col, tmp);
    }

    fn cnv_pairwise_apply_dft_tmp_bytes_default(
        _module: &Module<BE>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize
    where
        BE: Backend<ScalarPrep = f64>,
    {
        convolution_pairwise_apply_dft_tmp_bytes(res_size, a_size, b_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_pairwise_apply_dft_default<'s, R>(
        _module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, BE>,
        b: &CnvPVecRBackendRef<'_, BE>,
        i: usize,
        j: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: Backend<ScalarPrep = f64> + ReimArith + Reim4BlkMatVec + Reim4Convolution,
        BE::BufMut<'s>: HostBufMut<'s>,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
        for<'x> <BE as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        R: VecZnxDftToBackendMut<BE>,
    {
        let mut res_ref = res.to_backend_mut();
        let bytes = convolution_pairwise_apply_dft_tmp_bytes(res_ref.size(), a.size(), b.size());
        let (tmp, _) = take_host_typed::<BE, f64>(scratch.borrow(), bytes / size_of::<f64>());
        convolution_pairwise_apply_dft(cnv_offset, &mut res_ref, res_col, a, b, i, j, tmp);
    }

    fn cnv_prepare_self_tmp_bytes_default(module: &Module<BE>, res_size: usize, a_size: usize) -> usize
    where
        BE: Backend<ScalarPrep = f64>,
    {
        BE::bytes_of_vec_znx_dft(module.n(), 1, res_size.min(a_size))
    }

    fn cnv_prepare_self_default<'s, 'l, 'r>(
        module: &Module<BE>,
        left: &mut CnvPVecLBackendMut<'l, BE>,
        right: &mut CnvPVecRBackendMut<'r, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        Module<BE>: FFTModuleHandle<f64> + ModuleN + VecZnxDftBytesOf,
        BE: Backend<ScalarPrep = f64> + ReimArith + Reim4BlkMatVec + ReimFFTExecute<ReimFFTTable<f64>, f64> + 'static,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let tmp_size = left.size().min(a.size());
        let (tmp_bytes, _) = take_host_typed::<BE, u8>(scratch.borrow(), BE::bytes_of_vec_znx_dft(module.n(), 1, tmp_size));
        let mut tmp = VecZnxDft::from_data(tmp_bytes, module.n(), 1, tmp_size);
        let mut tmp_ref = vec_znx_dft_backend_mut_from_mut::<BE>(&mut tmp);
        convolution_prepare_self(module.get_fft_table(), left, right, a, mask, &mut tmp_ref);
    }
}

impl<BE: Backend> FFT64ConvolutionDefault<BE> for BE where BE::OwnedBuf: poulpy_hal::layouts::HostDataMut {}

#[doc(hidden)]
pub trait NTT120ConvolutionDefault<BE: Backend>: Backend
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    fn cnv_prepare_left_tmp_bytes_default(module: &Module<BE>, _res_size: usize, _a_size: usize) -> usize
    where
        BE: Backend<ScalarPrep = Q120bScalar>,
    {
        ntt120_cnv_prepare_left_tmp_bytes(module.n())
    }

    fn cnv_prepare_left_default<'s, 'r>(
        module: &Module<BE>,
        res: &mut CnvPVecLBackendMut<'r, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar> + NttFromZnx64 + NttDFTExecute<NttTable<Primes30>> + 'static,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let bytes = ntt120_cnv_prepare_left_tmp_bytes(module.n());
        let (tmp, _) = take_host_typed::<BE, u8>(scratch.borrow(), bytes);
        ntt120_cnv_prepare_left::<BE>(module, res, a, mask, tmp);
    }

    fn cnv_prepare_right_tmp_bytes_default(module: &Module<BE>, _res_size: usize, _a_size: usize) -> usize
    where
        BE: Backend<ScalarPrep = Q120bScalar>,
    {
        ntt120_cnv_prepare_right_tmp_bytes(module.n())
    }

    fn cnv_prepare_right_default<'s, 'r>(
        module: &Module<BE>,
        res: &mut CnvPVecRBackendMut<'r, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar> + NttFromZnx64 + NttDFTExecute<NttTable<Primes30>> + NttCFromB + 'static,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let bytes = ntt120_cnv_prepare_right_tmp_bytes(module.n());
        let (tmp, _) = take_host_typed::<BE, u64>(scratch.borrow(), bytes / size_of::<u64>());
        ntt120_cnv_prepare_right::<BE>(module, res, a, mask, tmp);
    }

    fn cnv_apply_dft_tmp_bytes_default(
        _module: &Module<BE>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize
    where
        BE: Backend<ScalarPrep = Q120bScalar>,
    {
        ntt120_cnv_apply_dft_tmp_bytes(res_size, a_size, b_size)
    }

    fn cnv_by_const_apply_tmp_bytes_default(
        _module: &Module<BE>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize
    where
        BE: Backend<ScalarBig = i128, ScalarPrep = Q120bScalar>,
    {
        ntt120_cnv_by_const_apply_tmp_bytes(res_size, a_size, b_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_by_const_apply_default<'s, R>(
        _module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
        b_coeff: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: Backend<ScalarBig = i128, ScalarPrep = Q120bScalar> + 'static,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8]>,
        for<'x> <BE as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        BE::BufMut<'s>: HostBufMut<'s>,
        R: VecZnxBigToBackendMut<BE>,
    {
        let mut res_ref = res.to_backend_mut();
        let bytes = ntt120_cnv_by_const_apply_tmp_bytes(0, 0, 0);
        let (tmp, _) = take_host_typed::<BE, u8>(scratch.borrow(), bytes);
        ntt120_cnv_by_const_apply::<BE>(cnv_offset, &mut res_ref, res_col, a, a_col, b, b_col, b_coeff, tmp);
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft_default<'s, R>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, BE>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, BE>,
        b_col: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar>
            + NttAddAssign
            + NttMulBbc1ColX2
            + NttMulBbc2ColsX2
            + NttPackLeft1BlkX2
            + NttPackRight1BlkX2,
        BE::BufMut<'s>: HostBufMut<'s>,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
        for<'x> <BE as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        R: VecZnxDftToBackendMut<BE>,
    {
        let mut res_ref = res.to_backend_mut();
        let bytes = ntt120_cnv_apply_dft_tmp_bytes(res_ref.size(), a.size(), b.size());
        let (tmp, _) = take_host_typed::<BE, u8>(scratch.borrow(), bytes);
        ntt120_cnv_apply_dft::<BE>(module, cnv_offset, &mut res_ref, res_col, a, a_col, b, b_col, tmp);
    }

    fn cnv_pairwise_apply_dft_tmp_bytes_default(
        _module: &Module<BE>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize
    where
        BE: Backend<ScalarPrep = Q120bScalar>,
    {
        ntt120_cnv_pairwise_apply_dft_tmp_bytes(res_size, a_size, b_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_pairwise_apply_dft_default<'s, R>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, BE>,
        b: &CnvPVecRBackendRef<'_, BE>,
        i: usize,
        j: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar>
            + NttAddAssign
            + NttMulBbc1ColX2
            + NttMulBbc2ColsX2
            + NttPackLeft1BlkX2
            + NttPackRight1BlkX2
            + NttPairwisePackLeft1BlkX2
            + NttPairwisePackRight1BlkX2,
        BE::BufMut<'s>: HostBufMut<'s>,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
        for<'x> <BE as Backend>::BufMut<'x>: poulpy_hal::layouts::HostDataMut,
        R: VecZnxDftToBackendMut<BE>,
    {
        let mut res_ref = res.to_backend_mut();
        let bytes = ntt120_cnv_pairwise_apply_dft_tmp_bytes(res_ref.size(), a.size(), b.size());
        let (tmp, _) = take_host_typed::<BE, u8>(scratch.borrow(), bytes);
        ntt120_cnv_pairwise_apply_dft::<BE>(module, cnv_offset, &mut res_ref, res_col, a, b, i, j, tmp);
    }

    fn cnv_prepare_self_tmp_bytes_default(module: &Module<BE>, _res_size: usize, _a_size: usize) -> usize
    where
        BE: Backend<ScalarPrep = Q120bScalar>,
    {
        ntt120_cnv_prepare_self_tmp_bytes(module.n())
    }

    fn cnv_prepare_self_default<'s, 'l, 'r>(
        module: &Module<BE>,
        left: &mut CnvPVecLBackendMut<'l, BE>,
        right: &mut CnvPVecRBackendMut<'r, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar> + NttFromZnx64 + NttDFTExecute<NttTable<Primes30>> + NttCFromB + 'static,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let bytes = ntt120_cnv_prepare_self_tmp_bytes(module.n());
        let (tmp, _) = take_host_typed::<BE, u8>(scratch.borrow(), bytes);
        ntt120_cnv_prepare_self::<BE>(module, left, right, a, mask, tmp);
    }
}

impl<BE: Backend> NTT120ConvolutionDefault<BE> for BE where BE::OwnedBuf: poulpy_hal::layouts::HostDataMut {}
