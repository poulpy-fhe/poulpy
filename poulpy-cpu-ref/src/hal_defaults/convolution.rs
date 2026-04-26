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
    api::{ModuleN, ScratchTakeBasic, TakeSlice, VecZnxDftBytesOf},
    layouts::{
        Backend, CnvPVecL, CnvPVecLToMut, CnvPVecLToRef, CnvPVecR, CnvPVecRToMut, CnvPVecRToRef, Module, Scratch, VecZnx,
        VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftToMut, VecZnxToRef, ZnxInfos,
    },
};
#[doc(hidden)]
pub trait FFT64ConvolutionDefaults<BE: Backend>: Backend {
    fn cnv_prepare_left_tmp_bytes_default(module: &Module<BE>, res_size: usize, a_size: usize) -> usize
    where
        BE: Backend<ScalarPrep = f64>,
    {
        BE::bytes_of_vec_znx_dft(module.n(), 1, res_size.min(a_size))
    }

    fn cnv_prepare_left_default<R, A>(module: &Module<BE>, res: &mut R, a: &A, mask: i64, scratch: &mut Scratch<BE>)
    where
        Module<BE>: FFTModuleHandle<f64> + ModuleN + VecZnxDftBytesOf,
        BE: Backend<ScalarPrep = f64> + ReimArith + Reim4BlkMatVec + ReimFFTExecute<ReimFFTTable<f64>, f64>,
        Scratch<BE>: ScratchTakeBasic,
        R: CnvPVecLToMut<BE>,
        A: VecZnxToRef,
    {
        let mut res: CnvPVecL<&mut [u8], BE> = res.to_mut();
        let a: VecZnx<&[u8]> = a.to_ref();
        let (mut tmp, _) = scratch.take_vec_znx_dft(module, 1, res.size().min(a.size()));
        convolution_prepare_left(module.get_fft_table(), &mut res, &a, mask, &mut tmp);
    }

    fn cnv_prepare_right_tmp_bytes_default(module: &Module<BE>, res_size: usize, a_size: usize) -> usize
    where
        BE: Backend<ScalarPrep = f64>,
    {
        BE::bytes_of_vec_znx_dft(module.n(), 1, res_size.min(a_size))
    }

    fn cnv_prepare_right_default<R, A>(module: &Module<BE>, res: &mut R, a: &A, mask: i64, scratch: &mut Scratch<BE>)
    where
        Module<BE>: FFTModuleHandle<f64> + ModuleN + VecZnxDftBytesOf,
        BE: Backend<ScalarPrep = f64> + ReimArith + Reim4BlkMatVec + ReimFFTExecute<ReimFFTTable<f64>, f64>,
        Scratch<BE>: ScratchTakeBasic,
        R: CnvPVecRToMut<BE>,
        A: VecZnxToRef,
    {
        let mut res: CnvPVecR<&mut [u8], BE> = res.to_mut();
        let a: VecZnx<&[u8]> = a.to_ref();
        let (mut tmp, _) = scratch.take_vec_znx_dft(module, 1, res.size().min(a.size()));
        convolution_prepare_right(module.get_fft_table(), &mut res, &a, mask, &mut tmp);
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
    fn cnv_by_const_apply_default<R, A>(
        _module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &[i64],
        scratch: &mut Scratch<BE>,
    ) where
        BE: Backend<ScalarBig = i64> + I64Ops,
        Scratch<BE>: TakeSlice,
        R: VecZnxBigToMut<BE>,
        A: VecZnxToRef,
    {
        let mut res: VecZnxBig<&mut [u8], BE> = res.to_mut();
        let a: VecZnx<&[u8]> = a.to_ref();
        let bytes = convolution_by_const_apply_tmp_bytes(res.size(), a.size(), b.len());
        let (tmp, _) = scratch.take_slice::<i64>(bytes / size_of::<i64>());
        convolution_by_const_apply(cnv_offset, &mut res, res_col, &a, a_col, b, tmp);
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft_default<R, A, B>(
        _module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        BE: Backend<ScalarPrep = f64> + Reim4BlkMatVec + Reim4Convolution,
        Scratch<BE>: TakeSlice,
        R: VecZnxDftToMut<BE>,
        A: CnvPVecLToRef<BE>,
        B: CnvPVecRToRef<BE>,
    {
        let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
        let a: CnvPVecL<&[u8], BE> = a.to_ref();
        let b: CnvPVecR<&[u8], BE> = b.to_ref();
        let bytes = convolution_apply_dft_tmp_bytes(res.size(), a.size(), b.size());
        let (tmp, _) = scratch.take_slice::<f64>(bytes / size_of::<f64>());
        convolution_apply_dft(cnv_offset, &mut res, res_col, &a, a_col, &b, b_col, tmp);
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
    fn cnv_pairwise_apply_dft_default<R, A, B>(
        _module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        b: &B,
        i: usize,
        j: usize,
        scratch: &mut Scratch<BE>,
    ) where
        BE: Backend<ScalarPrep = f64> + ReimArith + Reim4BlkMatVec + Reim4Convolution,
        Scratch<BE>: TakeSlice,
        R: VecZnxDftToMut<BE>,
        A: CnvPVecLToRef<BE>,
        B: CnvPVecRToRef<BE>,
    {
        let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
        let a: CnvPVecL<&[u8], BE> = a.to_ref();
        let b: CnvPVecR<&[u8], BE> = b.to_ref();
        let bytes = convolution_pairwise_apply_dft_tmp_bytes(res.size(), a.size(), b.size());
        let (tmp, _) = scratch.take_slice::<f64>(bytes / size_of::<f64>());
        convolution_pairwise_apply_dft(cnv_offset, &mut res, res_col, &a, &b, i, j, tmp);
    }

    fn cnv_prepare_self_tmp_bytes_default(module: &Module<BE>, res_size: usize, a_size: usize) -> usize
    where
        BE: Backend<ScalarPrep = f64>,
    {
        BE::bytes_of_vec_znx_dft(module.n(), 1, res_size.min(a_size))
    }

    fn cnv_prepare_self_default<L, R, A>(
        module: &Module<BE>,
        left: &mut L,
        right: &mut R,
        a: &A,
        mask: i64,
        scratch: &mut Scratch<BE>,
    ) where
        Module<BE>: FFTModuleHandle<f64> + ModuleN + VecZnxDftBytesOf,
        BE: Backend<ScalarPrep = f64> + ReimArith + Reim4BlkMatVec + ReimFFTExecute<ReimFFTTable<f64>, f64>,
        Scratch<BE>: ScratchTakeBasic,
        L: CnvPVecLToMut<BE>,
        R: CnvPVecRToMut<BE>,
        A: VecZnxToRef + ZnxInfos,
    {
        let mut left: CnvPVecL<&mut [u8], BE> = left.to_mut();
        let mut right: CnvPVecR<&mut [u8], BE> = right.to_mut();
        let a: VecZnx<&[u8]> = a.to_ref();
        let (mut tmp, _) = scratch.take_vec_znx_dft(module, 1, left.size().min(a.size()));
        convolution_prepare_self(module.get_fft_table(), &mut left, &mut right, &a, mask, &mut tmp);
    }
}

impl<BE: Backend> FFT64ConvolutionDefaults<BE> for BE {}

#[doc(hidden)]
pub trait NTT120ConvolutionDefaults<BE: Backend>: Backend {
    fn cnv_prepare_left_tmp_bytes_default(module: &Module<BE>, _res_size: usize, _a_size: usize) -> usize
    where
        BE: Backend<ScalarPrep = Q120bScalar>,
    {
        ntt120_cnv_prepare_left_tmp_bytes(module.n())
    }

    fn cnv_prepare_left_default<R, A>(module: &Module<BE>, res: &mut R, a: &A, mask: i64, scratch: &mut Scratch<BE>)
    where
        Module<BE>: NttModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar> + NttFromZnx64 + NttDFTExecute<NttTable<Primes30>>,
        Scratch<BE>: TakeSlice,
        R: CnvPVecLToMut<BE>,
        A: VecZnxToRef,
    {
        let bytes = ntt120_cnv_prepare_left_tmp_bytes(module.n());
        let (tmp, _) = scratch.take_slice::<u8>(bytes);
        ntt120_cnv_prepare_left::<R, A, BE>(module, res, a, mask, tmp);
    }

    fn cnv_prepare_right_tmp_bytes_default(module: &Module<BE>, _res_size: usize, _a_size: usize) -> usize
    where
        BE: Backend<ScalarPrep = Q120bScalar>,
    {
        ntt120_cnv_prepare_right_tmp_bytes(module.n())
    }

    fn cnv_prepare_right_default<R, A>(module: &Module<BE>, res: &mut R, a: &A, mask: i64, scratch: &mut Scratch<BE>)
    where
        Module<BE>: NttModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar> + NttFromZnx64 + NttDFTExecute<NttTable<Primes30>> + NttCFromB,
        Scratch<BE>: TakeSlice,
        R: CnvPVecRToMut<BE>,
        A: VecZnxToRef + ZnxInfos,
    {
        let bytes = ntt120_cnv_prepare_right_tmp_bytes(module.n());
        let (tmp, _) = scratch.take_slice::<u64>(bytes / size_of::<u64>());
        ntt120_cnv_prepare_right::<R, A, BE>(module, res, a, mask, tmp);
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
    fn cnv_by_const_apply_default<R, A>(
        _module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &[i64],
        scratch: &mut Scratch<BE>,
    ) where
        BE: Backend<ScalarBig = i128, ScalarPrep = Q120bScalar>,
        Scratch<BE>: TakeSlice,
        R: VecZnxBigToMut<BE>,
        A: VecZnxToRef,
    {
        let bytes = ntt120_cnv_by_const_apply_tmp_bytes(0, 0, 0);
        let (tmp, _) = scratch.take_slice::<u8>(bytes);
        ntt120_cnv_by_const_apply::<R, A, BE>(cnv_offset, res, res_col, a, a_col, b, tmp);
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft_default<R, A, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar>
            + NttAddAssign
            + NttMulBbc1ColX2
            + NttMulBbc2ColsX2
            + NttPackLeft1BlkX2
            + NttPackRight1BlkX2,
        Scratch<BE>: TakeSlice,
        R: VecZnxDftToMut<BE>,
        A: CnvPVecLToRef<BE>,
        B: CnvPVecRToRef<BE>,
    {
        let mut res_ref: VecZnxDft<&mut [u8], BE> = res.to_mut();
        let a_ref: CnvPVecL<&[u8], BE> = a.to_ref();
        let b_ref: CnvPVecR<&[u8], BE> = b.to_ref();
        let bytes = ntt120_cnv_apply_dft_tmp_bytes(res_ref.size(), a_ref.size(), b_ref.size());
        let (tmp, _) = scratch.take_slice::<u8>(bytes);
        ntt120_cnv_apply_dft::<_, _, _, BE>(module, cnv_offset, &mut res_ref, res_col, &a_ref, a_col, &b_ref, b_col, tmp);
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
    fn cnv_pairwise_apply_dft_default<R, A, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        b: &B,
        i: usize,
        j: usize,
        scratch: &mut Scratch<BE>,
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
        Scratch<BE>: TakeSlice,
        R: VecZnxDftToMut<BE>,
        A: CnvPVecLToRef<BE>,
        B: CnvPVecRToRef<BE>,
    {
        let res_ref: VecZnxDft<&mut [u8], BE> = res.to_mut();
        let a_ref: CnvPVecL<&[u8], BE> = a.to_ref();
        let b_ref: CnvPVecR<&[u8], BE> = b.to_ref();
        let bytes = ntt120_cnv_pairwise_apply_dft_tmp_bytes(res_ref.size(), a_ref.size(), b_ref.size());
        let (tmp, _) = scratch.take_slice::<u8>(bytes);
        ntt120_cnv_pairwise_apply_dft::<R, A, B, BE>(module, cnv_offset, res, res_col, a, b, i, j, tmp);
    }

    fn cnv_prepare_self_tmp_bytes_default(module: &Module<BE>, _res_size: usize, _a_size: usize) -> usize
    where
        BE: Backend<ScalarPrep = Q120bScalar>,
    {
        ntt120_cnv_prepare_self_tmp_bytes(module.n())
    }

    fn cnv_prepare_self_default<L, R, A>(
        module: &Module<BE>,
        left: &mut L,
        right: &mut R,
        a: &A,
        mask: i64,
        scratch: &mut Scratch<BE>,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar> + NttFromZnx64 + NttDFTExecute<NttTable<Primes30>> + NttCFromB,
        Scratch<BE>: TakeSlice,
        L: CnvPVecLToMut<BE>,
        R: CnvPVecRToMut<BE>,
        A: VecZnxToRef + ZnxInfos,
    {
        let bytes = ntt120_cnv_prepare_self_tmp_bytes(module.n());
        let (tmp, _) = scratch.take_slice::<u8>(bytes);
        ntt120_cnv_prepare_self::<L, R, A, BE>(module, left, right, a, mask, tmp);
    }
}

impl<BE: Backend> NTT120ConvolutionDefaults<BE> for BE {}
