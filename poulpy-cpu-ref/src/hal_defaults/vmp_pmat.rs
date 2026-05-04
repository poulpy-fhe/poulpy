//! Backend extension points for vector-matrix product (VMP) operations
//! on [`poulpy_hal::layouts::VmpPMat`].

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
    api::{TakeSlice, VecZnxDftAddAssign, VecZnxDftBytesOf},
    layouts::{
        Backend, MatZnxToRef, Module, Scratch, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VmpPMat, VmpPMatToMut, VmpPMatToRef,
        ZnxInfos, ZnxZero,
    },
};

#[doc(hidden)]
pub trait FFT64VmpDefaults<BE: Backend>: Backend {
    fn vmp_prepare_tmp_bytes_default(module: &Module<BE>, _rows: usize, _cols_in: usize, _cols_out: usize, _size: usize) -> usize
    where
        BE: Backend<ScalarPrep = f64>,
    {
        fft64_vmp_prepare_tmp_bytes(module.n())
    }

    fn vmp_prepare_default<R, A>(module: &Module<BE>, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        Module<BE>: FFTModuleHandle<f64>,
        BE: Backend<ScalarPrep = f64> + ReimArith + Reim4BlkMatVec + ReimFFTExecute<ReimFFTTable<f64>, f64>,
        Scratch<BE>: TakeSlice,
        R: VmpPMatToMut<BE>,
        A: MatZnxToRef,
    {
        let bytes = fft64_vmp_prepare_tmp_bytes(module.n());
        let (tmp, _) = scratch.take_slice::<f64>(bytes / size_of::<f64>());
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

    fn vmp_apply_dft_to_dft_default<R, A, C>(
        _module: &Module<BE>,
        res: &mut R,
        a: &A,
        b: &C,
        limb_offset: usize,
        scratch: &mut Scratch<BE>,
    ) where
        BE: Backend<ScalarPrep = f64> + ReimArith + Reim4BlkMatVec,
        Scratch<BE>: TakeSlice,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
        C: VmpPMatToRef<BE>,
    {
        let mut res_ref: VecZnxDft<&mut [u8], BE> = res.to_mut();
        let a_ref: VecZnxDft<&[u8], BE> = a.to_ref();
        let b_ref: VmpPMat<&[u8], BE> = b.to_ref();

        let bytes = fft64_vmp_apply_dft_to_dft_tmp_bytes(a_ref.size(), b_ref.rows(), b_ref.cols_in());
        let (tmp, _) = scratch.take_slice::<f64>(bytes / size_of::<f64>());
        fft64_vmp_apply_dft_to_dft(&mut res_ref, &a_ref, &b_ref, limb_offset, tmp);
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
        BE: Backend<ScalarPrep = f64>,
        Module<BE>: VecZnxDftBytesOf,
    {
        let _ = b_size;
        module.bytes_of_vec_znx_dft(b_cols_out, res_size) + fft64_vmp_apply_dft_to_dft_tmp_bytes(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft_accumulate_default<R, A, C>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        b: &C,
        limb_offset: usize,
        scratch: &mut Scratch<BE>,
    ) where
        BE: Backend<ScalarPrep = f64> + ReimArith + Reim4BlkMatVec,
        Module<BE>: VecZnxDftBytesOf + VecZnxDftAddAssign<BE>,
        Scratch<BE>: TakeSlice,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
        C: VmpPMatToRef<BE>,
    {
        let res_ref: VecZnxDft<&mut [u8], BE> = res.to_mut();
        let cols = res_ref.cols();
        let size = res_ref.size();
        let bytes = module.bytes_of_vec_znx_dft(cols, size);
        let (tmp_bytes, scratch_1) = scratch.take_slice::<u8>(bytes);
        let mut tmp = VecZnxDft::<&mut [u8], BE>::from_data(tmp_bytes, module.n(), cols, size);
        tmp.zero();
        Self::vmp_apply_dft_to_dft_default(module, &mut tmp, a, b, limb_offset, scratch_1);
        for col in 0..cols {
            module.vec_znx_dft_add_assign(res, col, &tmp, col);
        }
    }

    fn vmp_zero_default<R>(_module: &Module<BE>, res: &mut R)
    where
        R: VmpPMatToMut<BE>,
    {
        fft64_vmp_zero(res);
    }
}

impl<BE: Backend> FFT64VmpDefaults<BE> for BE {}

#[doc(hidden)]
pub trait NTT120VmpDefaults<BE: Backend>: Backend {
    fn vmp_prepare_tmp_bytes_default(module: &Module<BE>, _rows: usize, _cols_in: usize, _cols_out: usize, _size: usize) -> usize
    where
        BE: Backend<ScalarPrep = Q120bScalar>,
    {
        ntt120_vmp_prepare_tmp_bytes(module.n())
    }

    fn vmp_prepare_default<R, A>(module: &Module<BE>, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        Module<BE>: NttModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar> + NttDFTExecute<NttTable<Primes30>> + NttFromZnx64 + NttCFromB,
        Scratch<BE>: TakeSlice,
        R: VmpPMatToMut<BE>,
        A: MatZnxToRef,
    {
        let bytes = ntt120_vmp_prepare_tmp_bytes(module.n());
        let (tmp, _) = scratch.take_slice::<u64>(bytes / size_of::<u64>());
        ntt120_vmp_prepare::<R, A, BE>(module, res, a, tmp);
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

    fn vmp_apply_dft_to_dft_default<R, A, C>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        b: &C,
        limb_offset: usize,
        scratch: &mut Scratch<BE>,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar> + NttExtract1BlkContiguous + NttMulBbc1ColX2 + NttMulBbc2ColsX2,
        Scratch<BE>: TakeSlice,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
        C: VmpPMatToRef<BE>,
    {
        let mut res_ref: VecZnxDft<&mut [u8], BE> = res.to_mut();
        let a_ref: VecZnxDft<&[u8], BE> = a.to_ref();
        let b_ref: VmpPMat<&[u8], BE> = b.to_ref();

        let bytes = ntt120_vmp_apply_dft_to_dft_tmp_bytes(a_ref.size(), b_ref.rows(), b_ref.cols_in());
        let (tmp, _) = scratch.take_slice::<u64>(bytes / size_of::<u64>());
        ntt120_vmp_apply_dft_to_dft::<_, _, _, BE>(module, &mut res_ref, &a_ref, &b_ref, limb_offset, tmp);
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

    fn vmp_apply_dft_to_dft_accumulate_default<R, A, C>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        b: &C,
        limb_offset: usize,
        scratch: &mut Scratch<BE>,
    ) where
        Module<BE>: NttModuleHandle + VecZnxDftBytesOf + VecZnxDftAddAssign<BE>,
        BE: Backend<ScalarPrep = Q120bScalar> + NttExtract1BlkContiguous + NttMulBbc1ColX2 + NttMulBbc2ColsX2,
        Scratch<BE>: TakeSlice,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
        C: VmpPMatToRef<BE>,
    {
        let res_ref: VecZnxDft<&mut [u8], BE> = res.to_mut();
        let cols = res_ref.cols();
        let size = res_ref.size();
        let bytes = module.bytes_of_vec_znx_dft(cols, size);
        let (tmp_bytes, scratch_1) = scratch.take_slice::<u8>(bytes);
        let mut tmp = VecZnxDft::<&mut [u8], BE>::from_data(tmp_bytes, module.n(), cols, size);
        tmp.zero();
        Self::vmp_apply_dft_to_dft_default(module, &mut tmp, a, b, limb_offset, scratch_1);
        for col in 0..cols {
            module.vec_znx_dft_add_assign(res, col, &tmp, col);
        }
    }

    fn vmp_zero_default<R>(_module: &Module<BE>, res: &mut R)
    where
        R: VmpPMatToMut<BE>,
    {
        ntt120_vmp_zero::<R, BE>(res);
    }
}

impl<BE: Backend> NTT120VmpDefaults<BE> for BE {}
