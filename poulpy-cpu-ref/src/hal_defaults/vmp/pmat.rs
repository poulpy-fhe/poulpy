//! Reference kernels whose prepared operand is the packed cold-prep [`VmpPMat`](poulpy_hal::layouts::VmpPMat).

use super::take_host_typed;
use crate::reference::{
    fft64::{
        module::FFTModuleHandle,
        reim::{ReimArith, ReimFFTExecute, ReimFFTTable},
        reim4::Reim4BlkMatVec,
        vmp as fft64_vmp,
    },
    ntt4x30::{
        NttCFromB, NttDFTExecute, NttExtract1BlkContiguous, NttFromZnx64, NttMulBbc1ColX2, NttMulBbc2ColsX2, ntt::NttTable,
        primes::Primes30, types::Q120bScalar, vec_znx_dft::NttModuleHandle, vmp as ntt4x30_vmp,
    },
};
use poulpy_hal::{
    api::{
        HostBufMut, ModuleN, ScratchArenaTakeBasic, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftAddAssign, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftZero, VecZnxIdftApplyTmpA, VmpTMatBytesOf,
    },
    layouts::{
        Backend, HostDataMut, HostDataRef, MatZnxBackendRef, Module, ScratchArena, VecZnxDftBackendMut, VecZnxDftBackendRef,
        VecZnxDftToBackendMut, VecZnxDftToBackendRef, VmpPMatBackendMut, VmpPMatBackendRef,
    },
};
use std::mem::size_of;

#[doc(hidden)]
pub trait FFT64VmpPMatDefault<BE: Backend<ZnxWord = i64>>: Backend {
    #[allow(clippy::too_many_arguments)]
    fn vmp_zero_default(module: &Module<BE>, res: &mut VmpPMatBackendMut<'_, BE>)
    where
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + Reim4BlkMatVec + ReimFFTExecute<ReimFFTTable<f64>, f64> + 'static,
        for<'x> BE::BufMut<'x>: HostDataMut + HostBufMut<'x>,
    {
        let _ = module;
        fft64_vmp::vmp_zero::<BE>(res);
    }

    #[allow(clippy::too_many_arguments)]
    fn vmp_prepare_pmat_tmp_bytes_default(
        module: &Module<BE>,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> usize {
        let _ = (rows, cols_in, cols_out, size);
        fft64_vmp::vmp_prepare_pmat_tmp_bytes(module.n())
    }
    #[allow(clippy::too_many_arguments)]
    fn vmp_prepare_pmat_default(
        module: &Module<BE>,
        res: &mut VmpPMatBackendMut<'_, BE>,
        a: &MatZnxBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: FFTModuleHandle<f64>,
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + Reim4BlkMatVec + ReimFFTExecute<ReimFFTTable<f64>, f64> + 'static,
        for<'x> BE::BufMut<'x>: HostDataMut + HostBufMut<'x>,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        let bytes = fft64_vmp::vmp_prepare_pmat_tmp_bytes(module.n());
        let (tmp, _) = take_host_typed::<BE, f64>(scratch.borrow(), bytes / size_of::<f64>());
        fft64_vmp::vmp_prepare_pmat::<BE>(module.get_fft_table(), res, a, tmp);
    }
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_dft_to_dft_tmp_bytes_default(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        let _ = (module, res_size, a_cols_out, a_size);
        fft64_vmp::vmp_apply_pmat_dft_to_dft_tmp_bytes(b_size, a_rows, a_cols_in)
    }
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_dft_to_dft_default(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        a: &VmpPMatBackendRef<'_, BE>,
        b: &VecZnxDftBackendRef<'_, BE>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + Reim4BlkMatVec,
        for<'x> BE::BufMut<'x>: HostDataMut + HostBufMut<'x>,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        let bytes = fft64_vmp::vmp_apply_pmat_dft_to_dft_tmp_bytes(b.size(), a.rows(), a.cols_in());
        let (tmp, _) = take_host_typed::<BE, f64>(scratch.borrow(), bytes / size_of::<f64>());
        fft64_vmp::vmp_apply_pmat_dft_to_dft::<true, BE>(res, a, b, limb_offset, tmp);
    }
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_dft_to_dft_accumulate_tmp_bytes_default(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        module.bytes_of_vec_znx_dft(a_cols_out, res_size)
            + Self::vmp_apply_pmat_dft_to_dft_tmp_bytes_default(module, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
    }
    /// Accumulates through a contiguous temporary rather than straight into the
    /// scattered output, which measured about twice as fast.
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_dft_to_dft_accumulate_default(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        a: &VmpPMatBackendRef<'_, BE>,
        b: &VecZnxDftBackendRef<'_, BE>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: VecZnxDftAddAssign<BE> + VecZnxDftZero<BE>,
        BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + Reim4BlkMatVec,
        for<'x> BE::BufMut<'x>: HostDataMut + HostBufMut<'x>,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        let cols_out = res.cols();
        let res_size = res.size();
        let (mut tmp, mut scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, cols_out, res_size);
        for col in 0..cols_out {
            module.vec_znx_dft_zero(&mut tmp.to_backend_mut(), col);
        }
        Self::vmp_apply_pmat_dft_to_dft_default(module, &mut tmp.to_backend_mut(), a, b, limb_offset, &mut scratch_1);
        let tmp_ref = tmp.to_backend_ref();
        for col in 0..cols_out {
            module.vec_znx_dft_add_assign(res, col, &tmp_ref, col);
        }
    }
}

impl<BE: Backend<ZnxWord = i64>> FFT64VmpPMatDefault<BE> for BE {}

#[doc(hidden)]
pub trait NTT4x30VmpPMatDefault<BE: Backend<ZnxWord = i64>>: Backend {
    #[allow(clippy::too_many_arguments)]
    fn vmp_zero_default(module: &Module<BE>, res: &mut VmpPMatBackendMut<'_, BE>)
    where
        for<'x> BE::BufMut<'x>: HostDataMut + HostBufMut<'x>,
    {
        let _ = module;
        ntt4x30_vmp::ntt4x30_vmp_zero::<BE>(res);
    }

    #[allow(clippy::too_many_arguments)]
    fn vmp_prepare_pmat_tmp_bytes_default(
        module: &Module<BE>,
        rows: usize,
        cols_in: usize,
        cols_out: usize,
        size: usize,
    ) -> usize {
        let _ = (rows, cols_in, cols_out, size);
        ntt4x30_vmp::ntt4x30_vmp_prepare_pmat_tmp_bytes(module.n())
    }
    #[allow(clippy::too_many_arguments)]
    fn vmp_prepare_pmat_default(
        module: &Module<BE>,
        res: &mut VmpPMatBackendMut<'_, BE>,
        a: &MatZnxBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttDFTExecute<NttTable<Primes30>> + NttFromZnx64 + NttCFromB,
        for<'x> BE::BufMut<'x>: HostDataMut + HostBufMut<'x>,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        let bytes = ntt4x30_vmp::ntt4x30_vmp_prepare_pmat_tmp_bytes(module.n());
        let (tmp, _) = take_host_typed::<BE, u64>(scratch.borrow(), bytes / size_of::<u64>());
        ntt4x30_vmp::ntt4x30_vmp_prepare_pmat::<BE>(module, res, a, tmp);
    }
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_dft_to_dft_tmp_bytes_default(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        let _ = (module, res_size, a_cols_out, a_size);
        ntt4x30_vmp::ntt4x30_vmp_apply_pmat_dft_to_dft_tmp_bytes(b_size, a_rows, a_cols_in)
    }
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_dft_to_dft_default(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        a: &VmpPMatBackendRef<'_, BE>,
        b: &VecZnxDftBackendRef<'_, BE>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttExtract1BlkContiguous + NttMulBbc1ColX2 + NttMulBbc2ColsX2,
        for<'x> BE::BufMut<'x>: HostDataMut + HostBufMut<'x>,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        let bytes = ntt4x30_vmp::ntt4x30_vmp_apply_pmat_dft_to_dft_tmp_bytes(b.size(), a.rows(), a.cols_in());
        let (tmp, _) = take_host_typed::<BE, u64>(scratch.borrow(), bytes / size_of::<u64>());
        ntt4x30_vmp::ntt4x30_vmp_apply_pmat_dft_to_dft::<BE>(module, res, a, b, limb_offset, tmp);
    }
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_dft_to_dft_accumulate_tmp_bytes_default(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        module.bytes_of_vec_znx_dft(a_cols_out, res_size)
            + Self::vmp_apply_pmat_dft_to_dft_tmp_bytes_default(module, res_size, a_rows, a_cols_in, a_cols_out, a_size, b_size)
    }
    /// Accumulates through a contiguous temporary rather than straight into the
    /// scattered output, which measured about twice as fast.
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_dft_to_dft_accumulate_default(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        a: &VmpPMatBackendRef<'_, BE>,
        b: &VecZnxDftBackendRef<'_, BE>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: NttModuleHandle + VecZnxDftAddAssign<BE> + VecZnxDftZero<BE>,
        BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttExtract1BlkContiguous + NttMulBbc1ColX2 + NttMulBbc2ColsX2,
        for<'x> BE::BufMut<'x>: HostDataMut + HostBufMut<'x>,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        let cols_out = res.cols();
        let res_size = res.size();
        let (mut tmp, mut scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, cols_out, res_size);
        for col in 0..cols_out {
            module.vec_znx_dft_zero(&mut tmp.to_backend_mut(), col);
        }
        Self::vmp_apply_pmat_dft_to_dft_default(module, &mut tmp.to_backend_mut(), a, b, limb_offset, &mut scratch_1);
        let tmp_ref = tmp.to_backend_ref();
        for col in 0..cols_out {
            module.vec_znx_dft_add_assign(res, col, &tmp_ref, col);
        }
    }
}

impl<BE: Backend<ZnxWord = i64>> NTT4x30VmpPMatDefault<BE> for BE {}
