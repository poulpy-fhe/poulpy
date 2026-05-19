use std::mem::size_of;

use crate::{FFT64Avx, NTT120Avx};
use poulpy_cpu_ref::hal_defaults::{
    FFT64ConvolutionDefault, FFT64ModuleDefault, FFT64SvpDefault, FFT64VecZnxBigDefault, FFT64VecZnxDftDefault, FFT64VmpDefault,
    HalVecZnxDefault, NTT120ConvolutionDefault, NTT120ModuleDefault, NTT120SvpDefault, NTT120VecZnxBigDefault,
    NTT120VecZnxDftDefault, NTT120VmpDefault, VecZnxMatMulDefault,
};
use poulpy_hal::{
    api::{HostBufMut, ScratchArenaTakeBasic, VecZnxDftApply, VecZnxDftZero, VmpApplyDftToDft},
    layouts::{
        Backend, CoeffMatPMatBackendMut, CoeffMatPMatBackendRef, MatZnxBackendRef, Module, NoiseInfos, ScratchArena,
        VecZnxBackendMut, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftToBackendMut,
        VecZnxDftToBackendRef, VmpPMatBackendMut, VmpPMatBackendRef, ZnxInfos,
    },
    oep::{
        HalCoeffMatImpl, HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl,
        HalVecZnxMatMulImpl, HalVmpImpl,
    },
};

#[inline]
fn take_host_typed<'a, BE, T>(arena: ScratchArena<'a, BE>, len: usize) -> (&'a mut [T], ScratchArena<'a, BE>)
where
    BE: Backend + 'a,
    BE::BufMut<'a>: HostBufMut<'a>,
    T: Copy,
{
    debug_assert!(BE::SCRATCH_ALIGN.is_multiple_of(std::mem::align_of::<T>()));
    let (buf, arena) = arena.take_region(len * std::mem::size_of::<T>());
    let bytes: &'a mut [u8] = buf.into_bytes();
    let slice = unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, len) };
    (slice, arena)
}

unsafe impl HalVecZnxImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_vec_znx!();
}

unsafe impl HalModuleImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_module!(FFT64ModuleDefault);
}

unsafe impl HalVmpImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_vmp!(FFT64VmpDefault);
}

unsafe impl HalCoeffMatImpl<FFT64Avx> for FFT64Avx {
    fn coeff_mat_prepare_tmp_bytes(module: &Module<Self>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        crate::coeff_mat::coeff_mat_prepare_tmp_bytes::<Self>(module, rows, cols_in, cols_out, size)
    }

    fn coeff_mat_prepare(
        module: &Module<Self>,
        res: &mut CoeffMatPMatBackendMut<'_, Self>,
        matrix: &VecZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        crate::coeff_mat::coeff_mat_prepare::<Self>(module, res, matrix, scratch)
    }

    fn coeff_mat_apply_big_tmp_bytes(_module: &Module<Self>, rows_in: usize, rows_out: usize) -> usize {
        crate::coeff_mat::coeff_mat_apply_big_tmp_bytes(rows_in, rows_out)
    }

    fn coeff_mat_apply_big(
        _module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_limb: usize,
        pmat: &CoeffMatPMatBackendRef<'_, Self>,
        pmat_limb: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        a_limb: usize,
        rows_in: usize,
        rows_out: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        crate::coeff_mat::coeff_mat_apply_big_i64(res, res_limb, pmat, pmat_limb, a, a_col, a_limb, rows_in, rows_out)
    }
}

unsafe impl HalVecZnxMatMulImpl<FFT64Avx> for FFT64Avx {
    fn vec_znx_matmul_tmp_bytes(
        module: &Module<Self>,
        rows_in: usize,
        rows_out: usize,
        cols: usize,
        res_size: usize,
        u_size: usize,
        a_size: usize,
    ) -> usize {
        <Self as VecZnxMatMulDefault<Self>>::vec_znx_matmul_tmp_bytes_default(
            module, rows_in, rows_out, cols, res_size, u_size, a_size,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_matmul(
        module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        res_base2k: usize,
        u: &VecZnxBackendRef<'_, Self>,
        u_base2k: usize,
        u_bound_bits: u32,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        cols: usize,
        a_base2k: usize,
        rows_in: usize,
        rows_out: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        use poulpy_cpu_ref::hal_defaults::vec_znx_matmul::matmul_gemm;
        // FFT64 entries always fit i32 (FFT precision); cap the kernel width.
        let w = u_bound_bits.min(32);
        if w <= 16 {
            matmul_gemm::<Self, crate::gemm::AvxK16I64>(
                module, res, res_col, res_base2k, u, u_base2k, a, a_col, cols, a_base2k, rows_in, rows_out,
            );
        } else {
            matmul_gemm::<Self, crate::gemm::AvxK32I64>(
                module, res, res_col, res_base2k, u, u_base2k, a, a_col, cols, a_base2k, rows_in, rows_out,
            );
        }
    }
}

unsafe impl HalConvolutionImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_convolution!(FFT64ConvolutionDefault);
}

unsafe impl HalVecZnxBigImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_vec_znx_big!(FFT64VecZnxBigDefault);
}

unsafe impl HalSvpImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_svp!(FFT64SvpDefault);
}

unsafe impl HalVecZnxDftImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_vec_znx_dft!(FFT64VecZnxDftDefault);
}

unsafe impl HalVecZnxImpl<NTT120Avx> for NTT120Avx {
    poulpy_cpu_ref::hal_impl_vec_znx!();
}

unsafe impl HalModuleImpl<NTT120Avx> for NTT120Avx {
    poulpy_cpu_ref::hal_impl_module!(NTT120ModuleDefault);
}

unsafe impl HalVmpImpl<NTT120Avx> for NTT120Avx {
    fn vmp_apply_dft_tmp_bytes(
        module: &Module<Self>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        let a_dft_size = a_size.min(b_rows);
        <Self as Backend>::bytes_of_vec_znx_dft(module.n(), b_cols_in, a_dft_size)
            + Self::vmp_apply_dft_to_dft_tmp_bytes(module, res_size, a_dft_size, b_rows, b_cols_in, b_cols_out, b_size)
    }

    fn vmp_apply_dft<R>(
        module: &Module<Self>,
        res: &mut R,
        a: &VecZnxBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: VecZnxDftToBackendMut<Self>,
    {
        let a_cols = <VecZnxBackendRef<'_, Self> as ZnxInfos>::cols(a);
        let a_size = <VecZnxBackendRef<'_, Self> as ZnxInfos>::size(a);
        let b_rows = <VmpPMatBackendRef<'_, Self> as ZnxInfos>::rows(b);
        let cols_to_copy = a_cols.min(b.cols_in());
        let a_start_col = a_cols - cols_to_copy;
        let a_dft_size = a_size.min(b_rows);
        let offset = b.cols_in() - cols_to_copy;

        scratch.consume(|scratch| {
            let (mut a_dft, mut scratch) = scratch.take_vec_znx_dft_scratch(module, b.cols_in(), a_dft_size);
            for j in 0..offset {
                module.vec_znx_dft_zero(&mut a_dft, j);
            }
            for j in 0..cols_to_copy {
                module.vec_znx_dft_apply(1, 0, &mut a_dft, offset + j, a, a_start_col + j);
            }
            let mut res_ref = res.to_backend_mut();
            module.vmp_apply_dft_to_dft(&mut res_ref, &a_dft.to_backend_ref(), b, 0, &mut scratch);
            ((), scratch)
        })
    }

    fn vmp_prepare_tmp_bytes(module: &Module<Self>, _rows: usize, _cols_in: usize, _cols_out: usize, _size: usize) -> usize {
        crate::ntt120::vmp::vmp_prepare_tmp_bytes_avx(module.n())
    }

    fn vmp_prepare(
        module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &MatZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt120::vmp::vmp_prepare_tmp_bytes_avx(module.n());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt120::vmp::vmp_prepare_avx_pm(module, res, a, tmp);
    }

    fn vmp_apply_dft_to_dft_tmp_bytes(
        _module: &Module<Self>,
        _res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize {
        crate::ntt120::vmp::vmp_apply_tmp_bytes_avx(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt120::vmp::vmp_apply_tmp_bytes_avx(a.size(), b.rows(), b.cols_in());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt120::vmp::vmp_apply_dft_to_dft_avx(module, res, a, b, limb_offset, tmp);
    }

    fn vmp_apply_dft_to_dft_accumulate_tmp_bytes(
        _module: &Module<Self>,
        _res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize {
        crate::ntt120::vmp::vmp_apply_tmp_bytes_avx(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft_accumulate(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt120::vmp::vmp_apply_tmp_bytes_avx(a.size(), b.rows(), b.cols_in());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt120::vmp::vmp_apply_dft_to_dft_accumulate_avx(module, res, a, b, limb_offset, tmp);
    }

    fn vmp_zero(module: &Module<Self>, res: &mut VmpPMatBackendMut<'_, Self>) {
        <Self as NTT120VmpDefault<Self>>::vmp_zero_default(module, res)
    }
}

unsafe impl HalConvolutionImpl<NTT120Avx> for NTT120Avx {
    fn cnv_prepare_left_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
        <Self as NTT120ConvolutionDefault<Self>>::cnv_prepare_left_tmp_bytes_default(module, res_size, a_size)
    }

    fn cnv_prepare_left(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::CnvPVecLBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        mask: i64,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow();
        <Self as NTT120ConvolutionDefault<Self>>::cnv_prepare_left_default(module, res, a, mask, &mut scratch);
    }

    fn cnv_prepare_right_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
        <Self as NTT120ConvolutionDefault<Self>>::cnv_prepare_right_tmp_bytes_default(module, res_size, a_size)
    }

    fn cnv_prepare_right(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::CnvPVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        mask: i64,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow();
        <Self as NTT120ConvolutionDefault<Self>>::cnv_prepare_right_default(module, res, a, mask, &mut scratch);
    }

    fn cnv_apply_dft_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        _res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        crate::ntt120::convolution::cnv_apply_dft_avx_tmp_bytes(a_size, b_size)
    }

    fn cnv_by_const_apply_tmp_bytes(
        module: &Module<Self>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        <Self as NTT120ConvolutionDefault<Self>>::cnv_by_const_apply_tmp_bytes_default(
            module, cnv_offset, res_size, a_size, b_size,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_by_const_apply(
        module: &Module<Self>,
        cnv_offset: usize,
        mut res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
        b_coeff: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow();
        <Self as NTT120ConvolutionDefault<Self>>::cnv_by_const_apply_default(
            module,
            cnv_offset,
            &mut res,
            res_col,
            a,
            a_col,
            b,
            b_col,
            b_coeff,
            &mut scratch,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
        a_col: usize,
        b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt120::convolution::cnv_apply_dft_avx_tmp_bytes(a.size(), b.size());
        let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        unsafe {
            crate::ntt120::convolution::cnv_apply_dft_avx(module, res, cnv_offset, res_col, a, a_col, b, b_col, tmp);
        }
    }

    fn cnv_pairwise_apply_dft_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        crate::ntt120::convolution::cnv_pairwise_apply_dft_avx_tmp_bytes(res_size, a_size, b_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_pairwise_apply_dft(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
        b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
        i: usize,
        j: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt120::convolution::cnv_pairwise_apply_dft_avx_tmp_bytes(res.size(), a.size(), b.size());
        let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        unsafe {
            crate::ntt120::convolution::cnv_pairwise_apply_dft_avx(module, res, cnv_offset, res_col, a, b, i, j, tmp);
        }
    }

    fn cnv_prepare_self_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
        <Self as NTT120ConvolutionDefault<Self>>::cnv_prepare_self_tmp_bytes_default(module, res_size, a_size)
    }

    fn cnv_prepare_self(
        module: &Module<Self>,
        left: &mut poulpy_hal::layouts::CnvPVecLBackendMut<'_, Self>,
        right: &mut poulpy_hal::layouts::CnvPVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        mask: i64,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow();
        <Self as NTT120ConvolutionDefault<Self>>::cnv_prepare_self_default(module, left, right, a, mask, &mut scratch);
    }
}

unsafe impl HalVecZnxBigImpl<NTT120Avx> for NTT120Avx {
    poulpy_cpu_ref::hal_impl_vec_znx_big!(NTT120VecZnxBigDefault);
}

unsafe impl HalCoeffMatImpl<NTT120Avx> for NTT120Avx {
    fn coeff_mat_prepare_tmp_bytes(module: &Module<Self>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        crate::coeff_mat::coeff_mat_prepare_tmp_bytes::<Self>(module, rows, cols_in, cols_out, size)
    }

    fn coeff_mat_prepare(
        module: &Module<Self>,
        res: &mut CoeffMatPMatBackendMut<'_, Self>,
        matrix: &VecZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        crate::coeff_mat::coeff_mat_prepare::<Self>(module, res, matrix, scratch)
    }

    fn coeff_mat_apply_big_tmp_bytes(_module: &Module<Self>, rows_in: usize, rows_out: usize) -> usize {
        crate::coeff_mat::coeff_mat_apply_big_tmp_bytes(rows_in, rows_out)
    }

    fn coeff_mat_apply_big(
        _module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_limb: usize,
        pmat: &CoeffMatPMatBackendRef<'_, Self>,
        pmat_limb: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        a_limb: usize,
        rows_in: usize,
        rows_out: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        crate::coeff_mat::coeff_mat_apply_big_i128(res, res_limb, pmat, pmat_limb, a, a_col, a_limb, rows_in, rows_out)
    }
}

unsafe impl HalVecZnxMatMulImpl<NTT120Avx> for NTT120Avx {
    fn vec_znx_matmul_tmp_bytes(
        module: &Module<Self>,
        rows_in: usize,
        rows_out: usize,
        cols: usize,
        res_size: usize,
        u_size: usize,
        a_size: usize,
    ) -> usize {
        <Self as VecZnxMatMulDefault<Self>>::vec_znx_matmul_tmp_bytes_default(
            module, rows_in, rows_out, cols, res_size, u_size, a_size,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_matmul(
        module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        res_base2k: usize,
        u: &VecZnxBackendRef<'_, Self>,
        u_base2k: usize,
        u_bound_bits: u32,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        cols: usize,
        a_base2k: usize,
        rows_in: usize,
        rows_out: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        use poulpy_cpu_ref::hal_defaults::vec_znx_matmul::matmul_gemm;
        let w = u_bound_bits.min(64);
        if w <= 16 {
            matmul_gemm::<Self, crate::gemm::AvxK16I128>(
                module, res, res_col, res_base2k, u, u_base2k, a, a_col, cols, a_base2k, rows_in, rows_out,
            );
        } else if w <= 32 {
            matmul_gemm::<Self, crate::gemm::AvxK32I128S>(
                module, res, res_col, res_base2k, u, u_base2k, a, a_col, cols, a_base2k, rows_in, rows_out,
            );
        } else {
            matmul_gemm::<Self, crate::gemm::AvxK32I128D>(
                module, res, res_col, res_base2k, u, u_base2k, a, a_col, cols, a_base2k, rows_in, rows_out,
            );
        }
    }
}

unsafe impl HalSvpImpl<NTT120Avx> for NTT120Avx {
    poulpy_cpu_ref::hal_impl_svp!(NTT120SvpDefault);
}

unsafe impl HalVecZnxDftImpl<NTT120Avx> for NTT120Avx {
    poulpy_cpu_ref::hal_impl_vec_znx_dft!(NTT120VecZnxDftDefault);
}
