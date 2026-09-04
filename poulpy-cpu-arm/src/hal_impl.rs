#[allow(unused_imports)]
use std::mem::size_of;

use crate::{FFT64Neon, NTT4x30Neon};
use poulpy_cpu_ref::hal_defaults::{
    FFT64ConvolutionDefault, FFT64ModuleDefault, FFT64SvpDefault, FFT64VecZnxBigDefault, FFT64VecZnxDftDefault, FFT64VmpDefault,
    HalVecZnxDefault, NTT4x30ConvolutionDefault, NTT4x30ModuleDefault, NTT4x30SvpDefault, NTT4x30VecZnxBigDefault,
    NTT4x30VecZnxDftDefault, NTT4x30VmpDefault,
};
#[allow(unused_imports)]
use poulpy_hal::{
    api::{HostBufMut, ScratchArenaTakeBasic, VecZnxDftApply, VecZnxDftZero, VmpApplyDftToDft},
    layouts::{
        Backend, MatZnxBackendRef, MatZnxInfos, Module, NoiseInfos, ScratchArena, VecZnxBackendMut, VecZnxBackendRef,
        VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef, VecZnxInfos, VmpPMatBackendMut,
        VmpPMatBackendRef, ZnxInfos,
    },
    oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
};

#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn take_host_typed<'a, BE, T>(arena: ScratchArena<'a, BE>, len: usize) -> (&'a mut [T], ScratchArena<'a, BE>)
where
    BE: Backend<ZnxWord = i64> + 'a,
    BE::BufMut<'a>: HostBufMut<'a>,
    T: Copy,
{
    assert!(BE::SCRATCH_ALIGN.is_multiple_of(std::mem::align_of::<T>()));
    let byte_len = len
        .checked_mul(std::mem::size_of::<T>())
        .expect("typed scratch byte size overflows usize");
    let (buf, arena) = arena.take_region(byte_len);
    let bytes: &'a mut [u8] = buf.into_bytes();
    assert!((bytes.as_mut_ptr() as usize).is_multiple_of(std::mem::align_of::<T>()));
    let slice = unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, len) };
    (slice, arena)
}

unsafe impl HalVecZnxImpl<FFT64Neon> for FFT64Neon {
    poulpy_cpu_ref::hal_impl_vec_znx_without_normalize!();
    poulpy_cpu_ref::hal_impl_vec_znx_normalize!();
    poulpy_cpu_ref::hal_impl_vec_znx_canonicalize!();

    fn vec_znx_transpose_backend(module: &Module<Self>, res: &mut VecZnxBackendMut<'_, Self>, a: &VecZnxBackendRef<'_, Self>) {
        <Self as HalVecZnxDefault<Self>>::vec_znx_transpose_backend_default(module, res, a)
    }
}

unsafe impl HalModuleImpl<FFT64Neon> for FFT64Neon {
    poulpy_cpu_ref::hal_impl_module!(FFT64ModuleDefault);
}

unsafe impl HalVmpImpl<FFT64Neon> for FFT64Neon {
    poulpy_cpu_ref::hal_impl_vmp!(FFT64VmpDefault);
}

unsafe impl HalConvolutionImpl<FFT64Neon> for FFT64Neon {
    poulpy_cpu_ref::hal_impl_convolution!(FFT64ConvolutionDefault);
}

unsafe impl HalVecZnxBigImpl<FFT64Neon> for FFT64Neon {
    poulpy_cpu_ref::hal_impl_vec_znx_big!(FFT64VecZnxBigDefault);
}

unsafe impl HalSvpImpl<FFT64Neon> for FFT64Neon {
    poulpy_cpu_ref::hal_impl_svp!(FFT64SvpDefault);
}

unsafe impl HalVecZnxDftImpl<FFT64Neon> for FFT64Neon {
    poulpy_cpu_ref::hal_impl_vec_znx_dft!(FFT64VecZnxDftDefault);
}

unsafe impl HalVecZnxImpl<NTT4x30Neon> for NTT4x30Neon {
    poulpy_cpu_ref::hal_impl_vec_znx_without_normalize!();
    poulpy_cpu_ref::hal_impl_vec_znx_normalize!();
    poulpy_cpu_ref::hal_impl_vec_znx_canonicalize!();

    fn vec_znx_transpose_backend(module: &Module<Self>, res: &mut VecZnxBackendMut<'_, Self>, a: &VecZnxBackendRef<'_, Self>) {
        <Self as HalVecZnxDefault<Self>>::vec_znx_transpose_backend_default(module, res, a)
    }
}

unsafe impl HalModuleImpl<NTT4x30Neon> for NTT4x30Neon {
    poulpy_cpu_ref::hal_impl_module!(NTT4x30ModuleDefault);
}

#[cfg(target_arch = "aarch64")]
unsafe impl HalVmpImpl<NTT4x30Neon> for NTT4x30Neon {
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
        let a_cols = <VecZnxBackendRef<'_, Self> as VecZnxInfos>::cols(a);
        let a_size = <VecZnxBackendRef<'_, Self> as ZnxInfos>::size(a);
        let b_rows = <VmpPMatBackendRef<'_, Self> as MatZnxInfos>::rows(b);
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
        crate::ntt4x30::vmp::vmp_prepare_tmp_bytes_neon(module.n())
    }

    fn vmp_prepare(
        module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &MatZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30::vmp::vmp_prepare_tmp_bytes_neon(module.n());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30::vmp::vmp_prepare_neon_pm(module, res, a, tmp);
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
        crate::ntt4x30::vmp::vmp_apply_tmp_bytes_neon(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30::vmp::vmp_apply_tmp_bytes_neon(a.size(), b.rows(), b.cols_in());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30::vmp::vmp_apply_dft_to_dft_neon::<poulpy_hal::execution::SerialTaskExecutor>(
            module,
            res,
            a,
            b,
            limb_offset,
            tmp,
        );
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
        crate::ntt4x30::vmp::vmp_apply_tmp_bytes_neon(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft_accumulate(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30::vmp::vmp_apply_tmp_bytes_neon(a.size(), b.rows(), b.cols_in());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30::vmp::vmp_apply_dft_to_dft_accumulate_neon::<poulpy_hal::execution::SerialTaskExecutor>(
            module,
            res,
            a,
            b,
            limb_offset,
            tmp,
        );
    }

    fn vmp_extract_selected_rows(
        _module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &VmpPMatBackendRef<'_, Self>,
        first_row: usize,
        row_step: usize,
    ) {
        crate::ntt4x30::vmp::vmp_extract_selected_rows_neon_pm(res, a, first_row, row_step)
    }

    fn vmp_zero(module: &Module<Self>, res: &mut VmpPMatBackendMut<'_, Self>) {
        <Self as NTT4x30VmpDefault<Self>>::vmp_zero_default(module, res)
    }
}

#[cfg(not(target_arch = "aarch64"))]
unsafe impl HalVmpImpl<NTT4x30Neon> for NTT4x30Neon {
    poulpy_cpu_ref::hal_impl_vmp!(NTT4x30VmpDefault);
}

#[cfg(target_arch = "aarch64")]
unsafe impl HalConvolutionImpl<NTT4x30Neon> for NTT4x30Neon {
    fn cnv_prepare_left_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_prepare_left_tmp_bytes_default(module, res_size, a_size)
    }

    fn cnv_prepare_left(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::CnvPVecLBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        mask: i64,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow();
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_prepare_left_default(module, res, a, mask, &mut scratch);
    }

    fn cnv_prepare_right_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_prepare_right_tmp_bytes_default(module, res_size, a_size)
    }

    fn cnv_prepare_right(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::CnvPVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        mask: i64,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow();
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_prepare_right_default(module, res, a, mask, &mut scratch);
    }

    fn cnv_apply_dft_tmp_bytes(module: &Module<Self>, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize {
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_apply_dft_tmp_bytes_default(module, cnv_offset, res_size, a_size, b_size)
    }

    fn cnv_by_const_apply_tmp_bytes(
        module: &Module<Self>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_by_const_apply_tmp_bytes_default(
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
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_by_const_apply_default(
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
    fn cnv_by_const_apply_add(
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
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_by_const_apply_add_default(
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
        mut res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
        a_col: usize,
        b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow();
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_apply_dft_default(
            module,
            cnv_offset,
            &mut res,
            res_col,
            a,
            a_col,
            b,
            b_col,
            &mut scratch,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft_accumulate(
        module: &Module<Self>,
        cnv_offset: usize,
        mut res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
        a_col: usize,
        b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow();
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_apply_dft_accumulate_default(
            module,
            cnv_offset,
            &mut res,
            res_col,
            a,
            a_col,
            b,
            b_col,
            &mut scratch,
        );
    }

    fn cnv_pairwise_apply_dft_tmp_bytes(
        module: &Module<Self>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_pairwise_apply_dft_tmp_bytes_default(
            module, cnv_offset, res_size, a_size, b_size,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_pairwise_apply_dft(
        module: &Module<Self>,
        cnv_offset: usize,
        mut res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
        b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
        i: usize,
        j: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow();
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_pairwise_apply_dft_default(
            module,
            cnv_offset,
            &mut res,
            res_col,
            a,
            b,
            i,
            j,
            &mut scratch,
        );
    }

    fn cnv_prepare_self_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_prepare_self_tmp_bytes_default(module, res_size, a_size)
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
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_prepare_self_default(module, left, right, a, mask, &mut scratch);
    }
}

#[cfg(not(target_arch = "aarch64"))]
unsafe impl HalConvolutionImpl<NTT4x30Neon> for NTT4x30Neon {
    poulpy_cpu_ref::hal_impl_convolution!(NTT4x30ConvolutionDefault);
}

unsafe impl HalVecZnxBigImpl<NTT4x30Neon> for NTT4x30Neon {
    poulpy_cpu_ref::hal_impl_vec_znx_big!(NTT4x30VecZnxBigDefault);
}

unsafe impl HalSvpImpl<NTT4x30Neon> for NTT4x30Neon {
    poulpy_cpu_ref::hal_impl_svp!(NTT4x30SvpDefault);
}

unsafe impl HalVecZnxDftImpl<NTT4x30Neon> for NTT4x30Neon {
    poulpy_cpu_ref::hal_impl_vec_znx_dft!(NTT4x30VecZnxDftDefault);
}
