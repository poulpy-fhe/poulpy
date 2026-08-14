#[allow(unused_imports)]
use std::mem::size_of;

use crate::{FFT64Neon, NTT4x30Neon};
use poulpy_cpu_ref::hal_defaults::{
    FFT64ConvolutionDefault, FFT64ModuleDefault, FFT64SvpPPolDefault, FFT64SvpTPolDefault, FFT64VecZnxBigDefault,
    FFT64VecZnxDftDefault, FFT64VmpPMatDefault, FFT64VmpTMatDefault, HalVecZnxDefault, NTT4x30ConvolutionDefault,
    NTT4x30ModuleDefault, NTT4x30SvpPPolDefault, NTT4x30SvpTPolDefault, NTT4x30VecZnxBigDefault, NTT4x30VecZnxDftDefault,
};
#[allow(unused_imports)]
use poulpy_hal::{
    api::{
        HostBufMut, ScratchArenaTakeBasic, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAddAssign,
        VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftZero, VecZnxIdftApplyTmpA, VmpApplyPMatDftToDft, VmpTMatBytesOf,
    },
    layouts::{
        Backend, MatZnxBackendRef, MatZnxInfos, Module, NoiseInfos, ScratchArena, SvpTPolToBackendMut, SvpTPolToBackendRef,
        VecZnxBackendMut, VecZnxBackendRef, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftBackendMut,
        VecZnxDftBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef, VecZnxInfos, VmpPMatBackendMut, VmpPMatBackendRef,
        VmpTMatBackendMut, VmpTMatBackendRef, VmpTMatToBackendMut, VmpTMatToBackendRef, ZnxInfos,
    },
    oep::{
        HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalSvpPPolImpl, HalSvpTPolImpl, HalVecZnxBigImpl, HalVecZnxDftImpl,
        HalVecZnxImpl, HalVmpImpl, HalVmpPMatImpl, HalVmpTMatImpl,
    },
};

#[cfg(target_arch = "aarch64")]
#[inline]
fn take_host_typed<'a, BE, T>(arena: ScratchArena<'a, BE>, len: usize) -> (&'a mut [T], ScratchArena<'a, BE>)
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
    poulpy_cpu_ref::hal_impl_vec_znx!();

    fn vec_znx_transpose_backend(module: &Module<Self>, res: &mut VecZnxBackendMut<'_, Self>, a: &VecZnxBackendRef<'_, Self>) {
        <Self as HalVecZnxDefault<Self>>::vec_znx_transpose_backend_default(module, res, a)
    }
}

unsafe impl HalModuleImpl<FFT64Neon> for FFT64Neon {
    poulpy_cpu_ref::hal_impl_module!(FFT64ModuleDefault);
}

unsafe impl HalVmpPMatImpl<FFT64Neon> for FFT64Neon {
    poulpy_cpu_ref::hal_impl_vmp_pmat!(FFT64VmpPMatDefault);
}

unsafe impl HalVmpTMatImpl<FFT64Neon> for FFT64Neon {
    poulpy_cpu_ref::hal_impl_vmp_tmat!(FFT64VmpTMatDefault);
}

unsafe impl HalVmpImpl<FFT64Neon> for FFT64Neon {
    poulpy_cpu_ref::hal_impl_vmp!();
}

unsafe impl HalConvolutionImpl<FFT64Neon> for FFT64Neon {
    poulpy_cpu_ref::hal_impl_convolution!(FFT64ConvolutionDefault);
}

unsafe impl HalVecZnxBigImpl<FFT64Neon> for FFT64Neon {
    poulpy_cpu_ref::hal_impl_vec_znx_big!(FFT64VecZnxBigDefault);
}

unsafe impl HalSvpPPolImpl<FFT64Neon> for FFT64Neon {
    poulpy_cpu_ref::hal_impl_svp_ppol!(FFT64SvpPPolDefault);
}

unsafe impl HalSvpTPolImpl<FFT64Neon> for FFT64Neon {
    poulpy_cpu_ref::hal_impl_svp_tpol!(FFT64SvpTPolDefault);
}

unsafe impl HalSvpImpl<FFT64Neon> for FFT64Neon {
    poulpy_cpu_ref::hal_impl_svp!();
}

unsafe impl HalVecZnxDftImpl<FFT64Neon> for FFT64Neon {
    poulpy_cpu_ref::hal_impl_vec_znx_dft!(FFT64VecZnxDftDefault);
}

unsafe impl HalVecZnxImpl<NTT4x30Neon> for NTT4x30Neon {
    poulpy_cpu_ref::hal_impl_vec_znx!();

    fn vec_znx_transpose_backend(module: &Module<Self>, res: &mut VecZnxBackendMut<'_, Self>, a: &VecZnxBackendRef<'_, Self>) {
        <Self as HalVecZnxDefault<Self>>::vec_znx_transpose_backend_default(module, res, a)
    }
}

unsafe impl HalModuleImpl<NTT4x30Neon> for NTT4x30Neon {
    poulpy_cpu_ref::hal_impl_module!(NTT4x30ModuleDefault);
}

#[cfg(target_arch = "aarch64")]
unsafe impl HalVmpPMatImpl<NTT4x30Neon> for NTT4x30Neon {
    fn vmp_prepare_pmat_tmp_bytes(module: &Module<Self>, _r: usize, _ci: usize, _co: usize, _s: usize) -> usize {
        crate::ntt4x30::vmp::vmp_prepare_pmat_tmp_bytes_neon(module.n())
    }

    fn vmp_prepare_pmat(
        module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &MatZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30::vmp::vmp_prepare_pmat_tmp_bytes_neon(module.n());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30::vmp::vmp_prepare_pmat_neon_pm(module, res, a, tmp);
    }

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_dft_to_dft_tmp_bytes(
        _module: &Module<Self>,
        _res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        _a_cols_out: usize,
        _a_size: usize,
        b_size: usize,
    ) -> usize {
        crate::ntt4x30::vmp::vmp_apply_tmp_bytes_neon(b_size, a_rows, a_cols_in)
    }

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_dft_to_dft(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VmpPMatBackendRef<'_, Self>,
        b: &VecZnxDftBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30::vmp::vmp_apply_tmp_bytes_neon(b.size(), a.rows(), a.cols_in());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30::vmp::vmp_apply_pmat_dft_to_dft_neon(module, res, b, a, limb_offset, tmp);
    }

    fn vmp_zero(_module: &Module<Self>, res: &mut VmpPMatBackendMut<'_, Self>) {
        poulpy_cpu_ref::reference::ntt4x30::vmp::ntt4x30_vmp_zero::<Self>(res);
    }

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_dft_to_dft_accumulate_tmp_bytes(
        _module: &Module<Self>,
        _res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        _a_cols_out: usize,
        _a_size: usize,
        b_size: usize,
    ) -> usize {
        crate::ntt4x30::vmp::vmp_apply_tmp_bytes_neon(b_size, a_rows, a_cols_in)
    }

    /// Uses this backend's fused accumulating kernel rather than the trait's
    /// temporary-plus-add default.
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_pmat_dft_to_dft_accumulate(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VmpPMatBackendRef<'_, Self>,
        b: &VecZnxDftBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30::vmp::vmp_apply_tmp_bytes_neon(b.size(), a.rows(), a.cols_in());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30::vmp::vmp_apply_pmat_dft_to_dft_accumulate_neon(module, res, b, a, limb_offset, tmp);
    }

    poulpy_cpu_ref::hal_impl_vmp_pmat!(kernels: skip);
}

#[cfg(target_arch = "aarch64")]
unsafe impl HalVmpTMatImpl<NTT4x30Neon> for NTT4x30Neon {
    /// This backend builds both tiers identically, so the hot-prep tier uses the
    /// same accelerated packing and kernels as the packed tier.
    fn vmp_prepare_tmat_tmp_bytes(module: &Module<Self>, _r: usize, _ci: usize, _co: usize, _s: usize) -> usize {
        crate::ntt4x30::vmp::vmp_prepare_pmat_tmp_bytes_neon(module.n())
    }

    fn vmp_prepare_tmat(
        module: &Module<Self>,
        res: &mut VmpTMatBackendMut<'_, Self>,
        a: &MatZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30::vmp::vmp_prepare_pmat_tmp_bytes_neon(module.n());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30::vmp::vmp_prepare_tmat_neon_pm(module, res, a, tmp);
    }

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_dft_to_dft_tmp_bytes(
        _module: &Module<Self>,
        _res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        _a_cols_out: usize,
        _a_size: usize,
        b_size: usize,
    ) -> usize {
        crate::ntt4x30::vmp::vmp_apply_tmp_bytes_neon(b_size, a_rows, a_cols_in)
    }

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_dft_to_dft(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VmpTMatBackendRef<'_, Self>,
        b: &VecZnxDftBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30::vmp::vmp_apply_tmp_bytes_neon(b.size(), a.rows(), a.cols_in());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30::vmp::vmp_apply_tmat_dft_to_dft_neon(module, res, b, a, limb_offset, tmp);
    }

    fn vmp_tmat_zero(_module: &Module<Self>, res: &mut VmpTMatBackendMut<'_, Self>) {
        poulpy_cpu_ref::reference::ntt4x30::vmp::ntt4x30_vmp_tmat_zero::<Self>(res);
    }

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_dft_to_dft_accumulate_tmp_bytes(
        _module: &Module<Self>,
        _res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        _a_cols_out: usize,
        _a_size: usize,
        b_size: usize,
    ) -> usize {
        crate::ntt4x30::vmp::vmp_apply_tmp_bytes_neon(b_size, a_rows, a_cols_in)
    }

    /// Uses this backend's fused accumulating kernel rather than the trait's
    /// temporary-plus-add default.
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_tmat_dft_to_dft_accumulate(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VmpTMatBackendRef<'_, Self>,
        b: &VecZnxDftBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30::vmp::vmp_apply_tmp_bytes_neon(b.size(), a.rows(), a.cols_in());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30::vmp::vmp_apply_tmat_dft_to_dft_accumulate_neon(module, res, b, a, limb_offset, tmp);
    }

    poulpy_cpu_ref::hal_impl_vmp_tmat!(kernels: skip);
}

#[cfg(target_arch = "aarch64")]
unsafe impl HalVmpImpl<NTT4x30Neon> for NTT4x30Neon {
    poulpy_cpu_ref::hal_impl_vmp!();
}

#[cfg(not(target_arch = "aarch64"))]
unsafe impl HalVmpPMatImpl<NTT4x30Neon> for NTT4x30Neon {
    poulpy_cpu_ref::hal_impl_vmp_pmat!(NTT4x30VmpPMatDefault);
}

#[cfg(not(target_arch = "aarch64"))]
unsafe impl HalVmpTMatImpl<NTT4x30Neon> for NTT4x30Neon {
    poulpy_cpu_ref::hal_impl_vmp_tmat!(NTT4x30VmpTMatDefault);
}

#[cfg(not(target_arch = "aarch64"))]
unsafe impl HalVmpImpl<NTT4x30Neon> for NTT4x30Neon {
    poulpy_cpu_ref::hal_impl_vmp!();
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

unsafe impl HalSvpPPolImpl<NTT4x30Neon> for NTT4x30Neon {
    poulpy_cpu_ref::hal_impl_svp_ppol!(NTT4x30SvpPPolDefault);
}

unsafe impl HalSvpTPolImpl<NTT4x30Neon> for NTT4x30Neon {
    poulpy_cpu_ref::hal_impl_svp_tpol!(NTT4x30SvpTPolDefault);
}

unsafe impl HalSvpImpl<NTT4x30Neon> for NTT4x30Neon {
    poulpy_cpu_ref::hal_impl_svp!();
}

unsafe impl HalVecZnxDftImpl<NTT4x30Neon> for NTT4x30Neon {
    poulpy_cpu_ref::hal_impl_vec_znx_dft!(NTT4x30VecZnxDftDefault);
}
