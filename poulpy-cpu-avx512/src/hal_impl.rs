use std::mem::size_of;

use crate::{FFT64Avx512, NTT4x30Avx512};
use poulpy_cpu_ref::hal_defaults::{
    FFT64ConvolutionDefault, FFT64ModuleDefault, FFT64SvpPPolDefault, FFT64SvpTPolDefault, FFT64VecZnxBigDefault,
    FFT64VecZnxDftDefault, FFT64VmpPMatDefault, FFT64VmpTMatDefault, HalVecZnxDefault, NTT4x30ConvolutionDefault,
    NTT4x30ModuleDefault, NTT4x30SvpPPolDefault, NTT4x30SvpTPolDefault, NTT4x30VecZnxBigDefault, NTT4x30VecZnxDftDefault,
};
use poulpy_hal::{
    api::{
        HostBufMut, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAddAssign, VecZnxDftBytesOf,
        VecZnxDftZero, VecZnxIdftApplyTmpA, VmpTMatBytesOf,
    },
    layouts::{
        Backend, MatZnxBackendRef, Module, NoiseInfos, ScratchArena, SvpTPolToBackendMut, SvpTPolToBackendRef, VecZnxBackendMut,
        VecZnxBackendRef, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef,
        VecZnxDftToBackendMut, VecZnxDftToBackendRef, VmpPMatBackendMut, VmpPMatBackendRef, VmpTMatBackendMut, VmpTMatBackendRef,
        VmpTMatToBackendMut, VmpTMatToBackendRef,
    },
    oep::{
        HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalSvpPPolImpl, HalSvpTPolImpl, HalVecZnxBigImpl, HalVecZnxDftImpl,
        HalVecZnxImpl, HalVmpImpl, HalVmpPMatImpl, HalVmpTMatImpl,
    },
};

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

unsafe impl HalVecZnxImpl<FFT64Avx512> for FFT64Avx512 {
    poulpy_cpu_ref::hal_impl_vec_znx!();

    // TODO: add an AVX-512-accelerated tiled transpose kernel; falls back to
    // the reference impl for now.
    fn vec_znx_transpose_backend(module: &Module<Self>, res: &mut VecZnxBackendMut<'_, Self>, a: &VecZnxBackendRef<'_, Self>) {
        <Self as HalVecZnxDefault<Self>>::vec_znx_transpose_backend_default(module, res, a)
    }
}

unsafe impl HalModuleImpl<FFT64Avx512> for FFT64Avx512 {
    poulpy_cpu_ref::hal_impl_module!(FFT64ModuleDefault);
}

unsafe impl HalVmpPMatImpl<FFT64Avx512> for FFT64Avx512 {
    poulpy_cpu_ref::hal_impl_vmp_pmat!(FFT64VmpPMatDefault);
}

unsafe impl HalVmpTMatImpl<FFT64Avx512> for FFT64Avx512 {
    poulpy_cpu_ref::hal_impl_vmp_tmat!(FFT64VmpTMatDefault);
}

unsafe impl HalVmpImpl<FFT64Avx512> for FFT64Avx512 {
    poulpy_cpu_ref::hal_impl_vmp!();
}

unsafe impl HalConvolutionImpl<FFT64Avx512> for FFT64Avx512 {
    poulpy_cpu_ref::cnv_impl_prepares_pvec!(FFT64ConvolutionDefault);
    poulpy_cpu_ref::cnv_impl_prepares_tvec!(FFT64ConvolutionDefault);
    poulpy_cpu_ref::cnv_impl_by_const!(FFT64ConvolutionDefault);
    poulpy_cpu_ref::cnv_impl_apply_pvec!(FFT64ConvolutionDefault);
    poulpy_cpu_ref::cnv_impl_apply_tvec!(FFT64ConvolutionDefault);
}

unsafe impl HalVecZnxBigImpl<FFT64Avx512> for FFT64Avx512 {
    poulpy_cpu_ref::hal_impl_vec_znx_big!(FFT64VecZnxBigDefault);
}

unsafe impl HalSvpPPolImpl<FFT64Avx512> for FFT64Avx512 {
    poulpy_cpu_ref::hal_impl_svp_ppol!(FFT64SvpPPolDefault);
}

unsafe impl HalSvpTPolImpl<FFT64Avx512> for FFT64Avx512 {
    poulpy_cpu_ref::hal_impl_svp_tpol!(FFT64SvpTPolDefault);
}

unsafe impl HalSvpImpl<FFT64Avx512> for FFT64Avx512 {
    poulpy_cpu_ref::hal_impl_svp!();
}

unsafe impl HalVecZnxDftImpl<FFT64Avx512> for FFT64Avx512 {
    poulpy_cpu_ref::hal_impl_vec_znx_dft!(FFT64VecZnxDftDefault, automorphism_with_plan: skip);

    fn vec_znx_dft_automorphism_with_plan(
        _module: &Module<Self>,
        plan: &Self::AutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        crate::fft64::fft64_vec_znx_dft_automorphism_avx512::<Self>(plan, res, res_col, a, a_col);
    }
}

unsafe impl HalVecZnxImpl<NTT4x30Avx512> for NTT4x30Avx512 {
    poulpy_cpu_ref::hal_impl_vec_znx!();

    // TODO: add an AVX-512-accelerated tiled transpose kernel; falls back to
    // the reference impl for now.
    fn vec_znx_transpose_backend(module: &Module<Self>, res: &mut VecZnxBackendMut<'_, Self>, a: &VecZnxBackendRef<'_, Self>) {
        <Self as HalVecZnxDefault<Self>>::vec_znx_transpose_backend_default(module, res, a)
    }
}

unsafe impl HalModuleImpl<NTT4x30Avx512> for NTT4x30Avx512 {
    poulpy_cpu_ref::hal_impl_module!(NTT4x30ModuleDefault);
}

unsafe impl HalVmpPMatImpl<NTT4x30Avx512> for NTT4x30Avx512 {
    fn vmp_prepare_pmat_tmp_bytes(module: &Module<Self>, _r: usize, _ci: usize, _co: usize, _s: usize) -> usize {
        crate::ntt4x30_avx512::vmp::vmp_prepare_pmat_tmp_bytes_avx(module.n())
    }

    fn vmp_prepare_pmat(
        module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &MatZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30_avx512::vmp::vmp_prepare_pmat_tmp_bytes_avx(module.n());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30_avx512::vmp::vmp_prepare_pmat_avx_pm(module, res, a, tmp);
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
        crate::ntt4x30_avx512::vmp::vmp_apply_tmp_bytes_avx(b_size, a_rows, a_cols_in)
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
        let bytes = crate::ntt4x30_avx512::vmp::vmp_apply_tmp_bytes_avx(b.size(), a.rows(), a.cols_in());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30_avx512::vmp::vmp_apply_pmat_dft_to_dft_avx(module, res, b, a, limb_offset, tmp);
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
        crate::ntt4x30_avx512::vmp::vmp_apply_tmp_bytes_avx(b_size, a_rows, a_cols_in)
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
        let bytes = crate::ntt4x30_avx512::vmp::vmp_apply_tmp_bytes_avx(b.size(), a.rows(), a.cols_in());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30_avx512::vmp::vmp_apply_pmat_dft_to_dft_accumulate_avx(module, res, b, a, limb_offset, tmp);
    }

    poulpy_cpu_ref::hal_impl_vmp_pmat!(kernels: skip);
}

unsafe impl HalVmpTMatImpl<NTT4x30Avx512> for NTT4x30Avx512 {
    /// This backend builds both tiers identically, so the hot-prep tier uses the
    /// same accelerated packing and kernels as the packed tier.
    fn vmp_prepare_tmat_tmp_bytes(module: &Module<Self>, _r: usize, _ci: usize, _co: usize, _s: usize) -> usize {
        crate::ntt4x30_avx512::vmp::vmp_prepare_pmat_tmp_bytes_avx(module.n())
    }

    fn vmp_prepare_tmat(
        module: &Module<Self>,
        res: &mut VmpTMatBackendMut<'_, Self>,
        a: &MatZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30_avx512::vmp::vmp_prepare_pmat_tmp_bytes_avx(module.n());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30_avx512::vmp::vmp_prepare_tmat_avx_pm(module, res, a, tmp);
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
        crate::ntt4x30_avx512::vmp::vmp_apply_tmp_bytes_avx(b_size, a_rows, a_cols_in)
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
        let bytes = crate::ntt4x30_avx512::vmp::vmp_apply_tmp_bytes_avx(b.size(), a.rows(), a.cols_in());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30_avx512::vmp::vmp_apply_tmat_dft_to_dft_avx(module, res, b, a, limb_offset, tmp);
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
        crate::ntt4x30_avx512::vmp::vmp_apply_tmp_bytes_avx(b_size, a_rows, a_cols_in)
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
        let bytes = crate::ntt4x30_avx512::vmp::vmp_apply_tmp_bytes_avx(b.size(), a.rows(), a.cols_in());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30_avx512::vmp::vmp_apply_tmat_dft_to_dft_accumulate_avx(module, res, b, a, limb_offset, tmp);
    }

    poulpy_cpu_ref::hal_impl_vmp_tmat!(kernels: skip);
}

unsafe impl HalVmpImpl<NTT4x30Avx512> for NTT4x30Avx512 {
    poulpy_cpu_ref::hal_impl_vmp!();
}

unsafe impl HalConvolutionImpl<NTT4x30Avx512> for NTT4x30Avx512 {
    fn cnv_prepare_left_pvec_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_prepare_left_pvec_tmp_bytes_default(module, res_size, a_size)
    }

    fn cnv_prepare_left_pvec(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::CnvPVecLBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        mask: i64,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow();
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_prepare_left_pvec_default(module, res, a, mask, &mut scratch);
    }

    fn cnv_prepare_right_pvec_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_prepare_right_pvec_tmp_bytes_default(module, res_size, a_size)
    }

    fn cnv_prepare_right_pvec(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::CnvPVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        mask: i64,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow();
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_prepare_right_pvec_default(module, res, a, mask, &mut scratch);
    }

    fn cnv_apply_pvec_to_dft_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        poulpy_cpu_ref::reference::ntt4x30::convolution::ntt4x30_cnv_apply_pvec_to_dft_tmp_bytes(res_size, a_size, b_size)
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
    fn cnv_apply_pvec_to_dft(
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
        let bytes = poulpy_cpu_ref::reference::ntt4x30::convolution::ntt4x30_cnv_apply_pvec_to_dft_tmp_bytes(
            res.size(),
            a.size(),
            b.size(),
        );
        let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        poulpy_cpu_ref::reference::ntt4x30::convolution::ntt4x30_cnv_apply_pvec_to_dft::<Self, _, _>(
            module, cnv_offset, res, res_col, a, a_col, b, b_col, tmp,
        );
    }

    fn cnv_prepare_left_tvec_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        poulpy_cpu_ref::reference::ntt4x30::convolution::ntt4x30_cnv_prepare_left_tvec_tmp_bytes(module.n())
    }

    fn cnv_prepare_left_tvec(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::CnvTVecLBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        mask: i64,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = poulpy_cpu_ref::reference::ntt4x30::convolution::ntt4x30_cnv_prepare_left_tvec_tmp_bytes(module.n());
        let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        poulpy_cpu_ref::reference::ntt4x30::convolution::ntt4x30_cnv_prepare_left_tvec::<Self>(module, res, a, mask, tmp);
    }

    fn cnv_prepare_right_tvec_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        poulpy_cpu_ref::reference::ntt4x30::convolution::ntt4x30_cnv_prepare_right_tvec_tmp_bytes(module.n())
    }

    fn cnv_prepare_right_tvec(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::CnvTVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        mask: i64,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let n_u64 = poulpy_cpu_ref::reference::ntt4x30::convolution::ntt4x30_cnv_prepare_right_tvec_tmp_bytes(module.n())
            / size_of::<u64>();
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), n_u64);
        poulpy_cpu_ref::reference::ntt4x30::convolution::ntt4x30_cnv_prepare_right_tvec::<Self>(module, res, a, mask, tmp);
    }

    fn cnv_apply_tvec_to_dft_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        _res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        crate::ntt4x30_avx512::convolution::cnv_apply_tvec_to_dft_avx_tmp_bytes(a_size, b_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_tvec_to_dft(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &poulpy_hal::layouts::CnvTVecLBackendRef<'_, Self>,
        a_col: usize,
        b: &poulpy_hal::layouts::CnvTVecRBackendRef<'_, Self>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30_avx512::convolution::cnv_apply_tvec_to_dft_avx_tmp_bytes(a.size(), b.size());
        let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        unsafe {
            crate::ntt4x30_avx512::convolution::cnv_apply_tvec_to_dft_avx(
                module, res, cnv_offset, res_col, a, a_col, b, b_col, tmp,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_pvec_to_dft_accumulate(
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
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_apply_pvec_to_dft_accumulate_default(
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

    fn cnv_pairwise_apply_pvec_to_dft_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        poulpy_cpu_ref::reference::ntt4x30::convolution::ntt4x30_cnv_pairwise_apply_pvec_to_dft_tmp_bytes(
            res_size, a_size, b_size,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_pairwise_apply_pvec_to_dft(
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
        let bytes = poulpy_cpu_ref::reference::ntt4x30::convolution::ntt4x30_cnv_pairwise_apply_pvec_to_dft_tmp_bytes(
            res.size(),
            a.size(),
            b.size(),
        );
        let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        poulpy_cpu_ref::reference::ntt4x30::convolution::ntt4x30_cnv_pairwise_apply_pvec_to_dft::<Self, _, _>(
            module, cnv_offset, res, res_col, a, b, i, j, tmp,
        );
    }

    fn cnv_prepare_self_pvec_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_prepare_self_pvec_tmp_bytes_default(module, res_size, a_size)
    }

    fn cnv_prepare_self_pvec(
        module: &Module<Self>,
        left: &mut poulpy_hal::layouts::CnvPVecLBackendMut<'_, Self>,
        right: &mut poulpy_hal::layouts::CnvPVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        mask: i64,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow();
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_prepare_self_pvec_default(module, left, right, a, mask, &mut scratch);
    }
    fn cnv_apply_pvec_to_dft_accumulate_tmp_bytes(
        module: &Module<Self>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_apply_pvec_to_dft_accumulate_tmp_bytes_default(
            module, cnv_offset, res_size, a_size, b_size,
        )
    }

    fn cnv_accumulate_pvec_to_dft_tmp_bytes(
        module: &Module<Self>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_accumulate_pvec_to_dft_tmp_bytes_default(
            module, cnv_offset, res_size, a_size, b_size,
        )
    }

    fn cnv_accumulate_pvec_to_dft<'a>(
        module: &Module<Self>,
        cnv_offset: usize,
        mut res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        terms: &[poulpy_hal::layouts::CnvDftAccTermPvec<'a, Self>],
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Self: 'a,
    {
        let mut scratch = scratch.borrow();
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_accumulate_pvec_to_dft_default(
            module, cnv_offset, &mut res, res_col, terms, &mut scratch,
        )
    }

    fn cnv_accumulate_tvec_to_dft_tmp_bytes(
        module: &Module<Self>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_accumulate_tvec_to_dft_tmp_bytes_default(
            module, cnv_offset, res_size, a_size, b_size,
        )
    }

    fn cnv_accumulate_tvec_to_dft<'a>(
        module: &Module<Self>,
        cnv_offset: usize,
        mut res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        terms: &[poulpy_hal::layouts::CnvDftAccTermTvec<'a, Self>],
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Self: 'a,
    {
        let mut scratch = scratch.borrow();
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_accumulate_tvec_to_dft_default(
            module, cnv_offset, &mut res, res_col, terms, &mut scratch,
        )
    }

    fn cnv_apply_tvec_to_dft_accumulate_tmp_bytes(
        module: &Module<Self>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_apply_tvec_to_dft_accumulate_tmp_bytes_default(
            module, cnv_offset, res_size, a_size, b_size,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_tvec_to_dft_accumulate(
        module: &Module<Self>,
        cnv_offset: usize,
        mut res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &poulpy_hal::layouts::CnvTVecLBackendRef<'_, Self>,
        a_col: usize,
        b: &poulpy_hal::layouts::CnvTVecRBackendRef<'_, Self>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow();
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_apply_tvec_to_dft_accumulate_default(
            module, cnv_offset, &mut res, res_col, a, a_col, b, b_col, &mut scratch,
        )
    }

    fn cnv_pairwise_apply_tvec_to_dft_tmp_bytes(
        module: &Module<Self>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_pairwise_apply_tvec_to_dft_tmp_bytes_default(
            module, cnv_offset, res_size, a_size, b_size,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_pairwise_apply_tvec_to_dft(
        module: &Module<Self>,
        cnv_offset: usize,
        mut res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &poulpy_hal::layouts::CnvTVecLBackendRef<'_, Self>,
        b: &poulpy_hal::layouts::CnvTVecRBackendRef<'_, Self>,
        i: usize,
        j: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow();
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_pairwise_apply_tvec_to_dft_default(
            module, cnv_offset, &mut res, res_col, a, b, i, j, &mut scratch,
        )
    }

    fn cnv_prepare_self_tvec_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_prepare_self_tvec_tmp_bytes_default(module, res_size, a_size)
    }

    fn cnv_prepare_self_tvec(
        module: &Module<Self>,
        left: &mut poulpy_hal::layouts::CnvTVecLBackendMut<'_, Self>,
        right: &mut poulpy_hal::layouts::CnvTVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        mask: i64,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow();
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_prepare_self_tvec_default(module, left, right, a, mask, &mut scratch);
    }

}

unsafe impl HalVecZnxBigImpl<NTT4x30Avx512> for NTT4x30Avx512 {
    poulpy_cpu_ref::hal_impl_vec_znx_big!(NTT4x30VecZnxBigDefault);
}

unsafe impl HalSvpPPolImpl<NTT4x30Avx512> for NTT4x30Avx512 {
    poulpy_cpu_ref::hal_impl_svp_ppol!(NTT4x30SvpPPolDefault);
}

unsafe impl HalSvpTPolImpl<NTT4x30Avx512> for NTT4x30Avx512 {
    poulpy_cpu_ref::hal_impl_svp_tpol!(NTT4x30SvpTPolDefault);
}

unsafe impl HalSvpImpl<NTT4x30Avx512> for NTT4x30Avx512 {
    poulpy_cpu_ref::hal_impl_svp!();
}

unsafe impl HalVecZnxDftImpl<NTT4x30Avx512> for NTT4x30Avx512 {
    fn vec_znx_dft_apply(
        module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <Self as NTT4x30VecZnxDftDefault<Self>>::vec_znx_dft_apply_default(module, step, offset, res, res_col, a, a_col)
    }

    fn vec_znx_idft_apply_tmp_bytes(module: &Module<Self>) -> usize {
        <Self as NTT4x30VecZnxDftDefault<Self>>::vec_znx_idft_apply_tmp_bytes_default(module)
    }

    fn vec_znx_idft_apply(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow();
        <Self as NTT4x30VecZnxDftDefault<Self>>::vec_znx_idft_apply_default(module, res, res_col, a, a_col, &mut scratch);
    }

    fn vec_znx_idft_apply_tmpa(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, Self>,
        a_col: usize,
    ) {
        crate::ntt4x30_avx512::vec_znx_dft_consume::vec_znx_idft_apply_tmpa_avx512(module, res, res_col, a, a_col);
    }

    fn vec_znx_dft_add_into(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) {
        <Self as NTT4x30VecZnxDftDefault<Self>>::vec_znx_dft_add_into_default(module, res, res_col, a, a_col, b, b_col)
    }

    fn vec_znx_dft_add_scaled_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        a_scale: i64,
    ) {
        <Self as NTT4x30VecZnxDftDefault<Self>>::vec_znx_dft_add_scaled_assign_default(module, res, res_col, a, a_col, a_scale)
    }

    fn vec_znx_dft_add_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <Self as NTT4x30VecZnxDftDefault<Self>>::vec_znx_dft_add_assign_default(module, res, res_col, a, a_col)
    }

    fn vec_znx_dft_sub(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) {
        <Self as NTT4x30VecZnxDftDefault<Self>>::vec_znx_dft_sub_default(module, res, res_col, a, a_col, b, b_col)
    }

    fn vec_znx_dft_sub_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <Self as NTT4x30VecZnxDftDefault<Self>>::vec_znx_dft_sub_assign_default(module, res, res_col, a, a_col)
    }

    fn vec_znx_dft_sub_negate_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <Self as NTT4x30VecZnxDftDefault<Self>>::vec_znx_dft_sub_negate_assign_default(module, res, res_col, a, a_col)
    }

    fn vec_znx_dft_copy(
        module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <Self as NTT4x30VecZnxDftDefault<Self>>::vec_znx_dft_copy_default(module, step, offset, res, res_col, a, a_col)
    }

    fn vec_znx_dft_zero(module: &Module<Self>, res: &mut VecZnxDftBackendMut<'_, Self>, res_col: usize) {
        <Self as NTT4x30VecZnxDftDefault<Self>>::vec_znx_dft_zero_default(module, res, res_col)
    }

    type AutomorphismPlan = <Self as NTT4x30VecZnxDftDefault<Self>>::AutomorphismPlanDefault;

    fn vec_znx_dft_automorphism_plan(module: &Module<Self>, p: i64) -> Self::AutomorphismPlan {
        <Self as NTT4x30VecZnxDftDefault<Self>>::vec_znx_dft_automorphism_plan_default(module, p)
    }

    fn vec_znx_dft_automorphism_with_plan(
        module: &Module<Self>,
        plan: &Self::AutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <Self as NTT4x30VecZnxDftDefault<Self>>::vec_znx_dft_automorphism_with_plan_default(module, plan, res, res_col, a, a_col)
    }
}

#[cfg(feature = "enable-ifma")]
mod ifma_impl {
    use super::{ScratchArena, take_host_typed};
    use crate::NTT3x42Ifma;
    use crate::ntt3x42_ifma::svp::{
        svp_apply_ppol_dft_to_dft, svp_apply_ppol_dft_to_dft_assign, svp_apply_ppol_small_to_dft, svp_apply_tpol_dft_to_dft,
        svp_apply_tpol_dft_to_dft_assign, svp_apply_tpol_small_to_dft, svp_prepare_ppol, svp_prepare_tpol,
    };
    use poulpy_cpu_ref::hal_defaults::HalVecZnxDefault;
    use poulpy_hal::api::{
        VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAddAssign, VecZnxDftBytesOf, VecZnxDftZero,
        VecZnxIdftApplyTmpA, VmpTMatBytesOf,
    };
    use poulpy_hal::layouts::{
        ScalarZnxBackendRef, SvpPPolBackendMut, SvpPPolBackendRef, SvpTPolBackendMut, SvpTPolBackendRef, SvpTPolToBackendMut,
        SvpTPolToBackendRef, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftToBackendMut, VmpTMatToBackendMut,
        VmpTMatToBackendRef, ZnxView, ZnxViewMut,
    };
    use poulpy_hal::layouts::{VmpTMatBackendMut, VmpTMatBackendRef};
    use poulpy_hal::{
        layouts::{
            MatZnxBackendRef, Module, NoiseInfos, VecZnxBackendMut, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut,
            VecZnxDftBackendRef, VecZnxDftToBackendRef, VmpPMatBackendMut, VmpPMatBackendRef,
        },
        oep::{
            HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalSvpPPolImpl, HalSvpTPolImpl, HalVecZnxBigImpl, HalVecZnxDftImpl,
            HalVecZnxImpl, HalVmpImpl, HalVmpPMatImpl, HalVmpTMatImpl,
        },
    };
    use std::mem::size_of;

    unsafe impl HalVecZnxImpl<NTT3x42Ifma> for NTT3x42Ifma {
        poulpy_cpu_ref::hal_impl_vec_znx!();

        // TODO: add an AVX-512/IFMA-accelerated tiled transpose kernel; falls
        // back to the reference impl for now.
        fn vec_znx_transpose_backend(
            module: &Module<Self>,
            res: &mut VecZnxBackendMut<'_, Self>,
            a: &VecZnxBackendRef<'_, Self>,
        ) {
            <Self as HalVecZnxDefault<Self>>::vec_znx_transpose_backend_default(module, res, a)
        }
    }

    unsafe impl HalModuleImpl<NTT3x42Ifma> for NTT3x42Ifma {
        fn new(n: u64) -> Module<Self> {
            crate::ntt3x42_ifma::module::module_new(n)
        }
    }

    unsafe impl HalVmpPMatImpl<NTT3x42Ifma> for NTT3x42Ifma {
        fn vmp_prepare_pmat_tmp_bytes(module: &Module<Self>, _r: usize, _ci: usize, _co: usize, _s: usize) -> usize {
            crate::ntt3x42_ifma::vmp::vmp_prepare_pmat_tmp_bytes_ifma(module.n())
        }

        fn vmp_prepare_pmat(
            module: &Module<Self>,
            res: &mut VmpPMatBackendMut<'_, Self>,
            a: &MatZnxBackendRef<'_, Self>,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::vmp::vmp_prepare_pmat_tmp_bytes_ifma(module.n());
            let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
            crate::ntt3x42_ifma::vmp::vmp_prepare_pmat_ifma(module, res, a, tmp);
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
            crate::ntt3x42_ifma::vmp::vmp_apply_tmp_bytes_ifma(b_size, a_rows, a_cols_in)
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
            let bytes = crate::ntt3x42_ifma::vmp::vmp_apply_tmp_bytes_ifma(b.size(), a.rows(), a.cols_in());
            let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
            crate::ntt3x42_ifma::vmp::vmp_apply_pmat_dft_to_dft_ifma(module, res, b, a, limb_offset, tmp);
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
            crate::ntt3x42_ifma::vmp::vmp_apply_tmp_bytes_ifma(b_size, a_rows, a_cols_in)
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
            let bytes = crate::ntt3x42_ifma::vmp::vmp_apply_tmp_bytes_ifma(b.size(), a.rows(), a.cols_in());
            let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
            crate::ntt3x42_ifma::vmp::vmp_apply_pmat_dft_to_dft_accumulate_ifma(module, res, b, a, limb_offset, tmp);
        }

        poulpy_cpu_ref::hal_impl_vmp_pmat!(kernels: skip);
    }

    unsafe impl HalVmpTMatImpl<NTT3x42Ifma> for NTT3x42Ifma {
        /// The hot-prep tier is not accelerated on this backend, so it uses the
        /// reference encoding and reference kernels throughout. Prepare and apply
        /// stay on the same layout, so the tier is self-consistent.
        fn vmp_prepare_tmat_tmp_bytes(module: &Module<Self>, _r: usize, _ci: usize, _co: usize, _s: usize) -> usize {
            crate::ntt3x42_ifma::vmp::vmp_prepare_pmat_tmp_bytes_ifma(module.n())
        }

        fn vmp_prepare_tmat(
            module: &Module<Self>,
            res: &mut VmpTMatBackendMut<'_, Self>,
            a: &MatZnxBackendRef<'_, Self>,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::vmp::vmp_prepare_pmat_tmp_bytes_ifma(module.n());
            let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
            crate::ntt3x42_ifma::vmp::vmp_prepare_tmat_ifma(module, res, a, tmp);
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
            crate::ntt3x42_ifma::vmp::vmp_apply_tmp_bytes_ifma(b_size, a_rows, a_cols_in)
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
            let bytes = crate::ntt3x42_ifma::vmp::vmp_apply_tmp_bytes_ifma(b.size(), a.rows(), a.cols_in());
            let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
            crate::ntt3x42_ifma::vmp::vmp_apply_tmat_dft_to_dft_ifma(module, res, b, a, limb_offset, tmp);
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
            crate::ntt3x42_ifma::vmp::vmp_apply_tmp_bytes_ifma(b_size, a_rows, a_cols_in)
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
            let bytes = crate::ntt3x42_ifma::vmp::vmp_apply_tmp_bytes_ifma(b.size(), a.rows(), a.cols_in());
            let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
            crate::ntt3x42_ifma::vmp::vmp_apply_tmat_dft_to_dft_accumulate_ifma(module, res, b, a, limb_offset, tmp);
        }

        poulpy_cpu_ref::hal_impl_vmp_tmat!(kernels: skip);
    }

    unsafe impl HalVmpImpl<NTT3x42Ifma> for NTT3x42Ifma {
        poulpy_cpu_ref::hal_impl_vmp!();
    }

    use poulpy_cpu_ref::hal_defaults::NTT4x30VecZnxBigDefault;

    unsafe impl HalVecZnxBigImpl<NTT3x42Ifma> for NTT3x42Ifma {
        poulpy_cpu_ref::hal_impl_vec_znx_big!(NTT4x30VecZnxBigDefault);
    }

    unsafe impl HalSvpPPolImpl<NTT3x42Ifma> for NTT3x42Ifma {
        fn svp_prepare_ppol(
            module: &Module<Self>,
            res: &mut SvpPPolBackendMut<'_, Self>,
            res_col: usize,
            a: &ScalarZnxBackendRef<'_, Self>,
            a_col: usize,
        ) {
            svp_prepare_ppol(module, res, res_col, a, a_col);
        }

        fn svp_apply_ppol_small_to_dft(
            module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &SvpPPolBackendRef<'_, Self>,
            a_col: usize,
            b: &VecZnxBackendRef<'_, Self>,
            b_col: usize,
        ) {
            svp_apply_ppol_small_to_dft(module, res, res_col, a, a_col, b, b_col);
        }

        fn svp_apply_ppol_dft_to_dft(
            _module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &SvpPPolBackendRef<'_, Self>,
            a_col: usize,
            b: &VecZnxDftBackendRef<'_, Self>,
            b_col: usize,
        ) {
            svp_apply_ppol_dft_to_dft(res, res_col, a, a_col, b, b_col);
        }

        fn svp_apply_ppol_dft_to_dft_assign(
            _module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &SvpPPolBackendRef<'_, Self>,
            a_col: usize,
        ) {
            svp_apply_ppol_dft_to_dft_assign(res, res_col, a, a_col);
        }

        fn svp_ppol_copy_backend(
            _module: &Module<Self>,
            res: &mut SvpPPolBackendMut<'_, Self>,
            res_col: usize,
            a: &SvpPPolBackendRef<'_, Self>,
            a_col: usize,
        ) {
            res.at_mut(res_col, 0).copy_from_slice(a.at(a_col, 0));
        }

        poulpy_cpu_ref::hal_impl_svp_ppol!(kernels: skip);
    }

    unsafe impl HalSvpTPolImpl<NTT3x42Ifma> for NTT3x42Ifma {
        fn svp_prepare_tpol(
            module: &Module<Self>,
            res: &mut SvpTPolBackendMut<'_, Self>,
            res_col: usize,
            a: &ScalarZnxBackendRef<'_, Self>,
            a_col: usize,
        ) {
            svp_prepare_tpol(module, res, res_col, a, a_col);
        }

        fn svp_apply_tpol_small_to_dft(
            module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &SvpTPolBackendRef<'_, Self>,
            a_col: usize,
            b: &VecZnxBackendRef<'_, Self>,
            b_col: usize,
        ) {
            svp_apply_tpol_small_to_dft(module, res, res_col, a, a_col, b, b_col);
        }

        fn svp_apply_tpol_dft_to_dft(
            _module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &SvpTPolBackendRef<'_, Self>,
            a_col: usize,
            b: &VecZnxDftBackendRef<'_, Self>,
            b_col: usize,
        ) {
            svp_apply_tpol_dft_to_dft(res, res_col, a, a_col, b, b_col);
        }

        fn svp_apply_tpol_dft_to_dft_assign(
            _module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &SvpTPolBackendRef<'_, Self>,
            a_col: usize,
        ) {
            svp_apply_tpol_dft_to_dft_assign(res, res_col, a, a_col);
        }

        fn svp_tpol_copy_backend(
            _module: &Module<Self>,
            res: &mut SvpTPolBackendMut<'_, Self>,
            res_col: usize,
            a: &SvpTPolBackendRef<'_, Self>,
            a_col: usize,
        ) {
            res.at_mut(res_col, 0).copy_from_slice(a.at(a_col, 0));
        }

        poulpy_cpu_ref::hal_impl_svp_tpol!(kernels: skip);
    }

    unsafe impl HalSvpImpl<NTT3x42Ifma> for NTT3x42Ifma {
        poulpy_cpu_ref::hal_impl_svp!();
    }

    unsafe impl HalVecZnxDftImpl<NTT3x42Ifma> for NTT3x42Ifma {
        fn vec_znx_dft_apply(
            module: &Module<Self>,
            step: usize,
            offset: usize,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &VecZnxBackendRef<'_, Self>,
            a_col: usize,
        ) {
            crate::ntt3x42_ifma::vec_znx_dft::vec_znx_dft_apply(module, step, offset, res, res_col, a, a_col);
        }

        fn vec_znx_idft_apply_tmp_bytes(module: &Module<Self>) -> usize {
            crate::ntt3x42_ifma::vec_znx_dft::vec_znx_idft_apply_tmp_bytes(module.n())
        }

        fn vec_znx_idft_apply(
            module: &Module<Self>,
            res: &mut VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &VecZnxDftBackendRef<'_, Self>,
            a_col: usize,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::vec_znx_dft::vec_znx_idft_apply_tmp_bytes(module.n());
            let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
            crate::ntt3x42_ifma::vec_znx_dft::vec_znx_idft_apply(module, res, res_col, a, a_col, tmp);
        }

        fn vec_znx_idft_apply_tmpa(
            module: &Module<Self>,
            res: &mut VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &mut VecZnxDftBackendMut<'_, Self>,
            a_col: usize,
        ) {
            crate::ntt3x42_ifma::vec_znx_dft::vec_znx_idft_apply_tmpa_ifma(module, res, res_col, a, a_col);
        }

        fn vec_znx_dft_add_into(
            _module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &VecZnxDftBackendRef<'_, Self>,
            a_col: usize,
            b: &VecZnxDftBackendRef<'_, Self>,
            b_col: usize,
        ) {
            crate::ntt3x42_ifma::vec_znx_dft::vec_znx_dft_add_into(res, res_col, a, a_col, b, b_col);
        }

        fn vec_znx_dft_add_scaled_assign(
            _module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &VecZnxDftBackendRef<'_, Self>,
            a_col: usize,
            a_scale: i64,
        ) {
            crate::ntt3x42_ifma::vec_znx_dft::vec_znx_dft_add_scaled_assign(res, res_col, a, a_col, a_scale);
        }

        fn vec_znx_dft_add_assign(
            _module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &VecZnxDftBackendRef<'_, Self>,
            a_col: usize,
        ) {
            crate::ntt3x42_ifma::vec_znx_dft::vec_znx_dft_add_assign(res, res_col, a, a_col);
        }

        fn vec_znx_dft_sub(
            _module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &VecZnxDftBackendRef<'_, Self>,
            a_col: usize,
            b: &VecZnxDftBackendRef<'_, Self>,
            b_col: usize,
        ) {
            crate::ntt3x42_ifma::vec_znx_dft::vec_znx_dft_sub(res, res_col, a, a_col, b, b_col);
        }

        fn vec_znx_dft_sub_assign(
            _module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &VecZnxDftBackendRef<'_, Self>,
            a_col: usize,
        ) {
            crate::ntt3x42_ifma::vec_znx_dft::vec_znx_dft_sub_assign(res, res_col, a, a_col);
        }

        fn vec_znx_dft_sub_negate_assign(
            _module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &VecZnxDftBackendRef<'_, Self>,
            a_col: usize,
        ) {
            crate::ntt3x42_ifma::vec_znx_dft::vec_znx_dft_sub_negate_assign(res, res_col, a, a_col);
        }

        fn vec_znx_dft_copy(
            _module: &Module<Self>,
            step: usize,
            offset: usize,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &VecZnxDftBackendRef<'_, Self>,
            a_col: usize,
        ) {
            crate::ntt3x42_ifma::vec_znx_dft::vec_znx_dft_copy(step, offset, res, res_col, a, a_col);
        }

        fn vec_znx_dft_zero(_module: &Module<Self>, res: &mut VecZnxDftBackendMut<'_, Self>, res_col: usize) {
            crate::ntt3x42_ifma::vec_znx_dft::vec_znx_dft_zero(res, res_col);
        }

        type AutomorphismPlan = poulpy_cpu_ref::reference::ntt4x30::vec_znx_dft::NttAutomorphismPlan;

        fn vec_znx_dft_automorphism_plan(module: &Module<Self>, p: i64) -> Self::AutomorphismPlan {
            // The slot↔exponent map is determined by the DIF NTT structure
            // (bit-reversal over log2(n) bits + level-0 ω^i twiddle), not by
            // the prime set, so the NTT4x30 closed-form builder is identical
            // for NTT3x42.
            poulpy_cpu_ref::reference::ntt4x30::vec_znx_dft::build_ntt4x30_automorphism_plan(module.n(), p)
        }

        fn vec_znx_dft_automorphism_with_plan(
            _module: &Module<Self>,
            plan: &Self::AutomorphismPlan,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &VecZnxDftBackendRef<'_, Self>,
            a_col: usize,
        ) {
            crate::ntt3x42_ifma::vec_znx_dft::vec_znx_dft_automorphism(plan, res, res_col, a, a_col);
        }
    }

    unsafe impl HalConvolutionImpl<NTT3x42Ifma> for NTT3x42Ifma {
        fn cnv_prepare_left_pvec_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
            crate::ntt3x42_ifma::convolution::cnv_prepare_left_pvec_tmp_bytes(module.n())
        }

        fn cnv_prepare_left_pvec(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::CnvPVecLBackendMut<'_, Self>,
            a: &VecZnxBackendRef<'_, Self>,
            mask: i64,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::convolution::cnv_prepare_left_pvec_tmp_bytes(module.n());
            let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
            crate::ntt3x42_ifma::convolution::cnv_prepare_left_pvec(module, res, a, mask, tmp);
        }

        fn cnv_prepare_right_pvec_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
            crate::ntt3x42_ifma::convolution::cnv_prepare_right_pvec_tmp_bytes(module.n())
        }

        fn cnv_prepare_right_pvec(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::CnvPVecRBackendMut<'_, Self>,
            a: &VecZnxBackendRef<'_, Self>,
            mask: i64,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::convolution::cnv_prepare_right_pvec_tmp_bytes(module.n());
            let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
            crate::ntt3x42_ifma::convolution::cnv_prepare_right_pvec(module, res, a, mask, tmp);
        }

        fn cnv_apply_pvec_to_dft_tmp_bytes(
            _module: &Module<Self>,
            _cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            crate::ntt3x42_ifma::convolution::cnv_apply_pvec_to_dft_ifma_tmp_bytes(res_size, a_size, b_size)
        }

        fn cnv_by_const_apply_tmp_bytes(
            _module: &Module<Self>,
            _cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            crate::ntt3x42_ifma::convolution::cnv_by_const_apply_tmp_bytes(res_size, a_size, b_size)
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_by_const_apply(
            _module: &Module<Self>,
            cnv_offset: usize,
            res: &mut VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &VecZnxBackendRef<'_, Self>,
            a_col: usize,
            b: &VecZnxBackendRef<'_, Self>,
            b_col: usize,
            b_coeff: usize,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::convolution::cnv_by_const_apply_tmp_bytes(res.size(), a.size(), b.size());
            let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
            crate::ntt3x42_ifma::convolution::cnv_by_const_apply(cnv_offset, res, res_col, a, a_col, b, b_col, b_coeff, tmp);
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_apply_pvec_to_dft(
            _module: &Module<Self>,
            cnv_offset: usize,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
            b_col: usize,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::convolution::cnv_apply_pvec_to_dft_ifma_tmp_bytes(res.size(), a.size(), b.size());
            let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
            unsafe {
                crate::ntt3x42_ifma::convolution::cnv_apply_pvec_to_dft_ifma(res, cnv_offset, res_col, a, a_col, b, b_col, tmp);
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_apply_pvec_to_dft_accumulate(
            _module: &Module<Self>,
            cnv_offset: usize,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
            b_col: usize,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::convolution::cnv_apply_pvec_to_dft_ifma_tmp_bytes(res.size(), a.size(), b.size());
            let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
            unsafe {
                crate::ntt3x42_ifma::convolution::cnv_apply_pvec_to_dft_accumulate_ifma(
                    res, cnv_offset, res_col, a, a_col, b, b_col, tmp,
                );
            }
        }

        fn cnv_pairwise_apply_pvec_to_dft_tmp_bytes(
            _module: &Module<Self>,
            _cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            crate::ntt3x42_ifma::convolution::cnv_pairwise_apply_pvec_to_dft_ifma_tmp_bytes(res_size, a_size, b_size)
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_pairwise_apply_pvec_to_dft(
            _module: &Module<Self>,
            cnv_offset: usize,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
            i: usize,
            j: usize,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes =
                crate::ntt3x42_ifma::convolution::cnv_pairwise_apply_pvec_to_dft_ifma_tmp_bytes(res.size(), a.size(), b.size());
            let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
            unsafe {
                crate::ntt3x42_ifma::convolution::cnv_pairwise_apply_pvec_to_dft_ifma(res, cnv_offset, res_col, a, b, i, j, tmp);
            }
        }

        fn cnv_prepare_self_pvec_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
            crate::ntt3x42_ifma::convolution::cnv_prepare_self_pvec_tmp_bytes(module.n())
        }

        fn cnv_prepare_self_pvec(
            module: &Module<Self>,
            left: &mut poulpy_hal::layouts::CnvPVecLBackendMut<'_, Self>,
            right: &mut poulpy_hal::layouts::CnvPVecRBackendMut<'_, Self>,
            a: &VecZnxBackendRef<'_, Self>,
            mask: i64,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::convolution::cnv_prepare_self_pvec_tmp_bytes(module.n());
            let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
            crate::ntt3x42_ifma::convolution::cnv_prepare_self_pvec(module, left, right, a, mask, tmp);
        }
        fn cnv_prepare_left_tvec_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
            crate::ntt3x42_ifma::convolution::cnv_prepare_left_pvec_tmp_bytes(module.n())
        }

        fn cnv_prepare_left_tvec(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::CnvTVecLBackendMut<'_, Self>,
            a: &VecZnxBackendRef<'_, Self>,
            mask: i64,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::convolution::cnv_prepare_left_pvec_tmp_bytes(module.n());
            let (tmp, _) = take_host_typed::<Self, u8>(
                scratch.borrow(), bytes / size_of::<u8>());
            crate::ntt3x42_ifma::convolution::cnv_prepare_left_pvec(module, res, a, mask, tmp);
        }

        fn cnv_prepare_right_tvec_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
            crate::ntt3x42_ifma::convolution::cnv_prepare_right_pvec_tmp_bytes(module.n())
        }

        fn cnv_prepare_right_tvec(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::CnvTVecRBackendMut<'_, Self>,
            a: &VecZnxBackendRef<'_, Self>,
            mask: i64,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::convolution::cnv_prepare_right_pvec_tmp_bytes(module.n());
            let (tmp, _) = take_host_typed::<Self, u64>(
                scratch.borrow(), bytes / size_of::<u64>());
            crate::ntt3x42_ifma::convolution::cnv_prepare_right_pvec(module, res, a, mask, tmp);
        }

        fn cnv_apply_tvec_to_dft_tmp_bytes(
            _module: &Module<Self>,
            _cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            crate::ntt3x42_ifma::convolution::cnv_apply_pvec_to_dft_ifma_tmp_bytes(res_size, a_size, b_size)
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_apply_tvec_to_dft(
            _module: &Module<Self>,
            cnv_offset: usize,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::CnvTVecLBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::CnvTVecRBackendRef<'_, Self>,
            b_col: usize,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::convolution::cnv_apply_pvec_to_dft_ifma_tmp_bytes(res.size(), a.size(), b.size());
            let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
            unsafe {
                crate::ntt3x42_ifma::convolution::cnv_apply_pvec_to_dft_ifma(res, cnv_offset, res_col, a, a_col, b, b_col, tmp);
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_apply_tvec_to_dft_accumulate(
            _module: &Module<Self>,
            cnv_offset: usize,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::CnvTVecLBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::CnvTVecRBackendRef<'_, Self>,
            b_col: usize,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::convolution::cnv_apply_pvec_to_dft_ifma_tmp_bytes(res.size(), a.size(), b.size());
            let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
            unsafe {
                crate::ntt3x42_ifma::convolution::cnv_apply_pvec_to_dft_accumulate_ifma(res, cnv_offset, res_col, a, a_col, b, b_col, tmp);
            }
        }

        fn cnv_apply_tvec_to_dft_accumulate_tmp_bytes(
            _module: &Module<Self>,
            _cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            crate::ntt3x42_ifma::convolution::cnv_apply_pvec_to_dft_ifma_tmp_bytes(res_size, a_size, b_size)
        }

        fn cnv_apply_pvec_to_dft_accumulate_tmp_bytes(
            _module: &Module<Self>,
            _cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            crate::ntt3x42_ifma::convolution::cnv_apply_pvec_to_dft_ifma_tmp_bytes(res_size, a_size, b_size)
        }

        fn cnv_pairwise_apply_tvec_to_dft_tmp_bytes(
            _module: &Module<Self>,
            _cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            crate::ntt3x42_ifma::convolution::cnv_pairwise_apply_pvec_to_dft_ifma_tmp_bytes(res_size, a_size, b_size)
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_pairwise_apply_tvec_to_dft(
            _module: &Module<Self>,
            cnv_offset: usize,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::CnvTVecLBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::CnvTVecRBackendRef<'_, Self>,
            i: usize,
            j: usize,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::convolution::cnv_pairwise_apply_pvec_to_dft_ifma_tmp_bytes(res.size(), a.size(), b.size());
            let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
            unsafe {
                crate::ntt3x42_ifma::convolution::cnv_pairwise_apply_pvec_to_dft_ifma(res, cnv_offset, res_col, a, b, i, j, tmp);
            }
        }

        fn cnv_prepare_self_tvec_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
            crate::ntt3x42_ifma::convolution::cnv_prepare_self_pvec_tmp_bytes(module.n())
        }

        fn cnv_prepare_self_tvec(
            module: &Module<Self>,
            left: &mut poulpy_hal::layouts::CnvTVecLBackendMut<'_, Self>,
            right: &mut poulpy_hal::layouts::CnvTVecRBackendMut<'_, Self>,
            a: &VecZnxBackendRef<'_, Self>,
            mask: i64,
            scratch: &mut ScratchArena<'_, Self>,
        ) {
            let bytes = crate::ntt3x42_ifma::convolution::cnv_prepare_self_pvec_tmp_bytes(module.n());
            let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
            crate::ntt3x42_ifma::convolution::cnv_prepare_self_pvec(module, left, right, a, mask, tmp);
        }

        fn cnv_accumulate_pvec_to_dft_tmp_bytes(
            module: &Module<Self>,
            cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            Self::cnv_apply_pvec_to_dft_accumulate_tmp_bytes(module, cnv_offset, res_size, a_size, b_size)
        }

        fn cnv_accumulate_pvec_to_dft<'a>(
            module: &Module<Self>,
            cnv_offset: usize,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            terms: &[poulpy_hal::layouts::CnvDftAccTermPvec<'a, Self>],
            scratch: &mut ScratchArena<'_, Self>,
        ) where
            Self: 'a,
        {
            for j in 0..poulpy_hal::layouts::ZnxInfos::size(res) {
                poulpy_hal::layouts::ZnxZero::zero_at(res, res_col, j);
            }
            for t in terms {
                Self::cnv_apply_pvec_to_dft_accumulate(
                    module, cnv_offset, res, res_col, &t.a, t.a_col, &t.b, t.b_col, scratch);
            }
        }

        fn cnv_accumulate_tvec_to_dft_tmp_bytes(
            module: &Module<Self>,
            cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            Self::cnv_apply_tvec_to_dft_accumulate_tmp_bytes(module, cnv_offset, res_size, a_size, b_size)
        }

        fn cnv_accumulate_tvec_to_dft<'a>(
            module: &Module<Self>,
            cnv_offset: usize,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            terms: &[poulpy_hal::layouts::CnvDftAccTermTvec<'a, Self>],
            scratch: &mut ScratchArena<'_, Self>,
        ) where
            Self: 'a,
        {
            for j in 0..poulpy_hal::layouts::ZnxInfos::size(res) {
                poulpy_hal::layouts::ZnxZero::zero_at(res, res_col, j);
            }
            for t in terms {
                Self::cnv_apply_tvec_to_dft_accumulate(
                    module, cnv_offset, res, res_col, &t.a, t.a_col, &t.b, t.b_col, scratch);
            }
        }

    }
}
