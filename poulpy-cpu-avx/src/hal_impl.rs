use std::mem::size_of;

use crate::{FFT64Avx, NTT4x30Avx};
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

unsafe impl HalVecZnxImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_vec_znx!();

    // TODO: add an AVX-accelerated tiled transpose kernel; falls back to the
    // reference impl for now.
    fn vec_znx_transpose_backend(module: &Module<Self>, res: &mut VecZnxBackendMut<'_, Self>, a: &VecZnxBackendRef<'_, Self>) {
        <Self as HalVecZnxDefault<Self>>::vec_znx_transpose_backend_default(module, res, a)
    }
}

unsafe impl HalModuleImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_module!(FFT64ModuleDefault);
}

unsafe impl HalVmpPMatImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_vmp_pmat!(FFT64VmpPMatDefault);
}

unsafe impl HalVmpTMatImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_vmp_tmat!(FFT64VmpTMatDefault);
}

unsafe impl HalVmpImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_vmp!();
}

unsafe impl HalConvolutionImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_convolution!(FFT64ConvolutionDefault);
}

unsafe impl HalVecZnxBigImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_vec_znx_big!(FFT64VecZnxBigDefault);
}

unsafe impl HalSvpPPolImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_svp_ppol!(FFT64SvpPPolDefault);
}

unsafe impl HalSvpTPolImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_svp_tpol!(FFT64SvpTPolDefault);
}

unsafe impl HalSvpImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_svp!();
}

unsafe impl HalVecZnxDftImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_vec_znx_dft!(FFT64VecZnxDftDefault, automorphism_with_plan: skip);

    #[inline(always)]
    fn vec_znx_dft_automorphism_with_plan(
        _module: &Module<Self>,
        plan: &Self::AutomorphismPlan,
        res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        crate::fft64::fft64_vec_znx_dft_automorphism_avx::<Self>(plan, res, res_col, a, a_col);
    }
}

unsafe impl HalVecZnxImpl<NTT4x30Avx> for NTT4x30Avx {
    poulpy_cpu_ref::hal_impl_vec_znx!();

    // TODO: add an AVX-accelerated tiled transpose kernel; falls back to the
    // reference impl for now.
    fn vec_znx_transpose_backend(module: &Module<Self>, res: &mut VecZnxBackendMut<'_, Self>, a: &VecZnxBackendRef<'_, Self>) {
        <Self as HalVecZnxDefault<Self>>::vec_znx_transpose_backend_default(module, res, a)
    }
}

unsafe impl HalModuleImpl<NTT4x30Avx> for NTT4x30Avx {
    poulpy_cpu_ref::hal_impl_module!(NTT4x30ModuleDefault);
}

unsafe impl HalVmpPMatImpl<NTT4x30Avx> for NTT4x30Avx {
    fn vmp_prepare_pmat_tmp_bytes(module: &Module<Self>, _r: usize, _ci: usize, _co: usize, _s: usize) -> usize {
        crate::ntt4x30::vmp::vmp_prepare_pmat_tmp_bytes_avx(module.n())
    }

    fn vmp_prepare_pmat(
        module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &MatZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30::vmp::vmp_prepare_pmat_tmp_bytes_avx(module.n());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30::vmp::vmp_prepare_pmat_avx_pm(module, res, a, tmp);
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
        crate::ntt4x30::vmp::vmp_apply_tmp_bytes_avx(b_size, a_rows, a_cols_in)
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
        let bytes = crate::ntt4x30::vmp::vmp_apply_tmp_bytes_avx(b.size(), a.rows(), a.cols_in());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30::vmp::vmp_apply_pmat_dft_to_dft_avx(module, res, b, a, limb_offset, tmp);
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
        crate::ntt4x30::vmp::vmp_apply_tmp_bytes_avx(b_size, a_rows, a_cols_in)
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
        let bytes = crate::ntt4x30::vmp::vmp_apply_tmp_bytes_avx(b.size(), a.rows(), a.cols_in());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30::vmp::vmp_apply_pmat_dft_to_dft_accumulate_avx(module, res, b, a, limb_offset, tmp);
    }

    poulpy_cpu_ref::hal_impl_vmp_pmat!(kernels: skip);
}

unsafe impl HalVmpTMatImpl<NTT4x30Avx> for NTT4x30Avx {
    /// This backend builds both tiers identically, so the hot-prep tier uses the
    /// same accelerated packing and kernels as the packed tier.
    fn vmp_prepare_tmat_tmp_bytes(module: &Module<Self>, _r: usize, _ci: usize, _co: usize, _s: usize) -> usize {
        crate::ntt4x30::vmp::vmp_prepare_pmat_tmp_bytes_avx(module.n())
    }

    fn vmp_prepare_tmat(
        module: &Module<Self>,
        res: &mut VmpTMatBackendMut<'_, Self>,
        a: &MatZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30::vmp::vmp_prepare_pmat_tmp_bytes_avx(module.n());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30::vmp::vmp_prepare_tmat_avx_pm(module, res, a, tmp);
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
        crate::ntt4x30::vmp::vmp_apply_tmp_bytes_avx(b_size, a_rows, a_cols_in)
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
        let bytes = crate::ntt4x30::vmp::vmp_apply_tmp_bytes_avx(b.size(), a.rows(), a.cols_in());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30::vmp::vmp_apply_tmat_dft_to_dft_avx(module, res, b, a, limb_offset, tmp);
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
        crate::ntt4x30::vmp::vmp_apply_tmp_bytes_avx(b_size, a_rows, a_cols_in)
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
        let bytes = crate::ntt4x30::vmp::vmp_apply_tmp_bytes_avx(b.size(), a.rows(), a.cols_in());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt4x30::vmp::vmp_apply_tmat_dft_to_dft_accumulate_avx(module, res, b, a, limb_offset, tmp);
    }

    poulpy_cpu_ref::hal_impl_vmp_tmat!(kernels: skip);
}

unsafe impl HalVmpImpl<NTT4x30Avx> for NTT4x30Avx {
    poulpy_cpu_ref::hal_impl_vmp!();
}

unsafe impl HalConvolutionImpl<NTT4x30Avx> for NTT4x30Avx {
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

    fn cnv_apply_dft_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        poulpy_cpu_ref::reference::ntt4x30::convolution::ntt4x30_cnv_apply_dft_tmp_bytes(res_size, a_size, b_size)
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
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
        a_col: usize,
        b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes =
            poulpy_cpu_ref::reference::ntt4x30::convolution::ntt4x30_cnv_apply_dft_tmp_bytes(res.size(), a.size(), b.size());
        let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        poulpy_cpu_ref::reference::ntt4x30::convolution::ntt4x30_cnv_apply_dft::<Self>(
            module, cnv_offset, res, res_col, a, a_col, b, b_col, tmp,
        );
    }

    // glwe_mul_plain keeps the eager block-major path on AVX2 (faster than the
    // fused apply here), so the lazy methods inherit the eager default.

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

    fn cnv_accumulate_dft_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        _a_size: usize,
        _b_size: usize,
    ) -> usize {
        crate::ntt4x30::convolution::cnv_accumulate_dft_avx_tmp_bytes(res_size)
    }

    fn cnv_accumulate_dft<'a>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        terms: &[poulpy_hal::layouts::CnvDftAccTerm<'a, Self>],
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        Self: poulpy_hal::oep::HalVecZnxDftImpl<Self> + 'a,
    {
        let bytes = crate::ntt4x30::convolution::cnv_accumulate_dft_avx_tmp_bytes(res.size());
        let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        unsafe { crate::ntt4x30::convolution::cnv_accumulate_dft_avx(module, cnv_offset, res, res_col, terms, tmp) };
    }

    fn cnv_pairwise_apply_dft_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        poulpy_cpu_ref::reference::ntt4x30::convolution::ntt4x30_cnv_pairwise_apply_dft_tmp_bytes(res_size, a_size, b_size)
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
        let bytes = poulpy_cpu_ref::reference::ntt4x30::convolution::ntt4x30_cnv_pairwise_apply_dft_tmp_bytes(
            res.size(),
            a.size(),
            b.size(),
        );
        let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        poulpy_cpu_ref::reference::ntt4x30::convolution::ntt4x30_cnv_pairwise_apply_dft::<Self>(
            module, cnv_offset, res, res_col, a, b, i, j, tmp,
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

unsafe impl HalVecZnxBigImpl<NTT4x30Avx> for NTT4x30Avx {
    poulpy_cpu_ref::hal_impl_vec_znx_big!(NTT4x30VecZnxBigDefault);
}

unsafe impl HalSvpPPolImpl<NTT4x30Avx> for NTT4x30Avx {
    poulpy_cpu_ref::hal_impl_svp_ppol!(NTT4x30SvpPPolDefault);
}

unsafe impl HalSvpTPolImpl<NTT4x30Avx> for NTT4x30Avx {
    poulpy_cpu_ref::hal_impl_svp_tpol!(NTT4x30SvpTPolDefault);
}

unsafe impl HalSvpImpl<NTT4x30Avx> for NTT4x30Avx {
    poulpy_cpu_ref::hal_impl_svp!();
}

unsafe impl HalVecZnxDftImpl<NTT4x30Avx> for NTT4x30Avx {
    poulpy_cpu_ref::hal_impl_vec_znx_dft!(NTT4x30VecZnxDftDefault, automorphism_with_plan: skip);

    fn vec_znx_dft_automorphism_with_plan(
        _module: &Module<Self>,
        plan: &Self::AutomorphismPlan,
        res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        crate::ntt4x30::automorphism::ntt4x30_vec_znx_dft_automorphism_avx::<Self>(plan, res, res_col, a, a_col);
    }
}
