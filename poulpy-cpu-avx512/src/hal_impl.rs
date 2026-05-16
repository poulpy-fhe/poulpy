use std::mem::size_of;

use crate::{FFT64Avx512, NTT120Avx512};
use poulpy_cpu_ref::hal_defaults::{
    FFT64ConvolutionDefault, FFT64ModuleDefault, FFT64SvpDefault, FFT64VecZnxBigDefault, FFT64VecZnxDftDefault, FFT64VmpDefault,
    HalVecZnxDefault, NTT120ConvolutionDefault, NTT120ModuleDefault, NTT120SvpDefault, NTT120VecZnxBigDefault,
    NTT120VecZnxDftDefault, NTT120VmpDefault,
};
use poulpy_hal::{
    api::{HostBufMut, ScratchArenaTakeBasic, VecZnxDftApply, VecZnxDftZero, VmpApplyDftToDft},
    layouts::{
        Backend, MatZnxBackendRef, Module, NoiseInfos, ScratchArena, VecZnxBackendMut, VecZnxBackendRef, VecZnxDftBackendMut,
        VecZnxDftBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef, VmpPMatBackendMut, VmpPMatBackendRef, ZnxInfos,
    },
    oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
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

unsafe impl HalVecZnxImpl<FFT64Avx512> for FFT64Avx512 {
    poulpy_cpu_ref::hal_impl_vec_znx!();
}

unsafe impl HalModuleImpl<FFT64Avx512> for FFT64Avx512 {
    poulpy_cpu_ref::hal_impl_module!(FFT64ModuleDefault);
}

unsafe impl HalVmpImpl<FFT64Avx512> for FFT64Avx512 {
    poulpy_cpu_ref::hal_impl_vmp!(FFT64VmpDefault);
}

unsafe impl HalConvolutionImpl<FFT64Avx512> for FFT64Avx512 {
    poulpy_cpu_ref::hal_impl_convolution!(FFT64ConvolutionDefault);
}

unsafe impl HalVecZnxBigImpl<FFT64Avx512> for FFT64Avx512 {
    poulpy_cpu_ref::hal_impl_vec_znx_big!(FFT64VecZnxBigDefault);
}

unsafe impl HalSvpImpl<FFT64Avx512> for FFT64Avx512 {
    poulpy_cpu_ref::hal_impl_svp!(FFT64SvpDefault);
}

unsafe impl HalVecZnxDftImpl<FFT64Avx512> for FFT64Avx512 {
    poulpy_cpu_ref::hal_impl_vec_znx_dft!(FFT64VecZnxDftDefault);
}

unsafe impl HalVecZnxImpl<NTT120Avx512> for NTT120Avx512 {
    poulpy_cpu_ref::hal_impl_vec_znx!();
}

unsafe impl HalModuleImpl<NTT120Avx512> for NTT120Avx512 {
    poulpy_cpu_ref::hal_impl_module!(NTT120ModuleDefault);
}

unsafe impl HalVmpImpl<NTT120Avx512> for NTT120Avx512 {
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

    fn vmp_apply_dft<'s, R>(
        module: &Module<Self>,
        res: &mut R,
        a: &VecZnxBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'s, Self>,
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
        crate::ntt120_avx512::vmp::vmp_prepare_tmp_bytes_avx(module.n())
    }

    fn vmp_prepare<'s>(
        module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &MatZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'s, Self>,
    ) {
        let bytes = crate::ntt120_avx512::vmp::vmp_prepare_tmp_bytes_avx(module.n());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt120_avx512::vmp::vmp_prepare_avx_pm(module, res, a, tmp);
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
        crate::ntt120_avx512::vmp::vmp_apply_tmp_bytes_avx(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft<'s, 'r>(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'r, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'s, Self>,
    ) {
        let bytes = crate::ntt120_avx512::vmp::vmp_apply_tmp_bytes_avx(a.size(), b.rows(), b.cols_in());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt120_avx512::vmp::vmp_apply_dft_to_dft_avx(module, res, a, b, limb_offset, tmp);
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
        crate::ntt120_avx512::vmp::vmp_apply_tmp_bytes_avx(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft_accumulate<'s, 'r>(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'r, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'s, Self>,
    ) {
        let bytes = crate::ntt120_avx512::vmp::vmp_apply_tmp_bytes_avx(a.size(), b.rows(), b.cols_in());
        let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        crate::ntt120_avx512::vmp::vmp_apply_dft_to_dft_accumulate_avx(module, res, a, b, limb_offset, tmp);
    }

    fn vmp_zero(module: &Module<Self>, res: &mut VmpPMatBackendMut<'_, Self>) {
        <Self as NTT120VmpDefault<Self>>::vmp_zero_default(module, res)
    }
}

unsafe impl HalConvolutionImpl<NTT120Avx512> for NTT120Avx512 {
    fn cnv_prepare_left_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
        <Self as NTT120ConvolutionDefault<Self>>::cnv_prepare_left_tmp_bytes_default(module, res_size, a_size)
    }

    fn cnv_prepare_left<'s, 'r>(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::CnvPVecLBackendMut<'r, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        mask: i64,
        scratch: &mut ScratchArena<'s, Self>,
    ) {
        let mut scratch = scratch.borrow();
        <Self as NTT120ConvolutionDefault<Self>>::cnv_prepare_left_default(module, res, a, mask, &mut scratch);
    }

    fn cnv_prepare_right_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
        <Self as NTT120ConvolutionDefault<Self>>::cnv_prepare_right_tmp_bytes_default(module, res_size, a_size)
    }

    fn cnv_prepare_right<'s, 'r>(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::CnvPVecRBackendMut<'r, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        mask: i64,
        scratch: &mut ScratchArena<'s, Self>,
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
        crate::ntt120_avx512::convolution::cnv_apply_dft_avx_tmp_bytes(a_size, b_size)
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
    fn cnv_by_const_apply<'s>(
        module: &Module<Self>,
        cnv_offset: usize,
        mut res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
        b_coeff: usize,
        scratch: &mut ScratchArena<'s, Self>,
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
    fn cnv_apply_dft<'s>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
        a_col: usize,
        b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
        b_col: usize,
        scratch: &mut ScratchArena<'s, Self>,
    ) {
        let bytes = crate::ntt120_avx512::convolution::cnv_apply_dft_avx_tmp_bytes(a.size(), b.size());
        let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        unsafe {
            crate::ntt120_avx512::convolution::cnv_apply_dft_avx(module, res, cnv_offset, res_col, a, a_col, b, b_col, tmp);
        }
    }

    fn cnv_pairwise_apply_dft_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        crate::ntt120_avx512::convolution::cnv_pairwise_apply_dft_avx_tmp_bytes(res_size, a_size, b_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_pairwise_apply_dft<'s>(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
        b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
        i: usize,
        j: usize,
        scratch: &mut ScratchArena<'s, Self>,
    ) {
        let bytes = crate::ntt120_avx512::convolution::cnv_pairwise_apply_dft_avx_tmp_bytes(res.size(), a.size(), b.size());
        let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        unsafe {
            crate::ntt120_avx512::convolution::cnv_pairwise_apply_dft_avx(module, res, cnv_offset, res_col, a, b, i, j, tmp);
        }
    }

    fn cnv_prepare_self_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
        <Self as NTT120ConvolutionDefault<Self>>::cnv_prepare_self_tmp_bytes_default(module, res_size, a_size)
    }

    fn cnv_prepare_self<'s, 'l, 'r>(
        module: &Module<Self>,
        left: &mut poulpy_hal::layouts::CnvPVecLBackendMut<'l, Self>,
        right: &mut poulpy_hal::layouts::CnvPVecRBackendMut<'r, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        mask: i64,
        scratch: &mut ScratchArena<'s, Self>,
    ) {
        let mut scratch = scratch.borrow();
        <Self as NTT120ConvolutionDefault<Self>>::cnv_prepare_self_default(module, left, right, a, mask, &mut scratch);
    }
}

unsafe impl HalVecZnxBigImpl<NTT120Avx512> for NTT120Avx512 {
    poulpy_cpu_ref::hal_impl_vec_znx_big!(NTT120VecZnxBigDefault);
}

unsafe impl HalSvpImpl<NTT120Avx512> for NTT120Avx512 {
    poulpy_cpu_ref::hal_impl_svp!(NTT120SvpDefault);
}

unsafe impl HalVecZnxDftImpl<NTT120Avx512> for NTT120Avx512 {
    fn vec_znx_dft_apply(
        module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <Self as NTT120VecZnxDftDefault<Self>>::vec_znx_dft_apply_default(module, step, offset, res, res_col, a, a_col)
    }

    fn vec_znx_idft_apply_tmp_bytes(module: &Module<Self>) -> usize {
        <Self as NTT120VecZnxDftDefault<Self>>::vec_znx_idft_apply_tmp_bytes_default(module)
    }

    fn vec_znx_idft_apply<'s>(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, Self>,
    ) {
        let mut scratch = scratch.borrow();
        <Self as NTT120VecZnxDftDefault<Self>>::vec_znx_idft_apply_default(module, res, res_col, a, a_col, &mut scratch);
    }

    fn vec_znx_idft_apply_tmpa(
        module: &Module<Self>,
        res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, Self>,
        a_col: usize,
    ) {
        crate::ntt120_avx512::vec_znx_dft_consume::vec_znx_idft_apply_tmpa_avx512(module, res, res_col, a, a_col);
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
        <Self as NTT120VecZnxDftDefault<Self>>::vec_znx_dft_add_into_default(module, res, res_col, a, a_col, b, b_col)
    }

    fn vec_znx_dft_add_scaled_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        a_scale: i64,
    ) {
        <Self as NTT120VecZnxDftDefault<Self>>::vec_znx_dft_add_scaled_assign_default(module, res, res_col, a, a_col, a_scale)
    }

    fn vec_znx_dft_add_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <Self as NTT120VecZnxDftDefault<Self>>::vec_znx_dft_add_assign_default(module, res, res_col, a, a_col)
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
        <Self as NTT120VecZnxDftDefault<Self>>::vec_znx_dft_sub_default(module, res, res_col, a, a_col, b, b_col)
    }

    fn vec_znx_dft_sub_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <Self as NTT120VecZnxDftDefault<Self>>::vec_znx_dft_sub_assign_default(module, res, res_col, a, a_col)
    }

    fn vec_znx_dft_sub_negate_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <Self as NTT120VecZnxDftDefault<Self>>::vec_znx_dft_sub_negate_assign_default(module, res, res_col, a, a_col)
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
        <Self as NTT120VecZnxDftDefault<Self>>::vec_znx_dft_copy_default(module, step, offset, res, res_col, a, a_col)
    }

    fn vec_znx_dft_zero(module: &Module<Self>, res: &mut VecZnxDftBackendMut<'_, Self>, res_col: usize) {
        <Self as NTT120VecZnxDftDefault<Self>>::vec_znx_dft_zero_default(module, res, res_col)
    }
}

#[cfg(feature = "enable-ifma")]
mod ifma_impl {
    use super::{ScratchArena, take_host_typed};
    use crate::NTT126Ifma;
    use poulpy_cpu_ref::hal_defaults::HalVecZnxDefault;
    use poulpy_hal::{
        api::{ScratchArenaTakeBasic, VecZnxDftApply, VecZnxDftZero, VmpApplyDftToDft},
        layouts::{
            Backend, MatZnxBackendRef, Module, NoiseInfos, ScalarZnxBackendRef, SvpPPolBackendMut, SvpPPolBackendRef,
            VecZnxBackendMut, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut, VecZnxDftBackendRef,
            VecZnxDftToBackendMut, VecZnxDftToBackendRef, VmpPMatBackendMut, VmpPMatBackendRef, ZnxInfos,
        },
        oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
    };
    use std::mem::size_of;

    unsafe impl HalVecZnxImpl<NTT126Ifma> for NTT126Ifma {
        poulpy_cpu_ref::hal_impl_vec_znx!();
    }

    unsafe impl HalModuleImpl<NTT126Ifma> for NTT126Ifma {
        fn new(n: u64) -> Module<NTT126Ifma> {
            crate::ntt126_ifma::module::module_new(n)
        }
    }

    unsafe impl HalVmpImpl<NTT126Ifma> for NTT126Ifma {
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

        fn vmp_apply_dft<'s, R>(
            module: &Module<Self>,
            res: &mut R,
            a: &VecZnxBackendRef<'_, Self>,
            b: &VmpPMatBackendRef<'_, Self>,
            scratch: &mut ScratchArena<'s, Self>,
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
            crate::ntt126_ifma::vmp::vmp_prepare_tmp_bytes_ifma(module.n())
        }

        fn vmp_prepare<'s>(
            module: &Module<Self>,
            res: &mut VmpPMatBackendMut<'_, Self>,
            a: &MatZnxBackendRef<'_, Self>,
            scratch: &mut ScratchArena<'s, Self>,
        ) {
            let bytes = crate::ntt126_ifma::vmp::vmp_prepare_tmp_bytes_ifma(module.n());
            let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
            crate::ntt126_ifma::vmp::vmp_prepare_ifma(module, res, a, tmp);
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
            crate::ntt126_ifma::vmp::vmp_apply_tmp_bytes_ifma(a_size, b_rows, b_cols_in)
        }

        fn vmp_apply_dft_to_dft<'s, 'r>(
            module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'r, Self>,
            a: &VecZnxDftBackendRef<'_, Self>,
            b: &VmpPMatBackendRef<'_, Self>,
            limb_offset: usize,
            scratch: &mut ScratchArena<'s, Self>,
        ) {
            let bytes = crate::ntt126_ifma::vmp::vmp_apply_tmp_bytes_ifma(a.size(), b.rows(), b.cols_in());
            let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
            crate::ntt126_ifma::vmp::vmp_apply_dft_to_dft_ifma(module, res, a, b, limb_offset, tmp);
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
            crate::ntt126_ifma::vmp::vmp_apply_tmp_bytes_ifma(a_size, b_rows, b_cols_in)
        }

        fn vmp_apply_dft_to_dft_accumulate<'s, 'r>(
            module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'r, Self>,
            a: &VecZnxDftBackendRef<'_, Self>,
            b: &VmpPMatBackendRef<'_, Self>,
            limb_offset: usize,
            scratch: &mut ScratchArena<'s, Self>,
        ) {
            let bytes = crate::ntt126_ifma::vmp::vmp_apply_tmp_bytes_ifma(a.size(), b.rows(), b.cols_in());
            let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
            crate::ntt126_ifma::vmp::vmp_apply_dft_to_dft_accumulate_ifma(module, res, a, b, limb_offset, tmp);
        }

        fn vmp_zero(_module: &Module<Self>, res: &mut VmpPMatBackendMut<'_, Self>) {
            crate::ntt126_ifma::vmp::vmp_zero(res);
        }
    }

    use poulpy_cpu_ref::hal_defaults::{NTT120SvpDefault, NTT120VecZnxBigDefault};

    unsafe impl HalVecZnxBigImpl<NTT126Ifma> for NTT126Ifma {
        poulpy_cpu_ref::hal_impl_vec_znx_big!(NTT120VecZnxBigDefault);
    }

    unsafe impl HalSvpImpl<NTT126Ifma> for NTT126Ifma {
        fn svp_prepare(
            module: &Module<Self>,
            res: &mut SvpPPolBackendMut<'_, Self>,
            res_col: usize,
            a: &ScalarZnxBackendRef<'_, Self>,
            a_col: usize,
        ) {
            crate::ntt126_ifma::svp::svp_prepare(module, res, res_col, a, a_col);
        }

        fn svp_ppol_copy_backend(
            module: &Module<Self>,
            res: &mut SvpPPolBackendMut<'_, Self>,
            res_col: usize,
            a: &SvpPPolBackendRef<'_, Self>,
            a_col: usize,
        ) {
            <Self as NTT120SvpDefault<Self>>::svp_ppol_copy_backend_default(module, res, res_col, a, a_col);
        }

        fn svp_apply_dft(
            module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &SvpPPolBackendRef<'_, Self>,
            a_col: usize,
            b: &VecZnxBackendRef<'_, Self>,
            b_col: usize,
        ) {
            crate::ntt126_ifma::svp::svp_apply_dft(module, res, res_col, a, a_col, b, b_col);
        }

        fn svp_apply_dft_to_dft(
            module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &SvpPPolBackendRef<'_, Self>,
            a_col: usize,
            b: &VecZnxDftBackendRef<'_, Self>,
            b_col: usize,
        ) {
            crate::ntt126_ifma::svp::svp_apply_dft_to_dft(module, res, res_col, a, a_col, b, b_col);
        }

        fn svp_apply_dft_to_dft_assign(
            module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &SvpPPolBackendRef<'_, Self>,
            a_col: usize,
        ) {
            crate::ntt126_ifma::svp::svp_apply_dft_to_dft_assign(module, res, res_col, a, a_col);
        }
    }

    unsafe impl HalVecZnxDftImpl<NTT126Ifma> for NTT126Ifma {
        fn vec_znx_dft_apply(
            module: &Module<Self>,
            step: usize,
            offset: usize,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &VecZnxBackendRef<'_, Self>,
            a_col: usize,
        ) {
            crate::ntt126_ifma::vec_znx_dft::vec_znx_dft_apply(module, step, offset, res, res_col, a, a_col);
        }

        fn vec_znx_idft_apply_tmp_bytes(module: &Module<Self>) -> usize {
            crate::ntt126_ifma::vec_znx_dft::vec_znx_idft_apply_tmp_bytes(module.n())
        }

        fn vec_znx_idft_apply<'s>(
            module: &Module<Self>,
            res: &mut VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &VecZnxDftBackendRef<'_, Self>,
            a_col: usize,
            scratch: &mut ScratchArena<'s, Self>,
        ) {
            let bytes = crate::ntt126_ifma::vec_znx_dft::vec_znx_idft_apply_tmp_bytes(module.n());
            let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
            crate::ntt126_ifma::vec_znx_dft::vec_znx_idft_apply(module, res, res_col, a, a_col, tmp);
        }

        fn vec_znx_idft_apply_tmpa(
            module: &Module<Self>,
            res: &mut VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &mut VecZnxDftBackendMut<'_, Self>,
            a_col: usize,
        ) {
            crate::ntt126_ifma::vec_znx_dft::vec_znx_idft_apply_tmpa_ifma(module, res, res_col, a, a_col);
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
            crate::ntt126_ifma::vec_znx_dft::vec_znx_dft_add_into(res, res_col, a, a_col, b, b_col);
        }

        fn vec_znx_dft_add_scaled_assign(
            _module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &VecZnxDftBackendRef<'_, Self>,
            a_col: usize,
            a_scale: i64,
        ) {
            crate::ntt126_ifma::vec_znx_dft::vec_znx_dft_add_scaled_assign(res, res_col, a, a_col, a_scale);
        }

        fn vec_znx_dft_add_assign(
            _module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &VecZnxDftBackendRef<'_, Self>,
            a_col: usize,
        ) {
            crate::ntt126_ifma::vec_znx_dft::vec_znx_dft_add_assign(res, res_col, a, a_col);
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
            crate::ntt126_ifma::vec_znx_dft::vec_znx_dft_sub(res, res_col, a, a_col, b, b_col);
        }

        fn vec_znx_dft_sub_assign(
            _module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &VecZnxDftBackendRef<'_, Self>,
            a_col: usize,
        ) {
            crate::ntt126_ifma::vec_znx_dft::vec_znx_dft_sub_assign(res, res_col, a, a_col);
        }

        fn vec_znx_dft_sub_negate_assign(
            _module: &Module<Self>,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &VecZnxDftBackendRef<'_, Self>,
            a_col: usize,
        ) {
            crate::ntt126_ifma::vec_znx_dft::vec_znx_dft_sub_negate_assign(res, res_col, a, a_col);
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
            crate::ntt126_ifma::vec_znx_dft::vec_znx_dft_copy(step, offset, res, res_col, a, a_col);
        }

        fn vec_znx_dft_zero(_module: &Module<Self>, res: &mut VecZnxDftBackendMut<'_, Self>, res_col: usize) {
            crate::ntt126_ifma::vec_znx_dft::vec_znx_dft_zero(res, res_col);
        }
    }

    unsafe impl HalConvolutionImpl<NTT126Ifma> for NTT126Ifma {
        fn cnv_prepare_left_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
            crate::ntt126_ifma::convolution::cnv_prepare_left_tmp_bytes(module.n())
        }

        fn cnv_prepare_left<'s, 'r>(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::CnvPVecLBackendMut<'r, Self>,
            a: &VecZnxBackendRef<'_, Self>,
            mask: i64,
            scratch: &mut ScratchArena<'s, Self>,
        ) {
            let bytes = crate::ntt126_ifma::convolution::cnv_prepare_left_tmp_bytes(module.n());
            let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
            crate::ntt126_ifma::convolution::cnv_prepare_left(module, res, a, mask, tmp);
        }

        fn cnv_prepare_right_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
            crate::ntt126_ifma::convolution::cnv_prepare_right_tmp_bytes(module.n())
        }

        fn cnv_prepare_right<'s, 'r>(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::CnvPVecRBackendMut<'r, Self>,
            a: &VecZnxBackendRef<'_, Self>,
            mask: i64,
            scratch: &mut ScratchArena<'s, Self>,
        ) {
            let bytes = crate::ntt126_ifma::convolution::cnv_prepare_right_tmp_bytes(module.n());
            let (tmp, _) = take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
            crate::ntt126_ifma::convolution::cnv_prepare_right(module, res, a, mask, tmp);
        }

        fn cnv_apply_dft_tmp_bytes(
            _module: &Module<Self>,
            _cnv_offset: usize,
            _res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            crate::ntt126_ifma::convolution::cnv_apply_dft_ifma_tmp_bytes(a_size, b_size)
        }

        fn cnv_by_const_apply_tmp_bytes(
            _module: &Module<Self>,
            _cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            crate::ntt126_ifma::convolution::cnv_by_const_apply_tmp_bytes(res_size, a_size, b_size)
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_by_const_apply<'s>(
            _module: &Module<Self>,
            cnv_offset: usize,
            res: &mut VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &VecZnxBackendRef<'_, Self>,
            a_col: usize,
            b: &VecZnxBackendRef<'_, Self>,
            b_col: usize,
            b_coeff: usize,
            scratch: &mut ScratchArena<'s, Self>,
        ) {
            let bytes = crate::ntt126_ifma::convolution::cnv_by_const_apply_tmp_bytes(res.size(), a.size(), b.size());
            let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
            crate::ntt126_ifma::convolution::cnv_by_const_apply(cnv_offset, res, res_col, a, a_col, b, b_col, b_coeff, tmp);
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_apply_dft<'s>(
            _module: &Module<Self>,
            cnv_offset: usize,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
            b_col: usize,
            scratch: &mut ScratchArena<'s, Self>,
        ) {
            let bytes = crate::ntt126_ifma::convolution::cnv_apply_dft_ifma_tmp_bytes(a.size(), b.size());
            let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
            unsafe {
                crate::ntt126_ifma::convolution::cnv_apply_dft_ifma(res, cnv_offset, res_col, a, a_col, b, b_col, tmp);
            }
        }

        fn cnv_pairwise_apply_dft_tmp_bytes(
            _module: &Module<Self>,
            _cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            crate::ntt126_ifma::convolution::cnv_pairwise_apply_dft_ifma_tmp_bytes(res_size, a_size, b_size)
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_pairwise_apply_dft<'s>(
            _module: &Module<Self>,
            cnv_offset: usize,
            res: &mut VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
            i: usize,
            j: usize,
            scratch: &mut ScratchArena<'s, Self>,
        ) {
            let bytes = crate::ntt126_ifma::convolution::cnv_pairwise_apply_dft_ifma_tmp_bytes(res.size(), a.size(), b.size());
            let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
            unsafe {
                crate::ntt126_ifma::convolution::cnv_pairwise_apply_dft_ifma(res, cnv_offset, res_col, a, b, i, j, tmp);
            }
        }

        fn cnv_prepare_self_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
            crate::ntt126_ifma::convolution::cnv_prepare_self_tmp_bytes(module.n())
        }

        fn cnv_prepare_self<'s, 'l, 'r>(
            module: &Module<Self>,
            left: &mut poulpy_hal::layouts::CnvPVecLBackendMut<'l, Self>,
            right: &mut poulpy_hal::layouts::CnvPVecRBackendMut<'r, Self>,
            a: &VecZnxBackendRef<'_, Self>,
            mask: i64,
            scratch: &mut ScratchArena<'s, Self>,
        ) {
            let bytes = crate::ntt126_ifma::convolution::cnv_prepare_self_tmp_bytes(module.n());
            let (tmp, _) = take_host_typed::<Self, u8>(scratch.borrow(), bytes);
            crate::ntt126_ifma::convolution::cnv_prepare_self(module, left, right, a, mask, tmp);
        }
    }
}
