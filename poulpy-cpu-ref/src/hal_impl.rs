use crate::{
    FFT64Ref, NTT4x30Ref,
    hal_defaults::{
        FFT64ConvolutionDefault, FFT64ModuleDefault, FFT64SvpDefault, FFT64VecZnxBigDefault, FFT64VecZnxDftDefault,
        FFT64VmpDefault, HalVecZnxDefault, NTT4x30ConvolutionDefault, NTT4x30ModuleDefault, NTT4x30SvpDefault,
        NTT4x30VecZnxBigDefault, NTT4x30VecZnxDftDefault, NTT4x30VmpDefault,
    },
};
use poulpy_hal::{
    api::{VecZnxDftApply, VecZnxDftZero, VmpApplyDftToDft},
    layouts::{
        Backend, FitsIn, Module, NoiseInfos, NormalizationState, VecZnxBackendMut, VecZnxBackendRef, VecZnxDftToBackendMut,
        VecZnxDftToBackendRef, ZnxInfos,
    },
    oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
};

#[macro_use]
mod vec_znx;
#[macro_use]
mod module;
#[macro_use]
mod vmp;
#[macro_use]
mod convolution;
#[macro_use]
mod vec_znx_big;
#[macro_use]
mod svp;
#[macro_use]
mod vec_znx_dft;
#[macro_use]
#[cfg(all(test, feature = "enable-core"))]
pub(crate) mod delegating_backend;

unsafe impl HalVecZnxImpl<FFT64Ref> for FFT64Ref {
    hal_impl_vec_znx!();

    fn vec_znx_transpose_backend<S: NormalizationState>(
        module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self, S>,
        a: &VecZnxBackendRef<'_, Self, impl FitsIn<S>>,
    ) {
        <Self as HalVecZnxDefault<Self>>::vec_znx_transpose_backend_default(module, res, a)
    }
}

unsafe impl HalModuleImpl<FFT64Ref> for FFT64Ref {
    hal_impl_module!(FFT64ModuleDefault);
}

unsafe impl HalVmpImpl<FFT64Ref> for FFT64Ref {
    hal_impl_vmp!(FFT64VmpDefault);
}

unsafe impl HalConvolutionImpl<FFT64Ref> for FFT64Ref {
    hal_impl_convolution!(FFT64ConvolutionDefault);
}

unsafe impl HalVecZnxBigImpl<FFT64Ref> for FFT64Ref {
    hal_impl_vec_znx_big!(FFT64VecZnxBigDefault);
}

unsafe impl HalSvpImpl<FFT64Ref> for FFT64Ref {
    hal_impl_svp!(FFT64SvpDefault);
}

unsafe impl HalVecZnxDftImpl<FFT64Ref> for FFT64Ref {
    hal_impl_vec_znx_dft!(FFT64VecZnxDftDefault);
}

unsafe impl HalVecZnxImpl<NTT4x30Ref> for NTT4x30Ref {
    hal_impl_vec_znx!();

    fn vec_znx_transpose_backend<S: NormalizationState>(
        module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self, S>,
        a: &VecZnxBackendRef<'_, Self, impl FitsIn<S>>,
    ) {
        <Self as HalVecZnxDefault<Self>>::vec_znx_transpose_backend_default(module, res, a)
    }
}

unsafe impl HalModuleImpl<NTT4x30Ref> for NTT4x30Ref {
    hal_impl_module!(NTT4x30ModuleDefault);
}

unsafe impl HalVmpImpl<NTT4x30Ref> for NTT4x30Ref {
    hal_impl_vmp!(NTT4x30VmpDefault);
}

unsafe impl HalConvolutionImpl<NTT4x30Ref> for NTT4x30Ref {
    hal_impl_convolution!(NTT4x30ConvolutionDefault);

    fn cnv_accumulate_dft_tmp_bytes(
        module: &Module<Self>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_accumulate_dft_tmp_bytes_default(
            module, cnv_offset, res_size, a_size, b_size,
        )
    }

    fn cnv_accumulate_dft<'a>(
        module: &Module<Self>,
        cnv_offset: usize,
        mut res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        terms: &[poulpy_hal::layouts::CnvDftAccTerm<'a, Self>],
        scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
    ) where
        Self: HalVecZnxDftImpl<Self> + 'a,
    {
        let mut scratch = scratch.borrow();
        <Self as NTT4x30ConvolutionDefault<Self>>::cnv_accumulate_dft_default(
            module,
            cnv_offset,
            &mut res,
            res_col,
            terms,
            &mut scratch,
        );
    }
}

unsafe impl HalVecZnxBigImpl<NTT4x30Ref> for NTT4x30Ref {
    hal_impl_vec_znx_big!(NTT4x30VecZnxBigDefault);
}

unsafe impl HalSvpImpl<NTT4x30Ref> for NTT4x30Ref {
    hal_impl_svp!(NTT4x30SvpDefault);
}

unsafe impl HalVecZnxDftImpl<NTT4x30Ref> for NTT4x30Ref {
    hal_impl_vec_znx_dft!(NTT4x30VecZnxDftDefault);
}
