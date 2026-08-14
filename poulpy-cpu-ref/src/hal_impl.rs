use crate::{
    FFT64Ref, NTT4x30Ref,
    hal_defaults::{
        FFT64ConvolutionDefault, FFT64ModuleDefault, FFT64SvpPPolDefault, FFT64SvpTPolDefault, FFT64VecZnxBigDefault,
        FFT64VecZnxDftDefault, FFT64VmpPMatDefault, FFT64VmpTMatDefault, HalVecZnxDefault, NTT4x30ConvolutionDefault,
        NTT4x30ModuleDefault, NTT4x30SvpPPolDefault, NTT4x30SvpTPolDefault, NTT4x30VecZnxBigDefault, NTT4x30VecZnxDftDefault,
        NTT4x30VmpPMatDefault, NTT4x30VmpTMatDefault,
    },
};
use poulpy_hal::{
    api::{
        VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAddAssign, VecZnxDftBytesOf, VecZnxDftZero,
        VecZnxIdftApplyTmpA, VmpTMatBytesOf,
    },
    layouts::{
        Module, NoiseInfos, SvpTPolToBackendMut, SvpTPolToBackendRef, VecZnxBackendMut, VecZnxBackendRef, VecZnxBigToBackendMut,
        VecZnxBigToBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef, VmpTMatToBackendMut, VmpTMatToBackendRef,
    },
    oep::{
        HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalSvpPPolImpl, HalSvpTPolImpl, HalVecZnxBigImpl, HalVecZnxDftImpl,
        HalVecZnxImpl, HalVmpImpl, HalVmpPMatImpl, HalVmpTMatImpl,
    },
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

    fn vec_znx_transpose_backend(module: &Module<Self>, res: &mut VecZnxBackendMut<'_, Self>, a: &VecZnxBackendRef<'_, Self>) {
        <Self as HalVecZnxDefault<Self>>::vec_znx_transpose_backend_default(module, res, a)
    }
}

unsafe impl HalModuleImpl<FFT64Ref> for FFT64Ref {
    hal_impl_module!(FFT64ModuleDefault);
}

unsafe impl HalVmpPMatImpl<FFT64Ref> for FFT64Ref {
    hal_impl_vmp_pmat!(FFT64VmpPMatDefault);
}

unsafe impl HalVmpTMatImpl<FFT64Ref> for FFT64Ref {
    hal_impl_vmp_tmat!(FFT64VmpTMatDefault);
}

unsafe impl HalVmpImpl<FFT64Ref> for FFT64Ref {
    hal_impl_vmp!();
}

unsafe impl HalConvolutionImpl<FFT64Ref> for FFT64Ref {
    hal_impl_convolution!(FFT64ConvolutionDefault);
}

unsafe impl HalVecZnxBigImpl<FFT64Ref> for FFT64Ref {
    hal_impl_vec_znx_big!(FFT64VecZnxBigDefault);
}

unsafe impl HalSvpPPolImpl<FFT64Ref> for FFT64Ref {
    hal_impl_svp_ppol!(FFT64SvpPPolDefault);
}

unsafe impl HalSvpTPolImpl<FFT64Ref> for FFT64Ref {
    hal_impl_svp_tpol!(FFT64SvpTPolDefault);
}

unsafe impl HalSvpImpl<FFT64Ref> for FFT64Ref {
    hal_impl_svp!();
}

unsafe impl HalVecZnxDftImpl<FFT64Ref> for FFT64Ref {
    hal_impl_vec_znx_dft!(FFT64VecZnxDftDefault);
}

unsafe impl HalVecZnxImpl<NTT4x30Ref> for NTT4x30Ref {
    hal_impl_vec_znx!();

    fn vec_znx_transpose_backend(module: &Module<Self>, res: &mut VecZnxBackendMut<'_, Self>, a: &VecZnxBackendRef<'_, Self>) {
        <Self as HalVecZnxDefault<Self>>::vec_znx_transpose_backend_default(module, res, a)
    }
}

unsafe impl HalModuleImpl<NTT4x30Ref> for NTT4x30Ref {
    hal_impl_module!(NTT4x30ModuleDefault);
}

unsafe impl HalVmpPMatImpl<NTT4x30Ref> for NTT4x30Ref {
    hal_impl_vmp_pmat!(NTT4x30VmpPMatDefault);
}

unsafe impl HalVmpTMatImpl<NTT4x30Ref> for NTT4x30Ref {
    hal_impl_vmp_tmat!(NTT4x30VmpTMatDefault);
}

unsafe impl HalVmpImpl<NTT4x30Ref> for NTT4x30Ref {
    hal_impl_vmp!();
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

unsafe impl HalSvpPPolImpl<NTT4x30Ref> for NTT4x30Ref {
    hal_impl_svp_ppol!(NTT4x30SvpPPolDefault);
}

unsafe impl HalSvpTPolImpl<NTT4x30Ref> for NTT4x30Ref {
    hal_impl_svp_tpol!(NTT4x30SvpTPolDefault);
}

unsafe impl HalSvpImpl<NTT4x30Ref> for NTT4x30Ref {
    hal_impl_svp!();
}

unsafe impl HalVecZnxDftImpl<NTT4x30Ref> for NTT4x30Ref {
    hal_impl_vec_znx_dft!(NTT4x30VecZnxDftDefault);
}
