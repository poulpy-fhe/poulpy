//! HAL registrations for [`NTT4x30CIRef`](super::NTT4x30CIRef).

use super::NTT4x30CIRef;

use crate::hal_defaults::{
    HalVecZnxDefault, NTT4x30ConvolutionDefault, NTT4x30ModuleDefault, NTT4x30SvpDefault, NTT4x30VecZnxBigDefault,
    NTT4x30VecZnxDftDefault, NTT4x30VmpDefault,
};
use poulpy_hal::{
    layouts::{Module, VecZnxBackendMut, VecZnxBackendRef},
    oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
};

unsafe impl HalVecZnxImpl for NTT4x30CIRef {
    crate::hal_impl_vec_znx!();
}

unsafe impl HalModuleImpl for NTT4x30CIRef {
    crate::hal_impl_module!(NTT4x30ModuleDefault);
}

unsafe impl HalVmpImpl for NTT4x30CIRef {
    crate::hal_impl_vmp!(NTT4x30VmpDefault);
}

unsafe impl HalConvolutionImpl for NTT4x30CIRef {
    crate::hal_impl_convolution!(NTT4x30ConvolutionDefault);

    fn cnv_apply_dft_sum_tmp_bytes(
        module: &Module<Self>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        <Self as NTT4x30ConvolutionDefault>::cnv_apply_dft_sum_tmp_bytes_default(module, cnv_offset, res_size, a_size, b_size)
    }

    fn cnv_apply_dft_sum(
        module: &Module<Self>,
        cnv_offset: usize,
        mut res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        terms: &[poulpy_hal::layouts::CnvDftAccTerm<'_, Self>],
        scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
    ) {
        let mut scratch = scratch.borrow();
        <Self as NTT4x30ConvolutionDefault>::cnv_apply_dft_sum_default(
            module,
            cnv_offset,
            &mut res,
            res_col,
            terms,
            &mut scratch,
        );
    }
}

unsafe impl HalVecZnxBigImpl for NTT4x30CIRef {
    crate::hal_impl_vec_znx_big!(NTT4x30VecZnxBigDefault);
}

unsafe impl HalSvpImpl for NTT4x30CIRef {
    crate::hal_impl_svp!(NTT4x30SvpDefault);
}

unsafe impl HalVecZnxDftImpl for NTT4x30CIRef {
    crate::hal_impl_vec_znx_dft!(NTT4x30VecZnxDftDefault);
}
