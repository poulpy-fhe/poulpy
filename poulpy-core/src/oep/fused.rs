//! Core-owned low-level fusion seams.
//!
//! Their shapes come from GLWE tensor and gadget-key layouts, so HAL remains
//! limited to generic polynomial primitives. Portable implementations compose
//! those HAL primitives.

use poulpy_hal::{
    api::{Convolution, ScratchArenaTakeBasic, VecZnxDftCopy, VmpApplyDftToDft, VmpApplyDftToDftAccumulate},
    layouts::{
        Backend, CnvPVecLBackendRef, CnvPVecRBackendRef, Module, ScratchArena, VecZnxDftBackendMut, VecZnxDftBackendRef,
        VecZnxDftToBackendRef, VmpPMatBackendRef,
    },
};

/// Backend hook for the rank-one GLWE tensor product in the DFT domain.
///
/// The operation computes `(a0*b0, a0*b1 + a1*b0, a1*b1)`. Its semantics are
/// specific to the three-column tensor representation used by `poulpy-core`.
///
/// # Safety
///
/// Implementations must uphold the supplied backend layout, scratch, and
/// arithmetic contracts and produce the same three columns as the default
/// composition.
pub unsafe trait GLWETensorRank1DftImpl<BE: Backend>: Backend {
    fn glwe_tensor_rank1_dft_tmp_bytes(
        module: &Module<BE>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;

    /// Whether this backend provides a genuinely fused implementation.
    fn glwe_tensor_rank1_dft_is_fused(module: &Module<BE>) -> bool;

    fn glwe_tensor_rank1_dft(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        a: &CnvPVecLBackendRef<'_, BE>,
        b: &CnvPVecRBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    );
}

/// Backend hook for applying GGLWE gadget digits from interleaved DFT rows.
///
/// # Safety
///
/// Implementations must match [`gglwe_product_digits_strided_default`] and
/// respect all backend buffer and scratch bounds.
pub unsafe trait GGLWEProductDigitsStridedImpl<BE: Backend>: Backend {
    fn gglwe_product_digits_strided(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        a: &VecZnxDftBackendRef<'_, BE>,
        dsize: usize,
        pmat: &VmpPMatBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    );
}

/// Portable rank-one tensor fallback used by backend opt-in macros.
#[doc(hidden)]
pub fn glwe_tensor_rank1_dft_tmp_bytes_default<BE: Backend>(
    module: &Module<BE>,
    cnv_offset: usize,
    res_size: usize,
    a_size: usize,
    b_size: usize,
) -> usize
where
    Module<BE>: Convolution<BE>,
{
    module.cnv_apply_dft_tmp_bytes(cnv_offset, res_size, a_size, b_size)
}

/// Portable rank-one tensor fallback used by backend opt-in macros.
#[doc(hidden)]
pub fn glwe_tensor_rank1_dft_default<BE: Backend>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    a: &CnvPVecLBackendRef<'_, BE>,
    b: &CnvPVecRBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    Module<BE>: Convolution<BE>,
{
    module.cnv_apply_dft(cnv_offset, res, 0, a, 0, b, 0, scratch);
    module.cnv_apply_dft(cnv_offset, res, 1, a, 0, b, 1, scratch);
    module.cnv_apply_dft_accumulate(cnv_offset, res, 1, a, 1, b, 0, scratch);
    module.cnv_apply_dft(cnv_offset, res, 2, a, 1, b, 1, scratch);
}

/// Portable interleaved-digit GGLWE/VMP fallback used by backend opt-in
/// macros.
#[doc(hidden)]
pub fn gglwe_product_digits_strided_default<BE: Backend>(
    module: &Module<BE>,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    a: &VecZnxDftBackendRef<'_, BE>,
    dsize: usize,
    pmat: &VmpPMatBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    Module<BE>: VecZnxDftCopy<BE> + VmpApplyDftToDft<BE> + VmpApplyDftToDftAccumulate<BE>,
{
    let cols = a.cols();
    let a_size = a.size();
    let dnum = pmat.rows();
    for di in 0..dsize {
        let (mut digit, mut scratch_digit) =
            scratch
                .borrow()
                .take_vec_znx_dft_scratch(module, cols, ((a_size + di) / dsize).min(dnum));
        for col in 0..cols {
            module.vec_znx_dft_copy(dsize, dsize - di - 1, &mut digit, col, a, col);
        }

        if di == 0 {
            // The overwriting pass must initialize the entire destination;
            // only subsequent accumulating passes may use narrowed views.
            module.vmp_apply_dft_to_dft(res, &digit.to_backend_ref(), pmat, 0, &mut scratch_digit);
        } else {
            let pad = ((dsize - di) as isize - 2).max(0) as usize;
            let compute_size = res.size().min(pmat.size().saturating_sub(pad));
            let mut res_view = res.with_size_mut(compute_size);
            module.vmp_apply_dft_to_dft_accumulate(&mut res_view, &digit.to_backend_ref(), pmat, di, &mut scratch_digit);
        }
    }
}

/// Opts a backend into the portable Core rank-one tensor implementation.
#[macro_export]
macro_rules! impl_glwe_tensor_rank1_dft_default {
    ($be:ty) => {
        unsafe impl $crate::oep::GLWETensorRank1DftImpl<$be> for $be {
            fn glwe_tensor_rank1_dft_tmp_bytes(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res_size: usize,
                a_size: usize,
                b_size: usize,
            ) -> usize {
                $crate::oep::glwe_tensor_rank1_dft_tmp_bytes_default(module, cnv_offset, res_size, a_size, b_size)
            }

            fn glwe_tensor_rank1_dft_is_fused(_module: &::poulpy_hal::layouts::Module<$be>) -> bool {
                false
            }

            fn glwe_tensor_rank1_dft(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut ::poulpy_hal::layouts::VecZnxDftBackendMut<'_, $be>,
                a: &::poulpy_hal::layouts::CnvPVecLBackendRef<'_, $be>,
                b: &::poulpy_hal::layouts::CnvPVecRBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) {
                $crate::oep::glwe_tensor_rank1_dft_default(module, cnv_offset, res, a, b, scratch)
            }
        }
    };
}

/// Opts a backend into the portable Core interleaved-digit GGLWE product.
#[macro_export]
macro_rules! impl_gglwe_product_digits_strided_default {
    ($be:ty) => {
        unsafe impl $crate::oep::GGLWEProductDigitsStridedImpl<$be> for $be {
            fn gglwe_product_digits_strided(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut ::poulpy_hal::layouts::VecZnxDftBackendMut<'_, $be>,
                a: &::poulpy_hal::layouts::VecZnxDftBackendRef<'_, $be>,
                dsize: usize,
                pmat: &::poulpy_hal::layouts::VmpPMatBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) {
                $crate::oep::gglwe_product_digits_strided_default(module, res, a, dsize, pmat, scratch)
            }
        }
    };
}
