#[cfg(feature = "enable-ifma")]
use crate::NTT3x42Ifma;
use crate::{FFT64Avx512, NTT4x30Avx512};
use poulpy_core::{
    impl_conversion_defaults_full, impl_decryption_defaults_full, impl_encryption_defaults_full,
    impl_gglwe_automorphism_defaults_full, impl_gglwe_external_product_defaults_full, impl_gglwe_keyswitch_defaults_full,
    impl_gglwe_product_digits_strided_default, impl_ggsw_automorphism_defaults_full, impl_ggsw_external_product_defaults_full,
    impl_ggsw_keyswitch_defaults_full, impl_glwe_automorphism_defaults_full, impl_glwe_external_product_defaults_full,
    impl_glwe_keyswitch_defaults_full, impl_glwe_packing_defaults_full, impl_glwe_tensor_rank1_dft_default,
    impl_glwe_trace_defaults_full, impl_linear_transformation_defaults_full, impl_lwe_keyswitch_defaults_full,
};
use poulpy_hal::layouts::{
    CnvPVecLBackendRef, CnvPVecRBackendRef, Module, ScratchArena, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendRef,
};

impl_glwe_tensor_rank1_dft_default!(FFT64Avx512);
impl_gglwe_product_digits_strided_default!(FFT64Avx512);

unsafe impl poulpy_core::oep::GLWETensorRank1DftImpl<NTT4x30Avx512> for NTT4x30Avx512 {
    fn glwe_tensor_rank1_dft_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        crate::ntt4x30_avx512::convolution::cnv_tensor_rank1_dft_avx512_tmp_bytes(res_size, a_size, b_size)
    }

    fn glwe_tensor_rank1_dft_is_fused(_module: &Module<Self>) -> bool {
        true
    }

    fn glwe_tensor_rank1_dft(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &CnvPVecLBackendRef<'_, Self>,
        b: &CnvPVecRBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30_avx512::convolution::cnv_tensor_rank1_dft_avx512_tmp_bytes(res.size(), a.size(), b.size());
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        unsafe {
            crate::ntt4x30_avx512::convolution::cnv_tensor_rank1_dft_avx512(module, res, cnv_offset, a, b, tmp);
        }
    }
}

unsafe impl poulpy_core::oep::GGLWEProductDigitsStridedImpl<NTT4x30Avx512> for NTT4x30Avx512 {
    fn gglwe_product_digits_strided(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        dsize: usize,
        pmat: &VmpPMatBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt4x30_avx512::vmp::vmp_apply_digits_strided_tmp_bytes_avx(
            a.cols(),
            a.size(),
            dsize,
            pmat.rows(),
            pmat.cols_in(),
        );
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / std::mem::size_of::<u64>());
        crate::ntt4x30_avx512::vmp::vmp_apply_dft_to_dft_digits_strided_avx(module, res, a, dsize, pmat, tmp);
    }
}

#[cfg(feature = "enable-ifma")]
unsafe impl poulpy_core::oep::GLWETensorRank1DftImpl<NTT3x42Ifma> for NTT3x42Ifma {
    fn glwe_tensor_rank1_dft_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        crate::ntt3x42_ifma::convolution::cnv_tensor_rank1_dft_ifma_tmp_bytes(res_size, a_size, b_size)
    }

    fn glwe_tensor_rank1_dft_is_fused(_module: &Module<Self>) -> bool {
        true
    }

    fn glwe_tensor_rank1_dft(
        _module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &CnvPVecLBackendRef<'_, Self>,
        b: &CnvPVecRBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt3x42_ifma::convolution::cnv_tensor_rank1_dft_ifma_tmp_bytes(res.size(), a.size(), b.size());
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u8>(scratch.borrow(), bytes);
        unsafe {
            crate::ntt3x42_ifma::convolution::cnv_tensor_rank1_dft_ifma(res, cnv_offset, a, b, tmp);
        }
    }
}

#[cfg(feature = "enable-ifma")]
unsafe impl poulpy_core::oep::GGLWEProductDigitsStridedImpl<NTT3x42Ifma> for NTT3x42Ifma {
    fn gglwe_product_digits_strided(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        dsize: usize,
        pmat: &VmpPMatBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = crate::ntt3x42_ifma::vmp::vmp_apply_digits_strided_tmp_bytes_ifma(
            a.cols(),
            a.size(),
            dsize,
            pmat.rows(),
            pmat.cols_in(),
        );
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / std::mem::size_of::<u64>());
        crate::ntt3x42_ifma::vmp::vmp_apply_dft_to_dft_digits_strided_ifma(module, res, a, dsize, pmat, tmp);
    }
}

impl_glwe_automorphism_defaults_full!(FFT64Avx512);
impl_glwe_automorphism_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_glwe_automorphism_defaults_full!(NTT3x42Ifma);

impl_ggsw_automorphism_defaults_full!(FFT64Avx512);
impl_ggsw_automorphism_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_ggsw_automorphism_defaults_full!(NTT3x42Ifma);

impl_gglwe_automorphism_defaults_full!(FFT64Avx512);
impl_gglwe_automorphism_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_gglwe_automorphism_defaults_full!(NTT3x42Ifma);

impl_decryption_defaults_full!(FFT64Avx512);
impl_decryption_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_decryption_defaults_full!(NTT3x42Ifma);

impl_glwe_trace_defaults_full!(FFT64Avx512);
impl_glwe_trace_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_glwe_trace_defaults_full!(NTT3x42Ifma);

impl_glwe_packing_defaults_full!(FFT64Avx512);
impl_glwe_packing_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_glwe_packing_defaults_full!(NTT3x42Ifma);

impl_conversion_defaults_full!(FFT64Avx512);
impl_conversion_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_conversion_defaults_full!(NTT3x42Ifma);

impl_glwe_keyswitch_defaults_full!(FFT64Avx512);
impl_glwe_keyswitch_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_glwe_keyswitch_defaults_full!(NTT3x42Ifma);

impl_gglwe_keyswitch_defaults_full!(FFT64Avx512);
impl_gglwe_keyswitch_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_gglwe_keyswitch_defaults_full!(NTT3x42Ifma);

impl_ggsw_keyswitch_defaults_full!(FFT64Avx512);
impl_ggsw_keyswitch_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_ggsw_keyswitch_defaults_full!(NTT3x42Ifma);

impl_lwe_keyswitch_defaults_full!(FFT64Avx512);
impl_lwe_keyswitch_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_lwe_keyswitch_defaults_full!(NTT3x42Ifma);

impl_encryption_defaults_full!(FFT64Avx512);
impl_encryption_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_encryption_defaults_full!(NTT3x42Ifma);

impl_glwe_external_product_defaults_full!(FFT64Avx512);
impl_glwe_external_product_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_glwe_external_product_defaults_full!(NTT3x42Ifma);

impl_gglwe_external_product_defaults_full!(FFT64Avx512);
impl_gglwe_external_product_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_gglwe_external_product_defaults_full!(NTT3x42Ifma);

impl_ggsw_external_product_defaults_full!(FFT64Avx512);
impl_ggsw_external_product_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_ggsw_external_product_defaults_full!(NTT3x42Ifma);

impl_linear_transformation_defaults_full!(FFT64Avx512);
impl_linear_transformation_defaults_full!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_linear_transformation_defaults_full!(NTT3x42Ifma);
