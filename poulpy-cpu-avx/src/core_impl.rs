use crate::{FFT64Avx, NTT4x30Avx};
use poulpy_core::{
    impl_conversion_defaults_full, impl_decryption_defaults_full, impl_encryption_defaults_full,
    impl_gglwe_automorphism_defaults_full, impl_gglwe_external_product_defaults_full, impl_gglwe_keyswitch_defaults_full,
    impl_gglwe_product_digits_strided_default, impl_ggsw_automorphism_defaults_full, impl_ggsw_external_product_defaults_full,
    impl_ggsw_keyswitch_defaults_full, impl_glwe_automorphism_defaults_full, impl_glwe_external_product_defaults_full,
    impl_glwe_keyswitch_defaults_full, impl_glwe_packing_defaults_full, impl_glwe_tensor_rank1_dft_default,
    impl_glwe_trace_defaults_full, impl_linear_transformation_defaults_full, impl_lwe_keyswitch_defaults_full,
};
use poulpy_hal::layouts::{Module, ScratchArena, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendRef};

impl_glwe_tensor_rank1_dft_default!(FFT64Avx);
impl_glwe_tensor_rank1_dft_default!(NTT4x30Avx);
impl_gglwe_product_digits_strided_default!(FFT64Avx);

unsafe impl poulpy_core::oep::GGLWEProductDigitsStridedImpl<NTT4x30Avx> for NTT4x30Avx {
    fn gglwe_product_digits_strided(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        dsize: usize,
        pmat: &VmpPMatBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes =
            crate::ntt4x30::vmp::vmp_apply_digits_strided_tmp_bytes_avx(a.cols(), a.size(), dsize, pmat.rows(), pmat.cols_in());
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / std::mem::size_of::<u64>());
        crate::ntt4x30::vmp::vmp_apply_dft_to_dft_digits_strided_avx(module, res, a, dsize, pmat, tmp);
    }
}

impl_glwe_automorphism_defaults_full!(FFT64Avx);
impl_glwe_automorphism_defaults_full!(NTT4x30Avx);

impl_ggsw_automorphism_defaults_full!(FFT64Avx);
impl_ggsw_automorphism_defaults_full!(NTT4x30Avx);
impl_gglwe_automorphism_defaults_full!(FFT64Avx);
impl_gglwe_automorphism_defaults_full!(NTT4x30Avx);

impl_decryption_defaults_full!(FFT64Avx);
impl_decryption_defaults_full!(NTT4x30Avx);
impl_glwe_trace_defaults_full!(FFT64Avx);
impl_glwe_trace_defaults_full!(NTT4x30Avx);
impl_glwe_packing_defaults_full!(FFT64Avx);
impl_glwe_packing_defaults_full!(NTT4x30Avx);

impl_conversion_defaults_full!(FFT64Avx);
impl_conversion_defaults_full!(NTT4x30Avx);

impl_glwe_keyswitch_defaults_full!(FFT64Avx);
impl_glwe_keyswitch_defaults_full!(NTT4x30Avx);
impl_gglwe_keyswitch_defaults_full!(FFT64Avx);
impl_gglwe_keyswitch_defaults_full!(NTT4x30Avx);
impl_ggsw_keyswitch_defaults_full!(FFT64Avx);
impl_ggsw_keyswitch_defaults_full!(NTT4x30Avx);
impl_lwe_keyswitch_defaults_full!(FFT64Avx);
impl_lwe_keyswitch_defaults_full!(NTT4x30Avx);

impl_encryption_defaults_full!(FFT64Avx);
impl_encryption_defaults_full!(NTT4x30Avx);

impl_glwe_external_product_defaults_full!(FFT64Avx);
impl_glwe_external_product_defaults_full!(NTT4x30Avx);
impl_gglwe_external_product_defaults_full!(FFT64Avx);
impl_gglwe_external_product_defaults_full!(NTT4x30Avx);
impl_ggsw_external_product_defaults_full!(FFT64Avx);
impl_ggsw_external_product_defaults_full!(NTT4x30Avx);

impl_linear_transformation_defaults_full!(FFT64Avx);
impl_linear_transformation_defaults_full!(NTT4x30Avx);
