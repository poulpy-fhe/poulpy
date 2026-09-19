use crate::{FFT64Avx, NTT4x30Avx};
#[cfg(feature = "enable-rayon")]
use crate::{FFT64AvxRayon, NTT4x30AvxRayon};
use poulpy_core::{
    impl_conversion_reference_full, impl_decryption_reference_full, impl_encryption_reference_full,
    impl_gglwe_automorphism_reference_full, impl_gglwe_product_digits_strided_reference, impl_glwe_automorphism_reference_full,
    impl_glwe_external_product_reference_full, impl_glwe_keyswitch_reference_full, impl_glwe_packing_reference_full,
    impl_glwe_tensoring_reference, impl_glwe_trace_reference_full, impl_linear_transformation_reference_full,
    impl_lwe_keyswitch_reference_full,
};
use poulpy_hal::layouts::{Module, ScratchArena, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendRef};

impl_glwe_tensoring_reference!(FFT64Avx);
impl_glwe_tensoring_reference!(NTT4x30Avx);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(FFT64AvxRayon);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(NTT4x30AvxRayon);
impl_gglwe_product_digits_strided_reference!(FFT64Avx);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference!(FFT64AvxRayon);

unsafe impl poulpy_core::oep::GGLWEProductDigitsStridedImpl for NTT4x30Avx {
    fn gglwe_product_digits_strided_tmp_bytes(
        _module: &Module<Self>,
        _res_size: usize,
        a_cols: usize,
        a_size: usize,
        dsize: usize,
        pmat_rows: usize,
        pmat_cols_in: usize,
        _pmat_cols_out: usize,
        _pmat_size: usize,
    ) -> usize {
        crate::ntt4x30::vmp::vmp_apply_digits_strided_tmp_bytes_avx(a_cols, a_size, dsize, pmat_rows, pmat_cols_in, 1)
    }

    fn gglwe_product_digits_strided(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        dsize: usize,
        product_limbs: usize,
        pmat: &VmpPMatBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let bytes = Self::gglwe_product_digits_strided_tmp_bytes(
            module,
            res.size(),
            a.cols(),
            a.size(),
            dsize,
            pmat.rows(),
            pmat.cols_in(),
            pmat.cols_out(),
            pmat.size(),
        );
        let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / std::mem::size_of::<u64>());
        crate::ntt4x30::vmp::vmp_apply_dft_to_dft_digits_strided_avx::<poulpy_hal::execution::SerialTaskExecutor>(
            module,
            res,
            a,
            dsize,
            product_limbs,
            pmat,
            tmp,
        );
    }
}

impl_glwe_automorphism_reference_full!(FFT64Avx);
impl_glwe_automorphism_reference_full!(NTT4x30Avx);

impl_gglwe_automorphism_reference_full!(FFT64Avx);
impl_gglwe_automorphism_reference_full!(NTT4x30Avx);

impl_decryption_reference_full!(FFT64Avx);
impl_decryption_reference_full!(NTT4x30Avx);
impl_glwe_trace_reference_full!(FFT64Avx);
impl_glwe_trace_reference_full!(NTT4x30Avx);
impl_glwe_packing_reference_full!(FFT64Avx);
impl_glwe_packing_reference_full!(NTT4x30Avx);

impl_conversion_reference_full!(FFT64Avx);
impl_conversion_reference_full!(NTT4x30Avx);

impl_glwe_keyswitch_reference_full!(FFT64Avx);
impl_glwe_keyswitch_reference_full!(NTT4x30Avx);
impl_lwe_keyswitch_reference_full!(FFT64Avx);
impl_lwe_keyswitch_reference_full!(NTT4x30Avx);

impl_encryption_reference_full!(FFT64Avx);
poulpy_cpu_ref::impl_sampling_host!(FFT64Avx, fft64);
impl_encryption_reference_full!(NTT4x30Avx);
poulpy_cpu_ref::impl_sampling_host!(NTT4x30Avx, ntt4x30);

impl_glwe_external_product_reference_full!(FFT64Avx);
impl_glwe_external_product_reference_full!(NTT4x30Avx);

impl_linear_transformation_reference_full!(FFT64Avx);
impl_linear_transformation_reference_full!(NTT4x30Avx);

#[cfg(feature = "enable-rayon")]
mod rayon_defaults {
    use super::*;

    impl_glwe_automorphism_reference_full!(FFT64AvxRayon);
    impl_gglwe_automorphism_reference_full!(FFT64AvxRayon);
    impl_decryption_reference_full!(FFT64AvxRayon);
    impl_glwe_trace_reference_full!(FFT64AvxRayon);
    impl_glwe_packing_reference_full!(FFT64AvxRayon);
    impl_conversion_reference_full!(FFT64AvxRayon);
    impl_glwe_keyswitch_reference_full!(FFT64AvxRayon);
    impl_lwe_keyswitch_reference_full!(FFT64AvxRayon);
    impl_encryption_reference_full!(FFT64AvxRayon);
    poulpy_cpu_ref::impl_sampling_host!(FFT64AvxRayon, fft64);
    impl_glwe_external_product_reference_full!(FFT64AvxRayon);
    impl_linear_transformation_reference_full!(FFT64AvxRayon);

    impl_glwe_automorphism_reference_full!(NTT4x30AvxRayon);
    impl_gglwe_automorphism_reference_full!(NTT4x30AvxRayon);
    impl_decryption_reference_full!(NTT4x30AvxRayon);
    impl_glwe_trace_reference_full!(NTT4x30AvxRayon);
    impl_glwe_packing_reference_full!(NTT4x30AvxRayon);
    impl_conversion_reference_full!(NTT4x30AvxRayon);
    impl_glwe_keyswitch_reference_full!(NTT4x30AvxRayon);
    impl_lwe_keyswitch_reference_full!(NTT4x30AvxRayon);
    impl_encryption_reference_full!(NTT4x30AvxRayon);
    poulpy_cpu_ref::impl_sampling_host!(NTT4x30AvxRayon, ntt4x30);
    impl_glwe_external_product_reference_full!(NTT4x30AvxRayon);
    impl_linear_transformation_reference_full!(NTT4x30AvxRayon);
}
