use crate::{FFT64Avx, NTT4x30Avx};
use poulpy_core::default::keyswitching::glwe::with_bound_pmat;
use poulpy_core::layouts::{GGLWEActiveUse, GGLWEInfos, prepared::GGLWEPreparedBound};
use poulpy_core::{
    impl_conversion_defaults_full, impl_decryption_defaults_full, impl_encryption_defaults_full,
    impl_gglwe_automorphism_defaults_full, impl_gglwe_external_product_defaults_full, impl_gglwe_keyswitch_defaults_full,
    impl_gglwe_product_bound_default, impl_ggsw_automorphism_defaults_full, impl_ggsw_external_product_defaults_full,
    impl_ggsw_keyswitch_defaults_full, impl_glwe_automorphism_defaults_full, impl_glwe_external_product_defaults_full,
    impl_glwe_keyswitch_defaults_full, impl_glwe_packing_defaults_full, impl_glwe_tensoring_default,
    impl_glwe_trace_defaults_full, impl_linear_transformation_defaults_full, impl_lwe_keyswitch_defaults_full,
};
use poulpy_hal::api::VmpPMatBytesOf;
use poulpy_hal::layouts::{Module, ScratchArena, VecZnxDftBackendMut, VecZnxDftBackendRef};

impl_glwe_tensoring_default!(FFT64Avx);
impl_glwe_tensoring_default!(NTT4x30Avx);
impl_gglwe_product_bound_default!(FFT64Avx);

unsafe impl poulpy_core::oep::GGLWEProductBoundImpl<NTT4x30Avx> for NTT4x30Avx {
    fn gglwe_product_bound_tmp_bytes(
        module: &Module<Self>,
        _res_size: usize,
        a_cols: usize,
        a_size: usize,
        use_: &GGLWEActiveUse,
    ) -> usize {
        let logical = &use_.logical_layout;
        let kernel = crate::ntt4x30::vmp::vmp_apply_digits_strided_tmp_bytes_avx(
            a_cols,
            a_size,
            logical.dsize().as_usize(),
            logical.dnum().as_usize(),
            logical.rank_in().as_usize(),
        );
        if use_.is_dense() {
            return kernel;
        }
        // The bound is materialized before the kernel sees it.
        module.bytes_of_vmp_pmat(
            logical.dnum().as_usize(),
            logical.rank_in().as_usize(),
            (logical.rank_out() + 1).as_usize(),
            use_.logical_work_size,
        ) + kernel
    }

    fn gglwe_product_bound(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        bound: &GGLWEPreparedBound<'_, Self>,
        product_limbs: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let dsize = bound.use_().logical_layout.dsize().as_usize();
        with_bound_pmat(module, bound, scratch, |pmat, scratch| {
            let bytes = crate::ntt4x30::vmp::vmp_apply_digits_strided_tmp_bytes_avx(
                a.cols(),
                a.size(),
                dsize,
                pmat.rows(),
                pmat.cols_in(),
            );
            let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / std::mem::size_of::<u64>());
            crate::ntt4x30::vmp::vmp_apply_dft_to_dft_digits_strided_avx(module, res, a, dsize, product_limbs, pmat, tmp);
        });
        let _ = res;
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
