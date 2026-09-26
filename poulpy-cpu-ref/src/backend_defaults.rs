/// Registers shared Core defaults and host sampling for a CPU backend.
/// Tensoring and strided digit products are registered separately to allow optimized kernels.
#[macro_export]
macro_rules! impl_cpu_core_defaults {
    ($be:ty, $word_family:ident) => {
        ::poulpy_core::impl_automorphism_reference_full!($be);
        ::poulpy_core::impl_decryption_reference_full!($be);
        ::poulpy_core::impl_glwe_trace_derived_full!($be);
        ::poulpy_core::impl_glwe_packing_derived_full!($be);
        ::poulpy_core::impl_conversion_reference_full!($be);
        ::poulpy_core::impl_glwe_keyswitch_reference_full!($be);
        ::poulpy_core::impl_gglwe_keyswitch_derived_full!($be);
        ::poulpy_core::impl_ggsw_keyswitch_derived_full!($be);
        ::poulpy_core::impl_lwe_keyswitch_reference_full!($be);
        ::poulpy_core::impl_encryption_reference_full!($be);
        ::poulpy_core::impl_glwe_external_product_reference_full!($be);
        ::poulpy_core::impl_gglwe_external_product_derived_full!($be);
        ::poulpy_core::impl_ggsw_external_product_derived_full!($be);
        ::poulpy_core::impl_linear_transformation_reference_full!($be);
        ::poulpy_core::impl_operations_reference_full!($be);
        ::poulpy_core::impl_polynomial_evaluation_derived_full!($be);
        $crate::impl_sampling_host!($be, $word_family);
    };
}

/// Registers shared CKKS defaults and encoding for a CPU backend.
/// The backend selects its encoding transform and encapsulated ModUp implementation separately.
#[cfg(feature = "enable-ckks")]
#[macro_export]
macro_rules! impl_cpu_ckks_defaults {
    ($be:ty) => {
        ::poulpy_ckks::impl_ckks_conjugate_reference!($be);
        ::poulpy_ckks::impl_ckks_copy_reference!($be);
        ::poulpy_ckks::impl_ckks_encryption_reference!($be);
        ::poulpy_ckks::impl_ckks_imag_reference!($be);
        ::poulpy_ckks::impl_ckks_mul_reference!($be);
        ::poulpy_ckks::impl_ckks_neg_reference!($be);
        ::poulpy_ckks::impl_ckks_pow2_reference!($be);
        ::poulpy_ckks::impl_ckks_rotate_reference!($be);
        ::poulpy_ckks::impl_ckks_add_reference!($be);
        ::poulpy_ckks::impl_ckks_sub_reference!($be);
        ::poulpy_ckks::impl_ckks_plaintext_reference!($be);
        ::poulpy_ckks::impl_ckks_dft_reference!($be);
        ::poulpy_ckks::impl_ckks_eval_mod_reference!($be);
        ::poulpy_ckks::impl_ckks_polynomial_evaluation_reference!($be);
        $crate::impl_ckks_encoding!($be);
        $crate::impl_ckks_paco_coeff_encoding!($be);
        $crate::impl_ckks_ship_coeff_encoding!($be);
    };
}
