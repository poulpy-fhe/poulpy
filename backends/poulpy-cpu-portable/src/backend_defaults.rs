/// Registers shared Core defaults and host sampling for a CPU backend.
/// Tensoring and strided digit products are registered separately to allow optimized kernels.
///
/// `impl_cpu_core_defaults!(Backend, fft64 | ntt4x30)` registers both rings, and
/// `impl_cpu_core_defaults!(Backend, Ring, fft64 | ntt4x30)` registers `Ring` only,
/// for a backend that writes a family of the other ring by hand.
#[macro_export]
macro_rules! impl_cpu_core_defaults {
    ($be:ty, $word_family:ident) => {
        $crate::impl_cpu_core_defaults!($be, ::poulpy_hal::layouts::Standard, $word_family);
        $crate::impl_cpu_core_defaults!($be, ::poulpy_hal::layouts::ConjugateInvariant, $word_family);
    };
    ($be:ty, $ring:ty, $word_family:ident) => {
        ::poulpy_core::impl_automorphism_reference_full!($be, $ring);
        ::poulpy_core::impl_decompression_reference_full!($be, $ring);
        ::poulpy_core::impl_decryption_reference_full!($be, $ring);
        ::poulpy_core::impl_ggsw_conversion_reference_full!($be, $ring);
        ::poulpy_core::impl_glwe_keyswitch_reference_full!($be, $ring);
        ::poulpy_core::impl_gglwe_keyswitch_derived_full!($be, $ring);
        ::poulpy_core::impl_ggsw_keyswitch_derived_full!($be, $ring);
        ::poulpy_core::impl_encryption_reference_full!($be, $ring);
        ::poulpy_core::impl_glwe_external_product_reference_full!($be, $ring);
        ::poulpy_core::impl_gglwe_external_product_derived_full!($be, $ring);
        ::poulpy_core::impl_ggsw_external_product_derived_full!($be, $ring);
        ::poulpy_core::impl_glwe_linear_transformation_reference_full!($be, $ring);
        ::poulpy_core::impl_operations_reference_full!($be, $ring);
        ::poulpy_core::impl_glwe_polynomial_evaluation_derived_full!($be, $ring);
        ::poulpy_core::impl_preparation_reference_full!($be, $ring);
        ::poulpy_core::impl_secret_reference_full!($be, $ring);
        ::poulpy_core::impl_glwe_trace_derived_full!($be, $ring);
        $crate::impl_sampling_host!($be, $ring, $word_family);
    };
}

/// Registers the CKKS families shared by both rings (`impl_ckks_reference_full!`), with encoding, for a CPU backend.
/// The backend selects its encoding transform separately and registers the standard-only and cross-ring families
/// (the ring maps, the fold) itself.
///
/// `impl_cpu_ckks_defaults!(Backend)` registers both rings, and `impl_cpu_ckks_defaults!(Backend, Ring)` registers `Ring`
/// only, for a backend that writes a family of the other ring by hand.
#[cfg(feature = "enable-ckks")]
#[macro_export]
macro_rules! impl_cpu_ckks_defaults {
    ($be:ty) => {
        $crate::impl_cpu_ckks_defaults!($be, ::poulpy_hal::layouts::Standard);
        $crate::impl_cpu_ckks_defaults!($be, ::poulpy_hal::layouts::ConjugateInvariant);
    };
    ($be:ty, $ring:ty) => {
        ::poulpy_ckks::impl_ckks_reference_full!($be, $ring);
        $crate::impl_ckks_encoding!($be, $ring);
    };
}
