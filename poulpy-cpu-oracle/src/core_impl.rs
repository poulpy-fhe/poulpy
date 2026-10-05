//! Registers the generic Core compositions for the oracles.

use crate::{FFT64CIOracle, FFT64Oracle, NTT4x30CIOracle, NTT4x30Oracle};

/// The compositions shared by both rings.
macro_rules! impl_oracle_core {
    ($be:ty) => {
        ::poulpy_core::impl_glwe_tensoring_reference!($be);
        ::poulpy_core::impl_gglwe_product_digits_strided_reference!($be);
        ::poulpy_core::impl_automorphism_reference_full!($be);
        ::poulpy_core::impl_decryption_reference_full!($be);
        ::poulpy_core::impl_ggsw_conversion_reference_full!($be);
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
    };
}

/// The compositions that need monomial multiplication, on the standard ring only.
macro_rules! impl_oracle_core_standard {
    ($be:ty) => {
        ::poulpy_core::impl_conversion_reference_full!($be);
        ::poulpy_core::impl_glwe_packing_derived_full!($be);
        ::poulpy_core::impl_glwe_rotate_reference_full!($be);
        ::poulpy_core::impl_ggsw_rotate_derived_full!($be);
        ::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!($be);
        ::poulpy_core::impl_glwe_trace_derived_full!($be);
    };
}

impl_oracle_core!(FFT64Oracle);
impl_oracle_core_standard!(FFT64Oracle);
::poulpy_core::impl_glwe_ci_conversion_reference_full!(FFT64Oracle);
impl_oracle_core!(NTT4x30Oracle);
impl_oracle_core_standard!(NTT4x30Oracle);
::poulpy_core::impl_glwe_ci_conversion_reference_full!(NTT4x30Oracle);
impl_oracle_core!(FFT64CIOracle);
impl_oracle_core!(NTT4x30CIOracle);
