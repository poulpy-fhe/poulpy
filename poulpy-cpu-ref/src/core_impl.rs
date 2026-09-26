use super::{FFT64CIRef, FFT64Ref, NTT4x30CIRef, NTT4x30Ref};
use poulpy_core::{impl_gglwe_product_digits_strided_reference, impl_glwe_tensoring_reference};

impl_glwe_tensoring_reference!(FFT64Ref);
impl_glwe_tensoring_reference!(NTT4x30Ref);
impl_glwe_tensoring_reference!(FFT64CIRef);
impl_glwe_tensoring_reference!(NTT4x30CIRef);
impl_gglwe_product_digits_strided_reference!(FFT64Ref);
impl_gglwe_product_digits_strided_reference!(NTT4x30Ref);
impl_gglwe_product_digits_strided_reference!(FFT64CIRef);
impl_gglwe_product_digits_strided_reference!(NTT4x30CIRef);

crate::impl_cpu_core_defaults!(super::FFT64Ref, fft64);
::poulpy_core::impl_conversion_reference_full!(super::FFT64Ref);
::poulpy_core::impl_glwe_packing_derived_full!(super::FFT64Ref);
::poulpy_core::impl_glwe_rotate_reference_full!(super::FFT64Ref);
::poulpy_core::impl_ggsw_rotate_derived_full!(super::FFT64Ref);
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::FFT64Ref);
::poulpy_core::impl_glwe_trace_derived_full!(super::FFT64Ref);
crate::impl_cpu_core_defaults!(super::NTT4x30Ref, ntt4x30);
::poulpy_core::impl_conversion_reference_full!(super::NTT4x30Ref);
::poulpy_core::impl_glwe_packing_derived_full!(super::NTT4x30Ref);
::poulpy_core::impl_glwe_rotate_reference_full!(super::NTT4x30Ref);
::poulpy_core::impl_ggsw_rotate_derived_full!(super::NTT4x30Ref);
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::NTT4x30Ref);
::poulpy_core::impl_glwe_trace_derived_full!(super::NTT4x30Ref);
crate::impl_cpu_core_defaults!(super::FFT64CIRef, fft64);
crate::impl_cpu_core_defaults!(super::NTT4x30CIRef, ntt4x30);
