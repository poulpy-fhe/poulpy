use super::{FFT64CINeon, FFT64Neon, NTT4x30CINeon, NTT4x30Neon};
#[cfg(feature = "enable-rayon")]
use super::{FFT64CINeonRayon, FFT64NeonRayon, NTT4x30CINeonRayon, NTT4x30NeonRayon};
use poulpy_core::{impl_gglwe_product_digits_strided_reference, impl_glwe_tensoring_reference};

impl_glwe_tensoring_reference!(FFT64Neon);
impl_glwe_tensoring_reference!(NTT4x30Neon);
impl_gglwe_product_digits_strided_reference!(FFT64Neon);
impl_gglwe_product_digits_strided_reference!(NTT4x30Neon);
impl_glwe_tensoring_reference!(FFT64CINeon);
impl_glwe_tensoring_reference!(NTT4x30CINeon);
impl_gglwe_product_digits_strided_reference!(FFT64CINeon);
impl_gglwe_product_digits_strided_reference!(NTT4x30CINeon);

#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference!(FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference!(NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(FFT64CINeonRayon);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(NTT4x30CINeonRayon);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference!(FFT64CINeonRayon);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference!(NTT4x30CINeonRayon);

poulpy_cpu_portable::impl_cpu_core_defaults!(super::FFT64Neon, fft64);
::poulpy_core::impl_conversion_reference_full!(super::FFT64Neon);
::poulpy_core::impl_glwe_packing_derived_full!(super::FFT64Neon);
::poulpy_core::impl_glwe_rotate_reference_full!(super::FFT64Neon);
::poulpy_core::impl_ggsw_rotate_derived_full!(super::FFT64Neon);
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::FFT64Neon);
::poulpy_core::impl_glwe_trace_derived_full!(super::FFT64Neon);
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT4x30Neon, ntt4x30);
::poulpy_core::impl_conversion_reference_full!(super::NTT4x30Neon);
::poulpy_core::impl_glwe_packing_derived_full!(super::NTT4x30Neon);
::poulpy_core::impl_glwe_rotate_reference_full!(super::NTT4x30Neon);
::poulpy_core::impl_ggsw_rotate_derived_full!(super::NTT4x30Neon);
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::NTT4x30Neon);
::poulpy_core::impl_glwe_trace_derived_full!(super::NTT4x30Neon);
poulpy_cpu_portable::impl_cpu_core_defaults!(super::FFT64CINeon, fft64);
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::FFT64CINeon);
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT4x30CINeon, ntt4x30);
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::NTT4x30CINeon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_core_defaults!(super::FFT64NeonRayon, fft64);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_conversion_reference_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_packing_derived_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_rotate_reference_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_ggsw_rotate_derived_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_trace_derived_full!(super::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT4x30NeonRayon, ntt4x30);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_conversion_reference_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_packing_derived_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_rotate_reference_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_ggsw_rotate_derived_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_trace_derived_full!(super::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_core_defaults!(super::FFT64CINeonRayon, fft64);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::FFT64CINeonRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_portable::impl_cpu_core_defaults!(super::NTT4x30CINeonRayon, ntt4x30);
#[cfg(feature = "enable-rayon")]
::poulpy_core::impl_glwe_ci_conversion_reference_full!(super::NTT4x30CINeonRayon);
