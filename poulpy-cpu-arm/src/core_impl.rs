use super::{FFT64Neon, NTT4x30Neon};
#[cfg(feature = "enable-rayon")]
use super::{FFT64NeonRayon, NTT4x30NeonRayon};
use poulpy_core::{impl_gglwe_product_digits_strided_reference, impl_glwe_tensoring_reference};

impl_glwe_tensoring_reference!(FFT64Neon);
impl_glwe_tensoring_reference!(NTT4x30Neon);
impl_gglwe_product_digits_strided_reference!(FFT64Neon);
impl_gglwe_product_digits_strided_reference!(NTT4x30Neon);

#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference!(FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference!(NTT4x30NeonRayon);

poulpy_cpu_ref::impl_cpu_core_defaults!(super::FFT64Neon, fft64);
poulpy_cpu_ref::impl_cpu_core_defaults!(super::NTT4x30Neon, ntt4x30);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_core_defaults!(super::FFT64NeonRayon, fft64);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_core_defaults!(super::NTT4x30NeonRayon, ntt4x30);
