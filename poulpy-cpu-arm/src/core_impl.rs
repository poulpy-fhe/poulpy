use crate::{FFT64Neon, NTT4x30Neon};
#[cfg(feature = "enable-rayon")]
use crate::{FFT64NeonRayon, NTT4x30NeonRayon};
use poulpy_core::{impl_gglwe_product_digits_strided_reference, impl_glwe_tensoring_reference};

impl_glwe_tensoring_reference!(FFT64Neon);
impl_glwe_tensoring_reference!(crate::FFT64CINeon);
impl_glwe_tensoring_reference!(NTT4x30Neon);
impl_glwe_tensoring_reference!(crate::NTT4x30CINeon);
impl_gglwe_product_digits_strided_reference!(FFT64Neon);
impl_gglwe_product_digits_strided_reference!(crate::FFT64CINeon);
impl_gglwe_product_digits_strided_reference!(NTT4x30Neon);
impl_gglwe_product_digits_strided_reference!(crate::NTT4x30CINeon);

#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(crate::FFT64CINeonRayon);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_glwe_tensoring_reference!(crate::NTT4x30CINeonRayon);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference!(FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference!(crate::FFT64CINeonRayon);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference!(NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_gglwe_product_digits_strided_reference!(crate::NTT4x30CINeonRayon);

poulpy_cpu_ref::impl_cpu_core_defaults!(crate::FFT64Neon, fft64);
poulpy_cpu_ref::impl_cpu_core_defaults!(crate::FFT64CINeon, fft64);
poulpy_cpu_ref::impl_cpu_core_defaults!(crate::NTT4x30Neon, ntt4x30);
poulpy_cpu_ref::impl_cpu_core_defaults!(crate::NTT4x30CINeon, ntt4x30);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_core_defaults!(crate::FFT64NeonRayon, fft64);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_core_defaults!(crate::FFT64CINeonRayon, fft64);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_core_defaults!(crate::NTT4x30NeonRayon, ntt4x30);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_core_defaults!(crate::NTT4x30CINeonRayon, ntt4x30);
