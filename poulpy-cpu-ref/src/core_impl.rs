use crate::{FFT64CIRef, FFT64Ref, NTT4x30CIRef, NTT4x30Ref};
use poulpy_core::{impl_gglwe_product_digits_strided_reference, impl_glwe_tensoring_reference};

impl_glwe_tensoring_reference!(FFT64Ref);
impl_glwe_tensoring_reference!(FFT64CIRef);
impl_glwe_tensoring_reference!(NTT4x30Ref);
impl_glwe_tensoring_reference!(NTT4x30CIRef);
impl_gglwe_product_digits_strided_reference!(FFT64Ref);
impl_gglwe_product_digits_strided_reference!(FFT64CIRef);
impl_gglwe_product_digits_strided_reference!(NTT4x30Ref);
impl_gglwe_product_digits_strided_reference!(NTT4x30CIRef);

crate::impl_cpu_core_defaults!(crate::FFT64Ref, fft64);
crate::impl_cpu_core_defaults!(crate::FFT64CIRef, fft64);
crate::impl_cpu_core_defaults!(crate::NTT4x30Ref, ntt4x30);
crate::impl_cpu_core_defaults!(crate::NTT4x30CIRef, ntt4x30);
