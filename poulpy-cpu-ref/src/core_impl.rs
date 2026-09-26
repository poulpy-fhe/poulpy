use super::{FFT64Ref, NTT4x30Ref};
use poulpy_core::{impl_gglwe_product_digits_strided_reference, impl_glwe_tensoring_reference};

impl_glwe_tensoring_reference!(FFT64Ref);
impl_glwe_tensoring_reference!(NTT4x30Ref);
impl_gglwe_product_digits_strided_reference!(FFT64Ref);
impl_gglwe_product_digits_strided_reference!(NTT4x30Ref);

crate::impl_cpu_core_defaults!(super::FFT64Ref, fft64);
crate::impl_cpu_core_defaults!(super::NTT4x30Ref, ntt4x30);
