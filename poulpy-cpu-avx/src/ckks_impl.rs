use crate::{FFT64Avx, NTT4x30Avx};
use poulpy_ckks::impl_ckks_encapsulated_mod_up_reference;

impl_ckks_encapsulated_mod_up_reference!(FFT64Avx);
impl_ckks_encapsulated_mod_up_reference!(crate::FFT64CIAvx);
// `f64` encodes through the AVX2/FMA kernels; `Quad` has no accelerated
// transform and falls back to the generic scalar table. Rust has no
// specialization, so accelerated backends list their precisions explicitly.
macro_rules! select_avx_encoding_transform {
    ($be:ty) => {
        impl ::poulpy_cpu_ref::ckks_encoding::CKKSEncodingTransform<f64> for $be {
            type Fft = crate::FFT64AvxReimTable;
        }

        impl ::poulpy_cpu_ref::ckks_encoding::CKKSEncodingTransform<poulpy_ckks::Quad> for $be {
            type Fft = ::poulpy_cpu_ref::FFT64ReimTable<poulpy_ckks::Quad>;
        }
    };
}

select_avx_encoding_transform!(FFT64Avx);
select_avx_encoding_transform!(crate::FFT64CIAvx);
select_avx_encoding_transform!(NTT4x30Avx);
select_avx_encoding_transform!(crate::NTT4x30CIAvx);

poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::FFT64Avx);
poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::FFT64CIAvx);
poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::NTT4x30Avx);
poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::NTT4x30CIAvx);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::FFT64AvxRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::FFT64CIAvxRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::NTT4x30AvxRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::NTT4x30CIAvxRayon);
#[cfg(feature = "enable-rayon")]
select_avx_encoding_transform!(crate::FFT64AvxRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_encapsulated_mod_up_reference!(crate::FFT64AvxRayon);
#[cfg(feature = "enable-rayon")]
select_avx_encoding_transform!(crate::FFT64CIAvxRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_encapsulated_mod_up_reference!(crate::FFT64CIAvxRayon);
#[cfg(feature = "enable-rayon")]
select_avx_encoding_transform!(crate::NTT4x30AvxRayon);
#[cfg(feature = "enable-rayon")]
select_avx_encoding_transform!(crate::NTT4x30CIAvxRayon);
