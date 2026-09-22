use crate::{FFT64Neon, NTT4x30Neon};
use poulpy_ckks::impl_ckks_encapsulated_mod_up_reference;

impl_ckks_encapsulated_mod_up_reference!(FFT64Neon);
impl_ckks_encapsulated_mod_up_reference!(crate::FFT64CINeon);
impl_ckks_encapsulated_mod_up_reference!(NTT4x30Neon);
impl_ckks_encapsulated_mod_up_reference!(crate::NTT4x30CINeon);
// `f64` encodes through the NEON kernels; `Quad` has no accelerated transform
// and falls back to the generic scalar table. Rust has no specialization, so
// accelerated backends list their precisions explicitly.
macro_rules! select_neon_encoding_transform {
    ($be:ty) => {
        impl ::poulpy_cpu_ref::ckks_encoding::CKKSEncodingTransform<f64> for $be {
            type Fft = crate::FFT64NeonReimTable;
        }

        impl ::poulpy_cpu_ref::ckks_encoding::CKKSEncodingTransform<poulpy_ckks::Quad> for $be {
            type Fft = ::poulpy_cpu_ref::FFT64ReimTable<poulpy_ckks::Quad>;
        }
    };
}

select_neon_encoding_transform!(FFT64Neon);
select_neon_encoding_transform!(crate::FFT64CINeon);
select_neon_encoding_transform!(NTT4x30Neon);
select_neon_encoding_transform!(crate::NTT4x30CINeon);

poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::FFT64Neon);
poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::FFT64CINeon);
poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::NTT4x30Neon);
poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::NTT4x30CINeon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::FFT64CINeonRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::NTT4x30CINeonRayon);
#[cfg(feature = "enable-rayon")]
select_neon_encoding_transform!(crate::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_encapsulated_mod_up_reference!(crate::FFT64NeonRayon);
#[cfg(feature = "enable-rayon")]
select_neon_encoding_transform!(crate::FFT64CINeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_encapsulated_mod_up_reference!(crate::FFT64CINeonRayon);
#[cfg(feature = "enable-rayon")]
select_neon_encoding_transform!(crate::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_encapsulated_mod_up_reference!(crate::NTT4x30NeonRayon);
#[cfg(feature = "enable-rayon")]
select_neon_encoding_transform!(crate::NTT4x30CINeonRayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_encapsulated_mod_up_reference!(crate::NTT4x30CINeonRayon);
