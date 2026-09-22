#[cfg(feature = "enable-rayon")]
use crate::FFT64Avx512Rayon;
#[cfg(feature = "enable-ifma")]
use crate::NTT3x42Ifma;
use crate::{FFT64Avx512, NTT4x30Avx512};
use poulpy_ckks::impl_ckks_encapsulated_mod_up_reference;

impl_ckks_encapsulated_mod_up_reference!(FFT64Avx512);
impl_ckks_encapsulated_mod_up_reference!(crate::FFT64CIAvx512);
#[cfg(feature = "enable-rayon")]
impl_ckks_encapsulated_mod_up_reference!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
impl_ckks_encapsulated_mod_up_reference!(crate::FFT64CIAvx512Rayon);

// `f64` encodes through the AVX-512 kernels; `Quad` has no accelerated
// transform and falls back to the generic scalar table. Rust has no
// specialization, so accelerated backends list their precisions explicitly.
macro_rules! select_avx512_encoding_transform {
    ($be:ty) => {
        impl ::poulpy_cpu_ref::ckks_encoding::CKKSEncodingTransform<f64> for $be {
            type Fft = crate::FFT64Avx512ReimTable;
        }

        impl ::poulpy_cpu_ref::ckks_encoding::CKKSEncodingTransform<poulpy_ckks::Quad> for $be {
            type Fft = ::poulpy_cpu_ref::FFT64ReimTable<poulpy_ckks::Quad>;
        }
    };
}

select_avx512_encoding_transform!(FFT64Avx512);
select_avx512_encoding_transform!(crate::FFT64CIAvx512);
select_avx512_encoding_transform!(NTT4x30Avx512);
select_avx512_encoding_transform!(crate::NTT4x30CIAvx512);
#[cfg(feature = "enable-ifma")]
select_avx512_encoding_transform!(NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
select_avx512_encoding_transform!(crate::NTT3x42CIIfma);

#[cfg(feature = "enable-rayon")]
select_avx512_encoding_transform!(FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
select_avx512_encoding_transform!(crate::FFT64CIAvx512Rayon);

poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::FFT64Avx512);
poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::FFT64CIAvx512);
poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::NTT4x30Avx512);
poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::NTT4x30CIAvx512);
#[cfg(feature = "enable-ifma")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::NTT3x42CIIfma);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::FFT64CIAvx512Rayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::NTT4x30Avx512Rayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::NTT4x30CIAvx512Rayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::NTT3x42IfmaRayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(crate::NTT3x42CIIfmaRayon);
#[cfg(feature = "enable-rayon")]
select_avx512_encoding_transform!(crate::NTT4x30Avx512Rayon);
#[cfg(feature = "enable-rayon")]
select_avx512_encoding_transform!(crate::NTT4x30CIAvx512Rayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
select_avx512_encoding_transform!(crate::NTT3x42IfmaRayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
select_avx512_encoding_transform!(crate::NTT3x42CIIfmaRayon);
