#[cfg(feature = "enable-rayon")]
use super::FFT64Avx512Rayon;
#[cfg(feature = "enable-ifma")]
use super::NTT3x42Ifma;
use super::{FFT64Avx512, NTT4x30Avx512};
use poulpy_ckks::impl_ckks_encapsulated_mod_up_reference;

impl_ckks_encapsulated_mod_up_reference!(FFT64Avx512);
#[cfg(feature = "enable-rayon")]
impl_ckks_encapsulated_mod_up_reference!(FFT64Avx512Rayon);

// `f64` encodes through the AVX-512 kernels; `Quad` has no accelerated
// transform and falls back to the generic scalar table. Rust has no
// specialization, so accelerated backends list their precisions explicitly.
macro_rules! select_avx512_encoding_transform {
    ($be:ty) => {
        impl ::poulpy_cpu_ref::ckks_encoding::CKKSEncodingTransform<f64> for $be {
            type Fft = super::FFT64Avx512ReimTable;
        }

        impl ::poulpy_cpu_ref::ckks_encoding::CKKSEncodingTransform<poulpy_ckks::Quad> for $be {
            type Fft = ::poulpy_cpu_ref::FFT64ReimTable<poulpy_ckks::Quad>;
        }
    };
}

select_avx512_encoding_transform!(FFT64Avx512);
select_avx512_encoding_transform!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
select_avx512_encoding_transform!(NTT3x42Ifma);

#[cfg(feature = "enable-rayon")]
select_avx512_encoding_transform!(FFT64Avx512Rayon);

poulpy_cpu_ref::impl_cpu_ckks_defaults!(super::FFT64Avx512);
poulpy_cpu_ref::impl_cpu_ckks_defaults!(super::NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(super::NTT3x42Ifma);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(super::FFT64Avx512Rayon);
#[cfg(feature = "enable-rayon")]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(super::NTT4x30Avx512Rayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
poulpy_cpu_ref::impl_cpu_ckks_defaults!(super::NTT3x42IfmaRayon);
#[cfg(feature = "enable-rayon")]
select_avx512_encoding_transform!(super::NTT4x30Avx512Rayon);
#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
select_avx512_encoding_transform!(super::NTT3x42IfmaRayon);
