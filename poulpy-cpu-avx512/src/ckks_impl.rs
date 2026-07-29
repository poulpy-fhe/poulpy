#[cfg(feature = "enable-ifma")]
use crate::NTT3x42Ifma;
use crate::{FFT64Avx512, NTT4x30Avx512};
use poulpy_ckks::{
    impl_ckks_add_defaults, impl_ckks_bootstrapping_defaults, impl_ckks_conjugate_defaults, impl_ckks_copy_defaults,
    impl_ckks_dft_defaults, impl_ckks_encryption_defaults, impl_ckks_imag_defaults, impl_ckks_mul_defaults,
    impl_ckks_neg_defaults, impl_ckks_plaintext_defaults, impl_ckks_pow2_defaults, impl_ckks_rotate_defaults,
    impl_ckks_sub_defaults,
};

impl_ckks_conjugate_defaults!(FFT64Avx512);
impl_ckks_conjugate_defaults!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_ckks_conjugate_defaults!(NTT3x42Ifma);

impl_ckks_copy_defaults!(FFT64Avx512);
impl_ckks_copy_defaults!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_ckks_copy_defaults!(NTT3x42Ifma);

impl_ckks_encryption_defaults!(FFT64Avx512);
impl_ckks_encryption_defaults!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_ckks_encryption_defaults!(NTT3x42Ifma);

impl_ckks_imag_defaults!(FFT64Avx512);
impl_ckks_imag_defaults!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_ckks_imag_defaults!(NTT3x42Ifma);

impl_ckks_mul_defaults!(FFT64Avx512);
impl_ckks_mul_defaults!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_ckks_mul_defaults!(NTT3x42Ifma);

impl_ckks_neg_defaults!(FFT64Avx512);
impl_ckks_neg_defaults!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_ckks_neg_defaults!(NTT3x42Ifma);

impl_ckks_pow2_defaults!(FFT64Avx512);
impl_ckks_pow2_defaults!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_ckks_pow2_defaults!(NTT3x42Ifma);

impl_ckks_rotate_defaults!(FFT64Avx512);
impl_ckks_rotate_defaults!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_ckks_rotate_defaults!(NTT3x42Ifma);

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
select_avx512_encoding_transform!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
select_avx512_encoding_transform!(NTT3x42Ifma);

::poulpy_cpu_ref::impl_ckks_encoding!(FFT64Avx512);
::poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(FFT64Avx512);
::poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(FFT64Avx512);
::poulpy_cpu_ref::impl_ckks_encoding!(NTT4x30Avx512);
::poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(NTT4x30Avx512);
::poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
::poulpy_cpu_ref::impl_ckks_encoding!(NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
::poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(NTT3x42Ifma);
#[cfg(feature = "enable-ifma")]
::poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(NTT3x42Ifma);

impl_ckks_add_defaults!(FFT64Avx512);
impl_ckks_add_defaults!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_ckks_add_defaults!(NTT3x42Ifma);

impl_ckks_sub_defaults!(FFT64Avx512);
impl_ckks_sub_defaults!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_ckks_sub_defaults!(NTT3x42Ifma);

impl_ckks_plaintext_defaults!(FFT64Avx512);
impl_ckks_plaintext_defaults!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_ckks_plaintext_defaults!(NTT3x42Ifma);

impl_ckks_dft_defaults!(FFT64Avx512);
impl_ckks_dft_defaults!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_ckks_dft_defaults!(NTT3x42Ifma);

impl_ckks_bootstrapping_defaults!(FFT64Avx512);
impl_ckks_bootstrapping_defaults!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_ckks_bootstrapping_defaults!(NTT3x42Ifma);
