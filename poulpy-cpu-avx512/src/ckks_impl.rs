#[cfg(feature = "enable-ifma")]
use crate::NTT3x42Ifma;
use crate::{FFT64Avx512, NTT4x30Avx512};
use poulpy_ckks::{
    impl_ckks_add_defaults, impl_ckks_bootstrapping_defaults, impl_ckks_conjugate_defaults, impl_ckks_copy_defaults,
    impl_ckks_dft_defaults, impl_ckks_encryption_defaults, impl_ckks_imag_defaults, impl_ckks_mul_defaults,
    impl_ckks_neg_defaults, impl_ckks_plaintext_defaults, impl_ckks_pow2_defaults, impl_ckks_rotate_default,
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

impl_ckks_rotate_default!(FFT64Avx512);
impl_ckks_rotate_default!(NTT4x30Avx512);
#[cfg(feature = "enable-ifma")]
impl_ckks_rotate_default!(NTT3x42Ifma);

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
