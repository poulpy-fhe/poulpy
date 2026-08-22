use crate::{FFT64Avx, NTT4x30Avx};
#[cfg(feature = "enable-rayon")]
use crate::{FFT64AvxRayon, NTT4x30AvxRayon};
use poulpy_ckks::{
    impl_ckks_add_defaults, impl_ckks_conjugate_defaults, impl_ckks_copy_defaults, impl_ckks_dft_defaults,
    impl_ckks_encapsulated_mod_up_default, impl_ckks_encryption_defaults, impl_ckks_imag_defaults, impl_ckks_mul_defaults,
    impl_ckks_neg_defaults, impl_ckks_plaintext_defaults, impl_ckks_pow2_defaults, impl_ckks_rotate_defaults,
    impl_ckks_sub_defaults,
};

impl_ckks_encapsulated_mod_up_default!(FFT64Avx);
impl_ckks_encapsulated_mod_up_default!(NTT4x30Avx);
impl_ckks_conjugate_defaults!(FFT64Avx);
impl_ckks_conjugate_defaults!(NTT4x30Avx);
impl_ckks_copy_defaults!(FFT64Avx);
impl_ckks_copy_defaults!(NTT4x30Avx);
impl_ckks_encryption_defaults!(FFT64Avx);
impl_ckks_encryption_defaults!(NTT4x30Avx);
impl_ckks_imag_defaults!(FFT64Avx);
impl_ckks_imag_defaults!(NTT4x30Avx);
impl_ckks_mul_defaults!(FFT64Avx);
impl_ckks_mul_defaults!(NTT4x30Avx);
impl_ckks_neg_defaults!(FFT64Avx);
impl_ckks_neg_defaults!(NTT4x30Avx);
impl_ckks_pow2_defaults!(FFT64Avx);
impl_ckks_pow2_defaults!(NTT4x30Avx);
impl_ckks_rotate_defaults!(FFT64Avx);
impl_ckks_rotate_defaults!(NTT4x30Avx);
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
select_avx_encoding_transform!(NTT4x30Avx);

::poulpy_cpu_ref::impl_ckks_encoding!(FFT64Avx);
::poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(FFT64Avx);
::poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(FFT64Avx);
::poulpy_cpu_ref::impl_ckks_encoding!(NTT4x30Avx);
::poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(NTT4x30Avx);
::poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(NTT4x30Avx);
impl_ckks_add_defaults!(FFT64Avx);
impl_ckks_add_defaults!(NTT4x30Avx);
impl_ckks_sub_defaults!(FFT64Avx);
impl_ckks_sub_defaults!(NTT4x30Avx);
impl_ckks_plaintext_defaults!(FFT64Avx);
impl_ckks_plaintext_defaults!(NTT4x30Avx);
impl_ckks_dft_defaults!(FFT64Avx);
impl_ckks_dft_defaults!(NTT4x30Avx);

#[cfg(feature = "enable-rayon")]
mod rayon_defaults {
    use super::*;

    impl_ckks_encapsulated_mod_up_default!(FFT64AvxRayon);
    impl_ckks_conjugate_defaults!(FFT64AvxRayon);
    impl_ckks_copy_defaults!(FFT64AvxRayon);
    impl_ckks_encryption_defaults!(FFT64AvxRayon);
    impl_ckks_imag_defaults!(FFT64AvxRayon);
    impl_ckks_mul_defaults!(FFT64AvxRayon);
    impl_ckks_neg_defaults!(FFT64AvxRayon);
    impl_ckks_pow2_defaults!(FFT64AvxRayon);
    impl_ckks_rotate_defaults!(FFT64AvxRayon);
    select_avx_encoding_transform!(FFT64AvxRayon);
    ::poulpy_cpu_ref::impl_ckks_encoding!(FFT64AvxRayon);
    ::poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(FFT64AvxRayon);
    ::poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(FFT64AvxRayon);
    impl_ckks_add_defaults!(FFT64AvxRayon);
    impl_ckks_sub_defaults!(FFT64AvxRayon);
    impl_ckks_plaintext_defaults!(FFT64AvxRayon);
    impl_ckks_dft_defaults!(FFT64AvxRayon);

    impl_ckks_encapsulated_mod_up_default!(NTT4x30AvxRayon);
    impl_ckks_conjugate_defaults!(NTT4x30AvxRayon);
    impl_ckks_copy_defaults!(NTT4x30AvxRayon);
    impl_ckks_encryption_defaults!(NTT4x30AvxRayon);
    impl_ckks_imag_defaults!(NTT4x30AvxRayon);
    impl_ckks_mul_defaults!(NTT4x30AvxRayon);
    impl_ckks_neg_defaults!(NTT4x30AvxRayon);
    impl_ckks_pow2_defaults!(NTT4x30AvxRayon);
    impl_ckks_rotate_defaults!(NTT4x30AvxRayon);
    select_avx_encoding_transform!(NTT4x30AvxRayon);
    ::poulpy_cpu_ref::impl_ckks_encoding!(NTT4x30AvxRayon);
    ::poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(NTT4x30AvxRayon);
    ::poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(NTT4x30AvxRayon);
    impl_ckks_add_defaults!(NTT4x30AvxRayon);
    impl_ckks_sub_defaults!(NTT4x30AvxRayon);
    impl_ckks_plaintext_defaults!(NTT4x30AvxRayon);
    impl_ckks_dft_defaults!(NTT4x30AvxRayon);
}
