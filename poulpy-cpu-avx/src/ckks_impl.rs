use crate::{FFT64Avx, NTT4x30Avx};
#[cfg(feature = "enable-rayon")]
use crate::{FFT64AvxRayon, NTT4x30AvxRayon};
use poulpy_ckks::{
    impl_ckks_add_reference, impl_ckks_conjugate_reference, impl_ckks_copy_reference, impl_ckks_dft_reference,
    impl_ckks_encapsulated_mod_up_reference, impl_ckks_encryption_reference, impl_ckks_imag_reference, impl_ckks_mul_reference,
    impl_ckks_neg_reference, impl_ckks_plaintext_reference, impl_ckks_pow2_reference, impl_ckks_rotate_reference,
    impl_ckks_sub_reference,
};

impl_ckks_encapsulated_mod_up_reference!(FFT64Avx);
impl_ckks_conjugate_reference!(FFT64Avx);
impl_ckks_conjugate_reference!(NTT4x30Avx);
impl_ckks_copy_reference!(FFT64Avx);
impl_ckks_copy_reference!(NTT4x30Avx);
impl_ckks_encryption_reference!(FFT64Avx);
impl_ckks_encryption_reference!(NTT4x30Avx);
impl_ckks_imag_reference!(FFT64Avx);
impl_ckks_imag_reference!(NTT4x30Avx);
impl_ckks_mul_reference!(FFT64Avx);
impl_ckks_mul_reference!(NTT4x30Avx);
impl_ckks_neg_reference!(FFT64Avx);
impl_ckks_neg_reference!(NTT4x30Avx);
impl_ckks_pow2_reference!(FFT64Avx);
impl_ckks_pow2_reference!(NTT4x30Avx);
impl_ckks_rotate_reference!(FFT64Avx);
impl_ckks_rotate_reference!(NTT4x30Avx);
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
impl_ckks_add_reference!(FFT64Avx);
impl_ckks_add_reference!(NTT4x30Avx);
impl_ckks_sub_reference!(FFT64Avx);
impl_ckks_sub_reference!(NTT4x30Avx);
impl_ckks_plaintext_reference!(FFT64Avx);
impl_ckks_plaintext_reference!(NTT4x30Avx);
impl_ckks_dft_reference!(FFT64Avx);
impl_ckks_dft_reference!(NTT4x30Avx);

#[cfg(feature = "enable-rayon")]
mod rayon_defaults {
    use super::*;

    impl_ckks_encapsulated_mod_up_reference!(FFT64AvxRayon);
    impl_ckks_conjugate_reference!(FFT64AvxRayon);
    impl_ckks_copy_reference!(FFT64AvxRayon);
    impl_ckks_encryption_reference!(FFT64AvxRayon);
    impl_ckks_imag_reference!(FFT64AvxRayon);
    impl_ckks_mul_reference!(FFT64AvxRayon);
    impl_ckks_neg_reference!(FFT64AvxRayon);
    impl_ckks_pow2_reference!(FFT64AvxRayon);
    impl_ckks_rotate_reference!(FFT64AvxRayon);
    select_avx_encoding_transform!(FFT64AvxRayon);
    ::poulpy_cpu_ref::impl_ckks_encoding!(FFT64AvxRayon);
    ::poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(FFT64AvxRayon);
    ::poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(FFT64AvxRayon);
    impl_ckks_add_reference!(FFT64AvxRayon);
    impl_ckks_sub_reference!(FFT64AvxRayon);
    impl_ckks_plaintext_reference!(FFT64AvxRayon);
    impl_ckks_dft_reference!(FFT64AvxRayon);

    impl_ckks_conjugate_reference!(NTT4x30AvxRayon);
    impl_ckks_copy_reference!(NTT4x30AvxRayon);
    impl_ckks_encryption_reference!(NTT4x30AvxRayon);
    impl_ckks_imag_reference!(NTT4x30AvxRayon);
    impl_ckks_mul_reference!(NTT4x30AvxRayon);
    impl_ckks_neg_reference!(NTT4x30AvxRayon);
    impl_ckks_pow2_reference!(NTT4x30AvxRayon);
    impl_ckks_rotate_reference!(NTT4x30AvxRayon);
    select_avx_encoding_transform!(NTT4x30AvxRayon);
    ::poulpy_cpu_ref::impl_ckks_encoding!(NTT4x30AvxRayon);
    ::poulpy_cpu_ref::impl_ckks_paco_coeff_encoding!(NTT4x30AvxRayon);
    ::poulpy_cpu_ref::impl_ckks_ship_coeff_encoding!(NTT4x30AvxRayon);
    impl_ckks_add_reference!(NTT4x30AvxRayon);
    impl_ckks_sub_reference!(NTT4x30AvxRayon);
    impl_ckks_plaintext_reference!(NTT4x30AvxRayon);
    impl_ckks_dft_reference!(NTT4x30AvxRayon);
}
