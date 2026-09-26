use super::FFT64Ref;
use super::NTT4x30Ref;

use poulpy_ckks::impl_ckks_encapsulated_mod_up_reference;

impl_ckks_encapsulated_mod_up_reference!(FFT64Ref);
impl_ckks_encapsulated_mod_up_reference!(NTT4x30Ref);
// The reference backends have no accelerated transform, so they select the
// generic scalar table for every precision at once.
impl<F> crate::ckks_encoding::CKKSEncodingTransform<F> for FFT64Ref
where
    F: poulpy_ckks::api::CKKSEncodingScalar,
{
    type Fft = crate::FFT64ReimTable<F>;
}

impl<F> crate::ckks_encoding::CKKSEncodingTransform<F> for NTT4x30Ref
where
    F: poulpy_ckks::api::CKKSEncodingScalar,
{
    type Fft = crate::FFT64ReimTable<F>;
}

crate::impl_cpu_ckks_defaults!(super::FFT64Ref);
crate::impl_cpu_ckks_defaults!(super::NTT4x30Ref);
