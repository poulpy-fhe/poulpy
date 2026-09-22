use crate::FFT64RefBackend;
use crate::NTT4x30RefBackend;
use crate::ring::CpuRing;

use crate::{FFT64CIRef, FFT64Ref, NTT4x30CIRef, NTT4x30Ref};
use poulpy_ckks::impl_ckks_encapsulated_mod_up_reference;

impl_ckks_encapsulated_mod_up_reference!(FFT64Ref);
impl_ckks_encapsulated_mod_up_reference!(FFT64CIRef);
impl_ckks_encapsulated_mod_up_reference!(NTT4x30Ref);
impl_ckks_encapsulated_mod_up_reference!(NTT4x30CIRef);
// The reference backends have no accelerated transform, so they select the
// generic scalar table for every precision at once.
impl<R: CpuRing, F> crate::ckks_encoding::CKKSEncodingTransform<F> for FFT64RefBackend<R>
where
    F: poulpy_ckks::api::CKKSEncodingScalar,
{
    type Fft = crate::FFT64ReimTable<F>;
}

impl<R: CpuRing, F> crate::ckks_encoding::CKKSEncodingTransform<F> for NTT4x30RefBackend<R>
where
    F: poulpy_ckks::api::CKKSEncodingScalar,
{
    type Fft = crate::FFT64ReimTable<F>;
}

crate::impl_cpu_ckks_defaults!(crate::FFT64Ref);
crate::impl_cpu_ckks_defaults!(crate::FFT64CIRef);
crate::impl_cpu_ckks_defaults!(crate::NTT4x30Ref);
crate::impl_cpu_ckks_defaults!(crate::NTT4x30CIRef);
