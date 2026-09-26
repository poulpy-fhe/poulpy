//! Portable conjugate-invariant NTT backend.
//!
//! [`NTT4x30CIRef`] serves the conjugate invariant ring with its own module
//! handle and plans. It overrides the NTT execution for conjugate invariant
//! tables and forwards every ring-independent kernel to
//! [`NTT4x30Ref`](crate::NTT4x30Ref).

mod hal;
mod module;

pub use module::NTT4x30CIRefHandle;

use crate::{
    reference::ntt4x30::{
        NttDFTExecute,
        ntt::{NttTable, NttTableInv, intt_ref, ntt_ref},
        primes::Primes30,
    },
    ring::ConjugateInvariant,
};

/// Portable conjugate-invariant NTT backend.
#[cfg_attr(
    feature = "enable-core",
    doc = r"
Prepared keys retain their backend type:
```
use poulpy_cpu_ref::NTT4x30CIRef;
use poulpy_core::layouts::{GetTensorKey, GLWETensorKeyPrepared};
use poulpy_hal::AlignedBuf;
fn accepts_ci(_: &impl GetTensorKey<NTT4x30CIRef>) {}
fn prepared(key: &GLWETensorKeyPrepared<AlignedBuf, NTT4x30CIRef>) { accepts_ci(key); }
```
```compile_fail
use poulpy_cpu_ref::{NTT4x30CIRef, NTT4x30Ref};
use poulpy_core::layouts::{GetTensorKey, GLWETensorKeyPrepared};
use poulpy_hal::AlignedBuf;
fn accepts_ci(_: &impl GetTensorKey<NTT4x30CIRef>) {}
fn prepared(key: &GLWETensorKeyPrepared<AlignedBuf, NTT4x30Ref>) { accepts_ci(key); }
```"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NTT4x30CIRef;

crate::forward_znx_kernels!(NTT4x30CIRef => crate::NTT4x30Ref);
crate::forward_ntt_arith_kernels!(NTT4x30CIRef => crate::NTT4x30Ref);
crate::forward_ntt4x30_kernels!(NTT4x30CIRef => crate::NTT4x30Ref);
crate::forward_i128_kernels!(NTT4x30CIRef => crate::NTT4x30Ref);

impl NttDFTExecute<NttTable<Primes30, ConjugateInvariant>> for NTT4x30CIRef {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTable<Primes30, ConjugateInvariant>, data: &mut [u64]) {
        ntt_ref::<Primes30>(table, data);
    }
}

impl NttDFTExecute<NttTableInv<Primes30, ConjugateInvariant>> for NTT4x30CIRef {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTableInv<Primes30, ConjugateInvariant>, data: &mut [u64]) {
        intt_ref::<Primes30>(table, data);
    }
}

#[cfg(feature = "enable-core")]
mod core_impl {
    use super::NTT4x30CIRef;

    poulpy_core::impl_glwe_tensoring_reference!(NTT4x30CIRef);
    poulpy_core::impl_gglwe_product_digits_strided_reference!(NTT4x30CIRef);
    crate::impl_cpu_core_defaults!(NTT4x30CIRef, ntt4x30);
}
