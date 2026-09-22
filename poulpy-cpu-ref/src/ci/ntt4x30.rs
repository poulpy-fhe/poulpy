#[path = "../ntt4x30/module.rs"]
pub(crate) mod module;
#[path = "../ntt4x30/prim.rs"]
pub(crate) mod prim;
#[path = "../ntt4x30/vec_znx_big.rs"]
pub(crate) mod vec_znx_big;
#[path = "../ntt4x30/znx.rs"]
pub(crate) mod znx;
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
pub use NTT4x30CIRef as NTT4x30Ref;
