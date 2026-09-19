//! Open Extension Points (OEP) for backend crates.
//!
//! This module defines the `unsafe` backend extension layer as a set of
//! per-family traits. Backend crates implement only the families they own.
//!
//! All extension points in this module are `unsafe` because implementations
//! must uphold the backend safety contract.
//!
//! # Derived operations and the implementation order
//!
//! Operations classified *derived* or *variant* carry a default body composed
//! from the basis methods of the same backend (see [`crate::oep::derived`]); a backend
//! implements the basis and inherits the rest, then overrides where a fused
//! kernel is worth it, overriding a body and its `_tmp_bytes` together. The
//! default body is the definition of the operation and `test_suite::derived`
//! pins every override to it: an override is a faster route to the same
//! result, never a different one.
//!
//! The exception is a `_tmp_bytes` that sizes a whole family rather than one
//! body, `HalVecZnxImpl::vec_znx_lsh_tmp_bytes` and
//! `HalVecZnxImpl::vec_znx_rsh_tmp_bytes`, which must cover the temporary
//! every `_add` / `_sub` / `_assign` body of their family carves. There it is
//! correct to leave the family `_tmp_bytes` on its default while overriding
//! individual bodies, as a backend may do for `vec_znx_lsh_assign`; a
//! backend that does override such a `_tmp_bytes` must still return a value
//! large enough for the bodies of the family it left on the default.
//!
//! Those bodies cross family boundaries, so the family traits form a
//! composition graph, expressed as supertraits:
//!
//! ```text
//! HalVecZnxImpl        (root)
//! HalVecZnxBigImpl     (root)
//! HalVecZnxDftImpl    -> HalVecZnxBigImpl
//! HalSvpImpl          -> HalVecZnxDftImpl
//! HalVmpImpl          -> HalVecZnxDftImpl
//! HalConvolutionImpl  -> HalVecZnxDftImpl, HalVecZnxBigImpl
//! ```
//!
//! The graph is acyclic; a new backend implements the families in topological
//! order: `HalModuleImpl`, `HalVecZnxImpl`, `HalVecZnxBigImpl`,
//! `HalVecZnxDftImpl`, then `HalSvpImpl` / `HalVmpImpl` /
//! `HalConvolutionImpl` in any order.

pub mod derived;
mod hal_impl;

pub use derived::*;
pub use hal_impl::*;
