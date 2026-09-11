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
//! from the basis methods of the same backend (see [`derived`]); a backend
//! implements the basis and inherits the rest, then overrides where a fused
//! kernel is worth it — overriding a body and its `_tmp_bytes` together.
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
