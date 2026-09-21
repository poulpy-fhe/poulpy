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
//! Required methods form the backend implementation surface. Operations with
//! a default body compose those methods through backend-native views (see
//! [`crate::oep::derived`]); a backend inherits these compositions and may
//! override them with fused kernels. A *variant* classification does not imply
//! an inherited body: mutation variants such as
//! [`HalVecZnxBigImpl::vec_znx_big_add_small_assign`](crate::oep::HalVecZnxBigImpl::vec_znx_big_add_small_assign),
//! [`HalVecZnxDftImpl::vec_znx_idft_apply_tmpa`](crate::oep::HalVecZnxDftImpl::vec_znx_idft_apply_tmpa), and
//! [`HalSvpImpl::svp_apply_dft_to_dft_assign`](crate::oep::HalSvpImpl::svp_apply_dft_to_dft_assign) remain required because their
//! signatures cannot supply the temporary storage needed by a generic
//! composition. Their individual method docs explain these requirements.
//!
//! A derived default body defines the operation. Overrides must preserve its
//! result and scratch contract; override a body and its `_tmp_bytes` together
//! when the required scratch changes. [`crate::test_suite::derived`] compares
//! overrides directly against these bodies, and
//! [`crate::cross_backend_test_suite!`] compares backend implementations against
//! a reference backend. Backend crates must register and execute the applicable
//! tests for each implementation; exporting a generic test does not execute it.
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
