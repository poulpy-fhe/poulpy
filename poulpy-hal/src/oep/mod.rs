//! Open Extension Points (OEP) for backend crates.
//!
//! This module defines the `unsafe` backend extension layer as a set of
//! per-family traits. Backend crates implement only the families they own.
//!
//! All extension points in this module are `unsafe` because implementations
//! must uphold the backend safety contract.

mod hal_impl;

pub use hal_impl::*;
