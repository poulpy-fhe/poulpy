//! Blanket implementations connecting [`crate::api`] traits to [`crate::oep`] traits
//! on [`crate::layouts::Module`].
//!
//! This module contains no user-facing logic; it exists solely to wire the safe
//! API layer to the unsafe backend implementations via blanket `impl` blocks.

mod convolution;
mod module;
mod scratch;
mod svp_ppol;
mod vec_znx;
mod vec_znx_big;
mod vec_znx_dft;
mod vmp_pmat;
