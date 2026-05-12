//! Safe, user-facing trait definitions for polynomial arithmetic operations.
//!
//! Traits are organized by operation category:
//! - **module** -- module instantiation and ring degree queries.
//! - **vec\_znx** -- coefficient-domain arithmetic (add, sub, negate, shift, rotate, automorphism, normalization).
//! - **vec\_znx\_big** -- extended-precision accumulator operations.
//! - **vec\_znx\_dft** -- DFT-domain operations (forward/inverse transform, arithmetic).
//! - **svp\_ppol** -- scalar-vector product preparation and application.
//! - **vmp\_pmat** -- vector-matrix product preparation and application.
//! - **convolution** -- bivariate convolution preparation and application.
//! - **scratch** -- scratch buffer management.
//!
//! Scheme authors program against these traits; the actual computation is
//! dispatched to a backend via the [`oep`](crate::oep) extension points.

mod convolution;
mod module;
mod reim;
mod scratch;
mod svp_ppol;
mod vec_znx;
mod vec_znx_big;
mod vec_znx_dft;
mod vmp_pmat;

pub use convolution::*;
pub use module::*;
pub use reim::*;
pub use scratch::*;
pub use svp_ppol::*;
pub use vec_znx::*;
pub use vec_znx_big::*;
pub use vec_znx_dft::*;
pub use vmp_pmat::*;
