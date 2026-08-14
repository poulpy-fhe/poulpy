//! Backend extension points for scalar-vector product (SVP) operations.
//!
//! Each flavor trait carries the kernels only: prepare, copy, and the `tpol` /
//! `ppol` `_to_dft` and `_assign` variants, each taking its prepared operand as
//! the concrete layout type. The derived variants are emitted by
//! [`hal_impl_svp!`](crate::hal_impl_svp) inside each backend's `HalSvpImpl`
//! block, so they dispatch back to that backend's kernels.
//!
//! `SvpTPol` and `SvpPPol` currently hold the same bytes on every CPU backend,
//! so the paired `tpol` and `ppol` methods do the same work. They stay separate
//! methods over separate types: a backend that gains a cheaper hot-prep form
//! repoints its `tpol` methods alone, and no caller changes.

mod ppol;
mod tpol;

pub use ppol::{FFT64SvpPPolDefault, NTT4x30SvpPPolDefault};
pub use tpol::{FFT64SvpTPolDefault, NTT4x30SvpTPolDefault};
