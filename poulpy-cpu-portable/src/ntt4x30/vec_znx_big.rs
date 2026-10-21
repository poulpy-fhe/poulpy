//! Large-coefficient (i128) ring element vector support for [`NTT4x30Portable`](super::NTT4x30Portable).
//!
//! The shared `poulpy-cpu-portable` NTT4x30 defaults rely on backend-provided `I128BigOps`
//! hooks for vectorized i128 operations.

use super::NTT4x30Portable;
use poulpy_hal::layouts::Ring;

use crate::kernels::ntt4x30::I128BigOps;

impl<R: Ring> I128BigOps for NTT4x30Portable<R> {}
