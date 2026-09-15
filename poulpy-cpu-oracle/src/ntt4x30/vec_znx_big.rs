//! Large-coefficient (i128) ring element vector support for [`NTT4x30Oracle`](crate::NTT4x30Oracle).
//!
//! The shared `poulpy-cpu-oracle` NTT4x30 defaults rely on backend-provided `I128BigOps`
//! and `I128NormalizeOps` hooks for vectorized i128 operations.

use crate::NTT4x30Oracle;
use crate::reference::ntt4x30::{I128BigOps, I128NormalizeOps};

impl I128BigOps for NTT4x30Oracle {}
impl I128NormalizeOps for NTT4x30Oracle {
    const FUSE_NORMALIZE: bool = false;
}
