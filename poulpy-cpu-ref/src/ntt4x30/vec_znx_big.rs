//! Large-coefficient (i128) ring element vector support for [`NTT4x30Ref`](crate::NTT4x30Ref).
//!
//! The shared `poulpy-hal` NTT4x30 defaults rely on backend-provided `I128BigOps`
//! and `I128NormalizeOps` hooks for vectorized i128 operations.

use crate::NTT4x30Ref;
use crate::reference::ntt4x30::{I128BigOps, I128NormalizeOps};

impl I128BigOps for NTT4x30Ref {}
impl I128NormalizeOps for NTT4x30Ref {}
