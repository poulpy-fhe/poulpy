//! Roots, reciprocal, and division.

mod functions;

pub use functions::{CKKSRootOps, CbrtPlan, HypotPlan, RootOptions};

pub use crate::iterative::{CKKSDivSqrtOps, CKKSInverseDomainOps, CKKSInverseOps};
pub use crate::range::{COMPRESSION_L, IntervalNorm};
