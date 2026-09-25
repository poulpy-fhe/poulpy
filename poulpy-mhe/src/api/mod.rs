//! User-facing multiparty operation contracts.
//!
//! - [`pat`]: aggregation, normalization and finalization of every PAT shape.
//!
//! Every trait delegates through [`crate::oep`].
pub mod pat;
pub use pat::*;
