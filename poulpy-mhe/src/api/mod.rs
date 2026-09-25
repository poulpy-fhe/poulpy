//! User-facing multiparty operation contracts.
//!
//! - [`pat`]: aggregation, normalization and finalization of every PAT shape.
//! - [`public_key`]: the collective public key protocol.
//!
//! Every trait delegates through [`crate::oep`].
pub mod pat;
pub mod public_key;
pub use pat::*;
pub use public_key::*;
