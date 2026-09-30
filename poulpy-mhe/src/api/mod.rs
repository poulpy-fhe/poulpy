//! User-facing multiparty operation contracts.
//!
//! - [`evaluation_key`]: the collective switching and automorphism key protocols.
//! - [`pat`]: aggregation and finalization of every PAT shape.
//! - [`public_key`]: the collective public key protocol.
//!
//! A protocol trait holds the protocol's `_gen`, `_aggregate` and `_finalize`
//! operations on its share type.
//!
//! Every trait delegates through [`crate::oep`].
pub mod evaluation_key;
pub mod pat;
pub mod public_key;
pub use evaluation_key::*;
pub use pat::*;
pub use public_key::*;
