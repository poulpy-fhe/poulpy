//! User-facing multiparty operation contracts.
//!
//! - [`evaluation_key`]: the collective evaluation key protocols.
//! - [`keyswitch`]: the collective key switching protocols.
//! - [`pat`]: aggregation, normalization and finalization of every PAT shape.
//! - [`public_key`]: the collective public key protocol.
//!
//! Every trait delegates through [`crate::oep`].
pub mod evaluation_key;
pub mod keyswitch;
pub mod pat;
pub mod public_key;
pub use evaluation_key::*;
pub use keyswitch::*;
pub use pat::*;
pub use public_key::*;
