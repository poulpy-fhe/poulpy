//! User-facing multiparty operation contracts.
//!
//! - [`evaluation_key`]: the collective switching and automorphism key protocols.
//! - [`pat`]: aggregation and finalization of every PAT shape.
//! - [`public_key`]: the collective public key protocol.
//!
//! A protocol trait, `*MHEProtocol`, holds the protocol's `mhe_*_share_gen`,
//! `mhe_*_share_aggregate` and `mhe_*_share_finalize` operations on its share
//! type.
//!
//! Every trait delegates through [`crate::oep`].
pub mod evaluation_key;
pub mod pat;
pub mod public_key;
pub use evaluation_key::*;
pub use pat::*;
pub use public_key::*;
