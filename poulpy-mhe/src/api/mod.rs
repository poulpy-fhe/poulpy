//! User-facing multiparty operation contracts.
//!
//! - [`evaluation_key`]: the collective switching and automorphism key protocols.
//! - [`ggsw`]: the collective GGSW protocol.
//! - [`keyswitch`]: the collective key switching protocols.
//! - [`pat`]: aggregation and finalization of every PAT shape.
//! - [`public_key`]: the collective public key protocol.
//! - [`refresh`]: the collective refresh protocol.
//! - [`sharing`]: the encryption-to-shares and shares-to-encryption protocols.
//! - [`tensor_key`]: the collective tensor (relinearization) key protocol.
//! - [`threshold`]: Shamir thresholdization and threshold key switching.
//!
//! A protocol trait, `*MHEProtocol`, holds the protocol's `mhe_*_share_gen`,
//! `mhe_*_share_aggregate` and `mhe_*_share_finalize` operations on its share
//! type.
//!
//! Every trait delegates through [`crate::oep`].
pub mod evaluation_key;
pub mod ggsw;
pub mod keyswitch;
pub mod pat;
pub mod public_key;
pub mod refresh;
pub mod sharing;
pub mod tensor_key;
pub mod threshold;
pub use evaluation_key::*;
pub use ggsw::*;
pub use keyswitch::*;
pub use pat::*;
pub use public_key::*;
pub use refresh::*;
pub use sharing::*;
pub use tensor_key::*;
pub use threshold::*;
