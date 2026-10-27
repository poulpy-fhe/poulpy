//! User-facing multiparty operation contracts.
//!
//! Traits cover the collective switching and automorphism key protocols, the
//! collective GGSW, key switching, public key and tensor (relinearization) key
//! protocols, the encryption-to-shares and shares-to-encryption protocols, and
//! aggregation and finalization of every PAT shape.
//!
//! A protocol trait, `*MHEProtocol`, holds the protocol's `mhe_*_share_gen`,
//! `mhe_*_share_aggregate` and `mhe_*_share_finalize` operations on its share
//! type.
//!
//! Every trait delegates through [`crate::oep`].
mod evaluation_key;
mod ggsw;
mod keyswitch;
mod pat;
mod public_key;
mod sharing;
mod tensor_key;
pub use evaluation_key::*;
pub use ggsw::*;
pub use keyswitch::*;
pub use pat::*;
pub use public_key::*;
pub use sharing::*;
pub use tensor_key::*;
