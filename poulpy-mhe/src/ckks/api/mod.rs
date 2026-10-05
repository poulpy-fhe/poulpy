//! User-facing CKKS multiparty operation contracts.
//!
//! - [`refresh`]: the collective CKKS refresh protocol.
//!
//! Every trait delegates through [`crate::ckks::oep`].
pub mod refresh;
pub use refresh::*;
