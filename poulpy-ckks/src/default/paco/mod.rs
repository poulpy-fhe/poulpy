//! Backend-generic PaCo implementation: the coefficient-encoding host
//! default, the factor-chain generation, the slot-fold composites, and the
//! sequential/parallel bootstrap drivers.
//!
//! The invariant-bearing plans, compiled contexts, secret specifications, and
//! key containers live in [`crate::layouts::paco`]; the cleartext circle
//! embedding and coefficient encodings live in `crate::encoding::paco`;
//! evaluation is exposed through [`CKKSPaCoOps`](crate::api::CKKSPaCoOps).

pub(crate) mod bootstrap;
pub(crate) mod lt;
pub(crate) mod ops;
pub(crate) mod parallel;
pub(crate) mod preflight;
