//! PaCo plan material and slot helpers: the factor-chain generation and the
//! Galois and slot-order helpers.
//!
//! The invariant-bearing plans, compiled contexts, secret specifications, and
//! key containers live in [`crate::layouts::paco`]; the cleartext circle
//! embedding and coefficient encodings live in `crate::encoding::paco`;
//! evaluation is exposed through [`CKKSPaCoOps`](crate::api::CKKSPaCoOps), whose
//! drivers are the derived defaults of [`CKKSPaCoImpl`](crate::oep::CKKSPaCoImpl).

pub(crate) mod lt;
pub(crate) mod ops;
