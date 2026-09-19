//! The implementation of every `poulpy-core` operation.
//!
//! Each module composes one operation family from the HAL operations. These
//! compositions define what the operations compute and are the only validated
//! circuit: a backend runs them through the `impl_*_reference_full!` macros of
//! [`crate::oep`], or overrides a family through its `oep` trait with a faster
//! route to the same result (a fused kernel, device-native code, another
//! layout), which the parity suite pins to these bodies. An override that
//! computes anything else is a defect, not a variant.

pub mod automorphism;
pub mod conversion;
pub mod decryption;
pub mod encryption;
pub mod external_product;
pub mod glwe_packing;
pub mod glwe_trace;
pub mod keyswitching;
pub mod linear_transformation;
pub mod noise;
pub mod operations;
pub mod polynomial_evaluation;
