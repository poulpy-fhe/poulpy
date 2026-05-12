//! Blanket implementations connecting `poulpy-core` traits to the matching
//! backend family impl traits on [`poulpy_hal::layouts::Module`].
//!
//! This module contains no algorithmic logic; it only wires the safe public
//! traits to the backend-owned high-level extension point.

mod automorphism;
mod conversion;
mod decryption;
mod encryption;
mod external_product;
mod keyswitching;
mod operations;
