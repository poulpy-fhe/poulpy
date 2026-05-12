//! Open extension points for `poulpy-core`.
//!
//! Backends implement the per-family `*Impl` traits exported here to inherit or
//! override the high-level `poulpy-core` algorithms that are exposed through
//! safe traits on [`poulpy_hal::layouts::Module`].
//!
//! The `unsafe` marker on `*Impl` traits follows the same convention as the HAL:
//! implementors are taking responsibility for the core correctness contract of
//! the backend. In particular, implementations must preserve the mathematical
//! semantics and bit-parity requirements expected by end-to-end pipelines across
//! backends.

mod automorphism;
mod conversion;
mod decryption;
mod encryption;
mod external_product;
mod keyswitching;
mod operations;

pub use automorphism::*;
pub use conversion::*;
pub use decryption::*;
pub use encryption::*;
pub use external_product::*;
pub use keyswitching::*;
pub use operations::*;

pub use crate::impl_glwe_rotate_impl_from;

pub use crate::{
    impl_conversion_defaults_full, impl_decryption_defaults_full, impl_encryption_defaults_full,
    impl_gglwe_automorphism_defaults_full, impl_gglwe_external_product_defaults_full, impl_gglwe_keyswitch_defaults_full,
    impl_ggsw_automorphism_defaults_full, impl_ggsw_external_product_defaults_full, impl_ggsw_keyswitch_defaults_full,
    impl_glwe_automorphism_defaults_full, impl_glwe_external_product_defaults_full, impl_glwe_keyswitch_defaults_full,
    impl_glwe_packing_defaults_full, impl_glwe_trace_defaults_full, impl_lwe_keyswitch_defaults_full,
};
