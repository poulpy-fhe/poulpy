//! Open extension points for `poulpy-core`.
//!
//! The high-level algorithms are exposed through safe traits on
//! [`poulpy_hal::layouts::Module`], which resolve through two layers:
//!
//! - `*Impl` traits (this module), blanket-implemented for every backend whose
//!   `Module` implements the matching `*Default` traits. They are the seat the
//!   public API dispatches to, not the seat a backend takes.
//! - `*Default` traits (this module), implemented on `Module<BE>`. **This is the
//!   override surface.** They are abstract: no HAL supertraits and no default
//!   method bodies, so an implementor owes exactly the methods of one family.
//!
//! The `unsafe` marker on `*Impl` traits follows the same convention as the HAL:
//! implementors are taking responsibility for the core correctness contract of
//! the backend. In particular, implementations must preserve the mathematical
//! semantics and bit-parity requirements expected by end-to-end pipelines across
//! backends.
//!
//! # Taking the override surface
//!
//! A backend opts into the reference algorithms one family at a time, with the
//! `impl_*_defaults_full!` macros re-exported below. Each macro implements a
//! single `*Default` trait by forwarding every method to the corresponding
//! reference body, so a backend that accelerates one family hand-writes that
//! trait and macro-forwards the rest:
//!
//! ```ignore
//! use poulpy_core::oep::{GLWEKeyswitchDefault, impl_gglwe_keyswitch_defaults_full,
//!                        impl_ggsw_keyswitch_defaults_full, impl_lwe_keyswitch_defaults_full};
//!
//! impl GLWEKeyswitchDefault<MyBackend> for Module<MyBackend> {
//!     fn glwe_keyswitch_default<R, A, K>(&self, res: &mut R, a: &A, key: &K,
//!                                        scratch: &mut ScratchArena<'_, MyBackend>)
//!     where
//!         R: GLWEToBackendMut<MyBackend> + GLWEInfos,
//!         A: GLWEToBackendRef<MyBackend> + GLWEInfos,
//!         K: GGLWEPreparedToBackendRef<MyBackend> + GGLWEInfos,
//!     {
//!         let key_ref = key.to_backend_ref();
//!         my_fused_keyswitch(self, res, a, key_ref.data(), scratch);
//!     }
//!     // ... tmp_bytes and assign
//! }
//!
//! impl_gglwe_keyswitch_defaults_full!(MyBackend);
//! impl_ggsw_keyswitch_defaults_full!(MyBackend);
//! impl_lwe_keyswitch_defaults_full!(MyBackend);
//! ```
//!
//! `Module<MyBackend>` now implements the public `GLWEKeyswitch` trait, and the
//! override composes: the reference GGLWE and GGSW bodies call
//! `glwe_keyswitch_default`, so they route through the fused kernel too. The same
//! shape applies to families whose `*Impl` trait spans several sub-families
//! (`AutomorphismImpl` needs `GLWEAutomorphismDefault`, `GGSWAutomorphismDefault`
//! and `GGLWEAutomorphismDefault`): hand-write the accelerated one, macro-forward
//! its siblings.
//!
//! `poulpy-cpu-ref`'s `core_impl` module (feature `enable-core`) is the in-tree
//! worked example, forwarding every family.

mod automorphism;
mod conversion;
mod decryption;
mod encryption;
mod external_product;
mod keyswitching;
mod linear_transformation;
mod operations;
mod polynomial_evaluation;

pub use automorphism::*;
pub use conversion::*;
pub use decryption::*;
pub use encryption::*;
pub use external_product::*;
pub use keyswitching::*;
pub use linear_transformation::*;
pub use operations::*;
pub use polynomial_evaluation::*;

pub use crate::impl_glwe_rotate_impl_from;

pub use crate::{
    impl_conversion_defaults_full, impl_decryption_defaults_full, impl_encryption_defaults_full,
    impl_gglwe_automorphism_defaults_full, impl_gglwe_external_product_defaults_full, impl_gglwe_keyswitch_defaults_full,
    impl_ggsw_automorphism_defaults_full, impl_ggsw_external_product_defaults_full, impl_ggsw_keyswitch_defaults_full,
    impl_glwe_automorphism_defaults_full, impl_glwe_external_product_defaults_full, impl_glwe_keyswitch_defaults_full,
    impl_glwe_packing_defaults_full, impl_glwe_trace_defaults_full, impl_linear_transformation_defaults_full,
    impl_lwe_keyswitch_defaults_full,
};
