//! Open extension points for `poulpy-core`.
//!
//! The high-level algorithms are exposed through safe traits on
//! [`poulpy_hal::layouts::Module`], which resolve through two layers, with one exception:
//!
//! - `*Impl` traits (this module), blanket-implemented for every backend whose
//!   `Module` implements the matching `*Default` traits. They are the seat the
//!   public API dispatches to, not the seat a backend takes.
//! - `*Default` traits (this module), implemented on `Module<BE>`. **This is the
//!   override surface.** They are abstract: no HAL supertraits and no default
//!   method bodies, so an implementor owes exactly the methods of one family.
//! - [`SamplingImpl`] is the one exception: it has no `*Default` twin and no
//!   blanket impl, because `poulpy-core` has no reference body to offer. Every
//!   other family's reference body composes HAL operations; drawing from a
//!   distribution is not such a composition, and a backend's buffers are opaque
//!   to generic code, so only the backend can produce the values. A backend
//!   implements it directly, the CPU backends through
//!   `poulpy_cpu_ref::impl_sampling_host!`.
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
//! `impl_*_reference_full!` macros re-exported below. Each macro implements a
//! single `*Default` trait by forwarding every method to the corresponding
//! reference body, so a backend that accelerates one family hand-writes that
//! trait and macro-forwards the rest:
//!
//! ```ignore
//! use poulpy_core::oep::{GLWEKeyswitchReference, impl_gglwe_keyswitch_reference_full,
//!                        impl_ggsw_keyswitch_reference_full, impl_lwe_keyswitch_reference_full};
//!
//! impl GLWEKeyswitchReference<MyBackend> for Module<MyBackend> {
//!     fn glwe_keyswitch_reference<R, A>(&self, res: &mut R, a: &A,
//!                                     key: &GGLWEPreparedBackendRef<'_, MyBackend>,
//!                                     scratch: &mut ScratchArena<'_, MyBackend>)
//!     where
//!         R: GLWEToBackendMut<MyBackend> + GLWEInfos,
//!         A: GLWEToBackendRef<MyBackend> + GLWEInfos,
//!     {
//!         my_fused_keyswitch(self, res, a, key.data(), scratch);
//!     }
//!     // ... tmp_bytes and assign
//! }
//!
//! impl_gglwe_keyswitch_reference_full!(MyBackend);
//! impl_ggsw_keyswitch_reference_full!(MyBackend);
//! impl_lwe_keyswitch_reference_full!(MyBackend);
//! ```
//!
//! `Module<MyBackend>` now implements the public `GLWEKeyswitch` trait, and the
//! override composes: the reference GGLWE and GGSW bodies call
//! `glwe_keyswitch_reference`, so they route through the fused kernel too.
//!
//! The same shape applies where an `*Impl` trait spans several sub-families.
//! `AutomorphismImpl` needs all three of `GLWEAutomorphismReference`,
//! `GGSWAutomorphismReference` and `GGLWEAutomorphismReference`, so a backend with
//! only a fused GLWE automorphism hand-writes that one and macro-forwards the
//! other two:
//!
//! ```ignore
//! impl GLWEAutomorphismReference<MyBackend> for Module<MyBackend> { /* 9 methods */ }
//! impl_ggsw_automorphism_reference_full!(MyBackend);
//! impl_gglwe_automorphism_reference_full!(MyBackend);
//! ```
//!
//! Note the size of that first impl. A `*Default` trait is abstract, so an
//! override owes *every* method, not just the interesting one:
//! `GLWEKeyswitchReference` is 3 methods, but `GLWEAutomorphismReference` is 9 —
//! the plain and assign forms plus the `add`, `sub` and `sub_negate`
//! compositions. An accelerator that only wants to replace the core map still
//! writes the other six, forwarding them to
//! `crate::reference::automorphism::glwe`.
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
mod sampling;

pub use automorphism::*;
pub use conversion::*;
pub use decryption::*;
pub use encryption::*;
pub use external_product::*;
pub use keyswitching::*;
pub use linear_transformation::*;
pub use operations::*;
pub use polynomial_evaluation::*;
pub use sampling::*;

pub use crate::{
    impl_conversion_reference_full, impl_decryption_reference_full, impl_encryption_reference_full,
    impl_gglwe_automorphism_reference_full, impl_gglwe_external_product_reference_full, impl_gglwe_keyswitch_reference_full,
    impl_ggsw_automorphism_reference_full, impl_ggsw_external_product_reference_full, impl_ggsw_keyswitch_reference_full,
    impl_glwe_automorphism_reference_full, impl_glwe_external_product_reference_full, impl_glwe_keyswitch_reference_full,
    impl_glwe_packing_reference_full, impl_glwe_trace_reference_full, impl_linear_transformation_reference_full,
    impl_lwe_keyswitch_reference_full,
};
