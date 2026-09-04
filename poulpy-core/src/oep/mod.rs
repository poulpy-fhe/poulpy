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
//!     fn glwe_keyswitch_default<R, A>(&self, res: &mut R, a: &A,
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
//! impl_gglwe_keyswitch_defaults_full!(MyBackend);
//! impl_ggsw_keyswitch_defaults_full!(MyBackend);
//! impl_lwe_keyswitch_defaults_full!(MyBackend);
//! ```
//!
//! `Module<MyBackend>` now implements the public `GLWEKeyswitch` trait, and the
//! override composes: the reference GGLWE and GGSW bodies call
//! `glwe_keyswitch_default`, so they route through the fused kernel too.
//!
//! The same shape applies where an `*Impl` trait spans several sub-families.
//! `AutomorphismImpl` needs all three of `GLWEAutomorphismDefault`,
//! `GGSWAutomorphismDefault` and `GGLWEAutomorphismDefault`, so a backend with
//! only a fused GLWE automorphism hand-writes that one and macro-forwards the
//! other two:
//!
//! ```ignore
//! impl GLWEAutomorphismDefault<MyBackend> for Module<MyBackend> { /* 9 methods */ }
//! impl_ggsw_automorphism_defaults_full!(MyBackend);
//! impl_gglwe_automorphism_defaults_full!(MyBackend);
//! ```
//!
//! Note the size of that first impl. A `*Default` trait is abstract, so an
//! override owes *every* method, not just the interesting one:
//! `GLWEKeyswitchDefault` is 3 methods, but `GLWEAutomorphismDefault` is 9 —
//! the plain and assign forms plus the `add`, `sub` and `sub_negate`
//! compositions. An accelerator that only wants to replace the core map still
//! writes the other six, forwarding them to
//! `crate::default::automorphism::glwe`.
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

pub use crate::{
    impl_conversion_defaults_full, impl_decryption_defaults_full, impl_encryption_defaults_full,
    impl_gglwe_automorphism_defaults_full, impl_gglwe_external_product_defaults_full, impl_gglwe_keyswitch_defaults_full,
    impl_ggsw_automorphism_defaults_full, impl_ggsw_external_product_defaults_full, impl_ggsw_keyswitch_defaults_full,
    impl_glwe_automorphism_defaults_full, impl_glwe_external_product_defaults_full, impl_glwe_keyswitch_defaults_full,
    impl_glwe_packing_defaults_full, impl_glwe_trace_defaults_full, impl_linear_transformation_defaults_full,
    impl_lwe_keyswitch_defaults_full,
};

use poulpy_hal::layouts::{Backend, CoeffNormalized, CoeffUnnormalized, CoefficientState, Data, ZnxWord};
pub use poulpy_hal::oep::SetNormalizationState;

use crate::layouts::{GLWE, GLWEViewMut};

/// See [`SetNormalizationState`]: `set_normalized` is the backend-implementor
/// relabel with no normalization pass, reserved for fused kernels inside
/// backend crates. Scheme code must go through [`GLWE::normalize`].
impl<D: Data, W: ZnxWord, S: CoefficientState> SetNormalizationState for GLWE<D, W, S> {
    type WithState<T: CoefficientState> = GLWE<D, W, T>;

    fn set_unnormalized(self) -> GLWE<D, W, CoeffUnnormalized> {
        self.into_unnormalized()
    }

    unsafe fn set_normalized(self) -> GLWE<D, W, CoeffNormalized> {
        let GLWE { data, k, base2k } = self;
        GLWE {
            // SAFETY: forwarded caller contract.
            data: unsafe { data.set_normalized() },
            k,
            base2k,
        }
    }
}

impl<'a, BE: Backend + 'a, S: CoefficientState> SetNormalizationState for GLWEViewMut<'a, BE, S> {
    type WithState<T: CoefficientState> = GLWEViewMut<'a, BE, T>;

    fn set_unnormalized(self) -> GLWEViewMut<'a, BE, CoeffUnnormalized> {
        self.into_unnormalized()
    }

    unsafe fn set_normalized(self) -> GLWEViewMut<'a, BE, CoeffNormalized> {
        // SAFETY: forwarded caller contract.
        GLWEViewMut::from_inner(unsafe { self.into_inner().set_normalized() })
    }
}
