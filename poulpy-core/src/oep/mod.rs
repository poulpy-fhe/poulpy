//! Open extension points for `poulpy-core`.
//!
//! Public [`crate::api`] operations dispatch through backend `*Impl` traits.
//! Each backend implements these traits explicitly, either with its own methods
//! or with the forwarding macros provided here.
//!
//! Core follows the same layer distinction as HAL:
//!
//! - [`crate::reference`] contains algorithms built from HAL operations. Its
//!   `*Reference` methods and free functions remain independently callable.
//! - [`derived`] contains compositions of core operations. Their default bodies
//!   on `*Impl` traits preserve dispatch through the selected backend methods.
//!   A backend may override a derived body and its scratch query together.
//!
//! For example, `GLWERotateReference` rotates each GLWE polynomial through HAL.
//! Derived GGSW rotation calls core GLWE rotation on each row, so it reuses a
//! backend's GLWE rotation override. Trace and packing similarly call
//! other core operations. The polynomial-evaluation schedule composes the
//! caller's `BSGSOps` arithmetic policy, which owns scheme precision and rounding.
//!
//! # Selecting an implementation
//!
//! `impl_*_reference_full!` macros implement a backend's `*Impl` family with the
//! available reference algorithms and derived defaults. Purely derived families
//! have `impl_*_derived_full!` macros. [`crate::impl_core_reference_full!`] selects
//! the provided implementations for all core families except [`SamplingImpl`].
//!
//! To customize a family, implement its `*Impl` trait yourself. Forward unchanged
//! required methods to reference helpers and inherit the derived defaults you
//! want. Omit the corresponding family macro to avoid defining the trait twice.
//! Satisfying a reference helper's HAL bounds alone never selects a backend hook.
//! The compiled `poulpy-cpu-ref` example in `src/tests/delegating_backend.rs`
//! exercises reference forwarding and public dispatch through backend overrides.
//!
//! # Layout requirements
//!
//! Reference gadget digit products and GLWE external products form partial DFT
//! views and require [`poulpy_hal::layouts::Backend::DFT_LIMBS_CONTIGUOUS`]. Their
//! bodies assert that capability at monomorphization. A backend with another
//! representation supplies its own [`GGLWEProductDigitsStridedImpl`] and
//! [`GLWEExternalProductImpl`] methods. Other reference algorithms and derived
//! defaults remain available when their individual requirements are met.
//!
//! Prepared factories, decompression, and internal keyswitch helpers reuse core
//! and HAL operations through their documented bounds. Host-only noise
//! diagnostics have additional storage-access requirements.
//!
//! # Correctness
//!
//! Implementors of the unsafe `*Impl` traits are responsible for numerical,
//! layout, aliasing and scratch contracts. Deterministic parity tests compare
//! integer results and metadata with an explicit portable CPU reference execution.
//! Each backend prepares its own objects and uses its own advertised scratch
//! budget. Sampling streams may differ between backends; randomized parity tests
//! control sampled values separately.

mod automorphism;
mod conversion;
mod decryption;
pub mod derived;
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

/// Explicitly forwards every core operation family except backend-supplied sampling.
///
/// Select the individual family macros instead when replacing an operation. The
/// reference gadget-product and external-product bodies require contiguous DFT limbs.
#[macro_export]
macro_rules! impl_core_reference_full {
    ($be:ty) => {
        $crate::impl_conversion_reference_full!($be);
        $crate::impl_decryption_reference_full!($be);
        $crate::impl_encryption_reference_full!($be);
        $crate::impl_operations_reference_full!($be);
        $crate::impl_polynomial_evaluation_derived_full!($be);
        $crate::impl_gglwe_external_product_derived_full!($be);
        $crate::impl_gglwe_keyswitch_derived_full!($be);
        $crate::impl_ggsw_external_product_derived_full!($be);
        $crate::impl_ggsw_keyswitch_derived_full!($be);
        $crate::impl_automorphism_reference_full!($be);
        $crate::impl_glwe_external_product_reference_full!($be);
        $crate::impl_glwe_keyswitch_reference_full!($be);
        $crate::impl_glwe_packing_derived_full!($be);
        $crate::impl_glwe_trace_derived_full!($be);
        $crate::impl_linear_transformation_reference_full!($be);
        $crate::impl_lwe_keyswitch_reference_full!($be);
        $crate::impl_glwe_tensoring_reference!($be);
        $crate::impl_gglwe_product_digits_strided_reference!($be);
    };
}

pub use crate::{
    impl_automorphism_reference_full, impl_conversion_reference_full, impl_core_reference_full, impl_decryption_reference_full,
    impl_encryption_reference_full, impl_gglwe_external_product_derived_full, impl_gglwe_keyswitch_derived_full,
    impl_gglwe_product_digits_strided_reference, impl_ggsw_external_product_derived_full, impl_ggsw_keyswitch_derived_full,
    impl_ggsw_rotate_derived_full, impl_glwe_add_reference_full, impl_glwe_copy_reference_full,
    impl_glwe_external_product_reference_full, impl_glwe_keyswitch_reference_full, impl_glwe_mul_const_reference_full,
    impl_glwe_mul_plain_reference_full, impl_glwe_mul_xp_minus_one_reference_full, impl_glwe_negate_reference_full,
    impl_glwe_normalize_reference_full, impl_glwe_packing_derived_full, impl_glwe_rotate_reference_full,
    impl_glwe_shift_reference_full, impl_glwe_sub_reference_full, impl_glwe_tensoring_reference, impl_glwe_trace_derived_full,
    impl_glwe_zero_reference_full, impl_linear_transformation_reference_full, impl_lwe_keyswitch_reference_full,
    impl_operations_reference_full, impl_polynomial_evaluation_derived_full,
};

pub use crate::reference::{encryption::*, operations::*};
