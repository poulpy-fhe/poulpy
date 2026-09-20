//! Open extension points for `poulpy-core`.
//!
//! Public [`crate::api`] operations dispatch through backend `*Impl` traits.
//! Most families have a blanket `*Impl` implementation that calls an abstract
//! `*Reference` trait on [`poulpy_hal::layouts::Module`]. Implement that reference
//! trait to specialize the family. It has no HAL supertraits or default bodies;
//! each method can call the portable composition or provide its own algorithm.
//!
//! A backend can alternatively implement a `*Impl` trait directly and omit its
//! corresponding `*Reference` implementation. Public delegates require only the
//! selected `*Impl`; combining both choices for one family conflicts with the
//! blanket implementation.
//!
//! Portable bodies live in [`crate::reference`]. Arithmetic and encryption expose
//! separately named `*Composition` traits with `*_composition` methods, whose
//! blanket implementations carry the HAL requirements. The other families use
//! free functions. These helpers remain callable without occupying the override
//! contract. A reference body uses its own scratch query and dispatches through
//! other operation families where their contracts permit composition.
//!
//! Three families take a direct backend implementation instead: [`SamplingImpl`],
//! [`GLWETensoringImpl`], and [`GGLWEProductDigitsStridedImpl`]. Sampling is supplied
//! by the backend; tensoring and strided gadget products have explicit reference
//! forwarding macros. The existing `GLWETensoringReference` trait is a composition
//! helper, not the override boundary for tensoring.
//!
//! # Explicit reference opt-in
//!
//! `impl_*_reference_full!` macros forward a single reference contract. Group
//! macros [`crate::impl_operations_reference_full!`] and
//! [`crate::impl_encryption_reference_full!`] select all their subfamilies;
//! [`crate::impl_core_reference_full!`] selects every reference core family except
//! sampling. Use individual macros when replacing one family, and implement its
//! abstract trait by hand. Within that implementation, forward unchanged methods
//! through the corresponding `*Composition` trait or reference free function.
//!
//! Encryption's empty [`EncryptionReference`] aggregator requires all encryption
//! subfamilies; it supplies no implementations. After selecting those subfamilies
//! individually, implement the aggregator explicitly. Merely satisfying the HAL
//! requirements never opts a backend into an encryption or arithmetic override.
//!
//! A compiled example is the `poulpy-cpu-ref` test module
//! `src/tests/delegating_backend.rs`: it implements [`GLWERotateReference`] by
//! hand, forwards its scratch and assign methods, selects other arithmetic with
//! the family macros, and checks both parity and observable public-API dispatch.
//!
//! # Layout requirements
//!
//! Reference gadget digit products and GLWE external products form partial DFT
//! views and require [`poulpy_hal::layouts::Backend::DFT_LIMBS_CONTIGUOUS`]. Their
//! bodies assert that capability at monomorphization. A backend with another
//! representation must implement [`GGLWEProductDigitsStridedImpl`] and
//! [`GLWEExternalProductReference`] instead of opting into those reference bodies.
//! Other families can keep using their portable compositions. These restrictions
//! concern only the reference algorithms; the public contracts admit alternate
//! backend representations.
//!
//! Prepared factories, decompression, and internal keyswitch plumbing outside
//! this module are composition helpers, not independent backend override hooks.
//! Noise diagnostics requiring host-visible coefficients are separate from the
//! generic execution path.
//!
//! # Correctness
//!
//! The `unsafe` marker on `*Impl` traits assigns responsibility for the numerical,
//! layout, aliasing and scratch contracts to the backend implementor. Deterministic
//! parity tests compare canonical results and metadata directly with portable
//! `poulpy-cpu-ref` executions on identical logical inputs. Prepared byte layouts
//! and scratch byte counts need not match. Sampling streams may differ between
//! backends; randomized composition tests must control sampled values separately.

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

pub use crate::reference::{
    encryption::{
        GGLWECompressedEncryptSkReference, GGLWEEncryptSkReference, GGLWEToGGSWKeyCompressedEncryptSkReference,
        GGLWEToGGSWKeyEncryptSkReference, GGSWCompressedEncryptSkReference, GGSWEncryptSkReference,
        GLWEAutomorphismKeyCompressedEncryptSkReference, GLWEAutomorphismKeyEncryptSkReference, GLWECompressedEncryptSkReference,
        GLWEEncryptPkReference, GLWEEncryptSkReference, GLWEMaskFillReference, GLWEPublicKeyGenerateReference,
        GLWESwitchingKeyCompressedEncryptSkReference, GLWESwitchingKeyEncryptSkReference,
        GLWETensorKeyCompressedEncryptSkReference, GLWETensorKeyEncryptSkReference, GLWEToLWESwitchingKeyEncryptSkReference,
        LWEEncryptSkReference, LWEFillMaskReference, LWESwitchingKeyEncryptReference, LWEToGLWESwitchingKeyEncryptSkReference,
    },
    operations::{
        GGSWRotateReference, GLWEAddReference, GLWECopyReference, GLWEMulConstReference, GLWEMulPlainReference,
        GLWEMulXpMinusOneReference, GLWENegateReference, GLWENormalizeReference, GLWERotateReference, GLWEShiftReference,
        GLWESubReference, GLWEZeroReference,
    },
};

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
        $crate::impl_polynomial_evaluation_reference_full!($be);
        $crate::impl_gglwe_automorphism_reference_full!($be);
        $crate::impl_gglwe_external_product_reference_full!($be);
        $crate::impl_gglwe_keyswitch_reference_full!($be);
        $crate::impl_ggsw_automorphism_reference_full!($be);
        $crate::impl_ggsw_external_product_reference_full!($be);
        $crate::impl_ggsw_keyswitch_reference_full!($be);
        $crate::impl_glwe_automorphism_reference_full!($be);
        $crate::impl_glwe_external_product_reference_full!($be);
        $crate::impl_glwe_keyswitch_reference_full!($be);
        $crate::impl_glwe_packing_reference_full!($be);
        $crate::impl_glwe_trace_reference_full!($be);
        $crate::impl_linear_transformation_reference_full!($be);
        $crate::impl_lwe_keyswitch_reference_full!($be);
        $crate::impl_glwe_tensoring_reference!($be);
        $crate::impl_gglwe_product_digits_strided_reference!($be);
    };
}

pub use crate::{
    impl_core_reference_full, impl_gglwe_compressed_encrypt_sk_reference_full, impl_gglwe_encrypt_sk_reference_full,
    impl_gglwe_to_ggsw_key_compressed_encrypt_sk_reference_full, impl_gglwe_to_ggsw_key_encrypt_sk_reference_full,
    impl_ggsw_compressed_encrypt_sk_reference_full, impl_ggsw_encrypt_sk_reference_full, impl_ggsw_rotate_reference_full,
    impl_glwe_add_reference_full, impl_glwe_automorphism_key_compressed_encrypt_sk_reference_full,
    impl_glwe_automorphism_key_encrypt_sk_reference_full, impl_glwe_compressed_encrypt_sk_reference_full,
    impl_glwe_copy_reference_full, impl_glwe_encrypt_pk_reference_full, impl_glwe_encrypt_sk_reference_full,
    impl_glwe_mask_fill_reference_full, impl_glwe_mul_const_reference_full, impl_glwe_mul_plain_reference_full,
    impl_glwe_mul_xp_minus_one_reference_full, impl_glwe_negate_reference_full, impl_glwe_normalize_reference_full,
    impl_glwe_public_key_generate_reference_full, impl_glwe_rotate_reference_full, impl_glwe_shift_reference_full,
    impl_glwe_sub_reference_full, impl_glwe_switching_key_compressed_encrypt_sk_reference_full,
    impl_glwe_switching_key_encrypt_sk_reference_full, impl_glwe_tensor_key_compressed_encrypt_sk_reference_full,
    impl_glwe_tensor_key_encrypt_sk_reference_full, impl_glwe_to_lwe_switching_key_encrypt_sk_reference_full,
    impl_glwe_zero_reference_full, impl_lwe_encrypt_sk_reference_full, impl_lwe_mask_fill_reference_full,
    impl_lwe_switching_key_encrypt_reference_full, impl_lwe_to_glwe_switching_key_encrypt_sk_reference_full,
    impl_operations_reference_full, impl_polynomial_evaluation_reference_full,
};
