//! Backend contracts selected by the public CKKS API through delegates.
//!
//! Outputs follow the core [noise metadata rule](poulpy_core::oep#noise-metadata).
//! Delegates only forward.
//!
//! A backend explicitly implements each `*Impl` family. The
//! `impl_<family>_reference_full!` macros wire lower-layer algorithms from
//! [`crate::reference`]; a custom implementation can call those algorithms for
//! unchanged methods and replace other methods independently.
//!
//! Every API method dispatches through a same-named OEP method. Compositions of
//! selected operations (sums, multiply-accumulate and affine forms,
//! approximations, ring maps, encoding checks, the PaCo and SHIP drivers) are
//! crate-private derived defaults on these contracts. They call the selected
//! constituent operations, so their overrides and scratch queries remain
//! effective. A family's supertraits are exactly the families its defaults call.
//!
//! Scalar encoding and matrix-generation contracts preserve backend-resident
//! interfaces. Reference implementations carry their own lower-layer or host
//! staging requirements; these are not prerequisites for a custom override.
//!
//! An override must reproduce the reference circuit and pass parity against a
//! caller-selected comparison backend. Validation across backends is transitive.
//!
//! Conjugation, multiplication by `i`, complex polynomial evaluation, DFT,
//! encapsulated ModUp, the conjugate-invariant ring maps, PaCo and SHIP exist
//! only on the standard ring: conjugate-invariant backends do not implement them.
//! EvalMod runs on both rings; the conjugate-invariant ring refuses `ExpCmplx`.

mod add;
mod approximation;
mod bootstrapping;
mod carry_verb;
mod ckks_impl;
mod composite;
mod conjugate;
mod copy;
pub(crate) mod derived;
mod dft;
mod dot_product;
mod encoding;
mod encryption;
mod eval_mod;
mod fold;
mod imag;
mod inner_sum;
mod linear_transformation;
mod mul;
mod neg;
mod paco;
mod plaintext;
mod polynomial_evaluation;
mod pow2;
mod ring_map;
mod rotate;
mod ship;
mod sub;

pub use add::CKKSAddImpl;
pub use add::impl_ckks_add_reference_full;
pub use approximation::{CKKSApproximationImpl, impl_ckks_approximation_derived_full};
pub use bootstrapping::{
    CKKSBootstrappingCIImpl, CKKSBootstrappingImpl, CKKSEncapsulatedModUpImpl, CKKSModUpFirstFactorImpl,
    impl_ckks_bootstrapping_ci_reference_full, impl_ckks_bootstrapping_reference_full,
    impl_ckks_encapsulated_mod_up_reference_full, impl_ckks_mod_up_first_factor_reference_full,
};
pub use ckks_impl::CKKSImpl;
pub use composite::{
    CKKSAddManyImpl, CKKSAffineImpl, CKKSMulAddImpl, CKKSMulSubImpl, impl_ckks_add_many_derived_full,
    impl_ckks_affine_derived_full, impl_ckks_mul_add_derived_full, impl_ckks_mul_sub_derived_full,
};
pub use conjugate::CKKSConjugateImpl;
pub use conjugate::impl_ckks_conjugate_reference_full;
pub use copy::CKKSCopyImpl;
pub use copy::impl_ckks_copy_reference_full;
pub use dft::{DFTImpl, DFTMatrixImpl, impl_dft_matrix_reference_full, impl_dft_reference_full};
pub use dot_product::{CKKSDotProductImpl, impl_ckks_dot_product_reference_full};
pub use encoding::CKKSEncodingImpl;
pub use encryption::CKKSEncryptionImpl;
pub use encryption::impl_ckks_encryption_reference_full;
pub use eval_mod::{CKKSEvalModImpl, impl_ckks_eval_mod_reference_full};
pub use fold::{CKKSFoldImpl, CKKSFoldLayoutImpl, impl_ckks_fold_layout_reference_full, impl_ckks_fold_reference_full};
pub use imag::CKKSImagImpl;
pub use imag::impl_ckks_imag_reference_full;
pub use inner_sum::{CKKSInnerSumImpl, impl_ckks_inner_sum_derived_full};
pub use linear_transformation::{CKKSLinearTransformationImpl, impl_ckks_linear_transformation_reference_full};
pub use mul::CKKSMulImpl;
pub use mul::impl_ckks_mul_reference_full;
pub use neg::CKKSNegImpl;
pub use neg::impl_ckks_neg_reference_full;
pub use paco::{CKKSPaCoCoeffEncodingImpl, CKKSPaCoImpl, impl_ckks_paco_derived_full};
pub use plaintext::CKKSPlaintextZnxImpl;
pub use plaintext::impl_ckks_plaintext_znx_reference_full;
pub use polynomial_evaluation::{
    CKKSComplexPolynomialEvaluationImpl, CKKSPolynomialEvaluationImpl, impl_ckks_complex_polynomial_evaluation_reference_full,
    impl_ckks_polynomial_evaluation_reference_full,
};
pub use pow2::CKKSPow2Impl;
pub use pow2::impl_ckks_pow2_reference_full;
pub use ring_map::{CKKSRingMapImpl, impl_ckks_ring_map_derived_full};
pub use rotate::CKKSRotateImpl;
pub use rotate::impl_ckks_rotate_reference_full;
pub use ship::{CKKSShipCoeffEncodingImpl, CKKSShipImpl, impl_ckks_ship_reference_full};
pub use sub::CKKSSubImpl;
pub use sub::impl_ckks_sub_reference_full;

/// Selects the reference implementation of every scalar-independent CKKS family that serves both rings, the
/// families [`CKKSImpl`] bundles minus the standard-only ones.
///
/// Left out, each with its own opt-in: the standard-only families (`CKKSImagImpl`, `CKKSConjugateImpl`,
/// `CKKSComplexPolynomialEvaluationImpl`, `DFTImpl`, `CKKSFoldLayoutImpl`, `CKKSFoldImpl<R>`, `CKKSBootstrappingImpl`,
/// `CKKSBootstrappingCIImpl`, `CKKSEncapsulatedModUpImpl`, `CKKSModUpFirstFactorImpl`, `CKKSRingMapImpl`,
/// `CKKSPaCoImpl`, `CKKSShipImpl`), the scalar-generic seams (`CKKSEncodingImpl<F>`, `DFTMatrixImpl<F>`) and the
/// PaCo and SHIP coefficient encodings, which have no reference opt-in.
#[macro_export]
macro_rules! impl_ckks_reference_full {
    ($be:ty) => {
        $crate::impl_ckks_plaintext_znx_reference_full!($be);
        $crate::impl_ckks_copy_reference_full!($be);
        $crate::impl_ckks_add_reference_full!($be);
        $crate::impl_ckks_sub_reference_full!($be);
        $crate::impl_ckks_encryption_reference_full!($be);
        $crate::impl_ckks_neg_reference_full!($be);
        $crate::impl_ckks_pow2_reference_full!($be);
        $crate::impl_ckks_rotate_reference_full!($be);
        $crate::impl_ckks_inner_sum_derived_full!($be);
        $crate::impl_ckks_mul_reference_full!($be);
        $crate::impl_ckks_dot_product_reference_full!($be);
        $crate::impl_ckks_linear_transformation_reference_full!($be);
        $crate::impl_ckks_polynomial_evaluation_reference_full!($be);
        $crate::impl_ckks_eval_mod_reference_full!($be);
        $crate::impl_ckks_add_many_derived_full!($be);
        $crate::impl_ckks_mul_add_derived_full!($be);
        $crate::impl_ckks_affine_derived_full!($be);
        $crate::impl_ckks_mul_sub_derived_full!($be);
        $crate::impl_ckks_approximation_derived_full!($be);
    };
}
pub use crate::impl_ckks_reference_full;
