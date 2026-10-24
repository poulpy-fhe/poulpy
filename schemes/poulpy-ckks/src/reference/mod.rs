//! Callable CKKS algorithms built from core and HAL operations.
//!
//! These bodies define the circuit for each reference operation. They are
//! available independently of backend wiring: a backend opts into the matching
//! `impl_<family>_reference_full!` macro or calls a reference body from a custom `*Impl`.
//! Overrides must reproduce the circuit and pass the reusable parity tests
//! against a caller-selected comparison backend.
//!
//! Same-layer compositions, the PaCo and SHIP drivers included, are private
//! derived defaults in [`crate::oep`].

pub mod add;
pub mod bootstrapping;
pub(crate) mod carry_verb;
pub mod conjugate;
pub mod copy;
pub mod dft;
pub mod dot_product;
pub mod encoding;
pub mod encryption;
pub mod eval_mod;
mod eval_mod_scratch;
pub mod first_factor;
pub mod fold;
pub mod imag;
pub mod linear_transformation;
pub mod linear_transformation_diagonals;
pub mod mul;
pub mod neg;
pub mod paco;
pub mod plaintext;
pub mod polynomial_evaluation;
pub mod pow2;
pub mod rotate;
pub mod ship;
pub mod sub;

pub use add::CKKSAddReference;
pub use dft::gen_dft_matrices;
pub use dot_product::CKKSDotProductReference;
pub use eval_mod::CKKSEvalModOpsReference;
pub use linear_transformation::CKKSLinearTransformationReference;
pub use linear_transformation_diagonals::{
    ckks_encode_linear_transformation_from_diagonals, ckks_encode_linear_transformation_into,
    ckks_encode_linear_transformation_tmp_bytes,
};
pub use plaintext::CKKSPlaintextReference;
pub use polynomial_evaluation::PolynomialEvaluationReference;
pub use sub::CKKSSubReference;
