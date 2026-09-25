//! Callable CKKS algorithms built from core and HAL operations.
//!
//! These bodies define the circuit for each reference operation. They are
//! available independently of backend wiring: a backend opts into the matching
//! `impl_ckks_*_reference!` macro or calls a reference body from a custom `*Impl`.
//! Overrides must reproduce the circuit and pass the reusable parity tests
//! against a caller-selected comparison backend.
//!
//! Simple same-layer compositions are private derived defaults in
//! [`crate::oep`]. API-only protocol pipelines retain private implementation
//! helpers alongside their lower-layer building blocks.

pub mod add;
pub mod bootstrapping;
pub(crate) mod carry_verb;
pub(crate) mod ci_bootstrapping;
pub mod conjugate;
pub mod copy;
pub mod dft;
pub mod encoding;
pub mod encryption;
pub mod eval_mod;
mod eval_mod_scratch;
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
pub use eval_mod::CKKSEvalModOpsReference;
pub use linear_transformation_diagonals::ckks_encode_linear_transformation_from_diagonals;
pub use plaintext::CKKSPlaintextReference;
pub use polynomial_evaluation::PolynomialEvaluationReference;
pub use sub::CKKSSubReference;
