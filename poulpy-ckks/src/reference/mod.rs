//! The implementation of every CKKS operation.
//!
//! Each module composes one operation family from `poulpy-core` and the HAL.
//! These compositions define what the operations compute and are the only
//! validated circuit: a backend runs them through the `impl_ckks_*_reference!`
//! macros of [`crate::oep`], or overrides a family through its `oep` trait
//! with a faster route to the same result, which the parity suite pins to
//! these bodies. An override that computes anything else is a defect.

pub mod add;
pub mod bootstrapping;
pub(crate) mod carry_verb;
pub mod conjugate;
pub mod copy;
pub mod dft;
pub mod encryption;
pub mod eval_mod;
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
