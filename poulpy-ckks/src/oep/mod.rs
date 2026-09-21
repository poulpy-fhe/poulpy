//! Backend contracts selected by the public CKKS API through delegates.
//!
//! A backend explicitly implements each `*Impl` family. The
//! `impl_ckks_*_reference!` macros wire lower-layer algorithms from
//! [`crate::reference`]; a custom implementation can call those algorithms for
//! unchanged methods and replace other methods independently.
//!
//! Simple compositions of CKKS operations are crate-private derived defaults
//! on these contracts. They call the selected constituent operations, so their
//! overrides and scratch queries remain effective. Complete bootstrap, affine,
//! dot-product, and linear-transformation pipelines are API-only compositions.
//!
//! Scalar encoding and matrix-generation contracts preserve backend-resident
//! interfaces. Reference implementations carry their own lower-layer or host
//! staging requirements; these are not prerequisites for a custom override.
//!
//! An override must reproduce the reference circuit and pass parity against a
//! caller-selected comparison backend. Validation across backends is transitive.

mod add;
mod bootstrapping;
mod carry_verb;
mod ckks_impl;
mod conjugate;
mod copy;
pub(crate) mod derived;
mod dft;
mod encoding;
mod encryption;
mod eval_mod;
mod imag;
mod mul;
mod neg;
mod paco;
mod plaintext;
mod polynomial_evaluation;
mod pow2;
mod rotate;
mod ship;
mod sub;

pub use add::CKKSAddImpl;
pub use add::impl_ckks_add_reference;
pub use bootstrapping::{CKKSEncapsulatedModUpImpl, impl_ckks_encapsulated_mod_up_reference};
pub use ckks_impl::CKKSImpl;
pub use conjugate::CKKSConjugateImpl;
pub use conjugate::impl_ckks_conjugate_reference;
pub use copy::CKKSCopyImpl;
pub use copy::impl_ckks_copy_reference;
pub use dft::{DFTImpl, DFTMatrixImpl, impl_ckks_dft_reference};
pub use encoding::CKKSEncodingImpl;
pub use encryption::CKKSEncryptionImpl;
pub use encryption::impl_ckks_encryption_reference;
pub use eval_mod::{CKKSEvalModImpl, impl_ckks_eval_mod_reference};
pub use imag::CKKSImagImpl;
pub use imag::impl_ckks_imag_reference;
pub use mul::CKKSMulImpl;
pub use mul::impl_ckks_mul_reference;
pub use neg::CKKSNegImpl;
pub use neg::impl_ckks_neg_reference;
pub use paco::CKKSPaCoCoeffEncodingImpl;
pub use plaintext::CKKSPlaintextZnxImpl;
pub use plaintext::impl_ckks_plaintext_reference;
pub use polynomial_evaluation::{CKKSPolynomialEvaluationImpl, impl_ckks_polynomial_evaluation_reference};
pub use pow2::CKKSPow2Impl;
pub use pow2::impl_ckks_pow2_reference;
pub use rotate::CKKSRotateImpl;
pub use rotate::impl_ckks_rotate_reference;
pub use ship::CKKSShipCoeffEncodingImpl;
pub use sub::CKKSSubImpl;
pub use sub::impl_ckks_sub_reference;
