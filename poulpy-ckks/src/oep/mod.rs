//! Backend override seams: one `unsafe trait CKKS*Impl<BE>` per op family.
//!
//! Each OEP trait is keyed on the backend type and is the contract a backend implements (or inherits) to power the corresponding [`api`](crate::api) trait through the delegates layer.
//! The layering is `api::CKKS*Ops` → delegates on `Module<BE>` → `oep::CKKS*Impl` (this module, bound-free) ← `default::CKKS*Default` (the reference implementation, carrying per-method bounds).
//!
//! # Wiring patterns
//!
//! Most families follow the **opt-in marker** pattern: the family's blanket `unsafe impl<BE> CKKS*Impl<BE> for BE` is gated on `Module<BE>: CKKS*Default<BE>`, and a backend opts in with the family's one-line `impl_ckks_*_defaults!` macro (or bypasses the reference chain entirely by implementing the OEP trait natively).
//! Three kinds of family are deliberate exceptions:
//!
//! - **Unconditional blankets** — [`CKKSEvalModImpl`]: pure compositions of already-wired families, so they blanket over any backend whose constituent families are wired; there is no per-backend macro because there is nothing backend-specific to opt into.
//! - **Scalar-generic encoding seams** — [`CKKSEncodingImpl<BE, F>`], [`DFTMatrixImpl<BE, F>`], and [`CKKSPaCoCoeffEncodingImpl`]: parameterized by the encoding scalar and tied to the backend's FFT/codec plumbing, they are wired by backend-crate-side macros (e.g. `impl_ckks_encoding_*!` in the CPU backends) rather than by crate-side default markers, keeping host/FFT bounds out of this crate's API per the no-host-bounds rule.
//! - **Narrow protocol seam** — [`CKKSEncapsulatedModUpImpl`] lets backends optimize bootstrapping's dense-to-sparse → ModUp → sparse-to-dense stage.
//! - **No-OEP families** — the remaining composite ops (`CKKSMulAddOps`, `CKKSMulSubOps`, `CKKSAffineOps`, `CKKSAddManyOps`, `CKKSDotProductOps`, linear transformations): pure api-level compositions of other families' ops, implemented directly on `Module<BE>` in the delegates layer with no override seam of their own — overriding their constituents overrides them.

mod add;
mod bootstrapping;
mod carry_verb;
mod ckks_impl;
mod conjugate;
mod copy;
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
pub use add::impl_ckks_add_defaults;
pub use bootstrapping::{CKKSEncapsulatedModUpImpl, impl_ckks_encapsulated_mod_up_default};
pub use ckks_impl::CKKSImpl;
pub use conjugate::CKKSConjugateImpl;
pub use conjugate::impl_ckks_conjugate_defaults;
pub use copy::CKKSCopyImpl;
pub use copy::impl_ckks_copy_defaults;
pub use dft::{DFTDefault, DFTImpl, DFTMatrixDefault, DFTMatrixImpl, impl_ckks_dft_defaults};
pub use encoding::CKKSEncodingImpl;
pub use encryption::CKKSEncryptionImpl;
pub use encryption::impl_ckks_encryption_defaults;
pub use eval_mod::CKKSEvalModImpl;
pub use imag::CKKSImagImpl;
pub use imag::impl_ckks_imag_defaults;
pub use mul::CKKSMulImpl;
pub use mul::impl_ckks_mul_defaults;
pub use neg::CKKSNegImpl;
pub use neg::impl_ckks_neg_defaults;
pub use paco::CKKSPaCoCoeffEncodingImpl;
pub use plaintext::CKKSPlaintextZnxImpl;
pub use plaintext::impl_ckks_plaintext_defaults;
pub use polynomial_evaluation::CKKSPolynomialEvaluationImpl;
pub use pow2::CKKSPow2Impl;
pub use pow2::impl_ckks_pow2_defaults;
pub use rotate::CKKSRotateImpl;
pub use rotate::impl_ckks_rotate_defaults;
pub use ship::CKKSShipCoeffEncodingImpl;
pub use sub::CKKSSubImpl;
pub use sub::impl_ckks_sub_defaults;

pub use poulpy_hal::oep::SetNormalizationState;

use poulpy_hal::layouts::{Data, NormalizationState, Normalized, Unnormalized, ZnxWord};

use crate::layouts::CKKSCiphertext;

/// See [`SetNormalizationState`]: `set_normalized` is the backend-implementor
/// relabel with no normalization pass, reserved for fused kernels inside
/// backend crates. Scheme code must go through [`CKKSCiphertext::normalize`].
impl<D: Data, W: ZnxWord, S: NormalizationState> SetNormalizationState for CKKSCiphertext<D, W, S> {
    type WithState<T: NormalizationState> = CKKSCiphertext<D, W, T>;

    fn set_unnormalized(self) -> CKKSCiphertext<D, W, Unnormalized> {
        self.into_unnormalized()
    }

    unsafe fn set_normalized(self) -> CKKSCiphertext<D, W, Normalized> {
        let meta = self.meta;
        // SAFETY: forwarded caller contract.
        CKKSCiphertext::from_inner(unsafe { self.inner.set_normalized() }, meta)
    }
}
