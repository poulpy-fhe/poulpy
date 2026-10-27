use crate::CKKSResult as Result;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_hal::layouts::{Backend, ScratchArena};

use poulpy_core::layouts::{BSGSMeta, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta};

use crate::{CKKSCtBounds, SetCKKSInfos, layouts::CKKSCiphertextOwned, polynomial::ComplexBSGSPolynomial};

pub use poulpy_core::layouts::{BSGSPolynomialInfos, BabyStep, Basis, Parity, PolynomialInputTransform, PowerBasisHelper};

pub trait CKKSPolynomialEvaluationOps<BE: Backend> {
    /// Evaluates the real-coefficient polynomial `poly` on the powers of `power_basis` into `res`.
    ///
    /// Scratch is bounded by [`CKKSAllOpsTmpBytes::ckks_all_ops_tmp_bytes`](crate::api::CKKSAllOpsTmpBytes::ckks_all_ops_tmp_bytes).
    ///
    /// # Errors
    ///
    /// `InvalidOperand` when `poly` has no baby step or a basis other than that of `power_basis`, before anything is
    /// written. Then the errors of the evaluation: an untyped error for a power `power_basis` lacks, and those of its
    /// products and additions (`MissingRelinearizationKey`, `MultiplicationPrecisionUnderflow`,
    /// `InsufficientHomomorphicCapacity`), the last of which can come after `res` is written: `res` is untouched only
    /// on the `InvalidOperand` refusals.
    fn ckks_eval_poly_real_const_coeffs_from_power_basis<R, B, A, G, H>(
        &self,
        res: &mut R,
        power_basis: &G,
        poly: &B,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds,
        A: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        H: GetTensorKey<BE>;

    /// Builds the power basis internally then evaluates a real-coefficient
    /// polynomial.
    ///
    /// Scratch is bounded by [`CKKSAllOpsTmpBytes::ckks_all_ops_tmp_bytes`](crate::api::CKKSAllOpsTmpBytes::ckks_all_ops_tmp_bytes).
    ///
    /// # Errors
    ///
    /// `InvalidOperand` when `bsgs` has no baby step, before any work. Then the errors of the input transform and of
    /// the power basis' products, which write only temporaries, then those of
    /// [`Self::ckks_eval_poly_real_const_coeffs_from_power_basis`] and, for the transforms times the input, of the
    /// final product by `src`, which can come after `dst` is written: `dst` is untouched only on the errors before
    /// the evaluation.
    fn ckks_eval_poly_real_const_coeffs<R, S, B, H>(
        &self,
        dst: &mut R,
        src: &S,
        bsgs: &B,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds,
        H: GetTensorKey<BE>,
        CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos;
}

/// Complex-coefficient polynomial evaluation; standard ring only.
pub trait CKKSComplexPolynomialEvaluationOps<BE: Backend> {
    /// Evaluates a complex-coefficient polynomial `Σ_k (a_k + i·b_k)·z^k`,
    /// where `poly.re`/`poly.im` are the matched real/imag BSGS decompositions
    /// (identical baby-step schedule).
    /// Scratch is bounded by [`CKKSAllOpsTmpBytes::ckks_all_ops_tmp_bytes`](crate::api::CKKSAllOpsTmpBytes::ckks_all_ops_tmp_bytes).
    ///
    /// # Errors
    ///
    /// `InvalidOperand` when `poly.re` has no baby step, when `poly.re` and `poly.im` differ in baby-step count, in the
    /// length of a baby step, in degree, parity or basis, or when their basis is not that of `power_basis`, before
    /// anything is written. Then the errors of the evaluation as
    /// [`CKKSPolynomialEvaluationOps::ckks_eval_poly_real_const_coeffs_from_power_basis`], `res` being untouched only
    /// on the `InvalidOperand` refusals.
    fn ckks_eval_poly_complex_const_coeffs_from_power_basis<R, C, A, G, H>(
        &self,
        res: &mut R,
        power_basis: &G,
        poly: &ComplexBSGSPolynomial<C>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds + IntPolyInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        H: GetTensorKey<BE>;

    /// Builds the power basis internally then evaluates a complex-coefficient
    /// polynomial.
    ///
    /// Scratch is bounded by [`CKKSAllOpsTmpBytes::ckks_all_ops_tmp_bytes`](crate::api::CKKSAllOpsTmpBytes::ckks_all_ops_tmp_bytes).
    ///
    /// # Errors
    ///
    /// `InvalidOperand` for the shapes of `poly` that
    /// [`Self::ckks_eval_poly_complex_const_coeffs_from_power_basis`] refuses, other than the basis of the power basis it
    /// builds, and when `poly.re` and `poly.im` differ in input transform, before any work. Then the errors of the
    /// evaluation as [`CKKSPolynomialEvaluationOps::ckks_eval_poly_real_const_coeffs`], `dst` being untouched only on
    /// the errors before the evaluation.
    fn ckks_eval_poly_complex_const_coeffs<R, S, C, H>(
        &self,
        dst: &mut R,
        src: &S,
        poly: &ComplexBSGSPolynomial<C>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<BE> + CKKSCtBounds,
        C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds + IntPolyInfos,
        H: GetTensorKey<BE>,
        CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos;
}
