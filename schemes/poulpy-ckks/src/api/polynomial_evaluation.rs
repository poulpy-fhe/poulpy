use crate::CKKSResult as Result;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_hal::layouts::{Backend, Ring, ScratchArena, Standard};

use poulpy_core::layouts::{BSGSMeta, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta};

use crate::{CKKSCtBounds, SetCKKSInfos, layouts::CKKSCiphertextOwned, polynomial::ComplexBSGSPolynomial};

pub use poulpy_core::layouts::{BSGSPolynomialInfos, BabyStep, Basis, Parity, PolynomialInputTransform, PowerBasisHelper};

pub trait CKKSPolynomialEvaluationOps<BE: Backend, Rg: Ring = Standard> {
    /// Evaluates the real-coefficient polynomial `poly` on the powers of `power_basis` into `res`.
    ///
    /// Scratch is bounded by [`CKKSAllOpsTmpBytes::ckks_all_ops_tmp_bytes`](crate::api::CKKSAllOpsTmpBytes::ckks_all_ops_tmp_bytes).
    ///
    /// # Errors
    ///
    /// `InvalidOperand` when `poly` has no baby step, an empty one or a basis other than that of `power_basis`, or when
    /// its constants are of another [slot layout](crate::SlotLayout) than the power basis, and `PlaintextDegreeMismatch`
    /// for a constant whose degree does not embed into the power basis', before anything is written.
    /// Then the errors of the evaluation: an untyped error for a power `power_basis` lacks, and those of its products
    /// and additions (`MissingRelinearizationKey` or `RelinearizationKeyTooNarrow`, `MultiplicationPrecisionUnderflow`,
    /// `InsufficientHomomorphicCapacity`), the last of which can come after `res` is written: `res` is untouched only
    /// on these first refusals.
    ///
    /// # Metadata
    ///
    /// ```text
    /// log_delta_out  = log_delta of the powers
    /// log_budget_out = log_budget of the powers − poly.consumed_bits(log_delta, coefficient log_delta)
    /// k_out          = min(log_delta_out + log_budget_out, res.k)   (a narrower res drops budget)
    /// slots          = join of the powers' and every constant c's
    /// slot_layout    = the powers'
    /// log_sparsity   = min(the powers', log2(n / c.n) of every constant c), n the powers' degree
    /// canonical flag = set after an odd polynomial of one baby step summed by one dot product, or when the final copy
    ///                  into a res narrower than the evaluated value normalizes, unless the leading term is added
    ///                  after it (a last baby step of one constant, whose power the basis holds); cleared otherwise
    /// ```
    ///
    /// The slot layout of the constants picks the circuit of each baby step, both routes giving that metadata:
    /// integer constants on the full layout, plaintext products rounded once with shaping on a replicated layout.
    fn ckks_eval_poly_real_const_coeffs_from_power_basis<R, B, A, G, H>(
        &self,
        res: &mut R,
        power_basis: &G,
        poly: &B,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        B: BSGSPolynomialInfos<BE, Rg>,
        B::Coeffs: CKKSCtBounds + GLWEToBackendRef<BE, Rg>,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + BSGSMeta,
        G: PowerBasisHelper<BE, A, Rg>,
        H: GetTensorKey<BE, Rg>;

    /// Builds the power basis internally then evaluates a real-coefficient
    /// polynomial.
    ///
    /// Scratch is bounded by [`CKKSAllOpsTmpBytes::ckks_all_ops_tmp_bytes`](crate::api::CKKSAllOpsTmpBytes::ckks_all_ops_tmp_bytes).
    ///
    /// # Errors
    ///
    /// `InvalidOperand` when `bsgs` has no baby step or an empty one, or when its constants are of another slot layout
    /// than `src`, and `PlaintextDegreeMismatch` for a constant whose degree does not embed into `src`'s, before any
    /// work. Then the errors of the input transform and of
    /// the power basis' products, which write only temporaries, then those of
    /// [`Self::ckks_eval_poly_real_const_coeffs_from_power_basis`] and, for the transforms times the input, of the
    /// final product by `src`, which can come after `dst` is written: `dst` is untouched only on the errors before
    /// the evaluation.
    ///
    /// # Metadata
    ///
    /// As [`Self::ckks_eval_poly_real_const_coeffs_from_power_basis`] on the power basis of `src`, after its input
    /// transform, whose bits `bsgs.consumed_bits` counts.
    fn ckks_eval_poly_real_const_coeffs<R, S, B, H>(
        &self,
        dst: &mut R,
        src: &S,
        bsgs: &B,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        B: BSGSPolynomialInfos<BE, Rg>,
        B::Coeffs: CKKSCtBounds + GLWEToBackendRef<BE, Rg>,
        H: GetTensorKey<BE, Rg>,
        CKKSCiphertextOwned<BE, Rg>: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos;
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
    /// `InvalidOperand` when `poly.re` has no baby step, when `poly.re` and `poly.im` differ in baby-step count, in
    /// degree, parity or basis, when their basis is not that of
    /// `power_basis`, or when their constants are of another slot layout than the power basis, and
    /// `PlaintextDegreeMismatch` for a constant whose degree does not embed into the power basis', before anything is
    /// written. Then the errors of the evaluation as
    /// [`CKKSPolynomialEvaluationOps::ckks_eval_poly_real_const_coeffs_from_power_basis`], `res` being untouched only
    /// on these first refusals.
    ///
    /// # Metadata
    ///
    /// As [`CKKSPolynomialEvaluationOps::ckks_eval_poly_real_const_coeffs_from_power_basis`], except that each baby
    /// step adds `i` times its evaluation of `poly.im`, which stamps `slots = Complex` and clears the canonical flag,
    /// set again when the final copy into a `res` narrower than the evaluated value normalizes, unless the leading term
    /// is added after it.
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
        C: GLWEInfos + BSGSMeta + CKKSCtBounds + IntPolyInfos + GLWEToBackendRef<BE>,
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
    /// builds, when `poly.re` and `poly.im` differ in input transform, and when their constants are of another slot
    /// layout than `src`, and `PlaintextDegreeMismatch` for a constant whose degree does not embed into `src`'s, before
    /// any work. Then the errors of the
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
        C: GLWEInfos + BSGSMeta + CKKSCtBounds + IntPolyInfos + GLWEToBackendRef<BE>,
        H: GetTensorKey<BE>,
        CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos;
}
