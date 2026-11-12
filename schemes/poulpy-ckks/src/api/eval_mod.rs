use crate::CKKSResult as Result;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_hal::layouts::{Backend, Ring, ScratchArena, Standard};

use poulpy_core::layouts::{BSGSMeta, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta};

use crate::{CKKSCtBounds, SetCKKSInfos, layouts::eval_mod::EvalMod};

/// Homomorphic modular reduction (`x mod 1`) via a periodic-function polynomial
/// approximation, the core non-linear step of CKKS bootstrapping.
///
/// The reduction is configured by an [`EvalMod`] (compiled from an
/// [`EvalModPlan`](crate::layouts::eval_mod::EvalModPlan)
/// and uploaded to this backend); see the [`eval_mod`](crate::reference::eval_mod)
/// module for the base-polynomial / range-extension / inverse pipeline.
pub trait CKKSEvalModOps<BE: Backend, Rg: Ring = Standard> {
    /// Scratch space, in bytes, required by [`Self::ckks_eval_mod`] for an output
    /// shaped like `res` for input `ct` with relinearization key `tsk`.
    /// Pass the same `res`/`ct`/`params`/`tsk` you will evaluate with, since
    /// EvalMod raises the input to `params.plan.f_mod_log_delta` internally.
    fn ckks_eval_mod_tmp_bytes<R, C, P, F, T>(&self, res: &R, ct: &C, params: &EvalMod<F, P>, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        C: CKKSCtBounds,
        P: CKKSCtBounds,
        T: GGLWEInfos;

    /// Evaluates the configured `x mod 1` approximation of `ct` into `res`.
    ///
    /// `ct` is working storage: it becomes the first power of the evaluation, with no copy, so its content and metadata
    /// are unspecified on a successful return, and a caller that needs it again passes a copy. An allocation narrower
    /// than the `log_budget + f_mod_log_delta` bits EvalMod works at costs a copy in `scratch`, which the query counts.
    ///
    /// Consumes `params.consumed_bits()` bits of `log_budget`. `tsk` is the tensor (relinearization) key of the
    /// products, and `scratch` must hold at least [`Self::ckks_eval_mod_tmp_bytes`] bytes.
    ///
    /// # Errors
    ///
    /// Before any write, so that a refusal leaves `ct` and `res` unchanged:
    ///
    /// - [`InvalidOperand`](crate::CKKSCompositionError::InvalidOperand): a replicated slot layout or a rank other
    ///   than 1, and [`InvalidPlan`](crate::CKKSCompositionError::InvalidPlan): an `ExpCmplx` EvalMod on a ring of
    ///   real slots.
    /// - [`InsufficientHomomorphicCapacity`](crate::CKKSCompositionError::InsufficientHomomorphicCapacity): `ct` holds
    ///   fewer than `params.consumed_bits()` bits of budget.
    /// - [`MissingRelinearizationKey`](crate::CKKSCompositionError::MissingRelinearizationKey): an evaluation with a
    ///   ciphertext product whose `tsk` does not cover its widest product, of `log_budget + f_mod_log_delta` bits.
    /// - `CKKSError::Internal`: an evaluation with a ciphertext product given an arena below the query at the layout of
    ///   the key `tsk` answers for that width.
    fn ckks_eval_mod<R, C, P, F, H>(
        &self,
        res: &mut R,
        ct: &mut C,
        params: &EvalMod<F, P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, Rg> + IntPolyInfos + CKKSCtBounds + BSGSMeta + crate::layouts::CKKSMaskedConstants<BE, Rg>,
        H: GetTensorKey<BE, Rg>;
}
