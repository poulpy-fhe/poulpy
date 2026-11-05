use crate::CKKSResult as Result;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_hal::layouts::{Backend, ScratchArena};

use poulpy_core::layouts::{BSGSMeta, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    layouts::{CKKSCiphertextOwned, eval_mod::EvalMod},
};

/// Homomorphic modular reduction (`x mod 1`) via a periodic-function polynomial
/// approximation, the core non-linear step of CKKS bootstrapping.
///
/// The reduction is configured by an [`EvalMod`] (compiled from an
/// [`EvalModPlan`](crate::layouts::eval_mod::EvalModPlan)
/// and uploaded to this backend); see the [`eval_mod`](crate::reference::eval_mod)
/// module for the base-polynomial / range-extension / inverse pipeline.
pub trait CKKSEvalModOps<BE: Backend> {
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
    /// Consumes `params.consumed_bits()` bits of `log_budget`; errors if
    /// `ct` has insufficient remaining capacity. `tsk` is the tensor
    /// (relinearization) key used by the squaring steps, and `scratch` must hold
    /// at least [`Self::ckks_eval_mod_tmp_bytes`] bytes.
    fn ckks_eval_mod<R, C, P, F, H>(
        &self,
        res: &mut R,
        ct: &C,
        params: &EvalMod<F, P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
        H: GetTensorKey<BE>;

    /// [`Self::ckks_eval_mod`] on an owned `input`, which becomes the first power of the evaluation in place of a copy.
    /// Its allocation holds the `log_budget + f_mod_log_delta` bits EvalMod works at. Sized by
    /// [`Self::ckks_eval_mod_tmp_bytes`].
    ///
    /// # Errors
    ///
    /// As [`Self::ckks_eval_mod`], and [`InvalidOperand`](crate::CKKSCompositionError::InvalidOperand) for an allocation
    /// narrower than that width, each before any write to `res`.
    fn ckks_eval_mod_owned<R, P, F, H>(
        &self,
        res: &mut R,
        input: CKKSCiphertextOwned<BE>,
        params: &EvalMod<F, P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
        H: GetTensorKey<BE>;
}
