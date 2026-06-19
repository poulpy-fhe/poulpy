use anyhow::Result;
use poulpy_hal::layouts::{Backend, ScratchArena};

use poulpy_core::layouts::{BSGSMeta, GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta};

use crate::{CKKSCtBounds, SetCKKSInfos, layouts::eval_mod::EvalMod};

/// Homomorphic modular reduction (`x mod 1`) via a periodic-function polynomial
/// approximation, the core non-linear step of CKKS bootstrapping.
///
/// The reduction is configured by an [`EvalMod`] (compiled from an
/// [`EvalModPlan`](crate::layouts::eval_mod::EvalModPlan)
/// and uploaded to this backend); see the [`eval_mod`](crate::default::eval_mod)
/// module for the base-polynomial / range-extension / inverse pipeline.
pub trait CKKSEvalModOps<BE: Backend> {
    /// Scratch space, in bytes, required by [`Self::ckks_eval_mod`] for an output
    /// shaped like `res` with relinearization key `tsk`. The size is independent
    /// of `params`; pass the same `res`/`tsk` you will evaluate with.
    fn ckks_eval_mod_tmp_bytes<R, P, F, T>(&self, res: &R, params: &EvalMod<F, P>, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        P: CKKSCtBounds,
        T: GGLWEInfos;

    /// Evaluates the configured `x mod 1` approximation of `ct` into `res`.
    ///
    /// Consumes `params.eval_depth() * log_delta` bits of `log_budget`; errors if
    /// `ct` has insufficient remaining capacity. `tsk` is the tensor
    /// (relinearization) key used by the squaring steps, and `scratch` must hold
    /// at least [`Self::ckks_eval_mod_tmp_bytes`] bytes.
    fn ckks_eval_mod<R, C, P, F>(
        &self,
        res: &mut R,
        ct: &C,
        params: &EvalMod<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta;
}
