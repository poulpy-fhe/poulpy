use crate::CKKSResult as Result;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_hal::layouts::{Backend, ScratchArena};

use poulpy_core::layouts::{BSGSMeta, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta};

use crate::{CKKSCtBounds, SetCKKSInfos, layouts::eval_mod::EvalMod};

/// Homomorphic modular reduction (`x mod 1`) via a periodic-function polynomial
/// approximation, the core non-linear step of CKKS bootstrapping.
///
/// The reduction is configured by an [`EvalMod`] (compiled from an
/// [`EvalModPlan`](crate::layouts::eval_mod::EvalModPlan)
/// and uploaded to this backend); see the [`eval_mod`](crate::reference::eval_mod)
/// module for the base-polynomial / range-extension / inverse pipeline.
pub trait CKKSEvalModOps<BE: Backend> {
    /// Workspace for two independent evaluations sharing parameters and a key provider.
    fn ckks_eval_mod_pair_tmp_bytes<R1, R2, C1, C2, P, F, T>(
        &self,
        left_out: &R1,
        right_out: &R2,
        left_in: &C1,
        right_in: &C2,
        params: &EvalMod<F, P>,
        tsk: &T,
    ) -> usize
    where
        R1: CKKSCtBounds,
        R2: CKKSCtBounds,
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
        P: CKKSCtBounds,
        T: poulpy_core::layouts::GGLWEInfos;

    /// Evaluates both inputs with the single-input EvalMod semantics and metadata rules.
    /// Inputs and outputs must be disjoint. On error either output may have changed.
    /// The default runs selected single-input operations with the backend's task executor.
    #[allow(clippy::too_many_arguments)]
    fn ckks_eval_mod_pair<R1, R2, C1, C2, P, F, H>(
        &self,
        left_out: &mut R1,
        right_out: &mut R2,
        left_in: &C1,
        right_in: &C2,
        params: &EvalMod<F, P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R1: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + Send,
        R2: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + Send,
        C1: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
        C2: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta + Sync,
        F: Sync,
        H: GetTensorKey<BE> + Sync;

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
    /// Consumes `params.eval_depth() * log_delta` bits of `log_budget`; errors if
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
}
