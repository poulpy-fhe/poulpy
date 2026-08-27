use crate::CKKSResult as Result;
use poulpy_core::layouts::GLWERelinearizationKeyHelper;
use poulpy_core::layouts::GLWERelinearizationKeyLayoutHelper;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::prepared::GGLWEPreparedToBackendRef;
use poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef;
use poulpy_hal::layouts::{Backend, ScratchArena};

use poulpy_core::layouts::{BSGSMeta, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta};

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
    /// shaped like `res` for input `ct` with relinearization key `tsk`.
    /// Pass the same `res`/`ct`/`params`/`tsk` you will evaluate with, since
    /// EvalMod raises the input to `params.plan.f_mod_log_delta` internally.
    fn ckks_eval_mod_tmp_bytes<R, C, P, F, H>(&self, res: &R, ct: &C, params: &EvalMod<F, P>, tsk: &H) -> usize
    where
        R: CKKSCtBounds,
        C: CKKSCtBounds,
        P: CKKSCtBounds,
        H: GLWERelinearizationKeyLayoutHelper;

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
        H: GLWERelinearizationKeyHelper,
        H::Key: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>;

    /// Scratch space, in bytes, required by [`Self::ckks_eval_mod_pair`].
    ///
    /// Takes both branches' layouts, since the operation itself accepts
    /// independently shaped branches: the sequential default is the larger of
    /// the two single budgets, while a fused backend sizes whatever it holds
    /// live across both.
    #[allow(clippy::too_many_arguments)]
    fn ckks_eval_mod_pair_tmp_bytes<R0, R1, C0, C1, P, F, H>(
        &self,
        res_0: &R0,
        res_1: &R1,
        ct_0: &C0,
        ct_1: &C1,
        params: &EvalMod<F, P>,
        tsk: &H,
    ) -> usize
    where
        R0: CKKSCtBounds,
        R1: CKKSCtBounds,
        C0: CKKSCtBounds,
        C1: CKKSCtBounds,
        P: CKKSCtBounds,
        H: GLWERelinearizationKeyLayoutHelper;

    /// Evaluates the same `x mod 1` approximation on two independent inputs:
    /// `res_0 = f(ct_0)` and `res_1 = f(ct_1)`.
    ///
    /// Semantically two [`Self::ckks_eval_mod`] calls, and that is exactly what
    /// the default does. The point of the paired form is the borrow shape: both
    /// inputs, both outputs, the tensor key and one scratch lifetime are held
    /// for the whole operation, so a backend override sees both evaluation DAGs
    /// at once and can stream the shared key material once, batch the matching
    /// transforms and normalizations, and schedule the two branches together.
    /// The two branches must not alias.
    #[allow(clippy::too_many_arguments)]
    fn ckks_eval_mod_pair<R0, R1, C0, C1, P, F, H>(
        &self,
        res_0: &mut R0,
        res_1: &mut R1,
        ct_0: &C0,
        ct_1: &C1,
        params: &EvalMod<F, P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R0: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        R1: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C0: GLWEToBackendRef<BE> + CKKSCtBounds,
        C1: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
        H: GLWERelinearizationKeyHelper,
        H::Key: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>;
}
